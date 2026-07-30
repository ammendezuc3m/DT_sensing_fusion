#!/usr/bin/env python3
"""
Headless IEEE 802.11 beacon CSI receiver for a USRP B210.

Receives IEEE 802.11a/g frames with gr-ieee802-11, selects beacon frames from
a configured SSID and BSSID, and atomically overwrites a JSON file with the
latest CSI estimate.

Default target:
    SSID:       PRACIT-AIAAS
    BSSID:      04:42:1A:E4:DD:D0
    Channel:    11
    Frequency:  2462 MHz
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pmt
from gnuradio import blocks, fft, gr, uhd
from gnuradio.fft import window
import ieee802_11


DEFAULT_SSID = "PRACIT-AIAAS"
DEFAULT_BSSID = "04:42:1A:E4:DD:D0"
DEFAULT_FREQUENCY_HZ = 2_462_000_000.0
DEFAULT_SAMPLE_RATE = 20_000_000.0
DEFAULT_GAIN = 0.75
DEFAULT_OUTPUT = "results/wifi_csi/latest_csi.json"
DEFAULT_DATASET_OUTPUT = "results/wifi_csi/csi_dataset.txt"


def normalize_mac(value: str) -> str:
    """Validate and normalize a MAC address to lower-case colon notation."""
    compact = value.replace(":", "").replace("-", "").strip().lower()
    if len(compact) != 12 or any(c not in "0123456789abcdef" for c in compact):
        raise argparse.ArgumentTypeError(f"BSSID no válido: {value!r}")
    return ":".join(compact[i : i + 2] for i in range(0, 12, 2))


def mac_to_text(raw: bytes) -> str:
    return ":".join(f"{byte:02x}" for byte in raw)


def pmt_blob_to_bytes(value: Any) -> bytes:
    """Convert the data half of a GNU Radio PDU to bytes."""
    if pmt.is_blob(value):
        # pmt.blob_data() exposes a C pointer as a PyCapsule in GNU Radio 3.10,
        # so it cannot be passed directly to bytes(). Convert the complete PMT
        # object through GNU Radio's Python conversion layer instead.
        converted = pmt.to_python(value)

        if isinstance(converted, bytes):
            return converted
        if isinstance(converted, (bytearray, memoryview)):
            return bytes(converted)
        if isinstance(converted, np.ndarray):
            return converted.astype(np.uint8, copy=False).tobytes()
        if isinstance(converted, (list, tuple)):
            return bytes(converted)

        raise TypeError(
            "pmt.to_python(blob) devolvió un tipo no soportado: "
            f"{type(converted).__name__}"
        )

    if pmt.is_u8vector(value):
        return bytes(pmt.u8vector_elements(value))

    raise TypeError(f"Payload PDU no soportado: {pmt.write_string(value)}")


def pmt_dict_value(meta: Any, key: str, default: Any = None) -> Any:
    key_pmt = pmt.intern(key)
    value = pmt.dict_ref(meta, key_pmt, pmt.PMT_NIL)
    if pmt.eq(value, pmt.PMT_NIL):
        return default
    return value


def pmt_number(meta: Any, key: str) -> Optional[float]:
    value = pmt_dict_value(meta, key)
    if value is None:
        return None

    try:
        if pmt.is_real(value):
            return float(pmt.to_double(value))
        if pmt.is_integer(value):
            return float(pmt.to_long(value))
        if pmt.is_uint64(value):
            return float(pmt.to_uint64(value))
    except Exception:
        return None

    return None


def parse_beacon(frame: bytes) -> Optional[dict[str, Any]]:
    """
    Parse the fields needed from an IEEE 802.11 beacon MAC frame.

    decode_mac publishes the PSDU without the FCS. A normal beacon has:
      24-byte MAC header
      12-byte fixed beacon parameters
      variable Information Elements
    """
    if len(frame) < 36:
        return None

    frame_control = int.from_bytes(frame[0:2], byteorder="little")
    frame_type = (frame_control >> 2) & 0b11
    frame_subtype = (frame_control >> 4) & 0b1111

    # Management frame (type 0), beacon subtype (8).
    if frame_type != 0 or frame_subtype != 8:
        return None

    destination = mac_to_text(frame[4:10])
    source = mac_to_text(frame[10:16])
    bssid = mac_to_text(frame[16:22])

    timestamp_tsf = int.from_bytes(frame[24:32], byteorder="little")
    beacon_interval_tu = int.from_bytes(frame[32:34], byteorder="little")
    capability_info = int.from_bytes(frame[34:36], byteorder="little")

    ssid: Optional[str] = None
    channel: Optional[int] = None
    information_elements: dict[int, bytes] = {}

    offset = 36
    while offset + 2 <= len(frame):
        element_id = frame[offset]
        element_length = frame[offset + 1]
        element_start = offset + 2
        element_end = element_start + element_length

        if element_end > len(frame):
            # Truncated or malformed Information Element.
            break

        element_value = frame[element_start:element_end]
        information_elements[element_id] = element_value

        if element_id == 0:
            ssid = element_value.decode("utf-8", errors="replace")
        elif element_id == 3 and element_length >= 1:
            channel = int(element_value[0])

        offset = element_end

    return {
        "ssid": ssid,
        "bssid": bssid,
        "source": source,
        "destination": destination,
        "channel": channel,
        "timestamp_tsf": timestamp_tsf,
        "beacon_interval_tu": beacon_interval_tu,
        "capability_info": capability_info,
    }


class BeaconCsiSink(gr.basic_block):
    """Message-only GNU Radio block that filters beacons and saves their CSI."""

    def __init__(
        self,
        target_ssid: str,
        target_bssid: str,
        configured_frequency_hz: float,
        configured_channel: int,
        output_path: Path,
        dataset_output_path: Path,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            name="beacon_csi_sink",
            in_sig=None,
            out_sig=None,
        )

        self.target_ssid = target_ssid
        self.target_bssid = normalize_mac(target_bssid)
        self.configured_frequency_hz = float(configured_frequency_hz)
        self.configured_channel = int(configured_channel)
        self.output_path = output_path
        self.dataset_output_path = dataset_output_path
        self.verbose = verbose

        self.total_valid_pdus = 0
        self.total_beacons = 0
        self.total_matching_bssid = 0
        self.total_matching_ssid = 0
        self.total_saved = 0

        self._lock = threading.Lock()
        self._port_in = pmt.intern("in")
        self.message_port_register_in(self._port_in)
        self.set_msg_handler(self._port_in, self._handle_pdu)

    def _handle_pdu(self, message: Any) -> None:
        try:
            if not pmt.is_pair(message):
                return

            meta = pmt.car(message)
            payload = pmt.cdr(message)

            if not pmt.is_dict(meta):
                return

            frame = pmt_blob_to_bytes(payload)
            self.total_valid_pdus += 1

            beacon = parse_beacon(frame)
            if beacon is None:
                return

            self.total_beacons += 1

            if self.verbose:
                print(
                    "[beacon] "
                    f"ssid={beacon['ssid']!r} "
                    f"bssid={beacon['bssid']} "
                    f"channel={beacon['channel']}",
                    flush=True,
                )

            if beacon["bssid"] != self.target_bssid:
                return

            self.total_matching_bssid += 1

            if beacon["ssid"] != self.target_ssid:
                print(
                    "[warning] BSSID coincidente, pero SSID diferente: "
                    f"{beacon['ssid']!r}",
                    file=sys.stderr,
                    flush=True,
                )
                return

            self.total_matching_ssid += 1

            csi_pmt = pmt_dict_value(meta, "csi")
            if csi_pmt is None or not pmt.is_c32vector(csi_pmt):
                print(
                    "[warning] Beacon objetivo recibido sin metadata['csi'].",
                    file=sys.stderr,
                    flush=True,
                )
                return

            csi = np.asarray(pmt.c32vector_elements(csi_pmt), dtype=np.complex64)
            if csi.size == 0:
                print("[warning] CSI vacío.", file=sys.stderr, flush=True)
                return

            amplitude = np.abs(csi)
            phase = np.angle(csi)

            nominal_frequency = pmt_number(meta, "nominal frequency")
            frequency_offset = pmt_number(meta, "frequency offset")
            snr = pmt_number(meta, "snr")
            beta = pmt_number(meta, "beta")
            encoding = pmt_number(meta, "encoding")
            frame_bytes = pmt_number(meta, "frame bytes")

            now_ns = time.time_ns()
            result = {
                "timestamp_ns": now_ns,
                "timestamp_unix_s": now_ns / 1_000_000_000,
                "ssid": beacon["ssid"],
                "bssid": beacon["bssid"],
                "source": beacon["source"],
                "destination": beacon["destination"],
                "channel": (
                    beacon["channel"]
                    if beacon["channel"] is not None
                    else self.configured_channel
                ),
                "frequency_hz": (
                    nominal_frequency
                    if nominal_frequency is not None
                    else self.configured_frequency_hz
                ),
                "configured_frequency_hz": self.configured_frequency_hz,
                "frequency_offset_hz": frequency_offset,
                "snr_db": snr,
                "beta": beta,
                "encoding": int(encoding) if encoding is not None else None,
                "frame_bytes": int(frame_bytes) if frame_bytes is not None else len(frame),
                "beacon_interval_tu": beacon["beacon_interval_tu"],
                "timestamp_tsf": beacon["timestamp_tsf"],
                "num_subcarriers": int(csi.size),
                "csi_real": csi.real.astype(float).tolist(),
                "csi_imag": csi.imag.astype(float).tolist(),
                "csi_amplitude": amplitude.astype(float).tolist(),
                "csi_phase": phase.astype(float).tolist(),
            }

            self._atomic_write_json(result)
            self._append_csi_line(result, csi)
            self.total_saved += 1

            print(
                "[saved] "
                f"beacon #{self.total_saved} | "
                f"{beacon['ssid']} | {beacon['bssid']} | "
                f"CSI={csi.size} subcarriers | "
                f"SNR={snr if snr is not None else 'n/a'} | "
                f"{self.output_path}",
                flush=True,
            )

        except Exception as exc:
            print(
                f"[error] No se pudo procesar un PDU: {exc}",
                file=sys.stderr,
                flush=True,
            )

    def _atomic_write_json(self, value: dict[str, Any]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            temp_fd, temp_name = tempfile.mkstemp(
                prefix=f".{self.output_path.name}.",
                suffix=".tmp",
                dir=str(self.output_path.parent),
                text=True,
            )
            try:
                with os.fdopen(temp_fd, "w", encoding="utf-8") as handle:
                    json.dump(value, handle, indent=2, ensure_ascii=False)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())

                os.replace(temp_name, self.output_path)
            except Exception:
                try:
                    os.unlink(temp_name)
                except FileNotFoundError:
                    pass
                raise

    def _append_csi_line(self, result: dict[str, Any], csi: np.ndarray) -> None:
        """
        Append one CSI sample per line.

        Format:
            timestamp_ns<TAB>ssid<TAB>bssid<TAB>snr_db<TAB>c0,c1,...,cN

        Each complex coefficient is written as:
            real+imagj
        """
        self.dataset_output_path.parent.mkdir(parents=True, exist_ok=True)

        csi_text = ",".join(
            f"{float(value.real):.9g}{float(value.imag):+.9g}j"
            for value in csi
        )
        snr = result["snr_db"]
        snr_text = "" if snr is None else f"{float(snr):.9g}"

        line = (
            f"{result['timestamp_ns']}\t"
            f"{result['ssid']}\t"
            f"{result['bssid']}\t"
            f"{snr_text}\t"
            f"{csi_text}\n"
        )

        with self._lock:
            with self.dataset_output_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())

    def print_summary(self) -> None:
        print(
            "\nResumen:"
            f"\n  PDUs MAC válidos:        {self.total_valid_pdus}"
            f"\n  Beacons detectados:      {self.total_beacons}"
            f"\n  BSSID coincidente:       {self.total_matching_bssid}"
            f"\n  SSID+BSSID coincidentes: {self.total_matching_ssid}"
            f"\n  CSI guardados:           {self.total_saved}",
            flush=True,
        )


class WifiBeaconCsiReceiver(gr.top_block):
    """Headless form of the official gr-ieee802-11 wifi_rx flowgraph."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("WiFi beacon CSI receiver", catch_exceptions=True)

        self.frequency = float(args.frequency)
        self.sample_rate = float(args.sample_rate)
        self.gain = float(args.gain)
        self.lo_offset = float(args.lo_offset)
        self.sync_length = int(args.sync_length)
        self.window_size = int(args.window_size)

        # USRP source: same basic configuration as the official wifi_rx example.
        self.usrp_source = uhd.usrp_source(
            args.device_args,
            uhd.stream_args(
                cpu_format="fc32",
                args="",
                channels=[int(args.channel_index)],
            ),
        )
        self.usrp_source.set_samp_rate(self.sample_rate)
        self.usrp_source.set_time_unknown_pps(uhd.time_spec(0))

        tune_request = uhd.tune_request(
            self.frequency,
            rf_freq=self.frequency - self.lo_offset,
            rf_freq_policy=uhd.tune_request.POLICY_MANUAL,
        )
        self.usrp_source.set_center_freq(tune_request, 0)
        self.usrp_source.set_normalized_gain(self.gain, 0)

        if args.antenna:
            self.usrp_source.set_antenna(args.antenna, 0)

        if args.bandwidth > 0:
            self.usrp_source.set_bandwidth(float(args.bandwidth), 0)

        # Detection and synchronization chain copied from wifi_rx.
        self.complex_to_mag_squared = blocks.complex_to_mag_squared(1)
        self.delay_16 = blocks.delay(gr.sizeof_gr_complex, 16)
        self.conjugate = blocks.conjugate_cc()
        self.multiply = blocks.multiply_vcc(1)

        self.moving_average_power = blocks.moving_average_ff(
            self.window_size + 16,
            1,
            4000,
            1,
        )
        self.moving_average_corr = blocks.moving_average_cc(
            self.window_size,
            1,
            4000,
            1,
        )
        self.complex_to_mag = blocks.complex_to_mag(1)
        self.divide = blocks.divide_ff(1)

        self.sync_short = ieee802_11.sync_short(0.56, 2, False, False)
        self.delay_sync = blocks.delay(gr.sizeof_gr_complex, self.sync_length)
        self.sync_long = ieee802_11.sync_long(
            self.sync_length,
            False,
            False,
        )

        self.stream_to_vector = blocks.stream_to_vector(
            gr.sizeof_gr_complex,
            64,
        )
        self.fft = fft.fft_vcc(
            64,
            True,
            window.rectangular(64),
            True,
            1,
        )
        self.frame_equalizer = ieee802_11.frame_equalizer(
            ieee802_11.Equalizer(int(args.channel_estimator)),
            self.frequency,
            self.sample_rate,
            False,
            False,
        )
        self.decode_mac = ieee802_11.decode_mac(
            bool(args.log_mac),
            bool(args.debug_mac),
        )

        self.beacon_sink = BeaconCsiSink(
            target_ssid=args.ssid,
            target_bssid=args.bssid,
            configured_frequency_hz=self.frequency,
            configured_channel=int(args.wifi_channel),
            output_path=Path(args.output).expanduser().resolve(),
            dataset_output_path=Path(args.dataset_output).expanduser().resolve(),
            verbose=bool(args.verbose),
        )

        # Correlation / power detector.
        self.connect(self.usrp_source, self.complex_to_mag_squared)
        self.connect(self.complex_to_mag_squared, self.moving_average_power)
        self.connect(self.moving_average_power, (self.divide, 1))

        self.connect(self.usrp_source, self.delay_16)
        self.connect(self.delay_16, self.conjugate)
        self.connect((self.conjugate, 0), (self.multiply, 1))

        self.connect((self.usrp_source, 0), (self.multiply, 0))
        self.connect(self.multiply, self.moving_average_corr)
        self.connect(self.moving_average_corr, self.complex_to_mag)
        self.connect(self.complex_to_mag, (self.divide, 0))

        # 802.11 synchronization and decode.
        self.connect((self.delay_16, 0), (self.sync_short, 0))
        self.connect((self.moving_average_corr, 0), (self.sync_short, 1))
        self.connect((self.divide, 0), (self.sync_short, 2))

        self.connect((self.sync_short, 0), (self.delay_sync, 0))
        self.connect((self.sync_short, 0), (self.sync_long, 0))
        self.connect((self.delay_sync, 0), (self.sync_long, 1))

        self.connect(self.sync_long, self.stream_to_vector)
        self.connect(self.stream_to_vector, self.fft)
        self.connect(self.fft, self.frame_equalizer)
        self.connect(self.frame_equalizer, self.decode_mac)

        # decode_mac publishes one PDU containing both metadata (including CSI)
        # and the decoded IEEE 802.11 MAC frame.
        self.msg_connect(
            (self.decode_mac, "out"),
            (self.beacon_sink, "in"),
        )


def channel_to_frequency_hz(channel: int) -> float:
    """Map 2.4 GHz Wi-Fi channels to their centre frequency."""
    if 1 <= channel <= 13:
        return float((2407 + 5 * channel) * 1_000_000)
    if channel == 14:
        return 2_484_000_000.0
    raise argparse.ArgumentTypeError(
        "Este script solo calcula automáticamente canales 2.4 GHz (1-14). "
        "Para otros canales usa --frequency."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recibe beacons IEEE 802.11 con una USRP, filtra por SSID/BSSID "
            "y guarda el CSI de la última trama coincidente."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--ssid", default=DEFAULT_SSID)
    parser.add_argument("--bssid", type=normalize_mac, default=DEFAULT_BSSID)
    parser.add_argument("--wifi-channel", type=int, default=11)
    parser.add_argument(
        "--frequency",
        type=float,
        default=None,
        help="Frecuencia central en Hz. Si se omite se calcula desde --wifi-channel.",
    )
    parser.add_argument("--sample-rate", type=float, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument(
        "--gain",
        type=float,
        default=DEFAULT_GAIN,
        help="Ganancia normalizada UHD, entre 0.0 y 1.0.",
    )
    parser.add_argument("--lo-offset", type=float, default=0.0)
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=0.0,
        help="Ancho de banda RF UHD en Hz; 0 conserva el valor automático.",
    )
    parser.add_argument("--antenna", default="")
    parser.add_argument("--device-args", default="")
    parser.add_argument("--channel-index", type=int, default=0)
    parser.add_argument("--channel-estimator", type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument("--sync-length", type=int, default=320)
    parser.add_argument("--window-size", type=int, default=48)
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="JSON sobrescrito con el CSI más reciente.",
    )
    parser.add_argument(
        "--dataset-output",
        default=DEFAULT_DATASET_OUTPUT,
        help="TXT acumulativo: una línea por cada vector CSI.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log-mac", action="store_true")
    parser.add_argument("--debug-mac", action="store_true")

    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.gain <= 1.0:
        raise SystemExit("--gain debe estar entre 0.0 y 1.0.")

    if args.sample_rate not in (5_000_000.0, 10_000_000.0, 20_000_000.0):
        raise SystemExit(
            "--sample-rate debe ser 5000000, 10000000 o 20000000, "
            "igual que en el ejemplo oficial."
        )

    if args.frequency is None:
        args.frequency = channel_to_frequency_hz(args.wifi_channel)

    args.bssid = normalize_mac(args.bssid)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    receiver = WifiBeaconCsiReceiver(args)
    stop_event = threading.Event()

    def request_stop(signum: int, _frame: Any) -> None:
        print(f"\n[stop] Señal {signum}; deteniendo receptor...", flush=True)
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    print(
        "Iniciando receptor WiFi CSI:"
        f"\n  SSID:          {args.ssid}"
        f"\n  BSSID:         {args.bssid}"
        f"\n  Canal WiFi:    {args.wifi_channel}"
        f"\n  Frecuencia:    {args.frequency:.0f} Hz"
        f"\n  Sample rate:   {args.sample_rate:.0f} S/s"
        f"\n  Ganancia:      {args.gain:.2f} normalizada"
        f"\n  Último CSI:    {Path(args.output).expanduser().resolve()}"
        f"\n  Dataset CSI:   {Path(args.dataset_output).expanduser().resolve()}"
        "\nPulsa Ctrl+C para detener.\n",
        flush=True,
    )

    try:
        receiver.start()
        while not stop_event.wait(timeout=0.5):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        receiver.stop()
        receiver.wait()
        receiver.beacon_sink.print_summary()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
