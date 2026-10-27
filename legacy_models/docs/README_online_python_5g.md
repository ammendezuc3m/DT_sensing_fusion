# Online Python 5G SSB Pipeline

This document complements the existing repository documentation with the latest Python-only online path developed after the MATLAB-to-Python migration.

## Current status

The current working online chain is:

```text
USRP B210
  -> 20 ms IQ capture
  -> CFO warmup and correction
  -> PSS / NID2 / timing detection
  -> OFDM demodulation
  -> dataSSB   = 360 x 6
  -> rxGridSSB = 240 x 4
  -> PyTorch binary classifier (.pt)
  -> local JSON
  -> optional Mitsuba/Sionna XML
  -> optional SCP to the Digital Twin machine
```

Main online script:

```text
src/python/ssb_python/online_5g_python_cfo_json_scp.py
```

PyTorch model loader:

```text
src/python/ssb_python/rxgrid_torch_inference.py
```

Current checkpoint:

```text
results/binary_empty_vs_P5_rx/model_rxGridSSB/model.pt
```

The `.pt` is a PyTorch checkpoint, not TorchScript. It contains:

```text
model_state_dict
mean
std
model_name
input_shape = [2, 240, 4]
complex_mode = abs_phase
classes = ["empty", "P5"]
config
```

The model input is:

```text
[2, 240, 4]
channel 0 = abs(rxGridSSB)
channel 1 = angle(rxGridSSB)
```

## Environment

Activate the UHD environment:

```bash
cd DT_sensing_fusion
source .venv_uhd/bin/activate
```

Check UHD:

```bash
python - <<'PY'
import uhd
print("uhd OK")
PY
```

Check USRP:

```bash
uhd_find_devices
```

Expected device:

```text
USRP B210
serial: 34B73C3
```

If PyTorch is missing in `.venv_uhd`:

```bash
python -m pip install torch
```

## Short local test without SCP

```bash
cd DT_sensing_fusion
source .venv_uhd/bin/activate

python src/python/ssb_python/online_5g_python_cfo_json_scp.py \
  --serial 34B73C3 \
  --freq 3541.44e6 \
  --rate 15.36e6 \
  --gain 60 \
  --duration-ms 20 \
  --num-iters 30 \
  --warmup-iters 5 \
  --channel 0 \
  --force-nid2 0 \
  --enable-cfo-correction \
  --cfo-warmup-iters 30 \
  --cfo-correction-sign -1 \
  --inference-backend torch \
  --torch-model results/binary_empty_vs_P5_rx/model_rxGridSSB/model.pt \
  --torch-device cpu \
  --disable-scp \
  --progress-every 1
```

Check the generated JSON:

```bash
cat results/online/live_inference_state_5G.json
```

## Online run with SCP to the Digital Twin

Prepare the remote directory:

```bash
ssh <user>@<ip-or-hostname> \
  "mkdir -p <remote/path>"
```

Run:

```bash
cd DT_sensing_fusion
source .venv_uhd/bin/activate

python src/python/ssb_python/online_5g_python_cfo_json_scp.py \
  --serial 34B73C3 \
  --freq 3541.44e6 \
  --rate 15.36e6 \
  --gain 60 \
  --duration-ms 20 \
  --warmup-iters 5 \
  --channel 0 \
  --force-nid2 0 \
  --enable-cfo-correction \
  --cfo-warmup-iters 30 \
  --cfo-correction-sign -1 \
  --inference-backend torch \
  --torch-model results/binary_empty_vs_P5_rx/model_rxGridSSB/model.pt \
  --torch-device cpu \
  --remote-target "<user>@<ip-or-hostname>:<remote/path/live_inference_state_5G.json>" \
  --scp-every 1 \
  --progress-every 1
```

Watch the remote JSON:

```bash
watch -n 0.5 "ssh <user>@<ip-or-hostname> 'cat <remote/path/live_inference_state_5G.json>'"
```

If SCP is too slow, use:

```bash
--scp-every 3
```

or:

```bash
--scp-every 5
```

## Optional Mitsuba/Sionna XML export

To generate the XML scene as well, add:

```bash
--mitsuba-position-map-json config/sionna_mitsuba_position_map.json \
--enable-mitsuba-export
```

If SCP is enabled through `--remote-target`, the XML is sent automatically to the same remote directory as:

```text
live_person_sionna_scene.xml
```

Example target pair:

```text
<user>@<ip-or-hostname>:<remote/path/live_inference_state_5G.json>
<user>@<ip-or-hostname>:<remote/path/live_person_sionna_scene.xml>
```

## CFO behavior

The Python pipeline now applies CFO correction.

Typical warmup result:

```text
CFO median applied ~= -4.8 kHz to -5.2 kHz
```

Before CFO correction, PSS metric was usually:

```text
0.80-0.85
```

After CFO correction, PSS metric usually rises to:

```text
0.90-0.95
```

The current correction sign is:

```bash
--cfo-correction-sign -1
```

## Current timing

Typical timing with CFO + PyTorch inference:

```text
20 ms RF capture
~19-30 ms PSS/timing
~0.15 ms OFDM extraction
~50-60 ms total loop
~16-20 rxGridSSB/s
```

## Threshold fallback model

A simple threshold debug model exists:

```text
config/generic_5g_binary_model.json
```

It is not the real model. Use it only to test JSON/SCP:

```bash
--inference-backend threshold \
--model-config config/generic_5g_binary_model.json
```

For sensing, use:

```bash
--inference-backend torch \
--torch-model results/binary_empty_vs_P5_rx/model_rxGridSSB/model.pt
```

## Important note

The practical goal is not bit-exact MATLAB equivalence. The practical goal is:

```text
stable Python extraction
consistent rxGridSSB representation
online inference from the same Python feature pipeline
```

For future models, the preferred workflow is:

```text
capture Python rxGridSSB datasets
train on Python rxGridSSB datasets
infer online with the same Python extraction chain
```
