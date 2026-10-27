# DT_sensing_fusion

Passive 5G SSB sensing with a USRP B210, Python UHD processing, a Fisher + logistic-regression fingerprinting model and a CAPIF-exposed Sensing API.

This repository implements the **5G SSB and WiFi CSI sensing components** of a larger Digital Twin system. It receives 5G Synchronization Signal Blocks (SSBs), extracts an `rxGridSSB` representation, estimates the radio channel, classifies the scene with a model trained on your own labelled captures, converts the predicted class into objects with coordinates, and **publishes them through the Sensing API**: a REST API registered in a CAPIF Core Function (3GPP TS 29.222) so that any consumer (Digital Twin, Sionna, dashboards) can discover it and read the results.

The repository also includes tools for:

- collecting labeled 5G sensing datasets;
- analysing a dataset and training the Fisher + logistic-regression model;
- replaying a trained checkpoint on saved datasets;
- running the CAPIF Core Function and the Sensing API;
- consuming the Sensing API (polling or streaming) and converting results, for example to a Mitsuba/Sionna XML scene;
- comparing the current Python processing with the historical MATLAB reference workflow;
- collecting WiFi CSI from IEEE 802.11 OFDM beacon frames.

> [!IMPORTANT]
> A fingerprinting model is only valid for the deployment where its training data was captured: the same room, receiver and antenna placement, 5G cell and set of positions. It is not a general human detector. Collect a new dataset and retrain for every new deployment.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contents

1. [System architecture](#system-architecture)
2. [Recommended 5G SSB workflow](#recommended-5g-ssb-workflow)
3. [Initial setup — fresh computer](#initial-setup--fresh-computer)
4. [Collect a labeled 5G sensing dataset](#collect-a-labeled-5g-sensing-dataset)
5. [Analyze a collected dataset](#analyze-a-collected-dataset) and [train the model](#analyze-and-train-the-fisher-ssb-sensing-model)
6. [Map model labels to coordinates](#map-model-labels-to-objects-and-coordinates)
7. [Sensing API and CAPIF](#sensing-api-and-capif)
8. [Run online sensing](#run-online-5g-ssb-sensing)
9. [Consume the Sensing API](#consume-the-sensing-api)
10. [Mitsuba/Sionna XML export](#mitsubasionna-xml-export)
11. [WiFi CSI](#wifi-csi-receiver-files-and-recommended-execution)
12. [Main scripts](#main-scripts), [tests](#tests), [troubleshooting](#troubleshooting)

---

## System architecture

```text
Sensing computer
  USRP B210
    → Python UHD IQ capture
    → CFO estimation and correction
    → PSS/NID2 detection and timing synchronization
    → OFDM demodulation → dataSSB → rxGridSSB [240 x 4]
    → radio check: NID2, PCI and i_bar_SSB must match the model
    → channel estimate H (SSS + PBCH DM-RS) → Fisher features → PyTorch linear model
    → label → objects + coordinates                 (config/sensing_label_map.json)
    → POST /sensing/v1/reports  ──────────►  Sensing API (CAPIF AEF, :8090)
    → local latest-state JSON + CSV log                │ registers "sensing-api" in
    → optional local Mitsuba XML                       ▼
                                             CAPIF Core Function (:8080)
                                                       ▲
Consumers (Digital Twin, Sionna, dashboards)           │ 1. enrol as API invoker
  ─────────────────────────────────────────────────────┘ 2. discover "sensing-api" + routing info
  3. GET /sensing/v1/detections   or   GET /sensing/v1/stream  (Server-Sent Events)
```

Each published detection contains **object type, coordinates, confidence and the sensor that produced it** (for this pipeline: `sensor_type = 5g_ssb`, model `ssb_fisher_logreg`). Consumers convert this standard JSON into whatever representation they need; the repository includes a converter to Mitsuba XML as an example.

Main components:

| Component | Location | Role |
|---|---|---|
| Offline analysis and training | `src/python/run_ssb_sensing_analysis.py` | Analyses a labelled dataset, trains the Fisher + logistic-regression model and exports `model_fisher_logreg.pt` |
| Shared model frontend | `src/python/ssb_python/ssb_fisher_torch_inference.py` | Channel estimation, features, radio check and PyTorch model, shared by training and online inference |
| Online pipeline | `src/python/ssb_python/online_5g_ssb_sensing.py` | Capture → DSP → inference → Sensing API |
| Label map | `config/sensing_label_map.json` | Model label → objects and coordinates |
| CAPIF Core Function | `src/python/sensing_api/capif_core.py` | Provider/invoker enrolment, API publication, discovery and routing |
| Sensing API (AEF) | `src/python/sensing_api/server.py` | Receives reports and serves detections, history and an event stream |
| Example consumer | `src/python/sensing_api/consume.py` | CAPIF discovery + polling/streaming + JSON/Mitsuba conversion |

## Recommended 5G SSB workflow

For a new sensing deployment, follow this order:

```text
1. Capture labelled SSB sessions (one folder per label)
   collect_labeled_rxgridssb_dataset_cfo.py
          ↓
2. Analyse and train offline (labels discovered automatically from the folders)
   run_ssb_sensing_analysis.py  →  model_fisher_logreg.pt
          ↓
3. Replay-test the checkpoint on saved H5 data
   replay_fisher_checkpoint_on_h5.py
          ↓
4. Give every label its objects and coordinates
   config/sensing_label_map.json
          ↓
5. Start the CAPIF Core Function and the Sensing API
   python -m sensing_api.capif_core  /  python -m sensing_api.server
          ↓
6. Run online inference and publish
   online_5g_ssb_sensing.py
          ↓
7. Consumers discover the Sensing API through CAPIF and read the detections
```

---

## Repository scope

This repository currently contains the **5G sensing, inference and publication side**, together with an experimental WiFi CSI acquisition component.

It does include:

- USRP B210 capture;
- Python-only 5G SSB processing;
- CFO correction;
- `dataSSB` and `rxGridSSB` extraction;
- dataset collection and analysis;
- Fisher + logistic-regression training and PyTorch deployment;
- online inference with per-grid radio validation;
- the label-to-coordinates map;
- a CAPIF Core Function and the CAPIF-exposed Sensing API;
- an example Sensing API consumer;
- local JSON and CSV outputs;
- Mitsuba/Sionna XML generation;
- historical MATLAB validation utilities.

It does not currently include:

- the complete Isaac Sim application;
- Digital Twin scene update logic;
- MikroTik integration;
- complete orchestration of every Digital Twin component.

Older models (the binary CNN, threshold backend, ROI/hybrid experiments) and the SCP delivery scripts are kept for reference in [`legacy_models/`](legacy_models/README.md). They are not used by the current pipeline.

---

# Initial setup — fresh computer

Follow these steps in order on a new Ubuntu computer.

## Step 1 — Connect the USRP B210

1. Connect the USRP B210 to the computer using a USB 3 cable.
2. Connect the receiving antenna to the physical `RX2` port.
3. Do not start the sensing script yet.

A USB 3 connection is important because the receiver must continuously transfer large IQ sample blocks to the computer.

---

## Step 2 — Clone the repository

Run:

```bash
git clone https://github.com/ammendezuc3m/DT_sensing_fusion.git
cd DT_sensing_fusion
```

All commands in this guide must be executed from the repository root unless otherwise stated.

You can verify the current directory with:

```bash
pwd
```

The final part of the displayed path should be:

```text
DT_sensing_fusion
```

---

## Step 3 — Install system dependencies

Run:

```bash
sudo apt update

sudo apt install -y \
  git \
  python3 \
  python3-pip \
  python3-venv \
  build-essential \
  cmake \
  pkg-config \
  gnuradio \
  gnuradio-dev \
  libboost-all-dev \
  libcppunit-dev \
  liblog4cpp5-dev \
  libvolk-dev \
  pybind11-dev \
  uhd-host \
  python3-uhd
```

These packages install:

- Git;
- Python 3;
- `pip`;
- support for Python virtual environments;
- build tools required by GNU Radio out-of-tree modules;
- GNU Radio and development headers;
- the UHD command-line utilities;
- the UHD Python bindings required to control the USRP.

The UHD Python bindings are provided by Ubuntu through:

```text
python3-uhd
```

They are not normally installed through the project `requirements.txt`.

---

## Step 4 — Create and activate the Python environment

Create the virtual environment with access to the system Python packages:

```bash
python3 -m venv --system-site-packages .venv_uhd
```

Activate it:

```bash
source .venv_uhd/bin/activate
```

Upgrade `pip`:

```bash
python -m pip install --upgrade pip
```

After activation, the terminal prompt should begin with:

```text
(.venv_uhd)
```

The environment must be activated again whenever a new terminal is opened:

```bash
cd DT_sensing_fusion
source .venv_uhd/bin/activate
```

The `--system-site-packages` option is required because the UHD Python bindings are installed through Ubuntu using `python3-uhd`, rather than through `pip`.

---

## Step 5 — Install Python dependencies

Install all project Python dependencies using the single repository requirements file:

```bash
python -m pip install -r requirements.txt
```

The project must maintain only one Python requirements file so that a fresh computer can be prepared using a single command.

The UHD Python package itself is not installed through this file. It is provided by:

```text
python3-uhd
```

and made available to the virtual environment through:

```text
--system-site-packages
```

---

## Step 6 — Install the GNU Radio WiFi modules

The WiFi CSI receiver requires the GNU Radio Out-Of-Tree modules `gr-foo` and `gr-ieee802-11`.

Install them outside the repository to avoid adding third-party source trees to the project:

```bash
mkdir -p ~/src
cd ~/src
```

Install `gr-foo`:

```bash
git clone https://github.com/bastibl/gr-foo.git
cd gr-foo
git checkout maint-3.10

mkdir -p build
cd build

cmake ..
make -j"$(nproc)"
sudo make install
sudo ldconfig
```

Install `gr-ieee802-11`:

```bash
cd ~/src

git clone https://github.com/bastibl/gr-ieee802-11.git
cd gr-ieee802-11
git checkout maint-3.10

mkdir -p build
cd build

cmake ..
make -j"$(nproc)"
sudo make install
sudo ldconfig
```

Return to the repository root:

```bash
cd <dir_root>/DT_sensing_fusion
```

Replace `<dir_root>` with the directory where the repository was cloned.

Verify the GNU Radio WiFi dependencies:

```bash
python3 - <<'PY'
from gnuradio import blocks, fft, gr, uhd
import ieee802_11
import pmt

assert hasattr(ieee802_11, "frame_equalizer")
assert hasattr(ieee802_11, "decode_mac")

print("GNU Radio WiFi dependencies: OK")
PY
```

---

## Step 7 — Verify the Python UHD installation

Run:

```bash
python - <<'PY'
import uhd
print("uhd OK")
PY
```

A correct installation should produce:

```text
uhd OK
```

If the import fails, verify that:

- `python3-uhd` is installed;
- the virtual environment was created using `--system-site-packages`;
- the correct virtual environment is active.

Check the active Python executable with:

```bash
which python
```

It should point to a path inside:

```text
DT_sensing_fusion/.venv_uhd/
```

---

## Step 8 — Verify that UHD detects the USRP

Run:

```bash
uhd_find_devices
```

A correctly connected device should appear as a B210 and include information similar to:

```text
Device Address:
    serial: <serial_id>
    product: B210
    type: b200
```

Save the value shown next to:

```text
serial:
```

This identifier will be used later with:

```text
--serial <serial_id>
```

The exact serial number must not be hard-coded in the documentation because it is different for each USRP.

If no device is detected, verify:

- that the USRP is powered through the USB connection;
- that a USB 3 cable and USB 3 port are being used;
- that no other process is currently using the device;
- that the UHD packages were installed correctly.

You can inspect USB devices with:

```bash
lsusb
```

---

## Step 9 — Probe the USRP

Run:

```bash
uhd_usrp_probe
```

This command performs a more complete hardware and communication check.

A successful probe should confirm:

- `Detected Device: B210`;
- `Operating over USB 3`;
- successful register loopback tests;
- firmware and FPGA versions;
- two RX DSP channels;
- two TX DSP channels;
- RX frequency support;
- RX gain support;
- the available RX antenna ports `TX/RX` and `RX2`;
- no hardware or UHD communication errors.

The message:

```text
No GPSDO found
```

is not an error when no external GPSDO module is installed.

The RX frontend information should list:

```text
Antennas: TX/RX, RX2
```

This confirms that the physical `RX2` connector can be selected by the sensing script.

---

## Step 10 — Run a complete local 5G hardware test

This test verifies that the USRP, the 5G processing chain and the local outputs work **before any model exists**. It runs the online script in `--dsp-only` mode: every block is captured and processed, but no classifier is applied and nothing is published.

```text
USRP B210
  → IQ capture
  → CFO estimation and correction
  → PSS detection and timing synchronization
  → OFDM demodulation
  → dataSSB extraction
  → rxGridSSB extraction
  → validity checks
  → local JSON output
  → local CSV log
```

The Sensing API and Mitsuba export are disabled during this first test, so the radio chain can be validated on its own.

Replace `<serial_id>` with the serial number reported by `uhd_find_devices`:

```bash
python src/python/ssb_python/online_5g_ssb_sensing.py \
  --serial <serial_id> \
  --freq 3541.44e6 \
  --rate 15.36e6 \
  --gain 60 \
  --duration-ms 20 \
  --num-iters 30 \
  --warmup-iters 5 \
  --channel 0 \
  --antenna RX2 \
  --force-nid2 0 \
  --enable-cfo-correction \
  --cfo-warmup-iters 30 \
  --cfo-correction-sign -1 \
  --dsp-only \
  --label-map config/sensing_label_map.json \
  --disable-api \
  --progress-every 1
```

The command must be executed from the repository root because the configuration and output locations use relative paths.

### USRP and capture parameters

- `--serial <serial_id>` selects the USRP using its unique serial number.
- `--freq 3541.44e6` sets the receiver centre frequency to 3541.44 MHz.
- `--rate 15.36e6` requests a sampling rate of 15.36 MS/s.
- `--gain 60` applies 60 dB of RX gain.
- `--duration-ms 20` captures a block containing 20 ms of IQ samples.
- `--channel 0` selects logical RX channel 0.
- `--antenna RX2` selects the physical RX2 connector for that channel.

`--channel 0` and `--antenna RX2` do not represent the same setting:

- `--channel 0` selects the logical receive chain;
- `--antenna RX2` selects the physical antenna input used by that receive chain.

The frequency, sample rate, gain and NID2 values are specific to the current 5G deployment. They may need to be changed when receiving a different 5G cell or using another experimental configuration. Use [Automatic SSB Discovery](#automatic-ssb-discovery) when they are unknown.

### 5G processing parameters

- `--force-nid2 0` uses the expected PSS sequence corresponding to NID2 0.
- `--enable-cfo-correction` enables carrier frequency offset estimation and correction.
- `--cfo-warmup-iters 30` captures 30 initial blocks to obtain a stable CFO estimate.
- `--cfo-correction-sign -1` applies the estimated CFO using the sign expected by the current processing pipeline.

### Execution and output parameters

- `--num-iters 30` performs 30 sensing iterations (`0` runs forever).
- `--warmup-iters 5` discards the first 5 captured blocks before the loop starts, while the receiver settles.
- `--dsp-only` runs without a model.
- `--disable-api` does not publish to the Sensing API.
- `--progress-every 1` prints the result of every iteration.

### How to verify that the test succeeded

At startup, the script should report information similar to:

```text
=== Online 5G SSB sensing ===
model:              none (--dsp-only)
label map:          config/sensing_label_map.json (frame=world)
sensor id:          5g-ssb-rx-01
Sensing API:        disabled
local JSON:         results/online/live_inference_state_5G.json
mitsuba export:     disabled
```

The USRP configuration should report:

```text
Detected Device: B210
Operating over USB 3
```

It should also display the requested and actual values for:

- sampling rate;
- centre frequency;
- RX gain.

The requested and actual values should be identical or very close.

During CFO warmup, the script should produce valid estimates:

```text
=== CFO warmup result ===
valid estimates: <valid_estimates>/<total_estimates>
CFO median applied: <estimated_cfo> Hz
```

Ideally, all or most warmup iterations should be valid.

A non-zero CFO estimate is normal. It represents the frequency offset between the transmitter and receiver oscillators.

After warmup, the script enters the online loop. Each iteration produces a line similar to:

```text
[<iteration>] valid=1 label=NONE conf=- objs=0 rx_mean=<mean_amplitude> pss=<pss_metric> loop=<loop_time_ms> ms api=off err=
```

In `--dsp-only` mode `label` is `NONE` because no classifier runs; `valid=1` means the block passed the signal-processing checks.

The fields mean:

- `valid=1`: the captured block passed validation (and, with a model, the radio check).
- `label`: class predicted by the model; `NONE` in `--dsp-only` mode and `INVALID` when the iteration is invalid.
- `conf`: confidence associated with the selected class.
- `objs`: number of objects published for this iteration (from the label map).
- `rx_mean`: mean magnitude of the extracted `rxGridSSB`.
- `pss`: PSS detection metric.
- `loop`: total capture, processing, inference and output time for the iteration.
- `api=<sent>/<failed>`: reports delivered to / rejected by the Sensing API (`api=off` when disabled).
- `err=`: reason why an iteration is invalid; empty when there is no error.

A block is considered valid when:

- its PSS metric is greater than or equal to the configured minimum (`--min-pss-metric`, default 0.5);
- the expected six OFDM symbols are extracted;
- the generated `rxGridSSB` has shape `240 × 4`;
- with a model: NID2, PCI and i_bar_SSB match the model (see [the radio check](#radio-check-pci-and-i_bar_ssb)).

The complete test can be considered successful when:

- the B210 is detected and configured;
- USB 3 operation is confirmed;
- CFO warmup obtains valid estimates;
- the sensing iterations show `valid=1`;
- the PSS metric remains above the configured minimum;
- `err=` remains empty;
- the local JSON file is created;
- the CSV log is created.

### Prediction rate

The prediction period is represented by:

```text
loop=<loop_time_ms> ms
```

For example:

```text
loop=52.00 ms
```

means that the complete pipeline is producing approximately:

```text
1000 / 52.00 ≈ 19.2 predictions per second
```

This is the complete loop time, not only the model inference time. It includes capture, 5G processing, model inference and local output. Publishing to the Sensing API runs in a background thread and is not part of the loop time.

---

### Inspect the generated JSON

Run:

```bash
python -m json.tool results/online/live_inference_state_5G.json
```

The JSON contains the latest sensing state. It is replaced atomically during each iteration rather than storing every previous result.

After a test with 30 iterations, the latest state should normally contain:

```json
"iteration": 29
```

Top-level fields (`schema_version: python_5g_ssb_online_v2`):

- `timestamp_utc`;
- `iteration`;
- `valid`;
- `error`;
- `sensor`: sensor id, type (`5g_ssb`), technology, model and radio configuration;
- `label`, `confidence`, `probabilities`;
- `detections`: the same objects that are published to the Sensing API;
- `dsp`: detected NID2, PSS metric, timing offset and number of OFDM symbols;
- `radio_check`: expected and detected PCI and i_bar_SSB (with a model);
- `timing_ms`: capture, PSS, OFDM, DSP, inference and loop times.

A valid result should contain:

```json
"valid": true,
"error": ""
```

---

### Inspect the complete iteration log

Run:

```bash
cat results/online/python_5g_online_inference_log.csv
```

The CSV stores one row for every sensing iteration.

It includes:

- iteration number and UTC timestamp;
- validation result;
- predicted label and confidence;
- probability of the empty labels (`prob_empty`);
- number of published objects;
- mean `rxGridSSB` magnitude;
- PSS metric, NID2, detected PCI and detected i_bar_SSB;
- applied CFO;
- capture, PSS, OFDM, DSP, inference and loop times;
- Sensing API delivery counters;
- errors.

Unlike the JSON, the current CSV log is appended to rather than replaced. Repeated executions may therefore add new rows to the existing file.

At this point, the initial local setup and complete local radio validation are finished.

---

# Collect a labeled 5G sensing dataset

Every deployment needs its own labelled dataset: one class per scene state to recognise, for example `empty` and one class per target position (`P1`, `P2`, ...). The class name is the `--label` of the session and becomes the folder name under `--output-root`; the training script later discovers the classes from those folder names, so **use exactly the same label spelling in every session** and in the [label map](#map-model-labels-to-objects-and-coordinates).

Main dataset collection script:

```text
src/python/ssb_python/collect_labeled_rxgridssb_dataset_cfo.py
```

The script:

1. shows a preparation countdown;
2. performs CFO warmup;
3. continuously captures and processes SSB blocks;
4. accepts valid `dataSSB` and `rxGridSSB` samples;
5. writes an H5 dataset;
6. writes human-readable metadata;
7. writes a CSV capture log.

## Dataset parameter meanings

The most important labeling parameters are:

- `--label`: class assigned to every accepted sample in the session;
- `--scene`: description of the scene, for example `static`;
- `--person-id`: identifier for the person, or `none`;
- `--orientation`: person orientation, for example `front`, `sideways` or `none`;
- `--prep-sec`: countdown before capture starts;
- `--duration-sec`: total dataset collection duration;
- `--output-root`: root directory used to store the dataset.

The radio and 5G processing parameters should match the online inference deployment. In particular, use the same `--force-nid2`: it is stored in the H5 file and used by the training script to determine the cell PCI.

---

## Collect an empty-scene dataset

Make sure no target is present in the sensing area.

Run:

```bash
python src/python/ssb_python/collect_labeled_rxgridssb_dataset_cfo.py \
  --label empty \
  --scene static \
  --person-id none \
  --orientation none \
  --prep-sec 10 \
  --duration-sec 30 \
  --serial <serial_id> \
  --freq 3541.44e6 \
  --rate 15.36e6 \
  --gain 60 \
  --duration-ms 20 \
  --channel 0 \
  --antenna RX2 \
  --force-nid2 0 \
  --enable-cfo-correction \
  --cfo-warmup-iters 30 \
  --cfo-correction-sign -1 \
  --output-root data/python_ssb_datasets
```

During the preparation countdown, leave the monitored area empty.

---

## Collect a target-position dataset

Place the target at the desired position during the preparation countdown.

Example for position `P1`:

```bash
python src/python/ssb_python/collect_labeled_rxgridssb_dataset_cfo.py \
  --label P1 \
  --scene static \
  --person-id person_1 \
  --orientation sideways \
  --prep-sec 10 \
  --duration-sec 30 \
  --serial <serial_id> \
  --freq 3541.44e6 \
  --rate 15.36e6 \
  --gain 60 \
  --duration-ms 20 \
  --channel 0 \
  --antenna RX2 \
  --force-nid2 0 \
  --enable-cfo-correction \
  --cfo-warmup-iters 30 \
  --cfo-correction-sign -1 \
  --output-root data/python_ssb_datasets
```

To collect another position, replace:

```text
--label P1
```

with the desired class, for example:

```text
--label P2
```

Also record meaningful metadata:

```text
--person-id person_1
--orientation front
```

Use the same class naming convention consistently across all sessions.

---

## Dataset output structure

Each collection session creates:

```text
data/python_ssb_datasets/<label>/<session_id>/
```

Example:

```text
data/python_ssb_datasets/P1/session_<timestamp>_P1_static/
```

Each session contains:

```text
session_data.h5
metadata.json
capture_log.csv
```

### `session_data.h5`

Contains accepted valid samples.

Important arrays include:

```text
dataSSB          complex64, shape [360, 6, N]
rxGridSSB        complex64, shape [240, 4, N]
pss_metric       float32, shape [N]
timing_offset_samples
timing_offset_ms
capture_time_ms
pss_time_ms
ofdm_time_ms
dsp_time_ms
loop_time_ms
```

`N` is the number of accepted valid samples.

Important H5 attributes include:

```text
label
scene
person_id
orientation
session_id
serial
freq
rate
gain
cfo_hz_applied
```

### `metadata.json`

Contains human-readable information about the complete capture session.

Inspect it with:

```bash
python -m json.tool \
  data/python_ssb_datasets/<label>/<session_id>/metadata.json
```

### `capture_log.csv`

Contains one row for every attempted capture, including valid and invalid attempts.

Inspect it with:

```bash
cat data/python_ssb_datasets/<label>/<session_id>/capture_log.csv
```

---

# Analyze a collected dataset

Find the most recent H5 dataset:

```bash
LAST_H5="$(find data/python_ssb_datasets -name session_data.h5 | sort | tail -n 1)"
echo "$LAST_H5"
```

Analyze it:

```bash
python src/python/ssb_python/analyze_rxgrid_distributions.py \
  --input "$LAST_H5" \
  --dataset rxGridSSB \
  --label python_dataset
```

The analysis can generate plots such as:

```text
amplitude histogram
phase histogram
IQ scatter
mean amplitude heatmap
mean amplitude by subcarrier
mean amplitude by OFDM symbol
```

Use these plots to inspect whether:

- the dataset contains valid signal structure;
- amplitudes remain stable;
- phases behave consistently;
- some subcarriers or symbols look corrupted;
- different sessions appear reasonably comparable.

These plots do not by themselves prove that two classes are separable. They are diagnostic tools for identifying capture or preprocessing problems.

---

## Analyze and train the Fisher SSB sensing model

`src/python/run_ssb_sensing_analysis.py` is the offline analysis and training tool for a labelled collection of SSB sessions. It **automatically discovers the class names from the immediate subdirectories below `--data-root`**: every folder that contains at least one `session_data.h5` becomes a label. There is no fixed list of labels in the script, and any number of labels ≥ 2 is supported.

The expected local dataset layout is the one produced by the [collection script](#collect-a-labeled-5g-sensing-dataset):

```text
<data-root>/
  empty/
    session_<timestamp>_empty_static/
      session_data.h5
  P1/
    session_<timestamp>_P1_static/
      session_data.h5
  P2/
    session_<timestamp>_P2_static/
      session_data.h5
```

The directory is collected locally and is intentionally not included in Git. After activating the project virtual environment and installing `requirements.txt`, run:

```bash
python src/python/run_ssb_sensing_analysis.py \
  --data-root data/python_ssb_datasets \
  --output results/ssb_sensing_analysis
```

For a dataset stored elsewhere, pass its absolute path instead:

```bash
python src/python/run_ssb_sensing_analysis.py \
  --data-root /path/to/python_ssb_datasets \
  --output results/ssb_sensing_analysis
```

Optional arguments:

```text
--top-k N            number of Fisher-selected resource elements (default: 64)
--random-traces N    random channel-magnitude traces plotted per class (default: 10)
--target-ibar N      SSB beam (i_bar_SSB) to train on (default: most frequent)
--seed N             reproducible random seed (default: 42)
```

`detection_summary.json` lists, per session, the per-snapshot `i_bar_SSB` counts and how many snapshots were kept.

### What the script does

1. Loads `rxGridSSB [240, 4, N]` from every session.
2. Detects the cell's PCI (NID1 from SSS correlation, NID2 from the H5 attribute `force_nid2`) and the `i_bar_SSB` (PBCH DM-RS correlation) **of every snapshot**. The receiver can lock onto different SSB beams within one session; each beam has its own beamforming and DM-RS reference, so `H` is only comparable within one beam.
3. Keeps only the snapshots of the target beam (`--target-ibar`, default: the most frequent `i_bar_SSB` in the dataset) for every label, and fails if a label has (almost) no snapshots on that beam: that label was captured on another beam and cannot be compared with the others.
4. Estimates the channel `H` from SSS and PBCH DM-RS and computes `20·log10|H|` normalised by its own snapshot mean, which removes slow global gain changes.
5. Plots random snapshots, mean profiles, the temporal channel level and an exploratory PCA.
6. Splits the time-ordered samples of every class 60% / 20% / 20% (train / validation / test).
7. Ranks every resource element with a multi-class Fisher score **fitted on the training portion only** and keeps the top `--top-k`.
8. Builds the feature vector: the selected REs + 9 spectral features (low/mid/high band means and standard deviations and the band differences).
9. Trains `StandardScaler + LogisticRegression(class_weight="balanced")`, reports validation metrics, retrains on train + validation and reports the final test metrics and confusion matrix.
10. Exports the deployment artifacts.

### Outputs

Written below `--output`:

```text
01_random_H_magnitude_snapshots.pdf
02_mean_H_magnitude_profile.pdf
03_normalized_H_profile.pdf
04_global_H_level_temporal.pdf
05_fisher_RE_heatmap.pdf
06_PCA_normalized_H_magnitude.pdf
07_confusion_matrix.pdf
selected_resource_elements.csv
classification_report.txt
confusion_matrix.csv
detection_summary.json
metrics.json
model_fisher_logreg.joblib
model_fisher_logreg.pt
```

The two deployment artifacts are:

- `model_fisher_logreg.joblib`: the original scikit-learn pipeline, for offline evaluation;
- `model_fisher_logreg.pt`: the PyTorch checkpoint used online. The scaler is folded into a single linear layer, so its probabilities are identical to the scikit-learn pipeline (verified by the [tests](#tests)). It also stores the class names, the selected REs, the feature definition and the radio configuration (PCI, i_bar_SSB) of the training data.

Sessions from different cells (PCI) are refused. The checkpoint stores the target `i_bar_SSB`; online grids of other beams are rejected by the radio check, exactly as they were discarded for training.

The default evaluation splits the time-ordered samples of every class into 60% training, 20% validation and 20% test. This is useful as an initial within-session baseline, but samples from the same acquisition may still occur in different splits. Do not interpret its final-test metrics as performance on an independent experiment. For a deployment claim, reserve entire sessions or collection days for testing.

---

# Test a model checkpoint on a saved dataset

Find the latest dataset:

```bash
LAST_H5="$(find data/python_ssb_datasets -name session_data.h5 | sort | tail -n 1)"
```

Replay the trained model on up to 30 saved samples, with exactly the online inference path (radio check + features + PyTorch model):

```bash
python src/python/ssb_python/replay_fisher_checkpoint_on_h5.py \
  --model-pt results/ssb_sensing_analysis/model_fisher_logreg.pt \
  --input-h5 "$LAST_H5" \
  --max-samples 30
```

Use `--max-samples 0` to replay every sample and `--quiet` to print only the summary.

This verifies that:

- the H5 dataset can be opened;
- `rxGridSSB` can be read;
- the checkpoint can be loaded;
- the session has the PCI/i_bar_SSB the model expects (otherwise samples are reported as rejected);
- predictions can be produced for saved samples, with per-class mean probabilities.

A successful technical test does not guarantee that the model is appropriate for the new environment.

---

# Model checkpoint format

`model_fisher_logreg.pt` is a `torch.save` dictionary:

| Key | Content |
|---|---|
| `model_type` | `torch_fisher_logreg` |
| `model_name` | `ssb_fisher_logreg` |
| `model_state_dict` | weights of one `nn.Linear(n_features, n_classes)` |
| `classes` | class names, in output order (the dataset folder names) |
| `selected_resource_elements` | `[[subcarrier, symbol], ...]` Fisher-selected REs |
| `feature_spec` | description of the feature vector |
| `radio_configuration` | `{"pci": ..., "ibar": ...}` used to build the reference grid |

The online frontend (`ssb_fisher_torch_inference.py`) reconstructs `H` and the features from each received `rxGridSSB` with the same functions the trainer uses.

## Radio check (PCI and i_bar_SSB)

The channel estimate divides the received grid by the reference SSS and PBCH DM-RS of the training PCI and i_bar_SSB. If the receiver locks onto a different SSB (another beam, so another i_bar_SSB), the reference is wrong and `|H|` is distorted by several dB, which is of the same order as the differences between classes. Before every inference the online pipeline therefore:

1. checks that the NID2 used by the PSS search equals `PCI mod 3` of the model. With `--force-nid2` this is the configured value, so this step validates the configuration, not the signal;
2. correlates the SSS of the grid against the 336 NID1 candidates and checks that the detected PCI is the model's (rejects another cell and noise grids);
3. correlates the PBCH DM-RS against the 8 possible i_bar_SSB values and checks that the best one is the model's (rejects another SSB beam).

If any check fails, the iteration is marked invalid (`err=nid2_mismatch: ...`, `err=pci_mismatch: ...` or `err=ibar_mismatch: ...`) and nothing is published as a detection. `--fisher-pci` and `--fisher-ibar` override the checkpoint values when the transmitter configuration has deliberately changed.

---

# Map model labels to objects and coordinates

The model predicts a **label** (a dataset folder name such as `empty` or `P1`). The Digital Twin needs **objects with coordinates**. The translation is done by:

```text
config/sensing_label_map.json
```

so coordinates can be changed at any time without retraining or editing code. The file in the repository is a generic template with placeholder labels and coordinates; replace them with the labels of your dataset and the real positions once the Digital Twin layout is fixed.

```json
{
  "schema_version": "sensing_label_map_v1",
  "frame_id": "world",
  "units": "m",
  "default_object_type": "person",
  "labels": {
    "empty": [],
    "P1": [
      {"object_type": "person", "position": {"x": 1.0, "y": 2.0, "z": 0.0}, "yaw_deg": 0.0}
    ],
    "P2": {"position": {"x": 3.0, "y": 2.0}}
  }
}
```

Rules:

- every label of the model must appear in the map; the online pipeline refuses to start otherwise, so nothing is ever published at an invented position;
- `[]` means "no object" for that label (for example `empty`);
- a label may contain several objects (a list), or a single object written as a dict;
- `z` and `yaw_deg` default to 0; `object_type` defaults to `default_object_type`;
- positions are in metres in the frame `frame_id` (right-handed, Z up), angles in degrees.

Before using the map in a scene, verify the scene origin, the X/Y/Z axis directions, units, floor height and object orientation used by the Digital Twin.

Check a map against a trained model:

```bash
cd src/python
python -m sensing_api.label_map ../../config/sensing_label_map.json \
  --model ../../results/ssb_sensing_analysis/model_fisher_logreg.pt
```

```text
../../config/sensing_label_map.json: frame=world units=m
  empty: no objects
  P1: person@(1,2,0)
  P2: person@(3,2,0)
OK: every class of ../../results/ssb_sensing_analysis/model_fisher_logreg.pt is mapped ['P1', 'P2', 'empty']
```

---

# Sensing API and CAPIF

Results are published through a REST API instead of copying files to another computer. The API follows the CAPIF model (3GPP TS 29.222 / ETSI TS 123 222), adapted from the PRIME-6G Digital Twin implementation:

| Process | Default port | CAPIF role | Module |
|---|---:|---|---|
| CAPIF Core Function | 8080 | CCF: provider/invoker enrolment, publication, discovery, routing | `sensing_api.capif_core` |
| Sensing API | 8090 | API Exposing Function (AEF) for `sensing-api` | `sensing_api.server` |
| Online pipeline | — | producer of reports | `online_5g_ssb_sensing.py` |
| Consumers | — | API invokers | e.g. `sensing_api.consume` |

On startup the Sensing API enrols as an API provider in the CCF and publishes its `ServiceAPIDescription` as `sensing-api`. If the CCF is not running yet it keeps retrying every 5 s. Consumers enrol as invokers, discover `sensing-api` and obtain its address from the routing information; they never need a hard-coded host.

## Configuration

`config/sensing_api_config.json`:

```json
{
  "capif": {"host": "0.0.0.0", "port": 8080, "base_url": "http://127.0.0.1:8080/capif/v1"},
  "sensing_api": {"host": "0.0.0.0", "port": 8090, "advertised_host": "127.0.0.1",
                  "history_size": 1000, "register_with_capif": true}
}
```

- `capif.base_url`: where the Sensing API and consumers reach the CCF.
- `sensing_api.host` / `port`: bind address of the Sensing API.
- `sensing_api.advertised_host`: the address published in CAPIF. **Set it to the IP that consumers can reach** (not `0.0.0.0` or `127.0.0.1` when consumers run on other machines).
- `history_size`: number of reports kept in memory for `GET /reports`.

Environment variables override the file: `SENSING_CAPIF_URL`, `SENSING_API_HOST`, `SENSING_API_PORT`, `SENSING_API_ADVERTISED_HOST`, `SENSING_API_PRODUCER_TOKEN`, and `SENSING_API_CONFIG` (path of another config file).

## Start the services

In two terminals, from `src/python` with the virtual environment active:

```bash
cd src/python
python -m sensing_api.capif_core
```

```bash
cd src/python
SENSING_API_ADVERTISED_HOST=<ip_of_this_computer> python -m sensing_api.server
```

Check them:

```bash
curl -s http://127.0.0.1:8080/capif/v1/health
curl -s http://127.0.0.1:8090/sensing/v1/health
```

The Sensing API health must show `"capif_registered": true`. Interactive documentation:

```text
CAPIF CCF     http://<host>:8080/capif/v1/ui
Sensing API   http://<host>:8090/sensing/v1/ui
```

If a CAPIF Core Function already runs elsewhere, skip the first command and point `SENSING_CAPIF_URL` to it.

## Data contract

Conventions: JSON, UTF-8, snake_case, metres, degrees, RFC 3339 UTC timestamps.

A **report** is published by a producer once per inference:

```json
{
  "sensor": {
    "sensor_id": "5g-ssb-rx-01",
    "sensor_type": "5g_ssb",
    "technology": "5G-NR",
    "method": "passive_ssb_fingerprinting",
    "model_name": "ssb_fisher_logreg",
    "model_version": "model_fisher_logreg.pt",
    "metadata": {"center_frequency_hz": 3541440000.0, "sample_rate_sps": 15360000.0, "pci": 731, "ibar": 0}
  },
  "timestamp": "2026-10-26T10:15:30.123456+00:00",
  "sequence": 1532,
  "valid": true,
  "invalid_reason": null,
  "label": "P1",
  "confidence": 0.93,
  "probabilities": {"P1": 0.93, "P2": 0.05, "empty": 0.02},
  "detections": [
    {
      "object_id": "person_0",
      "object_type": "person",
      "position": {"x": 1.0, "y": 2.0, "z": 0.0},
      "yaw_deg": 0.0,
      "frame_id": "world",
      "confidence": 0.93,
      "label": "P1"
    }
  ],
  "extra": {"dsp": {"nid2": 0, "pss_metric": 0.91}, "cfo_hz": 1250.4}
}
```

- Invalid iterations (weak PSS, incomplete grid, radio-check failure) are published with `"valid": false`, the reason in `invalid_reason` and no detections, so consumers know the sensor is alive but has no trustworthy state.
- An empty-scene label is published as a valid report with `"detections": []`.
- `object_id` is stable per object type (`person_0`, `person_1`, ...), so a consumer moves the same object when the predicted position changes.
- `confidence` of a detection is the model probability of the predicted label.

## Endpoints

| Method | Resource | Meaning |
|---|---|---|
| POST | `/sensing/v1/reports` | Publish a report (producers) |
| GET | `/sensing/v1/detections` | Current detections of every sensor. Filters: `object_type`, `sensor_id`, `min_confidence`, `max_age_s` |
| GET | `/sensing/v1/reports` | Report history. Filters: `sensor_id`, `since`, `limit` |
| GET | `/sensing/v1/sensors` | Sensors that have published, with last report time and validity |
| GET | `/sensing/v1/sensors/{sensor_id}/latest` | Latest full report of one sensor |
| GET | `/sensing/v1/stream` | Server-Sent Events: one `report` event per published report |
| GET | `/sensing/v1/health` | Status and CAPIF registration |

`/detections` uses only the **latest** report of each sensor: if it is invalid or empty, that sensor contributes no objects (the previous state is not repeated, because it may be outdated). Use `max_age_s` to ignore sensors that stopped publishing.

## Security

The CAPIF descriptions declare `PKI` as interface metadata, as in the PRIME-6G implementation, but the processes do not enforce certificates. Optionally, set `SENSING_API_PRODUCER_TOKEN` on the server and the same variable (or `--api-token`) on the producer: `POST /reports` then requires `Authorization: Bearer <token>`; reads stay open. For deployments outside a trusted lab network, put both services behind TLS and real authorisation.

---

# Run online 5G SSB sensing

The repository does not include a trained model: a fingerprinting model is specific to its cell and room, so train one for your deployment first ([Analyze and train](#analyze-and-train-the-fisher-ssb-sensing-model)).

With the model trained, the label map completed and the services running:

```bash
python src/python/ssb_python/online_5g_ssb_sensing.py \
  --serial <serial_id> \
  --freq 3541.44e6 \
  --rate 15.36e6 \
  --gain 60 \
  --duration-ms 20 \
  --channel 0 \
  --antenna RX2 \
  --force-nid2 0 \
  --enable-cfo-correction \
  --cfo-warmup-iters 30 \
  --cfo-correction-sign -1 \
  --fisher-model results/ssb_sensing_analysis/model_fisher_logreg.pt \
  --label-map config/sensing_label_map.json \
  --sensor-id 5g-ssb-rx-01 \
  --capif-url http://<capif_host>:8080/capif/v1 \
  --progress-every 1
```

The radio and 5G processing parameters must be the same ones used to collect the training dataset.

Model, publication and output parameters:

- `--fisher-model`: checkpoint from `run_ssb_sensing_analysis.py` (default `results/ssb_sensing_analysis/model_fisher_logreg.pt`).
- `--fisher-pci`, `--fisher-ibar`: override the checkpoint radio configuration.
- `--torch-device cpu`: inference device.
- `--label-map`: label → objects and coordinates.
- `--sensor-id`: unique id of this receiver in the Sensing API.
- `--capif-url`: discover the Sensing API through CAPIF; or `--sensing-api-url http://<host>:8090` to call it directly.
- `--publish-every N`: publish one report every N iterations (default 1).
- `--api-token`: producer token, when the server requires one.
- `--disable-api`: run without publishing.
- `--local-json`, `--json-every`, `--log-csv`: local latest-state JSON and CSV log.
- `--enable-mitsuba-export`, `--mitsuba-output`, `--mitsuba-every`: optional local Mitsuba XML.
- `--min-pss-metric`: minimum PSS metric for a valid block (default 0.5).
- `--num-iters 0`: run forever (default); stop with `Ctrl+C`.

At startup the script prints the model classes and the radio configuration, and warns if `--force-nid2` does not match the model's PCI:

```text
=== Online 5G SSB sensing ===
Fisher model:       results/ssb_sensing_analysis/model_fisher_logreg.pt
classes:            ['P1', 'P2', 'empty']
radio config:       PCI=<pci> i_bar=<ibar> (NID2 must be <pci mod 3>)
label map:          config/sensing_label_map.json (frame=world)
sensor id:          5g-ssb-rx-01
Sensing API:        CAPIF http://<capif_host>:8080/capif/v1
```

Each iteration then prints, for example:

```text
[000153] valid=1 label=P1 conf=0.934 objs=1 rx_mean=31.204 pss=0.912 loop=48.70 ms api=153/0 err=
```

Publishing runs in a background thread that keeps only the newest pending report: if the API is slow or unreachable, the sensing loop does not wait and old states are dropped rather than queued. Delivery failures are counted in `api=<sent>/<failed>` and in the CSV, and a summary is printed at exit.

---

# Consume the Sensing API

## Discover through CAPIF

```bash
# 1. enrol as invoker
curl -s -X POST http://<capif_host>:8080/capif/v1/api-invoker-enrolments \
  -H "Content-Type: application/json" \
  -d '{"onboardingInformation": {"apiInvokerPublicKey": "my-app"}, "apiList": ["sensing-api"]}'

# 2. discover the API and note its apiId
curl -s "http://<capif_host>:8080/capif/v1/service-apis?api_name=sensing-api"

# 3. where to call it (ipv4Addr + port of the AEF)
curl -s http://<capif_host>:8080/capif/v1/routing-info/<apiId>
```

## Read detections

```bash
curl -s "http://<sensing_host>:8090/sensing/v1/detections?object_type=person&max_age_s=2"
```

```json
{
  "detections": [
    {
      "object_id": "person_0",
      "object_type": "person",
      "position": {"x": 1.0, "y": 2.0, "z": 0.0},
      "yaw_deg": 0.0,
      "frame_id": "world",
      "confidence": 0.93,
      "label": "P1",
      "sensor_id": "5g-ssb-rx-01",
      "sensor_type": "5g_ssb",
      "technology": "5G-NR",
      "model_name": "ssb_fisher_logreg",
      "timestamp": "2026-10-26T10:15:30.123456+00:00",
      "report_id": "4f0c..."
    }
  ],
  "count": 1,
  "generated_at": "2026-10-26T10:15:30.140000+00:00"
}
```

Other reads:

```bash
curl -s http://<sensing_host>:8090/sensing/v1/sensors
curl -s http://<sensing_host>:8090/sensing/v1/sensors/5g-ssb-rx-01/latest
curl -s "http://<sensing_host>:8090/sensing/v1/reports?sensor_id=5g-ssb-rx-01&limit=20"
curl -N http://<sensing_host>:8090/sensing/v1/stream        # live events, Ctrl+C to stop
```

## Python consumer

`sensing_api.consume` does the complete invoker flow (enrol → discover → routing) and then polls or streams:

```bash
cd src/python

# read the current detections once
python -m sensing_api.consume --capif-url http://<capif_host>:8080/capif/v1 --once

# follow the live stream and keep a Mitsuba scene up to date
python -m sensing_api.consume --capif-url http://<capif_host>:8080/capif/v1 --stream \
  --mitsuba-xml /path/to/live_person_sionna_scene.xml

# poll twice per second and keep the latest detections as JSON
python -m sensing_api.consume --api-url http://<sensing_host>:8090 --poll-interval 0.5 \
  --object-type person --json-out /path/to/latest_detections.json
```

In `--stream` mode the consumer keeps the latest detections of every sensor and writes the merged set on each event (sensors silent for longer than `--max-age-s` are dropped). Mitsuba shape ids include the sensor id, so several sensors can report `person_0` at the same time.

Minimal code for another application:

```python
import httpx
from sensing_api.capif_client import CapifInvoker

with CapifInvoker("http://<capif_host>:8080/capif/v1") as capif:
    capif.enrol("my-digital-twin")
    base_url = capif.resolve_base_url("sensing-api")

for det in httpx.get(f"{base_url}/sensing/v1/detections", params={"max_age_s": 2}).json()["detections"]:
    print(det["object_type"], det["position"], det["confidence"], det["sensor_id"])
```

---

# Mitsuba/Sionna XML export

The Sensing API publishes object types and coordinates; converting them to a scene description is a consumer decision. The repository provides one converter, `sensing_api/mitsuba.py`, usable in two places:

- by a consumer: `python -m sensing_api.consume ... --mitsuba-xml <path>` (recommended: the XML is produced where Sionna runs);
- locally by the online pipeline: `--enable-mitsuba-export` (default output `results/online/live_person_sionna_scene.xml`, next to `--local-json`).

Each detected `person` becomes a body ellipsoid and a head sphere sharing the BSDF id `human_body_standard`. Plain Mitsuba BSDFs cannot encode radio conductivity, so the radio material is written as an XML comment that the Sionna side maps to a `RadioMaterial`:

```text
SIONNA_RADIO_MATERIAL name=human_body_standard relative_permittivity=38 relative_permeability=1 conductivity_s_per_m=1.46 frequency_hz=3541440000
```

Geometry and material can be changed in the online pipeline with `--mitsuba-body-radius-x-m`, `--mitsuba-body-radius-y-m`, `--mitsuba-body-half-height-m`, `--mitsuba-body-center-z-m`, `--mitsuba-head-radius-m`, `--mitsuba-head-center-z-m`, `--mitsuba-human-material-name`, `--mitsuba-human-epsilon-r`, `--mitsuba-human-mu-r` and `--mitsuba-human-conductivity-s-per-m`. An empty scene (no person, or an invalid iteration) is exported without shapes.

The shape transforms are written as scale → rotate → translate, so the person is centred on the published coordinates; this is checked by loading the XML with Mitsuba in the tests. Mitsuba warns that its analytic `sphere` does not officially support non-uniform scaling (the body ellipsoid); if Sionna needs an exact ellipsoid, replace the body with a mesh.

Generate a scene locally without any remote machine:

```bash
python src/python/ssb_python/online_5g_ssb_sensing.py \
  --serial <serial_id> \
  --freq 3541.44e6 \
  --rate 15.36e6 \
  --gain 60 \
  --duration-ms 20 \
  --num-iters 30 \
  --warmup-iters 5 \
  --channel 0 \
  --antenna RX2 \
  --force-nid2 0 \
  --enable-cfo-correction \
  --cfo-warmup-iters 30 \
  --cfo-correction-sign -1 \
  --fisher-model results/ssb_sensing_analysis/model_fisher_logreg.pt \
  --label-map config/sensing_label_map.json \
  --disable-api \
  --enable-mitsuba-export \
  --progress-every 1
```

```bash
cat results/online/live_person_sionna_scene.xml
```

---

# Recommended workflow for a new deployment

For a new room, factory or antenna arrangement:

1. Install the repository and UHD environment.
2. Connect the USRP B210 and verify UHD communication.
3. Determine the correct frequency, sample rate, gain, channel, antenna and NID2 ([Automatic SSB Discovery](#automatic-ssb-discovery) helps).
4. Validate the radio chain with `online_5g_ssb_sensing.py --dsp-only` (Step 10).
5. Define the target classes, for example `empty`, `P1`, `P2`, `P3`.
6. Collect several labeled sessions for every class.
7. Inspect the datasets for capture and preprocessing problems.
8. Run `run_ssb_sensing_analysis.py` to analyse the signal and train the model.
9. Replay-test `model_fisher_logreg.pt` on saved H5 data, ideally on sessions not used for training.
10. Add every class with its objects and coordinates to `config/sensing_label_map.json` and check it against the model.
11. Start the CAPIF Core Function and the Sensing API with the correct `advertised_host`.
12. Run online inference publishing to the Sensing API.
13. Validate what the Digital Twin consumer reads from `/sensing/v1/detections` or `/sensing/v1/stream`.

---

# WiFi CSI receiver files and recommended execution

The WiFi CSI component currently contains two receiver scripts:

```text
src/python/wifi_csi/beacon_csi_receiver.py
src/python/wifi_csi/beacon_csi_dataset_receiver.py
```

Both scripts implement the same basic GNU Radio receive chain:

```text
USRP B210
  → IEEE 802.11 short preamble detection
  → short synchronization
  → long synchronization
  → 64-point FFT
  → gr-ieee802-11 frame equalizer
  → MAC frame decoding
  → beacon filtering by SSID and BSSID
  → CSI extraction
```

The CSI is obtained from the channel estimate generated by the
`gr-ieee802-11` `frame_equalizer` block and propagated in the decoded MAC
PDU metadata.

### `beacon_csi_receiver.py`

This is the recommended script for normal WiFi CSI acquisition.

It performs continuous reception and generates two outputs:

```text
latest CSI JSON
accumulative CSI dataset
```

The JSON file is atomically overwritten every time a matching beacon is
received, while the dataset file appends one CSI vector per matching beacon.

Recommended script:

```text
src/python/wifi_csi/beacon_csi_receiver.py
```

### `beacon_csi_dataset_receiver.py`

This script implements the same beacon-based CSI acquisition workflow and
also writes the latest CSI plus an accumulative dataset.

It is currently retained as an alternative entry point for the WiFi CSI
acquisition component.

For new experiments, use:

```text
src/python/wifi_csi/beacon_csi_receiver.py
```

unless a specific experiment explicitly requires the dataset receiver entry
point.

---

### Tested WiFi configuration

The receiver has been validated with the following access point:

```text
SSID:       PRACIT-AIAAS_5G
BSSID:      04:42:1A:E4:DD:D4
Channel:    100
Frequency:  5500 MHz
PHY rate:   6 Mbps beacon
```

The access point transmits OFDM-compatible beacon frames that can be decoded
by `gr-ieee802-11`.

The USRP B210 was configured with:

```text
Sample rate: 20 MS/s
Antenna:     RX2
Gain:        0.60 normalized UHD gain
```

Before running the receiver, activate the project environment:

```bash
cd DT_sensing_fusion
source .venv_uhd/bin/activate
```

If the environment was created following the initial setup section, the
GNU Radio, UHD and `gr-ieee802-11` modules should already be available.

Verify them with:

```bash
python - <<'PY'
from gnuradio import blocks, fft, gr, uhd
import ieee802_11
import pmt

print("GNU Radio: OK")
print("UHD: OK")
print("gr-ieee802-11: OK")
print("PMT: OK")
PY
```

---

### Run the validated 5 GHz beacon receiver

From the repository root:

```bash
python src/python/wifi_csi/beacon_csi_receiver.py \
  --ssid "PRACIT-AIAAS_5G" \
  --bssid "04:42:1A:E4:DD:D4" \
  --wifi-channel 100 \
  --frequency 5500000000 \
  --sample-rate 20000000 \
  --gain 0.60 \
  --antenna RX2 \
  --output results/wifi_csi/pracit_5g_latest.json \
  --dataset-output results/wifi_csi/pracit_5g_dataset.txt \
  --verbose
```

The receiver runs continuously until interrupted with:

```text
Ctrl+C
```

A successfully received target beacon produces output similar to:

```text
[beacon] ssid='PRACIT-AIAAS_5G' bssid=04:42:1a:e4:dd:d4 channel=100
[saved] beacon #1 | PRACIT-AIAAS_5G | 04:42:1a:e4:dd:d4 | CSI=52 subcarriers | SNR=<snr> | <output_path>
```

When the receiver is stopped, it prints a summary such as:

```text
Resumen:
  PDUs MAC válidos:        <count>
  Beacons detectados:      <count>
  BSSID coincidente:       <count>
  SSID+BSSID coincidentes: <count>
  CSI guardados:           <count>
```

The number shown as:

```text
CSI guardados
```

should match the number of lines appended to the dataset file during that
execution, assuming the dataset file was empty before the receiver started.

---

### Receiver parameters

The most relevant parameters are:

| Parameter | Meaning |
|---|---|
| `--ssid` | SSID that must be present in the decoded beacon. |
| `--bssid` | MAC address of the specific AP radio interface to accept. |
| `--wifi-channel` | Logical WiFi channel. For 5 GHz channels, also provide `--frequency`. |
| `--frequency` | USRP centre frequency in Hz. Required for 5 GHz because automatic channel-to-frequency conversion currently only covers 2.4 GHz channels 1–14. |
| `--sample-rate` | GNU Radio / USRP sample rate. The normal 802.11 OFDM configuration is `20000000` S/s. |
| `--gain` | Normalized UHD RX gain in the range `0.0` to `1.0`. |
| `--antenna` | Physical USRP RX antenna input, normally `RX2`. |
| `--device-args` | Optional UHD device arguments. |
| `--channel-index` | Logical USRP RX channel index. The default is `0`. |
| `--channel-estimator` | `gr-ieee802-11` channel estimator selection, values `0` to `3`. |
| `--sync-length` | Synchronization delay used by the GNU Radio receiver. |
| `--window-size` | Moving-average window used by the packet detector. |
| `--output` | JSON file containing only the most recent matching CSI sample. |
| `--dataset-output` | Append-only text file containing all matching CSI samples. |
| `--verbose` | Prints every decoded beacon before SSID/BSSID filtering. |
| `--log-mac` | Enables MAC logging in `gr-ieee802-11`. |
| `--debug-mac` | Enables MAC debugging in `gr-ieee802-11`. |

The complete argument list can always be inspected with:

```bash
python src/python/wifi_csi/beacon_csi_receiver.py --help
```

---

### 2.4 GHz versus 5 GHz execution

For 2.4 GHz channels 1–14, the script can derive the centre frequency from
`--wifi-channel`.

Example:

```bash
python src/python/wifi_csi/beacon_csi_receiver.py \
  --ssid "<SSID>" \
  --bssid "<BSSID>" \
  --wifi-channel 11 \
  --gain 0.60 \
  --antenna RX2 \
  --dataset-output results/wifi_csi/csi_dataset.txt
```

For 5 GHz operation, explicitly provide the frequency because the current
`channel_to_frequency_hz()` helper only converts 2.4 GHz channels.

Example for channel 100:

```text
channel 100 → 5500 MHz → 5500000000 Hz
```

Therefore use:

```bash
--wifi-channel 100 \
--frequency 5500000000
```

---

### Verify a WiFi CSI capture

Count the accumulated CSI vectors:

```bash
wc -l results/wifi_csi/pracit_5g_dataset.txt
```

Inspect the first and last samples:

```bash
head -n 1 results/wifi_csi/pracit_5g_dataset.txt
tail -n 1 results/wifi_csi/pracit_5g_dataset.txt
```

Validate the complete dataset:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np

path = Path("results/wifi_csi/pracit_5g_dataset.txt")

rows = []

with path.open(encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, 1):
        fields = line.rstrip("\n").split("\t")

        if len(fields) != 5:
            raise RuntimeError(
                f"Invalid dataset line {line_number}: expected 5 fields, "
                f"received {len(fields)}"
            )

        timestamp_ns, ssid, bssid, snr_db, csi_text = fields

        csi = np.asarray(
            [complex(value) for value in csi_text.split(",")],
            dtype=np.complex64,
        )

        rows.append(
            (
                int(timestamp_ns),
                ssid,
                bssid,
                float(snr_db),
                csi,
            )
        )

print("Valid CSI samples:", len(rows))

if rows:
    H = np.stack([row[4] for row in rows])
    snr = np.asarray([row[3] for row in rows])

    print("CSI matrix shape:", H.shape)
    print("SSID values:", sorted(set(row[1] for row in rows)))
    print("BSSID values:", sorted(set(row[2] for row in rows)))
    print("NaN values:", np.isnan(H).sum())
    print("Inf values:", np.isinf(H).sum())
    print(
        "SNR min / mean / max:",
        float(snr.min()),
        float(snr.mean()),
        float(snr.max()),
    )
PY
```

For the current legacy OFDM receiver, a correct dataset should normally
produce:

```text
CSI matrix shape: (<number_of_beacons>, 52)
NaN values: 0
Inf values: 0
```

A validated laboratory capture using `PRACIT-AIAAS_5G` produced hundreds of
matching beacon CSI vectors, each containing 52 complex subcarrier channel
estimates.

---

### Important distinction: CSI versus raw L-LTF samples

The current GNU Radio WiFi receiver stores the frequency-domain channel
estimate produced from the legacy OFDM training field:

```text
CSI H[k] → 52 complex subcarrier coefficients
```

It does not currently store the complete raw time-domain L-LTF waveform:

```text
L-LTF → 160 complex time-domain samples
```

For the normal WiFi CSI sensing workflow in this repository, the 52-element
CSI vector is the intended output.

Raw L-LTF extraction would require an additional capture path around the
802.11 synchronization stage and is not part of the current
`beacon_csi_receiver.py` output.

---

# WiFi CSI Acquisition using USRP B210

This project includes a Python receiver that extracts Channel State Information (CSI) from IEEE 802.11 beacon frames using a USRP B210 and the GNU Radio `gr-ieee802-11` module.

The receiver filters beacon frames by SSID and BSSID and stores one CSI vector per received frame, allowing CSI datasets to be collected for later processing.

## Receiver script

The receiver is located at:

```text
src/python/wifi_csi/beacon_csi_receiver.py
```

The required GNU Radio, UHD and `gr-ieee802-11` dependencies must already be available in the project environment as part of the initial project setup.

## Selecting an access point

The receiver must be configured with:

- the SSID of the target access point;
- its BSSID;
- the WiFi channel;
- an appropriate USRP receive gain.

Available access points can be listed with:

```bash
nmcli -f SSID,BSSID,CHAN,FREQ,SIGNAL dev wifi list
```

The BSSID identifies the specific radio interface of the access point. Two networks with the same SSID may therefore require different BSSID values.

## Running the receiver

Example:

```bash
python3 src/python/wifi_csi/beacon_csi_receiver.py \
  --ssid "eduroam" \
  --bssid "5C:E1:76:D3:0B:20" \
  --wifi-channel 11 \
  --gain 0.60 \
  --dataset-output results/wifi_csi/csi_dataset.txt
```

The most relevant arguments are:

```text
--ssid              Target SSID.
--bssid             Target BSSID.
--wifi-channel      WiFi channel used by the access point.
--frequency         Optional centre frequency in Hz.
--sample-rate       Receiver sample rate.
--gain              Normalised UHD receive gain between 0.0 and 1.0.
--antenna           USRP receive antenna.
--device-args       Additional UHD device arguments.
--channel-index     USRP receive channel index.
--channel-estimator Channel estimator used by gr-ieee802-11.
--output             JSON file overwritten with the latest matching CSI sample.
--dataset-output    Accumulative CSI dataset output file.
--verbose           Print all decoded beacon frames.
```

Passing `--ssid`, `--bssid` and `--wifi-channel` explicitly is recommended. The script contains deployment-specific defaults that may not correspond to an OFDM-compatible access point in another environment.

The complete list of options can be displayed with:

```bash
python3 src/python/wifi_csi/beacon_csi_receiver.py --help
```

## Dataset output

The dataset file is append-only while the receiver is running.

Each matching beacon produces one new line containing one complete CSI vector.

The line format is:

```text
timestamp_ns<TAB>ssid<TAB>bssid<TAB>snr_db<TAB>csi_0,csi_1,...,csi_N
```

Each CSI coefficient is stored as a Python-compatible complex number:

```text
real+imagj
```

Example:

```text
1785420000123456789	eduroam	5c:e1:76:d3:0b:20	23.4	0.01907947+0.01698885j,0.01259829+0.01777455j,...
```

For the current legacy OFDM receiver, each decoded frame normally produces a CSI vector with 52 complex subcarrier values.

## Verifying the captured dataset

Count the number of collected CSI samples:

```bash
wc -l results/wifi_csi/csi_dataset.txt
```

Inspect the last captured sample:

```bash
tail -n 1 results/wifi_csi/csi_dataset.txt
```

Validate the structure and load the last CSI vector:

```bash
python3 - <<'PY'
from pathlib import Path

import numpy as np

dataset_path = Path("results/wifi_csi/csi_dataset.txt")
last_line = dataset_path.read_text(encoding="utf-8").splitlines()[-1]

timestamp_ns, ssid, bssid, snr_db, csi_text = last_line.split("\t")

csi = np.asarray(
    [complex(value) for value in csi_text.split(",")],
    dtype=np.complex64,
)

print("Timestamp:", timestamp_ns)
print("SSID:", ssid)
print("BSSID:", bssid)
print("SNR:", snr_db)
print("CSI shape:", csi.shape)
print("CSI:", csi)
PY
```

A successful capture with the current receiver should normally report:

```text
CSI shape: (52,)
```

## Output files

The receiver generates two runtime outputs:

```text
results/wifi_csi/latest_csi.json
results/wifi_csi/csi_dataset.txt
```

- `latest_csi.json` contains only the most recently received matching CSI sample and is overwritten atomically.
- `csi_dataset.txt` is append-only and contains one line for every matching beacon.

Both files are local runtime outputs and are excluded from Git.

The directory itself is retained in Git through:

```text
results/wifi_csi/.gitkeep
```

## Compatibility note

CSI extraction relies on the channel estimate generated by the `frame_equalizer` block in `gr-ieee802-11` and propagated through the decoded MAC PDU metadata.

The current receiver supports beacon frames transmitted using legacy IEEE 802.11 OFDM-compatible rates.

Some commercial 2.4 GHz access points transmit beacon frames using legacy DSSS/CCK rates associated with IEEE 802.11b. Such beacons may be visible to a conventional WiFi adapter but are not decoded by this receiver.

A strong signal reported by `nmcli` therefore does not guarantee that the access point beacon uses a PHY format supported by `gr-ieee802-11`.

---

# Automatic SSB Discovery

Discover unknown 5G NR FR1 SSBs without prior knowledge of the carrier frequency, band, SCS or NID2.

Example:

```bash
python src/python/ssb_python/scan_5g_ssb_auto.py \
    --serial <USRP_SERIAL> \
    --gain 60 \
    --mode quick
```
## Useful options
```text
--start-mhz 3300
--stop-mhz 3800
--mode quick|balanced|exhaustive
```
---

# Historical MATLAB reference

The recommended deployment is Python-only. MATLAB is not required for normal operation.

The older workflow used:

```text
MATLAB capture
  → MATLAB rxGridSSB extraction
  → TCP transfer to Python
  → Python inference
  → JSON
  → SCP
```

That workflow is preserved only for historical validation and comparison.

Relevant files include:

```text
legacy_models/docs/README_online_ssB_empty_p5.md
run_python_matlab_interleaved_capture.sh
src/python/ssb_python/compare_interleaved_python_matlab.py
```

The practical objective is not permanent bit-exact MATLAB equivalence.

The current objective is:

```text
stable Python extraction
consistent Python rxGridSSB representation
training on Python-generated rxGridSSB
online inference using the same Python processing chain
```

Use the MATLAB comparison utilities only when investigating extraction differences or validating future signal-processing changes.

---

# Main scripts

| Script | Purpose |
|---|---|
| `src/python/ssb_python/online_5g_ssb_sensing.py` | Online sensing: capture, DSP, radio check, Fisher inference, Sensing API publication, local JSON/CSV and optional Mitsuba XML |
| `src/python/ssb_python/ssb_fisher_torch_inference.py` | Shared channel/feature frontend, radio check, PyTorch model and sklearn → PyTorch export |
| `src/python/run_ssb_sensing_analysis.py` | Generic multi-label offline analysis, Fisher selection, training and deployment-checkpoint export |
| `src/python/ssb_python/replay_fisher_checkpoint_on_h5.py` | Replays a checkpoint on a saved H5 dataset |
| `src/python/ssb_python/collect_labeled_rxgridssb_dataset_cfo.py` | Operator-friendly labeled dataset collection |
| `src/python/ssb_python/capture_online_rxgridssb_dataset_cfo.py` | Lower-level CFO-corrected dataset capture |
| `src/python/ssb_python/analyze_rxgrid_distributions.py` | Generates diagnostic plots for saved datasets |
| `src/python/ssb_python/scan_5g_ssb_auto.py` | Automatic 5G SSB discovery |
| `src/python/ssb_python/capture_iq_blocks_uhd.py` | Raw IQ capture for low-level debugging |
| `src/python/ssb_python/compare_interleaved_python_matlab.py` | Historical Python/MATLAB comparison |
| `src/python/sensing_api/capif_core.py` | CAPIF Core Function |
| `src/python/sensing_api/server.py` | Sensing API (CAPIF AEF) |
| `src/python/sensing_api/consume.py` | Example consumer (CAPIF invoker) |
| `src/python/sensing_api/label_map.py` | Label map parser and checker |
| `src/python/sensing_api/producer.py` | Report builder and non-blocking publisher |
| `src/python/sensing_api/mitsuba.py` | Detections → Mitsuba XML |
| `src/python/wifi_csi/beacon_csi_receiver.py` | WiFi CSI receiver |
| `src/python/wifi_csi/beacon_csi_dataset_receiver.py` | WiFi CSI dataset receiver |

Configuration files:

| File | Purpose |
|---|---|
| `config/sensing_label_map.json` | Model label → objects and coordinates |
| `config/sensing_api_config.json` | CAPIF and Sensing API hosts and ports |
| `config/map_zones.json`, `config/target_ssb.json`, `config/sensing_config.json` | Used by the historical MATLAB workflow |

---

# Tests

The automated tests run without a USRP (the radio is simulated with synthetic SSB grids):

```bash
python -m pytest
```

They check that:

- the online features and the PyTorch probabilities are identical to the offline scikit-learn pipeline (multi-class and binary);
- the radio check accepts the training SSB and rejects another i_bar_SSB, another cell (PCI), noise grids and a wrong NID2;
- the label map is validated and model labels without coordinates are refused;
- the Sensing API accepts, validates, filters and serves reports, and the producer token works;
- end to end over real HTTP: the Sensing API registers in the CAPIF Core Function, a consumer discovers it, the producer publishes through CAPIF discovery and the consumer receives the report by polling and by the event stream;
- the online script `main()` runs with a simulated radio, rejects bad grids, publishes the right detections and stops with a clear message when no SSB is found;
- the trainer keeps only the snapshots of the target SSB beam and refuses a label captured on another beam;
- the Mitsuba XML places the person at the published coordinates (when Mitsuba is installed).

---

# Legacy models

Earlier models and experiments are kept in [`legacy_models/`](legacy_models/README.md): the binary CNN checkpoint, its loader and replay test, the threshold backend configuration, ROI/hybrid/CNN training pipelines, explainability and calibration scripts, the old online servers, the SCP launcher and the old runbooks. They are not used by the current pipeline and were moved without modification.

---

# Generated files and Git policy

Do not commit large generated data or runtime output folders.

Typical generated paths include:

```text
data/python_ssb_datasets/
results/ssb_sensing_analysis/
results/online/
results/python_online_rxgridssb_dataset_cfo/
results/python_rxgrid_distribution/
results/python_matlab_rxgrid_compare/
logs/
```

Trained models are deployment-specific and are not committed. Keep the `results/ssb_sensing_analysis/` output of each deployment together with its label map.

---

# Troubleshooting

## `ModuleNotFoundError: No module named 'uhd'`

Verify:

```bash
sudo apt install -y python3-uhd
```

The environment must have been created with:

```bash
python3 -m venv --system-site-packages .venv_uhd
```

Recreate it if necessary.

---

## `ModuleNotFoundError: No module named 'torch'`

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

Confirm:

```bash
python -c "import torch; print(torch.__version__)"
```

---

## No USRP is detected

Run:

```bash
uhd_find_devices
```

Check:

- USB cable;
- USB 3 port;
- device power;
- UHD installation;
- whether another application is using the B210.

---

## The script reports `valid=0`

Inspect:

- `err=`;
- PSS metric;
- selected NID2;
- frequency;
- gain;
- antenna selection;
- physical antenna connection;
- CFO estimate;
- the radio check, when the error is `nid2_mismatch`, `pci_mismatch` or `ibar_mismatch` (see below).

The deployment-specific parameters may not match the received 5G cell.

---

## `CFO warmup failed ... no SSB was detected`

The script stops before the online loop because none of the CFO warmup captures contains an SSB (PSS metric around 0.01 instead of 0.5–0.98). There is no 5G cell at that frequency/NID2, or nothing reaches the receiver:

- check that the gNB is transmitting;
- find the SSBs on air with [Automatic SSB Discovery](#automatic-ssb-discovery) and use the reported frequency and NID2;
- check the antenna port (`--antenna RX2`) and the gain.

The radio parameters of a trained model are the ones stored in its dataset (`freq_hz` and `force_nid2` attributes of `session_data.h5`). If the cell on air is not the one of the dataset, the radio check rejects every grid (see below) and a new dataset is needed.

---

## The script reports `err=ibar_mismatch`, `err=pci_mismatch` or `err=nid2_mismatch`

The received SSB does not match the PCI/i_bar_SSB the model was trained with (see [the radio check](#radio-check-pci-and-i_bar_ssb)).

- `nid2_mismatch`: `--force-nid2` differs from the model's `PCI mod 3`. Use the same NID2 as during dataset collection. If the only cell on air has another NID2, the model belongs to another cell and cannot be used here.
- `pci_mismatch`: the grid comes from another cell, or the SSB was not really detected (noise). Check frequency, NID2 and the PSS metric.
- `ibar_mismatch` on every iteration: the cell or SSB beam differs from the training data. Check the receiver position and the cell, compare with `detection_summary.json` of the training run, and retrain if the deployment changed. Use `--fisher-ibar` only if you know the transmitter configuration changed on purpose.
- `ibar_mismatch` on some iterations: the receiver alternates between SSB beams; those iterations are discarded on purpose. The CSV column `detected_ibar` shows the distribution.

---

## The model checkpoint cannot be loaded

Verify the path:

```bash
ls -l results/ssb_sensing_analysis/model_fisher_logreg.pt
```

The command must be executed from the repository root.

The checkpoint must be produced by `run_ssb_sensing_analysis.py` (`model_type = torch_fisher_logreg`). Checkpoints of the legacy CNN are not supported by the current pipeline.

---

## `LabelMapError: Labels without coordinates`

The model has classes that are not in `config/sensing_label_map.json`. Add them (use `[]` for labels without objects) and check:

```bash
cd src/python
python -m sensing_api.label_map ../../config/sensing_label_map.json \
  --model ../../results/ssb_sensing_analysis/model_fisher_logreg.pt
```

---

## The Sensing API is not registered in CAPIF

`GET /sensing/v1/health` shows `"capif_registered": false` and the server log repeats `CAPIF registration failed ... retrying`.

- Check that the CCF is running: `curl -s <capif_base_url>/health`.
- Check `capif.base_url` / `SENSING_CAPIF_URL` from the Sensing API computer.
- The CCF keeps its registry in memory. The Sensing API verifies its registration every 30 s and registers again automatically after a CCF restart, so allow up to 30 s before it is discoverable again.

---

## `address already in use` when starting the services

Another process uses port 8080 or 8090 (common on machines running other network functions). Start the services on other ports and point the producer and consumers to them:

```bash
cd src/python
python -m sensing_api.capif_core --port 18080
SENSING_CAPIF_URL=http://127.0.0.1:18080/capif/v1 python -m sensing_api.server --port 18090
```

---

## Consumers discover the API but cannot connect

The address published in CAPIF is `advertised_host:port`. If it is `127.0.0.1`, remote consumers cannot use it. Set `SENSING_API_ADVERTISED_HOST=<reachable ip>` and restart the Sensing API. Check the firewall for ports 8080 and 8090.

---

## The online script shows `api=<sent>/<failed>` with failures

- `http_401`: the server requires a producer token; pass `--api-token` or set `SENSING_API_PRODUCER_TOKEN`.
- `http_422`: the report does not match the data contract (the server response is in the CSV `error` column only for the first characters; check the Sensing API log).
- `ConnectError` / `capif_discovery`: the Sensing API or the CCF is not reachable from the sensing computer.

Sensing continues while the API is down; only the newest state is sent when it comes back.

---

## The CSV contains rows from previous runs

The current CSV log is appended to.

Inspect:

```bash
wc -l results/online/python_5g_online_inference_log.csv
```

Remove it before a clean test only when previous logs are no longer needed:

```bash
rm results/online/python_5g_online_inference_log.csv
```

The script will create it again during the next run.

---

# Current project status

Completed:

- Python-only USRP capture;
- CFO estimation and correction;
- PSS and timing detection;
- OFDM demodulation;
- `dataSSB` extraction;
- `rxGridSSB` extraction;
- labeled dataset collection;
- dataset analysis;
- generic multi-label Fisher + logistic-regression training;
- PyTorch deployment of the trained model, verified equal to the offline pipeline;
- per-grid radio check (NID2, i_bar_SSB);
- label → coordinates map;
- CAPIF Core Function and CAPIF-exposed Sensing API;
- non-blocking publication from the online pipeline;
- example consumer with CAPIF discovery, polling and streaming;
- local JSON output and CSV log;
- optional Mitsuba/Sionna XML output;
- automated tests;
- WiFi CSI receiver for OFDM beacon frames.

Future integration work:

- final coordinates in the label map, aligned with the Digital Twin frame;
- Digital Twin / Isaac Sim consumer of the Sensing API;
- publishing WiFi CSI results through the same Sensing API;
- TLS and real CAPIF authorisation for deployments outside the lab network;
- MikroTik integration;
- complete multi-component orchestration.
