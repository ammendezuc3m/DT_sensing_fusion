# Python 5G SSB Online Sensing Deployment Guide

This guide is written for someone starting from a new computer and wanting to run **online 5G sensing** with the Python pipeline.

The deployment target is:

```text
USRP B210
  -> Python UHD IQ capture
  -> CFO warmup/correction
  -> PSS/NID2/timing detection
  -> OFDM demodulation
  -> dataSSB / rxGridSSB extraction
  -> PyTorch model inference
  -> local JSON
  -> optional SCP to a remote Digital Twin machine
```

MATLAB is **not required** for deployment. MATLAB was only used as a reference during development to validate that the Python extraction produced consistent SSB/rxGridSSB representations.

---

## 1. What this repository currently does

The current repository supports two practical modes:

### A. Online inference mode

This is the real-time sensing mode.

```text
USRP B210 receives 5G SSB
Python extracts rxGridSSB
Python applies a PyTorch model
Python writes a JSON result
Python optionally sends that JSON to a remote machine by SCP
```

Main script:

```text
src/python/ssb_python/online_5g_python_cfo_json_scp.py
```

Current demo model:

```text
results/binary_empty_vs_P5_rx/model_rxGridSSB/model.pt
```

Current model classes:

```text
empty
P5
```

This means:

```text
empty = no target/person in the trained scene
P5    = target/person at position P5 in the trained scene
```

### B. Dataset collection mode

This is the mode used to record new labeled data for training or validating a new model.

Main script added for this workflow:

```text
src/python/ssb_python/collect_labeled_rxgridssb_dataset_cfo.py
```

It does:

```text
10 s preparation countdown
CFO warmup
live SSB extraction
save dataSSB / rxGridSSB samples into an H5 dataset
save metadata and a CSV capture log
```

---

## 2. Important limitation of the included model

The included `.pt` model is a **demo binary model** trained for the current lab setup:

```text
empty vs P5
```

It is **not a general human detector** and it is **not expected to generalize automatically** to a different factory, antenna position, receiver position, room, geometry, label set, or deployment environment.

For a new factory or scenario, the correct workflow is:

```text
1. Install the environment.
2. Connect and test the USRP B210.
3. Collect a new labeled dataset using the Python dataset collection script.
4. Train a new model using that dataset.
5. Replace the checkpoint passed to the online inference script.
6. Run online inference with the new checkpoint.
```

The current model is included so that users can see the full end-to-end pipeline working.

---

## 3. Hardware setup

Required hardware:

```text
USRP B210
USB 3 cable
RX antenna connected to the selected RX channel
5G source/cell transmitting SSB
Linux computer with UHD support
```

Current tested USRP parameters:

```text
serial: 34B73C3
center frequency: 3541.44 MHz
sample rate: 15.36 Msps
gain: 60 dB
capture duration: 20 ms
RX channel: 0
NID2: 0
```

These values are specific to the current setup and may need to be changed for another deployment.

---

## 4. Fresh computer setup

### 4.1 Clone the repository

```bash
git clone git@github.com:ammendezuc3m/DT_sensing_fusion.git
cd DT_sensing_fusion
```

If SSH is not configured for GitHub, use the HTTPS clone URL instead.

### 4.2 Install system packages

On Ubuntu/Debian-like systems:

```bash
sudo apt update

sudo apt install -y \
  git \
  python3 \
  python3-pip \
  python3-venv \
  uhd-host \
  python3-uhd
```

Check that UHD can see the USRP:

```bash
uhd_find_devices
```

You should see a B210 device. Example:

```text
type: b200
product: B210
serial: 34B73C3
```

You can also probe it:

```bash
uhd_usrp_probe
```

### 4.3 Create the Python UHD environment

This environment uses `--system-site-packages` so that Python can see the UHD Python bindings installed by the OS.

```bash
cd DT_sensing_fusion

python3 -m venv --system-site-packages .venv_uhd
source .venv_uhd/bin/activate

python -m pip install --upgrade pip
```

Install Python dependencies.

If the repository already has UHD requirements:

```bash
python -m pip install -r requirements/requirements-uhd.txt
```

If not, install the minimal set:

```bash
python -m pip install "numpy<2" scipy h5py matplotlib torch
```

Check imports:

```bash
python - <<'PY'
import numpy
import scipy
import h5py
import torch
import uhd

print("numpy:", numpy.__version__)
print("torch:", torch.__version__)
print("uhd OK")
PY
```

---

## 5. Useful scripts and what they do

### `online_5g_python_cfo_json_scp.py`

Full online inference script.

It performs:

```text
USRP capture
CFO warmup/correction
PSS/NID2/timing
OFDM demodulation
rxGridSSB extraction
PyTorch or threshold inference
local JSON write
optional SCP to remote host
```

Use this for real-time deployment.

---

### `rxgrid_torch_inference.py`

PyTorch model loader and inference wrapper.

It reconstructs the current CNN model and loads:

```text
model_state_dict
mean
std
classes
input_shape
complex_mode
```

from the `.pt` checkpoint.

The expected input is:

```text
rxGridSSB complex array with shape [240, 4]
```

It converts it into:

```text
[1, 2, 240, 4]
channel 0 = abs(rxGridSSB)
channel 1 = angle(rxGridSSB)
```

---

### `collect_labeled_rxgridssb_dataset_cfo.py`

Dataset collection script.

It gives the operator a preparation countdown, performs CFO warmup, collects valid SSB samples, and saves them as a labeled H5 dataset.

Use this to collect training/validation data for a new factory or scenario.

---

### `capture_online_rxgridssb_dataset_cfo.py`

Lower-level online dataset capture tool.

It captures CFO-corrected `dataSSB` and `rxGridSSB` datasets but is less operator-friendly than `collect_labeled_rxgridssb_dataset_cfo.py`.

---

### `analyze_rxgrid_distributions.py`

Analysis script for saved H5 datasets.

It generates plots such as:

```text
amplitude histogram
phase histogram
IQ scatter
mean amplitude heatmap
mean amplitude by subcarrier
mean amplitude by OFDM symbol
```

Use this to check whether a dataset looks stable.

---

### `test_rxgrid_torch_checkpoint_on_h5.py`

Tests a `.pt` model on a saved H5 dataset.

Use this before online deployment to verify that a checkpoint can be loaded and that it produces predictions on stored `rxGridSSB` samples.

---

### `capture_iq_blocks_uhd.py`

Raw IQ capture script.

Use this only for low-level debugging. It saves raw IQ blocks before SSB extraction.

---

## 6. Run online inference locally

Activate the environment:

```bash
cd DT_sensing_fusion
source .venv_uhd/bin/activate
```

Run a short local test without SCP:

```bash
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

Expected behavior:

```text
CFO warmup: valid estimates
CFO median applied: around -5 kHz in the current setup
Online loop: valid=1
PSS metric: usually around 0.90-0.95 after CFO correction
Loop time: usually around 50-60 ms
JSON written locally
```

Local JSON path:

```text
results/online/live_inference_state_5G.json
```

Inspect it:

```bash
cat results/online/live_inference_state_5G.json
```

---

## 7. Run online inference and send JSON by SCP

Create the remote folder:

```bash
ssh <user>@<ip-or-hostname> \
  "mkdir -p <remote/path>"
```

Run the pipeline with SCP enabled:

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

```text
<user>@<ip-or-hostname>:<remote/path/live_inference_state_5G.json>
```

Example:

```bash
--remote-target "<user>@<ip-or-hostname>:<remote/path/live_inference_state_5G.json>"
```

Make sure SSH works first:

```bash
ssh <user>@<ip-or-hostname>
```

If SCP is too slow, send less frequently:

```bash
--scp-every 3
```

or:

```bash
--scp-every 5
```

---

## 8. Online JSON output

The online script writes a JSON file with both simple fields and detailed diagnostic fields.

Example:

```json
{
  "schema_version": "python_5g_ssb_online_v1",
  "source": "python_uspr_b210_ssb_pipeline",
  "timestamp_unix": 1782915703.9528105,
  "timestamp_utc": "2026-07-01T14:21:43.952812+00:00",
  "iteration": 29,
  "valid": true,
  "label": "P5",
  "prediction": "P5",
  "class_name": "P5",
  "class_id": 1,
  "confidence": 0.9995,
  "person_detected": true,
  "position": "P5",
  "probabilities": {
    "empty": 0.00049,
    "P5": 0.99951
  },
  "dsp": {
    "cfo_enabled": true,
    "cfo_hz_applied": -4870.8,
    "nid2": 0,
    "timing_offset_samples": 298063,
    "timing_offset_ms": 19.405,
    "pss_metric": 0.9425,
    "n_symbols_extracted": 6
  },
  "timing_ms": {
    "capture": 20.97,
    "pss": 19.10,
    "ofdm": 0.16,
    "dsp_total": 19.26,
    "loop_total": 51.87
  },
  "grid": {
    "rxGridSSB_shape": [240, 4]
  }
}
```

For a Digital Twin consumer, the most useful simple fields are:

```text
valid
label
prediction
class_name
class_id
confidence
person_detected
position
```

---

## 9. Collect a labeled dataset with the new Python script

Use this when creating a dataset for a new factory or a new set of positions/classes.

The script gives a preparation countdown before collecting data, so the operator has time to move to the desired position.

### 9.1 Collect EMPTY data

```bash
cd DT_sensing_fusion
source .venv_uhd/bin/activate

python src/python/ssb_python/collect_labeled_rxgridssb_dataset_cfo.py \
  --label empty \
  --scene static \
  --person-id none \
  --orientation none \
  --prep-sec 10 \
  --duration-sec 30 \
  --serial 34B73C3 \
  --freq 3541.44e6 \
  --rate 15.36e6 \
  --gain 60 \
  --duration-ms 20 \
  --channel 0 \
  --force-nid2 0 \
  --enable-cfo-correction \
  --cfo-warmup-iters 30 \
  --cfo-correction-sign -1 \
  --output-root data/python_ssb_datasets
```

### 9.2 Collect P5 data

```bash
python src/python/ssb_python/collect_labeled_rxgridssb_dataset_cfo.py \
  --label P5 \
  --scene static \
  --person-id person_1 \
  --orientation sideways \
  --prep-sec 10 \
  --duration-sec 30 \
  --serial 34B73C3 \
  --freq 3541.44e6 \
  --rate 15.36e6 \
  --gain 60 \
  --duration-ms 20 \
  --channel 0 \
  --force-nid2 0 \
  --enable-cfo-correction \
  --cfo-warmup-iters 30 \
  --cfo-correction-sign -1 \
  --output-root data/python_ssb_datasets
```

### 9.3 Collect a custom position

Example:

```bash
python src/python/ssb_python/collect_labeled_rxgridssb_dataset_cfo.py \
  --label P1 \
  --scene static \
  --person-id person_1 \
  --orientation front \
  --prep-sec 10 \
  --duration-sec 30 \
  --serial 34B73C3 \
  --freq 3541.44e6 \
  --rate 15.36e6 \
  --gain 60 \
  --duration-ms 20 \
  --channel 0 \
  --force-nid2 0 \
  --enable-cfo-correction \
  --cfo-warmup-iters 30 \
  --cfo-correction-sign -1 \
  --output-root data/python_ssb_datasets
```

---

## 10. Dataset output structure

The dataset script creates a session folder:

```text
data/python_ssb_datasets/<label>/<session_id>/
```

Example:

```text
data/python_ssb_datasets/P5/session_20260701_153000_P5_static/
```

Inside:

```text
session_data.h5
metadata.json
capture_log.csv
```

### `session_data.h5`

Contains accepted valid samples:

```text
dataSSB          complex64, shape [360, 6, N]
rxGridSSB        complex64, shape [240, 4, N]
pss_metric       float32, shape [N]
timing_offset_samples int64, shape [N]
timing_offset_ms float32, shape [N]
capture_time_ms  float32, shape [N]
pss_time_ms      float32, shape [N]
ofdm_time_ms     float32, shape [N]
dsp_time_ms      float32, shape [N]
loop_time_ms     float32, shape [N]
```

H5 attributes include:

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

Human-readable session metadata.

### `capture_log.csv`

One row per attempted capture, including valid and invalid attempts.

---

## 11. Analyze a collected dataset

```bash
LAST_H5="$(find data/python_ssb_datasets -name session_data.h5 | sort | tail -n 1)"

python src/python/ssb_python/analyze_rxgrid_distributions.py \
  --input "$LAST_H5" \
  --dataset rxGridSSB \
  --label python_dataset
```

---

## 12. Test the current model on a collected dataset

```bash
LAST_H5="$(find data/python_ssb_datasets -name session_data.h5 | sort | tail -n 1)"

python src/python/ssb_python/test_rxgrid_torch_checkpoint_on_h5.py \
  --model-pt results/binary_empty_vs_P5_rx/model_rxGridSSB/model.pt \
  --input-h5 "$LAST_H5" \
  --max-samples 30
```

---

## 13. Replacing the model

The online script accepts a different checkpoint:

```bash
--torch-model path/to/new/model.pt
```

Example:

```bash
python src/python/ssb_python/online_5g_python_cfo_json_scp.py \
  ... \
  --inference-backend torch \
  --torch-model results/my_new_model/model.pt
```

The current loader expects a checkpoint with:

```text
model_state_dict
mean
std
input_shape = [2, 240, 4]
complex_mode = abs_phase
classes = [...]
```

The current model architecture is implemented in:

```text
src/python/ssb_python/rxgrid_torch_inference.py
```

If the new model uses the same architecture and same input convention, no code change is needed.

If the new model uses a different architecture or different input channels, update:

```text
src/python/ssb_python/rxgrid_torch_inference.py
```

---

## 14. Recommended new-factory workflow

For a new factory or environment:

```text
1. Install repo and UHD environment.
2. Connect the USRP B210 and verify UHD.
3. Decide the target classes:
   empty, P1, P2, P3...
4. Collect labeled datasets with collect_labeled_rxgridssb_dataset_cfo.py.
5. Inspect the datasets with analyze_rxgrid_distributions.py.
6. Train a model using the collected Python rxGridSSB datasets.
7. Save the model checkpoint using the expected format.
8. Run online_5g_python_cfo_json_scp.py with --torch-model pointing to the new checkpoint.
9. Update --remote-target to the Digital Twin machine IP/path.
10. Validate the JSON consumed by the Digital Twin.
```

---

## 15. Git policy

Do not commit generated data:

```text
data/python_ssb_datasets/
results/online/
results/python_online_rxgridssb_dataset_cfo/
results/python_rxgrid_distribution/
logs/
```

Commit code, configs, and documentation only.

The small demo checkpoint may be committed if needed for demonstration:

```text
results/binary_empty_vs_P5_rx/model_rxGridSSB/model.pt
```

---

## 16. Coordinates and Mitsuba/Sionna XML export

The online inference script can export two synchronized outputs:

```text
1. JSON inference state
2. Mitsuba XML scene for Sionna/ray-tracing workflows
```

The JSON remains the main state interface. The Mitsuba XML is generated from the same prediction and the same position coordinates.

Main script:

```text
src/python/ssb_python/online_5g_python_cfo_json_scp.py
```

Position map:

```text
config/sionna_mitsuba_position_map.json
```

Local outputs:

```text
results/online/live_inference_state_5G.json
results/online/live_person_sionna_scene.xml
```

### 16.1 Position coordinates

The current map is stored in:

```text
config/sionna_mitsuba_position_map.json
```

Current coordinates are derived from the deployment grid:

```json
{
  "P1": {"translation_m": [6.0, 7.0, 0.0], "yaw_deg": 0.0},
  "P2": {"translation_m": [6.0, 5.0, 0.0], "yaw_deg": 0.0},
  "P3": {"translation_m": [4.0, 7.0, 0.0], "yaw_deg": 0.0},
  "P4": {"translation_m": [3.0, 5.0, 0.0], "yaw_deg": 0.0},
  "P5": {"translation_m": [2.0, 6.0, 0.0], "yaw_deg": 0.0}
}
```

Coordinate convention:

```text
x_m = top grid label
y_m = left grid label
z_m = floor height
```

If the Sionna scene uses another origin, scale, or axis convention, update this JSON file.

### 16.2 Local run with JSON + XML

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
  --mitsuba-position-map-json config/sionna_mitsuba_position_map.json \
  --enable-mitsuba-export \
  --disable-scp \
  --progress-every 1
```

Generated files:

```text
results/online/live_inference_state_5G.json
results/online/live_person_sionna_scene.xml
```

### 16.3 Send JSON and XML by SCP

Use the `--remote-target` argument for the JSON:

```bash
--remote-target "<user>@<ip-or-hostname>:<remote/path/live_inference_state_5G.json>"
```

Enable XML export with:

```bash
--enable-mitsuba-export
```

If `--remote-target` is provided for the JSON and no explicit XML target is provided, the XML is automatically sent to the same remote directory using the default filename:

```text
live_person_sionna_scene.xml
```

Example:

```bash
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
  --mitsuba-position-map-json config/sionna_mitsuba_position_map.json \
  --enable-mitsuba-export \
  --remote-target "<user>@<ip-or-hostname>:<remote/path/live_inference_state_5G.json>" \
  --progress-every 1
```

This sends:

```text
<user>@<ip-or-hostname>:<remote/path/live_inference_state_5G.json>
<user>@<ip-or-hostname>:<remote/path/live_person_sionna_scene.xml>
```

To use a custom XML destination, provide:

```bash
--mitsuba-scp-target "<user>@<ip-or-hostname>:<remote/path/custom_person_scene.xml>"
```

Check the remote files:

```bash
ssh <user>@<ip-or-hostname> \
  "ls -lh <remote/path/live_inference_state_5G.json> <remote/path/live_person_sionna_scene.xml>"
```

### 16.4 JSON coordinates

For a `P5` prediction, the JSON contains:

```json
"coordinates": {
  "available": true,
  "frame": "map_grid_m",
  "label": "P5",
  "translation_m": [2.0, 6.0, 0.0],
  "x_m": 2.0,
  "y_m": 6.0,
  "z_m": 0.0,
  "yaw_deg": 0.0
}
```

### 16.5 Mitsuba XML

For a `P5` prediction, the XML contains a standard human proxy placed at the same coordinates:

```xml
<shape type="sphere" id="person_P5_body">
    <transform name="to_world">
        <translate value="2,6,0.92"/>
        <scale value="0.28,0.18,0.62"/>
    </transform>
    <ref id="human_body_standard"/>
</shape>

<shape type="sphere" id="person_P5_head">
    <transform name="to_world">
        <translate value="2,6,1.62"/>
        <scale value="0.12,0.12,0.12"/>
    </transform>
    <ref id="human_body_standard"/>
</shape>
```

The XML is a Mitsuba-compatible scene file using standard scene elements such as:

```text
scene
shape
transform
translate
scale
bsdf
```

### 16.6 Radio material metadata

The XML includes radio-material metadata as comments:

```xml
<!--SIONNA_RADIO_MATERIAL name=human_body_standard relative_permittivity=38 relative_permeability=1 conductivity_s_per_m=1.46 frequency_hz=3.54144e+09-->
```

These values are placeholders. For accurate Sionna ray tracing, replace them with validated parameters for the target frequency and human/body material model.
