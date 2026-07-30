# DT_sensing_fusion

Passive 5G SSB sensing with a USRP B210, Python UHD processing and PyTorch inference.

This repository implements the **5G sensing component** of a larger Digital Twin system. It receives 5G Synchronization Signal Blocks (SSBs), extracts an `rxGridSSB` representation, applies a trained model and exports the latest inferred state as JSON. The JSON can remain local or be transferred automatically to another computer through SCP.

The repository also includes tools for:

- collecting labeled 5G sensing datasets;
- inspecting and analyzing captured datasets;
- testing compatible PyTorch checkpoints;
- generating a synchronized Mitsuba/Sionna XML scene;
- comparing the current Python processing with the historical MATLAB reference workflow.

> [!IMPORTANT]
> The included `empty` versus `P5` model is a demonstration model trained for one specific laboratory deployment. It is not a general human detector and is not expected to generalize automatically to another room, factory, antenna arrangement, 5G cell or set of positions.

---

## System architecture

The recommended online pipeline is:

```text
USRP B210
  → Python UHD IQ capture
  → CFO estimation and correction
  → PSS/NID2 detection and timing synchronization
  → OFDM demodulation
  → dataSSB extraction
  → rxGridSSB extraction
  → PyTorch inference
  → local JSON and CSV outputs
  → optional SCP transfer
  → optional Mitsuba/Sionna XML export
```

Main online script:

```text
src/python/ssb_python/online_5g_python_cfo_json_scp.py
```

Current model loader:

```text
src/python/ssb_python/rxgrid_torch_inference.py
```

Included demonstration checkpoint:

```text
results/binary_empty_vs_P5_rx/model_rxGridSSB/model.pt
```

---

## Repository scope

This repository currently contains the **5G sensing and inference side** of the system.

It does include:

- USRP B210 capture;
- Python-only 5G SSB processing;
- CFO correction;
- `dataSSB` and `rxGridSSB` extraction;
- PyTorch inference;
- local JSON generation;
- local CSV logging;
- SCP delivery;
- dataset collection and analysis;
- Mitsuba/Sionna XML generation;
- historical MATLAB validation utilities.

It does not currently include:

- the complete Isaac Sim application;
- the script that continuously reads the JSON inside Isaac Sim;
- Digital Twin scene update logic;
- MikroTik integration;
- complete orchestration of every Digital Twin component.

The remote path configured in this repository must later match the path read by the external Digital Twin consumer.

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
  uhd-host \
  python3-uhd
```

These packages install:

- Git;
- Python 3;
- `pip`;
- support for Python virtual environments;
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

## Step 6 — Verify the Python UHD installation

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

## Step 7 — Verify that UHD detects the USRP

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

## Step 8 — Probe the USRP

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

## Step 9 — Run a complete local 5G sensing test

This test runs the complete local sensing pipeline:

```text
USRP B210
  → IQ capture
  → CFO estimation and correction
  → PSS detection and timing synchronization
  → OFDM demodulation
  → dataSSB extraction
  → rxGridSSB extraction
  → PyTorch model inference
  → local JSON output
  → local CSV log
```

SCP transfer and Mitsuba export are disabled during this first test. This allows the complete sensing and inference pipeline to be validated locally before configuring a remote Digital Twin machine.

Replace `<serial_id>` with the serial number reported by `uhd_find_devices`:

```bash
python src/python/ssb_python/online_5g_python_cfo_json_scp.py \
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
  --inference-backend torch \
  --torch-model results/binary_empty_vs_P5_rx/model_rxGridSSB/model.pt \
  --torch-device cpu \
  --disable-scp \
  --progress-every 1
```

The command must be executed from the repository root because the model, configuration and output locations use relative paths.

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

The frequency, sample rate, gain and NID2 values are specific to the current 5G deployment. They may need to be changed when receiving a different 5G cell or using another experimental configuration.

### 5G processing parameters

- `--force-nid2 0` uses the expected PSS sequence corresponding to NID2 0.
- `--enable-cfo-correction` enables carrier frequency offset estimation and correction.
- `--cfo-warmup-iters 30` captures 30 initial blocks to obtain a stable CFO estimate.
- `--cfo-correction-sign -1` applies the estimated CFO using the sign expected by the current processing pipeline.

### Inference parameters

- `--inference-backend torch` selects the trained PyTorch neural network.
- `--torch-model results/binary_empty_vs_P5_rx/model_rxGridSSB/model.pt` loads the trained model included in the repository.
- `--torch-device cpu` performs inference using the CPU.

The selected model expects a complex `rxGridSSB` represented using amplitude and phase, with an effective input shape of:

```text
2 × 240 × 4
```

The two channels correspond to:

```text
channel 0 = amplitude
channel 1 = phase
```

### Execution and output parameters

- `--num-iters 30` performs 30 sensing iterations.
- `--warmup-iters 5` reserves initial iterations for execution warmup where applicable.
- `--progress-every 1` prints the result of every iteration.
- `--disable-scp` prevents remote JSON transfer during the local validation test.

### How to verify that the test succeeded

At startup, the script should report information similar to:

```text
=== Full Python 5G online inference ===
model type:         torch_cnn2d_abs_phase
inference backend:  torch
torch model:        results/binary_empty_vs_P5_rx/model_rxGridSSB/model.pt
local JSON:         results/online/live_inference_state_5G.json
remote target:      SCP disabled
```

This confirms that:

- the PyTorch inference backend was selected;
- the trained repository model was loaded;
- the local JSON output path was configured;
- SCP transfer was intentionally disabled.

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

After warmup, the script enters the online sensing loop. Each iteration produces a line similar to:

```text
[<iteration>] valid=1 label=<predicted_label> conf=<confidence> rx_mean=<mean_amplitude> pss=<pss_metric> loop=<loop_time_ms> ms scp=0 mitsuba=0 mitsuba_scp=0 err=
```

The fields mean:

- `valid=1`: the captured block passed the signal-processing validation.
- `label`: class predicted by the PyTorch model.
- `conf`: confidence associated with the selected class.
- `rx_mean`: mean magnitude of the extracted `rxGridSSB`.
- `pss`: PSS detection metric.
- `loop`: total capture, processing, inference and output time for the iteration.
- `scp=0`: SCP was disabled for this test.
- `mitsuba=0`: Mitsuba XML export was disabled.
- `mitsuba_scp=0`: no Mitsuba XML was sent remotely.
- `err=`: no error occurred.

A block is considered valid when:

- its PSS metric is greater than or equal to the configured minimum;
- the expected six OFDM symbols are extracted;
- the generated `rxGridSSB` has shape `240 × 4`.

The complete test can be considered successful when:

- the B210 is detected and configured;
- USB 3 operation is confirmed;
- CFO warmup obtains valid estimates;
- the sensing iterations show `valid=1`;
- the PSS metric remains above the configured minimum;
- the model produces labels and confidence values;
- `err=` remains empty;
- the local JSON file is created;
- the CSV log is created.

The value:

```text
scp=0
```

is expected and does not indicate an error because the test uses `--disable-scp`.

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

This is the complete loop time, not only the neural-network inference time. It includes capture, 5G processing, model inference, local output and, when enabled, SCP transfer.

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

Important top-level fields include:

- `timestamp_utc`;
- `iteration`;
- `valid`;
- `error`;
- `label`;
- `class_id`;
- `confidence`;
- `person_detected`;
- `position`;
- `probabilities`;
- `dsp`;
- `timing_ms`;
- `grid`;
- `inference`.

A valid result should contain:

```json
"valid": true,
"error": ""
```

The DSP section includes:

- whether CFO correction was enabled;
- the CFO applied in hertz;
- the detected NID2;
- the timing offset;
- the PSS metric;
- the number of extracted OFDM symbols.

The grid section should contain:

```json
"rxGridSSB_shape": [
    240,
    4
]
```

The inference section identifies the model and should report:

```json
"model_type": "torch_cnn2d_abs_phase"
```

It also records the model checkpoint path and the predicted class probabilities.

---

### Inspect the complete iteration log

Run:

```bash
cat results/online/python_5g_online_inference_log.csv
```

The CSV stores one row for every sensing iteration.

It includes:

- iteration number;
- validation result;
- predicted label;
- confidence;
- class probabilities;
- mean `rxGridSSB` magnitude;
- PSS metric;
- applied CFO;
- capture time;
- PSS processing time;
- OFDM processing time;
- complete DSP time;
- complete loop time;
- SCP status;
- Mitsuba status;
- errors.

Unlike the JSON, the current CSV log is appended to rather than replaced. Repeated executions may therefore add new rows to the existing file.

At this point, the initial local setup and complete local sensing validation are finished.

---

# Connect the sensing pipeline to a remote Digital Twin machine

The next objective is to send the generated inference JSON to the computer that will consume the sensing result.

The destination configured in the sensing command must be the same JSON path read by the external consumer:

```text
Sensing machine:
--remote-target <remote_user>@<remote_host>:<remote_json_path>

Remote consumer:
read JSON from <remote_json_path>
```

The two paths must refer to the same file.

## Step 10 — Verify SSH connectivity

Before testing the complete sensing pipeline, verify that the sensing computer can connect to the destination computer through SSH.

Run:

```bash
ssh <remote_user>@<remote_host>
```

The first connection may display:

```text
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```

Check that the displayed fingerprint corresponds to the destination computer and then enter:

```text
yes
```

SSH may then request the password of `<remote_user>`.

If the connection succeeds, exit the remote session:

```bash
exit
```

If the connection returns:

```text
Connection refused
```

the SSH server may not be installed or active on the destination computer.

On the destination computer, install and enable it:

```bash
sudo apt update
sudo apt install -y openssh-server
sudo systemctl enable --now ssh
```

Check its status:

```bash
sudo systemctl status ssh
```

The service should appear as:

```text
active (running)
```

---

## Step 11 — Configure passwordless SSH authentication

The sensing pipeline must transfer the JSON automatically. It cannot stop during every iteration to request the remote user password.

SSH key authentication must therefore be configured before running the pipeline with SCP enabled.

### 11.1 Check whether an SSH key already exists

On the sensing computer, run:

```bash
ls -la ~/.ssh
```

Look for a key pair such as:

```text
id_ed25519
id_ed25519.pub
```

The file without `.pub` is the private key and must never be shared.

The `.pub` file is the public key and can be copied to the destination computer.

### 11.2 Create an SSH key if necessary

If no suitable key exists, generate one:

```bash
ssh-keygen -t ed25519
```

When prompted for the key location, press Enter to use the default path:

```text
~/.ssh/id_ed25519
```

For fully automatic SCP execution, leave the passphrase empty by pressing Enter when prompted.

A key without a passphrase permits unattended execution. The private key must therefore be protected with appropriate file permissions and must not be copied or shared.

The generated files are:

```text
~/.ssh/id_ed25519
~/.ssh/id_ed25519.pub
```

### 11.3 Copy the public key to the destination computer

Run:

```bash
ssh-copy-id <remote_user>@<remote_host>
```

The command will request the remote account password one final time.

A successful installation should report that one or more keys were added.

If the machine has several SSH keys and the wrong public key is selected automatically, specify the desired key explicitly:

```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub <remote_user>@<remote_host>
```

If the SSH server uses a non-default port, run:

```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub \
  -p <ssh_port> \
  <remote_user>@<remote_host>
```

### 11.4 Verify passwordless SSH access

Run:

```bash
ssh <remote_user>@<remote_host>
```

The connection should open without requesting the remote account password.

Exit the session:

```bash
exit
```

For a non-interactive verification, run:

```bash
ssh -o BatchMode=yes <remote_user>@<remote_host> "echo SSH_OK"
```

Expected output:

```text
SSH_OK
```

The option:

```text
-o BatchMode=yes
```

prevents SSH from requesting a password interactively.

If key authentication is not working, the command will fail immediately instead of waiting for user input.

---

## Step 12 — Verify the remote destination directory

The directory that will contain the JSON must already exist on the destination computer.

Create it remotely if necessary:

```bash
ssh <remote_user>@<remote_host> \
  "mkdir -p <remote_directory>"
```

Verify that the directory exists and that the remote user can write to it:

```bash
ssh <remote_user>@<remote_host> \
  "test -d <remote_directory> && test -w <remote_directory> && echo DIRECTORY_OK"
```

Expected output:

```text
DIRECTORY_OK
```

The final destination file will be:

```text
<remote_directory>/<remote_json_filename>
```

The generic SCP target format is:

```text
<remote_user>@<remote_host>:<remote_directory>/<remote_json_filename>
```

---

## Step 13 — Test SCP independently

Before enabling SCP inside the sensing pipeline, test it with a small JSON file.

Create a temporary local JSON:

```bash
echo '{"scp_test": true}' > /tmp/test_scp.json
```

Transfer it:

```bash
scp /tmp/test_scp.json \
  <remote_user>@<remote_host>:<remote_directory>/<remote_json_filename>
```

The transfer must complete without requesting a password.

Verify the received file:

```bash
ssh <remote_user>@<remote_host> \
  "python3 -m json.tool <remote_directory>/<remote_json_filename>"
```

Expected output:

```json
{
    "scp_test": true
}
```

This confirms:

```text
local file
  → SSH authentication
  → SCP transfer
  → remote directory
  → remote JSON file
```

If this independent test fails, do not start the sensing pipeline yet. Resolve the SSH, authentication, permission or destination-path problem first.

---

## Step 14 — Run the 5G sensing pipeline with SCP enabled

Once passwordless SCP works, run the complete sensing pipeline without `--disable-scp`.

Replace:

- `<serial_id>` with the USRP serial number;
- `<remote_user>` with the destination user;
- `<remote_host>` with the destination hostname or IP address;
- `<remote_directory>` with the directory read by the remote consumer;
- `<remote_json_filename>` with the JSON filename read by the remote consumer.

```bash
python src/python/ssb_python/online_5g_python_cfo_json_scp.py \
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
  --inference-backend torch \
  --torch-model results/binary_empty_vs_P5_rx/model_rxGridSSB/model.pt \
  --torch-device cpu \
  --remote-target <remote_user>@<remote_host>:<remote_directory>/<remote_json_filename> \
  --scp-every 1 \
  --progress-every 1
```

### SCP parameters

- `--remote-target` defines the user, computer and complete destination path of the JSON.
- `--scp-every 1` transfers the JSON after every sensing iteration.
- `--disable-scp` must not be included when remote transfer is required.

The remote target has the following structure:

```text
<remote_user>@<remote_host>:<remote_json_path>
```

where:

```text
<remote_json_path> =
<remote_directory>/<remote_json_filename>
```

The remote consumer must read that same file.

---

## Step 15 — Verify the online SCP transfer

A successful sensing iteration should contain:

```text
scp=1
```

For example:

```text
[<iteration>] valid=1 label=<label> conf=<confidence> ... scp=1 ... err=
```

This confirms that:

- the local JSON was generated;
- the SCP process completed successfully;
- the remote JSON was updated.

If the output contains:

```text
scp=0
```

inspect the value following `err=`.

### Password or interactive authentication problem

Typical result:

```text
scp=0
err=scp_timeout
```

This commonly occurs when SCP is waiting for a password.

Verify passwordless authentication with:

```bash
ssh -o BatchMode=yes <remote_user>@<remote_host> "echo SSH_OK"
```

### Incorrect destination directory

Possible error:

```text
No such file or directory
```

Create the directory:

```bash
ssh <remote_user>@<remote_host> \
  "mkdir -p <remote_directory>"
```

### Insufficient permissions

Possible error:

```text
Permission denied
```

Check the directory ownership and permissions on the destination computer:

```bash
ssh <remote_user>@<remote_host> \
  "ls -ld <remote_directory>"
```

### Host verification problem

Possible error:

```text
Host key verification failed
```

Establish the SSH connection manually first:

```bash
ssh <remote_user>@<remote_host>
```

Verify the destination fingerprint before accepting it.

---

## Step 16 — Inspect the remote JSON

After the sensing test finishes, inspect the file on the destination computer:

```bash
ssh <remote_user>@<remote_host> \
  "python3 -m json.tool <remote_directory>/<remote_json_filename>"
```

For a 30-iteration test, the remote file should normally contain:

```json
"iteration": 29
```

It should also contain:

```json
"valid": true,
"error": ""
```

and an inference section describing the model prediction.

The local and remote files can be compared using checksums.

Local checksum:

```bash
sha256sum results/online/live_inference_state_5G.json
```

Remote checksum:

```bash
ssh <remote_user>@<remote_host> \
  "sha256sum <remote_directory>/<remote_json_filename>"
```

After the final transfer, both checksums should match.

---

## Step 17 — Monitor the remote JSON during execution

The destination computer can monitor the JSON while the sensing pipeline is running.

On the destination computer:

```bash
watch -n 0.5 \
  python3 -m json.tool <remote_directory>/<remote_json_filename>
```

The displayed fields should change as new sensing iterations are received.

A future Isaac Sim integration must use the same path:

```text
<remote_directory>/<remote_json_filename>
```

to read the current inferred state and update the Digital Twin.

---

# Collect a labeled 5G sensing dataset

The included demonstration model only distinguishes:

```text
empty
P5
```

and was trained for the current laboratory setup.

For another deployment, room, antenna location or set of positions, collect a new labeled dataset.

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

The radio and 5G processing parameters should match the online inference deployment.

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

Example for position `P5`:

```bash
python src/python/ssb_python/collect_labeled_rxgridssb_dataset_cfo.py \
  --label P5 \
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
--label P5
```

with the desired class, for example:

```text
--label P1
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
data/python_ssb_datasets/P5/session_<timestamp>_P5_static/
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

# Test a model checkpoint on a saved dataset

Find the latest dataset:

```bash
LAST_H5="$(find data/python_ssb_datasets -name session_data.h5 | sort | tail -n 1)"
```

Run the included model on up to 30 samples:

```bash
python src/python/ssb_python/test_rxgrid_torch_checkpoint_on_h5.py \
  --model-pt results/binary_empty_vs_P5_rx/model_rxGridSSB/model.pt \
  --input-h5 "$LAST_H5" \
  --max-samples 30
```

This verifies that:

- the H5 dataset can be opened;
- `rxGridSSB` can be read;
- the checkpoint can be loaded;
- the model accepts the expected input shape;
- predictions can be produced for saved samples.

A successful technical test does not guarantee that the model is appropriate for the new environment.

---

# Model checkpoint format

The current online loader expects a PyTorch checkpoint containing:

```text
model_state_dict
mean
std
model_name
input_shape
complex_mode
classes
config
```

The expected input convention is:

```text
input_shape = [2, 240, 4]
complex_mode = abs_phase
```

The model input tensor has shape:

```text
[batch, 2, 240, 4]
```

Its channels are:

```text
channel 0 = abs(rxGridSSB)
channel 1 = angle(rxGridSSB)
```

Normalization is performed using the values stored in the checkpoint:

```text
x = (x - mean) / std
```

The current architecture and loader are implemented in:

```text
src/python/ssb_python/rxgrid_torch_inference.py
```

---

# Replace the online model

To use another compatible checkpoint, change:

```text
--torch-model
```

Example:

```bash
python src/python/ssb_python/online_5g_python_cfo_json_scp.py \
  ... \
  --inference-backend torch \
  --torch-model results/my_new_model/model.pt
```

No loader change is required when the new checkpoint uses:

```text
input_shape = [2, 240, 4]
complex_mode = abs_phase
```

and the same implemented architecture.

If the architecture, input dimensions or channel representation change, update:

```text
src/python/ssb_python/rxgrid_torch_inference.py
```

The class list stored in the checkpoint must agree with the labels used during training and with any downstream consumer of the JSON.

---

# Recommended workflow for a new deployment

For a new room, factory or antenna arrangement:

1. Install the repository and UHD environment.
2. Connect the USRP B210 and verify UHD communication.
3. Determine the correct frequency, sample rate, gain, channel, antenna and NID2.
4. Define the target classes, for example `empty`, `P1`, `P2`, `P3`.
5. Collect several labeled sessions for every class.
6. Inspect the datasets for capture and preprocessing problems.
7. Split the data correctly into training, validation and test groups.
8. Train a model using Python-generated `rxGridSSB` data.
9. Save a checkpoint in the expected format.
10. Test the checkpoint on saved H5 data.
11. Run online inference with the new checkpoint.
12. Configure the remote JSON destination.
13. Validate the state read by the external Digital Twin consumer.

Do not assume that the demonstration checkpoint will work correctly in the new deployment.

---

# Mitsuba/Sionna XML export

The online inference script can export two synchronized outputs:

```text
1. JSON inference state
2. Mitsuba XML scene for Sionna or ray-tracing workflows
```

The JSON remains the main machine-readable sensing state.

The XML is generated from the same prediction and configured position coordinates.

Position map:

```text
config/sionna_mitsuba_position_map.json
```

Default local XML output:

```text
results/online/live_person_sionna_scene.xml
```

## Position-coordinate map

The coordinate file maps each predicted position to a translation and orientation.

Example structure:

```json
{
  "P1": {
    "translation_m": [6.0, 7.0, 0.0],
    "yaw_deg": 0.0
  },
  "P5": {
    "translation_m": [2.0, 6.0, 0.0],
    "yaw_deg": 0.0
  }
}
```

The current map uses a deployment-specific coordinate convention.

Before using it in another scene, verify:

- scene origin;
- X-axis direction;
- Y-axis direction;
- Z-axis direction;
- units;
- floor height;
- object orientation.

If the Sionna or Mitsuba scene uses another origin, scale or axis convention, update:

```text
config/sionna_mitsuba_position_map.json
```

---

## Generate JSON and XML locally

Run:

```bash
python src/python/ssb_python/online_5g_python_cfo_json_scp.py \
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

A successful iteration should show:

```text
mitsuba=1
```

Because SCP is disabled, the same line should show:

```text
mitsuba_scp=0
```

---

## Send JSON and XML to a remote machine

Use the normal JSON destination:

```text
--remote-target <remote_user>@<remote_host>:<remote_directory>/<remote_json_filename>
```

Enable XML generation:

```text
--enable-mitsuba-export
```

Example:

```bash
python src/python/ssb_python/online_5g_python_cfo_json_scp.py \
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
  --inference-backend torch \
  --torch-model results/binary_empty_vs_P5_rx/model_rxGridSSB/model.pt \
  --torch-device cpu \
  --remote-target <remote_user>@<remote_host>:<remote_directory>/<remote_json_filename> \
  --scp-every 1 \
  --mitsuba-position-map-json config/sionna_mitsuba_position_map.json \
  --enable-mitsuba-export \
  --progress-every 1
```

If no explicit XML target is configured, the current script can send the XML to the same remote directory using:

```text
live_person_sionna_scene.xml
```

The remote directory must therefore be writable and must be the directory expected by the external Sionna/Mitsuba workflow.

Successful iterations should show:

```text
scp=1
mitsuba=1
mitsuba_scp=1
```

---

# Threshold inference backend

The repository contains a simple threshold backend for debugging the JSON and SCP mechanisms.

Configuration file:

```text
config/generic_5g_binary_model.json
```

Example selection:

```bash
--inference-backend threshold \
--model-config config/generic_5g_binary_model.json
```

This is not the trained PyTorch sensing model.

Use the threshold backend only for development or transport testing. For the actual model pipeline, use:

```bash
--inference-backend torch \
--torch-model results/binary_empty_vs_P5_rx/model_rxGridSSB/model.pt
```

---

# Main scripts

| Script | Purpose |
|---|---|
| `src/python/ssb_python/online_5g_python_cfo_json_scp.py` | Complete online Python sensing, inference, JSON, SCP and optional XML export |
| `src/python/ssb_python/rxgrid_torch_inference.py` | Loads the PyTorch checkpoint and performs inference |
| `src/python/ssb_python/collect_labeled_rxgridssb_dataset_cfo.py` | Operator-friendly labeled dataset collection |
| `src/python/ssb_python/capture_online_rxgridssb_dataset_cfo.py` | Lower-level CFO-corrected dataset capture |
| `src/python/ssb_python/test_rxgrid_torch_checkpoint_on_h5.py` | Tests a checkpoint on a saved H5 dataset |
| `src/python/ssb_python/analyze_rxgrid_distributions.py` | Generates diagnostic plots for saved datasets |
| `src/python/ssb_python/capture_iq_blocks_uhd.py` | Raw IQ capture for low-level debugging |
| `src/python/ssb_python/compare_interleaved_python_matlab.py` | Historical Python/MATLAB comparison |

---

# Generated files and Git policy

Do not commit large generated data or runtime output folders.

Typical generated paths include:

```text
data/python_ssb_datasets/
results/online/
results/python_online_rxgridssb_dataset_cfo/
results/python_rxgrid_distribution/
results/python_matlab_rxgrid_compare/
logs/
```

The included demonstration checkpoint may remain in the repository when required for end-to-end testing:

```text
results/binary_empty_vs_P5_rx/model_rxGridSSB/model.pt
```

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
- CFO estimate.

The deployment-specific parameters may not match the received 5G cell.

---

## The model checkpoint cannot be loaded

Verify the path:

```bash
ls -l results/binary_empty_vs_P5_rx/model_rxGridSSB/model.pt
```

The command must be executed from the repository root.

A replacement checkpoint must use the format expected by:

```text
src/python/ssb_python/rxgrid_torch_inference.py
```

---

## SCP asks for a password

Configure SSH key authentication.

Verify:

```bash
ssh -o BatchMode=yes <remote_user>@<remote_host> "echo SSH_OK"
```

If multiple SSH keys exist, copy the intended public key explicitly:

```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub \
  <remote_user>@<remote_host>
```

---

## SCP reports `Permission denied`

Check the remote directory:

```bash
ssh <remote_user>@<remote_host> \
  "ls -ld <remote_directory>"
```

The remote user must have write permission.

---

## SCP reports `No such file or directory`

Create the directory:

```bash
ssh <remote_user>@<remote_host> \
  "mkdir -p <remote_directory>"
```

---

## `Host key verification failed`

Connect manually:

```bash
ssh <remote_user>@<remote_host>
```

Verify the fingerprint before accepting it.

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
## Automatic SSB Discovery

Discover unknown 5G NR FR1 SSBs without prior knowledge of the carrier frequency, band, SCS or NID2.

Example:

```bash
python scan_5g_ssb_auto.py \
    --serial <USRP_SERIAL> \
    --gain 60 \
    --mode quick
```
###Useful options:
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
README_online_ssB_empty_p5.md
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

# Current project status

Completed:

- Python-only USRP capture;
- CFO estimation and correction;
- PSS and timing detection;
- OFDM demodulation;
- `dataSSB` extraction;
- `rxGridSSB` extraction;
- PyTorch inference;
- local JSON output;
- local CSV log;
- passwordless SCP-compatible remote transfer;
- labeled dataset collection;
- dataset analysis;
- checkpoint testing;
- optional Mitsuba/Sionna XML output.

Future integration work:

- Isaac Sim JSON consumer;
- Digital Twin scene update logic;
- final coordinate alignment between sensing and the Digital Twin;
- MikroTik integration;
- complete multi-component orchestration.
