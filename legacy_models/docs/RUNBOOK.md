# DT_sensing_fusion Runbook

Operational runbook for capture, online inference, JSON export, Python UHD capture and Git workflow.

---

## 1. Online 5G EMPTY/P5 inference

Recommended command:

```bash
cd ~/AlbertoDir/DT_sensing_fusion

./run_online_5g_binary_json_scp.sh
```

Short test:

```bash
cd ~/AlbertoDir/DT_sensing_fusion

MAX_VALID_SSB=200 ./run_online_5g_binary_json_scp.sh
```

The script:

```text
- activates `.venv`
- starts the Python inference server
- waits a few seconds
- starts MATLAB streamer
- writes local JSON
- sends remote JSON by SCP
- kills Python when MATLAB exits
```

---

## 2. Check remote JSON

Check once:

```bash
ssh nextnet@163.117.140.146 \
  "cat ~/AlbertoDir/demo_5G/5G_inference/live_inference_state_5G.json"
```

Live check from the local machine:

```bash
watch -n 0.5 "ssh nextnet@163.117.140.146 'cat ~/AlbertoDir/demo_5G/5G_inference/live_inference_state_5G.json'"
```

Alternative without `watch` on the remote machine:

```bash
ssh nextnet@163.117.140.146 \
  'while true; do printf "\033c"; date; cat ~/AlbertoDir/demo_5G/5G_inference/live_inference_state_5G.json; sleep 0.5; done'
```

---

## 3. MATLAB dataset collection

Command:

```bash
matlab -batch "addpath('src/matlab'); collect_ssb_dataset('empty','static',30,'none','none',10)"
```

Meaning:

```text
label           = 'empty'
movementState   = 'static'
durationSeconds = 30
personId        = 'none'
orientation     = 'none'
pauseBeforeSec  = 10
```

Examples:

```bash
matlab -batch "addpath('src/matlab'); collect_ssb_dataset('empty','static',30,'none','none',10)"
matlab -batch "addpath('src/matlab'); collect_ssb_dataset('P5','static',30,'person_1','sideways',10)"
```

---

## 4. MATLAB online streamer only

Use when the Python inference server is already running:

```bash
cd ~/AlbertoDir/DT_sensing_fusion

SEND_EVERY_N=1 \
matlab -batch "run('src/matlab/stream_rxgridssb_online_to_python.m')"
```

Short test:

```bash
cd ~/AlbertoDir/DT_sensing_fusion

MAX_VALID_SSB=100 \
SEND_EVERY_N=1 \
matlab -batch "run('src/matlab/stream_rxgridssb_online_to_python.m')"
```

---

## 5. Python inference server only

```bash
cd ~/AlbertoDir/DT_sensing_fusion

.venv/bin/python src/python/online_rxgridssb_inference_server.py
```

---

## 6. UHD Python environment

Activate:

```bash
cd ~/AlbertoDir/DT_sensing_fusion

source .venv_uhd/bin/activate
```

Check USRP B210:

```bash
python - <<'PY'
import uhd

usrp = uhd.usrp.MultiUSRP("serial=34B73C3")
print("USRP OK")
print("RX channels:", usrp.get_rx_num_channels())
print("Motherboard:", usrp.get_mboard_name())
PY
```

---

## 7. Python UHD raw IQ capture

### 7.1. Single 20 ms capture

```bash
cd ~/AlbertoDir/DT_sensing_fusion
source .venv_uhd/bin/activate

python src/python/ssb_python/test_capture_iq_uhd.py \
  --serial 34B73C3 \
  --freq 3541.44e6 \
  --rate 15.36e6 \
  --gain 60 \
  --duration-ms 20 \
  --channel 0
```

Expected:

```text
waveform shape = (307200,)
dtype = complex64
```

### 7.2. Repeated 20 ms block capture

```bash
cd ~/AlbertoDir/DT_sensing_fusion
source .venv_uhd/bin/activate

python src/python/ssb_python/capture_iq_blocks_uhd.py \
  --serial 34B73C3 \
  --freq 3541.44e6 \
  --rate 15.36e6 \
  --gain 60 \
  --duration-ms 20 \
  --num-blocks 20 \
  --channel 0 \
  --progress-every 1
```

Expected:

```text
waveform shape = (20, 307200)
dtype = complex64
```

Recommended initial Python capture gain:

```text
gain = 60 dB
```

Current observed capture performance:

```text
capture duration/block ≈ 20.2 ms
samples/block = 307200
blocks_per_second ≈ 15.4 with compressed HDF5 writing
```

The HDF5 write overhead is only for offline testing. The future online pipeline will process blocks in memory.

---

## 8. Next Python-only implementation phase

The next implementation phase is offline IQ inspection.

Goal:

```text
read captured .h5/.npz IQ files
compute block-level signal metrics
plot power over samples
plot spectrum
detect obvious bursts/energy changes
prepare for PSS/NID2 detection
```

Target script:

```text
src/python/ssb_python/inspect_iq_capture.py
```

After that, the next DSP phase will be:

```text
PSS generation -> NID2 search -> timing estimate -> CFO -> OFDM demod -> dataSSB
```

---

## 9. Git workflow

Check status:

```bash
cd ~/AlbertoDir/DT_sensing_fusion

git status --short
```

Add selected files only:

```bash
git add README.md docs/ requirements/ src/ run_*.sh
```

Commit:

```bash
git commit -m "Message"
```

Push:

```bash
git push
```

Avoid:

```bash
git add .
```

unless `.gitignore` has been reviewed carefully.

---

## 10. Files that should not be committed

Do not commit:

```text
.venv/
.venv_uhd/
.venv_datassb/
data/
logs/
runtime/
backups/
*.mat
*.h5
*.hdf5
large raw captures
temporary online JSON files
```