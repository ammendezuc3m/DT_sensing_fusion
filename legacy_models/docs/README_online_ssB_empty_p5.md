# Online 5G SSB EMPTY/P5 inference notes

This document summarizes the current online inference status.

---

## 1. Current state

The online demo detects:

```text
EMPTY = no person
P5    = person at position P5
```

The recommended online launcher is:

```bash
cd DT_sensing_fusion

./run_online_5g_binary_json_scp.sh
```

Short test:

```bash
MAX_VALID_SSB=200 ./run_online_5g_binary_json_scp.sh
```

---

## 2. Current pipeline

```text
MATLAB captures one valid SSB
MATLAB extracts rxGridSSB = dataSSB(61:300, 2:5)
MATLAB sends rxGridSSB to Python by TCP
Python transforms real/imag to abs/phase
Python normalizes
Python infers EMPTY/P5
Python stabilizes temporally
Python writes a JSON
Python sends the JSON by SCP to the Digital Twin machine
```

There is no queue and no accumulated delay. MATLAB waits for Python before capturing the next valid SSB.

---

## 3. Model

Recommended model:

```bash
results/binary_empty_vs_P5_rx/model_rxGridSSB/model.pt
```

Normalization:

```bash
results/binary_empty_vs_P5_rx/model_rxGridSSB/normalization_rx_abs_phase.npz
```

Metadata:

```bash
results/binary_empty_vs_P5_rx/model_rxGridSSB/metadata.csv
```

---

## 4. Input sample

MATLAB extraction:

```matlab
rxGridSSB = dataSSB(61:300, 2:5, i);
```

Shape:

```text
240 subcarriers x 4 OFDM symbols
```

Python input:

```text
[2, 240, 4]
channel 0 = abs(rxGridSSB)
channel 1 = angle(rxGridSSB)
```

---

## 5. JSON export

Local:

```bash
results/online/live_inference_state_5G.json
```

Remote:

```text
<user>@<ip-or-hostname>:<remote/path/live_inference_state_5G.json>
```

Check remote:

```bash
watch -n 0.5 "ssh <user>@<ip-or-hostname> 'cat <remote/path/live_inference_state_5G.json>'"
```

---

## 6. Python-only migration plan

The Python-only version should be implemented in this order:

```text
1. UHD raw IQ capture from USRP B210.
2. Save 20 ms waveform to .npz/.h5.
3. Implement PSS generation and timing estimation.
4. Implement NID2 and CFO estimation.
5. Implement OFDM demodulation.
6. Extract dataSSB = 360 x 6.
7. Extract rxGridSSB = dataSSB[60:300, 1:5].
8. Reuse the existing PyTorch model.
9. Write local JSON and SCP remote JSON.
```
