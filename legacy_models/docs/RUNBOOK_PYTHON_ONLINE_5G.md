# RUNBOOK — Latest Python Online SSB Sensing

This runbook contains copy/paste commands for the latest Python-only online pipeline.

## 1. Check Git status before adding latest work

```bash
cd ~/AlbertoDir/DT_sensing_fusion
git status --short
```

You should see newly created/modified scripts such as:

```text
src/python/ssb_python/cfo_utils.py
src/python/ssb_python/capture_online_rxgridssb_dataset_cfo.py
src/python/ssb_python/rxgrid_torch_inference.py
src/python/ssb_python/online_5g_python_cfo_json_scp.py
src/python/ssb_python/test_rxgrid_torch_checkpoint_on_h5.py
config/generic_5g_binary_model.json
run_python_matlab_interleaved_capture.sh
```

Generated folders should not be committed.

## 2. Recommended .gitignore additions

Append only if these are not already present:

```bash
cd ~/AlbertoDir/DT_sensing_fusion

cat >> .gitignore <<'EOF'

# Generated Python online / CFO / comparison outputs
results/online/
results/python_online_profile/
results/python_online_rxgridssb_dataset/
results/python_online_rxgridssb_dataset_cfo/
results/python_rxgrid_distribution/
results/python_matlab_rxgrid_compare/
results/python_datassb_batch/
results/python_datassb_extract/
results/python_pss_detection/
results/python_iq_inspection/
logs/python_matlab_interleaved_*/

# Raw generated captures
data/python_iq_captures/
data/python_iq_blocks/
EOF
```

## 3. Add only the latest code/config/docs

```bash
cd ~/AlbertoDir/DT_sensing_fusion

git add .gitignore

# Latest Python online and CFO code
git add src/python/ssb_python/cfo_utils.py
git add src/python/ssb_python/capture_online_rxgridssb_dataset_cfo.py
git add src/python/ssb_python/capture_online_rxgridssb_dataset.py
git add src/python/ssb_python/profile_online_datassb_pipeline.py
git add src/python/ssb_python/online_5g_python_cfo_json_scp.py
git add src/python/ssb_python/rxgrid_torch_inference.py
git add src/python/ssb_python/test_rxgrid_torch_checkpoint_on_h5.py

# Analysis/comparison helpers
git add src/python/ssb_python/analyze_rxgrid_distributions.py
git add src/python/ssb_python/compare_interleaved_python_matlab.py
git add src/python/ssb_python/extract_datassb_batch_offline.py
git add src/python/ssb_python/compare_python_matlab_rxgrid.py 2>/dev/null || true

# Config and helper script
git add config/generic_5g_binary_model.json
git add run_python_matlab_interleaved_capture.sh

# Docs from this package after copying them into the repo
git add README_online_python_5g.md
git add docs/RUNBOOK_PYTHON_ONLINE_5G.md
git add docs/DATASET_MODEL_JSON_GUIDE.md
```

Ensure generated data is not staged:

```bash
git restore --staged results 2>/dev/null || true
git restore --staged data 2>/dev/null || true
git restore --staged logs 2>/dev/null || true
```

Check staged files:

```bash
git status --short
```

Commit:

```bash
git commit -m "Add Python-only CFO SSB online inference and documentation"
git push
```

## 4. Local online test with the real PyTorch model

```bash
cd ~/AlbertoDir/DT_sensing_fusion
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

Expected behavior:

```text
CFO warmup result: valid estimates 30/30
CFO median applied: around -5 kHz
Online loop: valid=1
PSS metric: around 0.90-0.95
loop: around 50-60 ms
```

## 5. Test the PyTorch checkpoint on a saved H5

```bash
LAST_CFO_H5="$(ls -t results/python_online_rxgridssb_dataset_cfo/*.h5 | head -n 1)"

python src/python/ssb_python/test_rxgrid_torch_checkpoint_on_h5.py \
  --model-pt results/binary_empty_vs_P5_rx/model_rxGridSSB/model.pt \
  --input-h5 "$LAST_CFO_H5" \
  --max-samples 30
```

## 6. Run online with SCP

```bash
ssh nextnet@163.117.140.146 \
  "mkdir -p ~/AlbertoDir/demo_5G/5G_inference"
```

```bash
cd ~/AlbertoDir/DT_sensing_fusion
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
  --remote-target "nextnet@163.117.140.146:~/AlbertoDir/demo_5G/5G_inference/live_inference_state_5G.json" \
  --scp-every 1 \
  --progress-every 1
```

Remote watch:

```bash
watch -n 0.5 "ssh nextnet@163.117.140.146 'cat ~/AlbertoDir/demo_5G/5G_inference/live_inference_state_5G.json'"
```

## 7. Capture a processed Python dataset

```bash
cd ~/AlbertoDir/DT_sensing_fusion
source .venv_uhd/bin/activate

python src/python/ssb_python/capture_online_rxgridssb_dataset_cfo.py \
  --serial 34B73C3 \
  --freq 3541.44e6 \
  --rate 15.36e6 \
  --gain 60 \
  --duration-ms 20 \
  --num-iters 200 \
  --warmup-iters 10 \
  --channel 0 \
  --force-nid2 0 \
  --enable-cfo-correction \
  --cfo-warmup-iters 30 \
  --cfo-correction-sign -1 \
  --progress-every 10
```

Output:

```text
results/python_online_rxgridssb_dataset_cfo/*.h5
```

## 8. Analyze distributions

```bash
LAST_H5="$(ls -t results/python_online_rxgridssb_dataset_cfo/*.h5 | head -n 1)"

python src/python/ssb_python/analyze_rxgrid_distributions.py \
  --input "$LAST_H5" \
  --dataset rxGridSSB \
  --label python_cfo
```

Most useful plots:

```text
hist_amplitude_db.png
hist_phase.png
iq_scatter.png
mean_amplitude_heatmap_db.png
mean_amplitude_by_subcarrier.png
mean_amplitude_by_symbol.png
```

## 9. Compare with MATLAB if needed

Run interleaved capture:

```bash
./run_python_matlab_interleaved_capture.sh empty static 30 none none 10
```

Compare:

```bash
python src/python/ssb_python/compare_interleaved_python_matlab.py \
  --python-pre path/to/python_pre.h5 \
  --matlab-mat path/to/session_data.mat \
  --python-post path/to/python_post.h5 \
  --normalize none \
  --out-dir results/python_matlab_rxgrid_compare/empty_static_raw
```

Important figures:

```text
compare_mean_amplitude_by_subcarrier.png
compare_mean_amplitude_by_symbol.png
compare_hist_amplitude_db.png
compare_hist_phase.png
heatmap_python_pre.png
heatmap_matlab.png
heatmap_python_post.png
```
