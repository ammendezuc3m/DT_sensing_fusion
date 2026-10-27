# Legacy models and experiments

This folder keeps the earlier sensing models, experiments and documentation for
reference. **None of it is used by the current pipeline.**

The current 5G SSB workflow uses only:

- `src/python/run_ssb_sensing_analysis.py` (offline analysis and Fisher +
  logistic-regression training);
- `src/python/ssb_python/online_5g_ssb_sensing.py` (online inference);
- the Sensing API under `src/python/sensing_api/`.

See the main `README.md`.

Contents:

| Path | What it was |
|---|---|
| `src/python/` | Older dataset builders, CNN/ROI/hybrid training pipelines, P3/P5 explainability and calibration scripts, older online inference servers |
| `src/python/ssb_python/rxgrid_torch_inference.py` | Loader of the legacy rxGridSSB CNN checkpoint |
| `src/python/ssb_python/test_rxgrid_torch_checkpoint_on_h5.py` | Replay test for the legacy CNN checkpoint |
| `results/binary_empty_vs_P5_rx/` | Legacy binary `empty`/`P5` CNN demonstration model |
| `config/generic_5g_binary_model.json` | Parameters of the legacy threshold backend |
| `config/sionna_mitsuba_position_map.json` | Legacy label-to-Mitsuba map (replaced by `config/sensing_label_map.json`) |
| `scripts/run_online_5g_binary_json_scp.sh` | Legacy online launcher with SCP delivery |
| `docs/` | Legacy READMEs and runbooks |

The scripts were moved without modification. Some of them use paths relative to
their original location (`src/python/`), so they may need path adjustments if
you run them again.
