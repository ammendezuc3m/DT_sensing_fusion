# Informe rapido dataSSB empty vs P5

Archivos encontrados: 10
Capturas validas: 99292

## empty
- Capturas validas: 49640
- Bloques: 5
- Potencia media: 30.956 dB
- Potencia mediana: 30.796 dB

## P5
- Capturas validas: 49652
- Bloques: 5
- Potencia media: 30.039 dB
- Potencia mediana: 30.077 dB

## Split por bloque
- test: P5:block_01:datassb_side_v1_6labels_P5_block01_20260624_142259
- train: P5:block_02:datassb_side_v1_6labels_P5_block02_20260624_144936
- train: P5:block_03:datassb_side_v1_6labels_P5_block03_20260624_153828
- train: P5:block_04:datassb_side_v1_6labels_P5_block04_20260624_160151
- val: P5:block_05:datassb_side_v1_6labels_P5_block05_20260624_161810
- train: empty:block_01:datassb_side_v1_6labels_empty_block01_20260624_141149
- test: empty:block_02:datassb_side_v1_6labels_empty_block02_20260624_143504
- train: empty:block_03:datassb_side_v1_6labels_empty_block03_20260624_145939
- train: empty:block_04:datassb_side_v1_6labels_empty_block04_20260624_151053
- val: empty:block_05:datassb_side_v1_6labels_empty_block05_20260624_152235

## Figuras
- figures/power_by_subcarrier_empty_vs_P5.png
- figures/delta_power_by_subcarrier_P5_minus_empty.png
- figures/effect_size_by_subcarrier_P5_vs_empty.png
- figures/delta_power_heatmap_rxGridSSB_P5_minus_empty.png
- figures/total_power_distribution_empty_vs_P5.png
- figures/pca_power_features_empty_vs_P5.png si scikit-learn esta instalado
