# FairSSL

Official implementation for the paper:

**FairSSL: Fair Multimodal Self-Supervised Learning**

## Repository Structure

```text
.
|-- MIMIC_Main.py
|-- MODMA_Main2.py
|-- mimic_cxr_main.py
|-- configs/
|   |-- mimic.yaml
|   |-- mimic_cxr.yaml
|   `-- modma.yaml
|-- DataSets/
|   |-- MIMIC.py
|   |-- MIMIC_CXR.py
|   `-- MODMA.py
|-- Models/
|   |-- Basic_ConvNet.py
|   `-- FeatNet.py
|-- SSL/
|   `-- VICReg.py
`-- Utils/
    |-- config.py
    |-- Eval.py
    |-- loss_type_dataloader.py
    `-- metric_calculate.py
```

## Training Entry Points

| Dataset | Script | Configuration |
| --- | --- | --- |
| MIMIC | `MIMIC_Main.py` | `configs/mimic.yaml` |
| MIMIC-CXR | `mimic_cxr_main.py` | `configs/mimic_cxr.yaml` |
| MODMA | `MODMA_Main2.py` | `configs/modma.yaml` |

## Configuration

Experiment settings are defined in YAML files under `configs/`. Detailed configurations used for each reported result is described inside the paper appendix. Dataset paths, W&B settings, sweep definitions, run names, and FairSSL variant selections are loaded at runtime by the training scripts.

## Running Experiments

```bash
python MIMIC_Main.py
python mimic_cxr_main.py
python MODMA_Main2.py
```

Each script reads its corresponding YAML configuration and launches the configured FairSSL sweeps or runs.
