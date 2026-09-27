# Dataset Card: Colorectal Cancer Tissue Classification

## 1. Benchmark Datasets

| Dataset Attribute | Source Domain (`NCT-CRC-HE-100K-NONORM`) | Target Domain (`CRC-VAL-HE-7K`) |
| :--- | :--- | :--- |
| **Patches & Resolution** | 100,000 patches (224 x 224 px, 0.5 µm/px) | 7,180 patches (224 x 224 px, 0.5 µm/px) |
| **Stain Shift** | Routine clinical H&E (unnormalized raw) | External hospital laboratory protocol & scanner |
| **Down load link** | [10.5281/zenodo.1214456](https://doi.org/10.5281/zenodo.1214456)![NCT-100K](./images/NCT-100k.png) | [10.5281/zenodo.1214456](https://doi.org/10.5281/zenodo.1214456)![CRC-7k](./images/CRC-7k.png) |

---

## 2. Nine Histological Classes

| Index | Code | Class Name | Morphological Description |
| :---: | :--- | :--- | :--- |
| 0 | `ADI` | Adipose | Lipid-filled fat cells with eccentric nuclei. |
| 1 | `BACK`| Background | Transparent glass slide areas. |
| 2 | `DEB` | Debris | Necrotic material, hemorrhage, cell fragments. |
| 3 | `LYM` | Lymphocytes | Dense aggregates of dark, round hyperchromatic immune nuclei. |
| 4 | `MUC` | Mucus | Pools of extracellular mucin. |
| 5 | `MUS` | Muscle | Eosinophilic smooth muscle bundles. |
| 6 | `NORM`| Normal Mucosa | Healthy non-dysplastic colorectal mucosal crypts. |
| 7 | `STR` | Stroma | Cancer-associated fibrous stroma and desmoplasia. |
| 8 | `TUM` | Tumor | Colorectal adenocarcinoma epithelium. |

---

## 3. Split & Firewall Protocol

1. **Source Domain**: Stratified 70% train / 15% val_id / 15% test_id with fixed `seed=42`. Default compute budget: standardized 25,000 patch subset (17,500 train / 3,750 val / 3,750 test).
2. **Canonical Reference**: Exactly one dense tumor tile chosen strictly from the **training split** only.
3. **Strict Domain Firewall**: `CRC-VAL-HE-7K` is strictly evaluated post-training as `test_ood`. Never used for model training, normalizer fitting, or checkpoint selection.
