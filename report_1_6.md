# Robust Histopathology Image Classification under Staining Variations
**Course Project Report — Deep Learning (DL2026)**  
**Group 1 · Project 32**  
**Repository:** `DL2026-1-32`

---

## 1. Abstract

Deep learning models achieve high accuracy in digital histopathology. However, when these models run on images from another hospital, their performance often drops sharply. This problem happens because staining chemicals, laboratory protocols, and slide scanners create large color variations between hospitals. In this project, we study how to make convolutional and transformer models robust against staining shifts in colorectal cancer tissue classification. We train models on 24,993 patches from the NCT-CRC-HE-100K-NONORM dataset (Heidelberg cohort) and test them on 7,180 unseen patches from the CRC-VAL-HE-7K dataset (Aachen cohort) across nine tissue classes. We design a 13-experiment ablation matrix to evaluate three defense axes: color normalization (Reinhard and Macenko), data augmentation (geometric transformations and biological HED stain jitter), and model backbones (ResNet-50, ConvNeXt-Tiny, and the Phikon foundation model). Our experiments reveal three key findings: stain normalization recovers up to 18 F1 points on external data; stain augmentation becomes redundant once normalization is applied; and pretrained foundation models show strong zero-shot stain robustness but suffer when artificial normalization alters their learned feature distributions. Our best-defended ResNet-50 model achieves an out-of-domain Macro-F1 score of 0.8689.

---

## 2. Introduction and Research Question

### 2.1. Clinical Background and Problem Statement
Colorectal cancer is one of the leading causes of cancer deaths worldwide. In clinical pathology, doctors examine tissue biopsies under a microscope to confirm diagnosis and plan patient treatment. To see cellular structures clearly, laboratory technicians stain thin tissue slices with two chemical dyes: **Hematoxylin** and **Eosin** (H&E).
- **Hematoxylin** stains cell nuclei deep purple-blue because it binds to acidic nucleic acids (DNA and RNA).
- **Eosin** stains the cytoplasm, extracellular matrix, and muscle fibers bright pink-red because it binds to basic proteins.

Digital pathology scanners convert these glass slides into high-resolution digital images called Whole Slide Images (WSI). In recent years, Deep Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) have shown impressive diagnostic accuracy on digital slides. 

However, a critical barrier stops these AI models from safe clinical deployment: **inter-laboratory staining variation**. Even though all hospitals follow standard H&E protocols, the visual appearance of slides differs greatly between medical centers. These color shifts happen due to:
1. Different chemical manufacturers, dye purity, and reagent batch ages.
2. Variations in tissue processing time, fixation quality, and microtome section thickness.
3. Differences in optical lenses, light sources, and sensor color calibration across digital slide scanner brands (such as Philips, Leica, or Hamamatsu).

```text
Source Hospital (Heidelberg)             Target Hospital (Aachen)
- Deep purple hematoxylin bias           - Bright pink eosin bias
- Specific scanner illumination profile  - Independent optical calibration
                 \                           /
                  \                         /
                   v                       v
               [Deep Learning Model Trained on Source]
                                |
             +------------------+------------------+
             |                                     |
             v                                     v
     In-Domain Evaluation                 Out-of-Domain Evaluation
   High Accuracy (~98.9% F1)            Severe Drop (~66.4% F1)
```

### 2.2. The Shortcut Learning Challenge
When a deep learning model trains on images from only one hospital, it easily learns "shortcuts" instead of true biological features. For example, if tumor regions at Hospital A happen to have slightly darker purple tones due to high nuclear density, the network may simply learn: *"Dark purple means cancer, light pink means benign tissue"*.

When this model is deployed at Hospital B, where normal muscle tissue is stained darker purple or cancer tissue is stained lighter pink, the model becomes confused and fails. In machine learning, this failure is called **domain shift** or **out-of-distribution (OOD) degradation**. In clinical medicine, such errors can lead to missed tumors or false cancer diagnoses.

### 2.3. Research Questions
To understand what interventions effectively protect deep models against staining variations, we formulate five direct research questions:

- **Research Question 1 (Anchor Baseline):** What is the exact performance drop when an undefended standard model (ResNet-50) is tested on an external hospital cohort?
- **Research Question 2 (Color Normalization):** Can algorithmic stain normalization (statistical Reinhard transfer vs. physical Macenko deconvolution) close the performance gap without retraining?
- **Research Question 3 (Data Augmentation):** Can synthetic stain perturbation (HED jitter) during training match or exceed the benefit of test-time stain normalization?
- **Research Question 4 (Defense Synergy):** Are color normalization and data augmentation complementary, or do they conflict when combined?
- **Research Question 5 (Architectural Robustness):** Does a modern convolutional network (ConvNeXt-Tiny) or a self-supervised pathology foundation model (Phikon) possess natural stain invariance without explicit defenses?

---

## 3. Related Work

Our research connects three major areas in computational pathology: stain normalization algorithms, data augmentation strategies, and pretrained vision backbones.

### 3.1. Stain Normalization Algorithms
Stain normalization algorithms adjust the color distribution of an input image to match a chosen reference image. Existing methods fall into two main families:

1. **Statistical Color Matching (Reinhard et al., 2001):**  
   Reinhard color transfer converts an RGB image into the perceptual $L\alpha\beta$ color space developed by Ruderman et al. (1998). In this space, the luminance channel ($L$) and the chromatic channels ($\alpha, \beta$) are statistically independent. The algorithm calculates the mean and standard deviation of each channel for both the source image and the target reference image. It then shifts and scales the source image channels to match the target statistics. This method is computationally fast and requires only simple matrix arithmetic.

2. **Optical Density Deconvolution (Macenko et al., 2009):**  
   Macenko normalization models the physical physics of light absorption using the **Beer-Lambert Law**:
   $$OD = -\log_{10}\left(\frac{I}{I_0}\right)$$
   where $I$ is the transmitted light intensity and $I_0$ is the incident light intensity. Macenko's method transforms RGB pixel values into Optical Density (OD) space. It applies Singular Value Decomposition (SVD) to find the two-dimensional plane containing the primary dye absorption vectors. By projecting pixel values onto this plane and identifying extreme angular percentiles (1st and 99th percentiles), the method calculates the distinct stain vectors for Hematoxylin and Eosin. It then normalizes stain concentration maps against a reference patch.

Other methods, such as sparse non-negative matrix factorization (Vahadane et al., 2016) and Generative Adversarial Networks (CycleGAN, StainGAN), also exist. However, Reinhard and Macenko remain the two most widely deployed algorithms in clinical workflows due to their deterministic behavior and lack of hallucinated tissue structures.

### 3.2. Data Augmentation Strategies
Instead of altering images during testing, data augmentation trains the network to ignore color variations by exposing it to diverse variations during training.
- **Spatial / Geometric Augmentation:** Standard transformations include random horizontal flips, vertical flips, and 90-degree rotations. Because tissue slides have no natural "up" or "down" orientation, geometric augmentation is a mandatory baseline for microscopy.
- **Stain Color Perturbation (Tellez et al., 2019):** Rather than simple RGB jittering, biologically-grounded stain jittering deconvolves images into separate Hematoxylin, Eosin, and background channels. The algorithm adds random scale and shift factors to individual dye concentrations. This forces the model to rely on structural morphology rather than specific dye intensity.

### 3.3. Vision Backbones and Foundation Models
Most prior benchmarks rely on classical ImageNet-pretrained CNNs:
- **ResNet-50 (He et al., 2016):** A 50-layer residual network that uses skip connections. It has served as the universal standard backbone in medical imaging for nearly a decade.
- **ConvNeXt-Tiny (Liu et al., 2022):** A modernized pure convolutional architecture that incorporates design choices from Vision Transformers, including $7 \times 7$ depthwise convolutions, inverted bottlenecks, LayerNorm, and GELU activations.
- **Pathology Foundation Models (Phikon, Filiot et al., 2023):** Built on the Vision Transformer (ViT-B/16) architecture, Phikon was trained by Owkin on 43 million histology tiles from The Cancer Genome Atlas (TCGA) using the self-supervised iBOT framework. Because it observed diverse slide preparations from hundreds of laboratories during pretraining, Phikon is hypothesized to have high intrinsic stain invariance.

---

## 4. Dataset and Data Preparation

### 4.1. Dataset Source and Official Benchmark Links
This project uses the established colorectal histology benchmark created by Kather et al. (2018, 2019). The benchmark provides two separate, non-overlapping collections of histological image tiles:

- **Official Zenodo Archive:** [https://doi.org/10.5281/zenodo.1214456](https://doi.org/10.5281/zenodo.1214456)
- **Dataset Version:** `v0.1` (Published: 2018-04-07; Verified: 2026-10-04)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

| Dataset Identifier | Clinical Role | Patches | Resolution | Source Centers |
| :--- | :--- | :---: | :---: | :--- |
| `NCT-CRC-HE-100K-NONORM` | Source Domain (Train / Val-ID / Test-ID) | 100,000 | $224 \times 224$ px ($0.5\,\mu\text{m/px}$) | NCT Heidelberg & UMM Mannheim, Germany ($N=86$ patients) |
| `CRC-VAL-HE-7K` | Target Domain (Test-OOD) | 7,180 | $224 \times 224$ px ($0.5\,\mu\text{m/px}$) | University Hospital RWTH Aachen, Germany ($N=50$ patients) |

> **Important Note on the NONORM Variant:** The Zenodo repository also hosts a Macenko-normalized archive (`NCT-CRC-HE-100K.zip`). In this study, we strictly use the unnormalized **`NONORM`** archive. Preserving raw clinical staining variations in the source domain is necessary to test our normalization and augmentation hypotheses.

![NCT-100K Source Domain Patches](docs/images/NCT-100k.png)  
*Figure 1: Representative histological patches from the NCT-CRC-HE-100K-NONORM source domain (Heidelberg/Mannheim cohort).*

![CRC-7K Target Domain Patches](docs/images/CRC-7k.png)  
*Figure 2: Representative histological patches from the CRC-VAL-HE-7K target domain (Aachen cohort).*

### 4.2. Nine Histological Classes
Both datasets contain nine mutually exclusive tissue classes commonly found in colorectal cancer specimens:

| Label Index | Code | Class Name | Biological Description | Target Count (`CRC-VAL-7K`) |
| :---: | :--- | :--- | :--- | :---: |
| 0 | `ADI` | Adipose | Lipid-filled fat cells with thin borders and clear cytoplasm | 1,338 |
| 1 | `BACK`| Background | Clear glass slide areas with no tissue | 847 |
| 2 | `DEB` | Debris | Necrotic cell fragments, hemorrhage, and cautery artifacts | 339 |
| 3 | `LYM` | Lymphocytes | Dense clusters of small, dark, round immune cell nuclei | 634 |
| 4 | `MUC` | Mucus | Amorphous extracellular pools of light basophilic mucus | 1,035 |
| 5 | `MUS` | Smooth Muscle | Compact bundles of pink-staining spindle-shaped muscle fibers | 592 |
| 6 | `NORM`| Normal Mucosa | Organized healthy glandular structures with goblet cells | 741 |
| 7 | `STR` | Stroma | Fibrous collagenous connective tissue and desmoplastic stroma | 421 |
| 8 | `TUM` | Colorectal Carcinoma | Malignant glandular epithelium with enlarged, irregular nuclei | 1,233 |
| **Total** | | | **All 9 tissue categories from 50 independent patients** | **7,180** |

### 4.3. Data Subsampling and Partitioning Protocol
To allow all 13 experimental configurations to complete within a single Kaggle GPU session (~3.5 hours on an Nvidia T4 GPU), we use a standardized subset budget of 25,000 patches from the 100,000-patch source pool:
1. **Per-Class Subsampling:** The budget is divided evenly across all 9 classes using integer division:
   $$\text{Patches per class} = \lfloor 25,000 / 9 \rfloor = 2,777$$
   $$2,777 \times 9 = \mathbf{24,993\text{ total source patches}}$$
2. **Stratified Split (70 / 15 / 15):** The 24,993 source patches are split using two consecutive stratified `train_test_split` calls (`random_state=42`):
   - **Training Set (`train.csv`):** 70% $\rightarrow$ **17,495 patches** (used for model weight optimization).
   - **In-Domain Validation Set (`val_id.csv`):** 15% $\rightarrow$ **3,749 patches** (used strictly for per-epoch checkpoint selection).
   - **In-Domain Test Set (`test_id.csv`):** 15% $\rightarrow$ **3,749 patches** (evaluated once post-training to measure in-domain performance).
3. **Out-of-Domain External Set (`test_ood.csv`):** 100% of the `CRC-VAL-HE-7K` dataset $\rightarrow$ **7,180 patches**.

```text
Source Pool: NCT-CRC-HE-100K-NONORM
  │
  └── Stratified Sampling (2,777 per class) ──► 24,993 patches
                                                    │
        ┌───────────────────────────────────────────┴───────────────────────────────────────────┐
        ▼                                           ▼                                           ▼
  Train Split (70%)                           Val-ID Split (15%)                          Test-ID Split (15%)
  17,495 patches                              3,749 patches                               3,749 patches
  (Model weight optimization)                 (Best checkpoint selection)                 (In-domain test benchmark)

Target Pool: CRC-VAL-HE-7K
  │
  └── Strict Domain Firewall ─────────────────────────────────────────────────────────────► Test-OOD Split (100%)
                                                                                          7,180 patches
                                                                                          (External hospital test)
```

### 4.4. The Strict Domain Firewall Rule
To guarantee valid scientific evaluation, the target dataset (`CRC-VAL-HE-7K`) is protected by a strict domain firewall:
- It is never included in training sets.
- It is never used to fit color normalizers.
- It is never consulted for hyperparameter selection or early stopping.
- It is loaded and evaluated exactly once per experiment after training is completely finished.

### 4.5. Canonical Stain Reference Tile Selection
All color normalization experiments require a single target reference image. In our pipeline, this reference tile is chosen deterministically as the **very first `TUM` patch in the training split** and saved to `results/splits/reference_stain.png`. Selecting a dense tumor patch from the training split ensures that the reference contains strong, balanced signals of both Hematoxylin (nuclei) and Eosin (stroma) without causing data leakage from test sets.

![Canonical Stain Reference Tile](results/splits/reference_stain.png)  
*Figure 3: The canonical reference patch (`results/splits/reference_stain.png`) used for all Reinhard and Macenko normalization steps.*

---

## 5. Methods

```mermaid
flowchart LR
    subgraph Data_Pipeline["Data Pipeline (PatchTransform)"]
        Raw["Raw Input Image (224x224)"] --> Norm{"Normalization?"}
        Norm -->|"Reinhard"| RNorm["Reinhard Color Transfer"]
        Norm -->|"Macenko"| MNorm["Macenko Deconvolution"]
        Norm -->|"None"| RawP["Raw RGB"]
        RNorm & MNorm & RawP --> Aug{"Train Augmentation?"}
        Aug -->|"Aug-Geo"| Geo["Flips & Rotations"]
        Aug -->|"Aug-Stain"| HED["HED Stain Jitter"]
        Aug -->|"Aug-Combined"| Comb["Geo + HED Jitter"]
        Aug -->|"None"| Clean["Clean Patch"]
        Geo & HED & Comb & Clean --> Tensor["Tensor & ImageNet Norm"]
    end
    subgraph Architecture["Model Family"]
        Tensor --> Net{"Backbone Choice"}
        Net -->|"ResNet-50"| RN["ResNet-50 (Fine-tune)"]
        Net -->|"ConvNeXt"| CN["ConvNeXt-Tiny (Fine-tune)"]
        Net -->|"Phikon"| PK["Phikon ViT (Frozen + Linear)"]
    end
    subgraph Loss_Opt["Loss & Optimization"]
        RN & CN & PK --> CE["Cross-Entropy (Label Smoothing 0.1)"]
        CE --> Opt["AdamW + Cosine LR Schedule + AMP"]
    end
```

### 5.1. Baseline Method
Our anchor baseline (**EXP-01**) uses a standard **ResNet-50** architecture initialized with ImageNet-1k weights (`resnet50.a1_in1k` from the `timm` library). 
- **Preprocessing:** Raw RGB images are resized to $224 \times 224$ pixels without color normalization (`norm=none`).
- **Augmentation:** Deterministic input pipeline without spatial transformations or color jittering (`aug=none`). Only standard ImageNet channel standardization is applied:
  $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$
- **Optimization:** AdamW optimizer with initial learning rate $\eta = 1 \times 10^{-3}$, weight decay $\lambda = 0.05$, and a Cosine Annealing learning rate schedule decaying to $\eta_{\min} = 1 \times 10^{-5}$ over 8 epochs.
- **Objective Function:** Cross-entropy loss with label smoothing ($\alpha = 0.1$) to prevent overconfident output distributions:
  $$\mathcal{L}_{\text{LS}} = (1 - \alpha)\mathcal{L}_{\text{CE}} + \frac{\alpha}{K}\sum_{k=1}^K -\log p_k$$
  where $K = 9$ classes.

### 5.2. Main Proposed Method
Our primary defense framework combines:
1. **Macenko Optical Density Normalization:** Every input patch is deconvolved into optical density space. Its stain vectors and 99th-percentile dye concentrations are scaled to match the canonical reference patch before entering the neural network.
2. **Combined Spatial and Stain Augmentation (`Aug-Combined`):** During training, patches undergo random horizontal flips ($p=0.5$), vertical flips ($p=0.5$), and random 90-degree rotations, combined with stochastic HED stain concentration jittering ($p=0.8$, concentration scale $\alpha \in [0.8, 1.2]$, concentration bias $\beta \in [-0.05, 0.05]$).
3. **Histology Foundation Model (Phikon):** A Vision Transformer (ViT-B/16) pretrained on 43 million histology tiles using self-supervised learning. The 86-million parameter ViT encoder is frozen (`requires_grad = False`), and a linear probe head:
   $$f_{\text{head}}(x) = W x_{[\text{CLS}]} + b \quad (W \in \mathbb{R}^{9 \times 768}, \; b \in \mathbb{R}^9)$$
   is trained on top of the extracted 768-dimensional `[CLS]` token representations using learning rate $\eta = 3 \times 10^{-4}$.

### 5.3. Comparison Strategy and Metrics
To fairly evaluate the impact of each technique, we measure classification performance across both domains using two primary metrics and two robustness quantifiers:

1. **Primary Metric — Macro-averaged F1 Score (`Macro-F1`):**  
   We calculate the harmonic mean of precision and recall for each class $c \in \{0, \dots, 8\}$, then take the unweighted arithmetic mean across all 9 classes:
   $$\text{Macro-F1} = \frac{1}{9}\sum_{c=0}^8 \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$
   This metric weights all tissue categories equally, preventing majority classes (such as `ADI`) from hiding poor diagnostic performance on minority classes (such as `DEB`).
2. **Secondary Metric — Top-1 Accuracy (`Acc`):**  
   The overall percentage of correctly classified patches.
3. **Absolute Performance Drop ($\Delta\text{-F1}$):**  
   The absolute drop in F1 score when moving from the in-domain test set to the external target test set:
   $$\Delta\text{-F1} = \text{F1}_{\text{ID}} - \text{F1}_{\text{OOD}}$$
   *(Lower is better; $\Delta\text{-F1} = 0$ represents perfect stain invariance).*
4. **Relative Retention Rate ($\text{RR}_{\text{F1}}$):**  
   The percentage of source diagnostic performance preserved in the external hospital domain:
   $$\text{RR}_{\text{F1}} = \left(\frac{\text{F1}_{\text{OOD}}}{\text{F1}_{\text{ID}}}\right) \times 100\%$$
   *(Higher is better; $100\%$ indicates zero performance loss across medical centers).*

---

## 6. Experimental Setup

The investigation is structured into three clear setups following the course project specifications.

### 6.1. Setup 1 — Baseline vs. Main Model
We contrast the anchor baseline against our primary defense candidates to measure the overall effectiveness of our engineering interventions:
- **Baseline Configuration (EXP-01):** ResNet-50 trained on raw patches without normalization or data augmentation.
- **Defended ResNet Configuration (EXP-09):** ResNet-50 trained with full defense (Macenko normalization + combined geometric and HED stain augmentation).
- **Foundation Model Configuration (EXP-12):** Undefended Phikon linear probe evaluating intrinsic representations learned from 43 million TCGA patches.

### 6.2. Setup 2 — Main Research Experiment (The 13-Cell Ablation Matrix)
All 13 experiments are organized across five progressive scientific stages. Each cell tests a controlled modification while keeping all other variables identical:

| Exp ID | Stage | Backbone | Pretraining Source | Normalization | Augmentation Policy | Primary Hypothesis Tested |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **EXP-01** | Stage 0 | ResNet-50 | ImageNet-1k | None | None | **Anchor Baseline:** Establishes the raw domain shift drop. |
| **EXP-02** | Stage 1 | ResNet-50 | ImageNet-1k | Reinhard | None | Statistical LAB mean/std matching improves OOD transfer. |
| **EXP-03** | Stage 1 | ResNet-50 | ImageNet-1k | Macenko | None | Physical optical density deconvolution outperforms statistical transfer. |
| **EXP-04** | Stage 2 | ResNet-50 | ImageNet-1k | None | Aug-Geo | Spatial orientation invariance alone improves OOD transfer. |
| **EXP-05** | Stage 2 | ResNet-50 | ImageNet-1k | None | Aug-Stain | Synthetic HED jitter matches explicit stain normalization. |
| **EXP-06** | Stage 2 | ResNet-50 | ImageNet-1k | None | Aug-Combined | Spatial and stain augmentations provide additive benefits. |
| **EXP-07** | Stage 3 | ResNet-50 | ImageNet-1k | Macenko | Aug-Geo | Combining Macenko with spatial transforms produces gains. |
| **EXP-08** | Stage 3 | ResNet-50 | ImageNet-1k | Macenko | Aug-Stain | Adding stain jitter to normalized patches creates conflict. |
| **EXP-09** | Stage 3 | ResNet-50 | ImageNet-1k | Macenko | Aug-Combined | **Defended ResNet Baseline:** Full defense policy for ResNet. |
| **EXP-10** | Stage 4 | ConvNeXt-T | ImageNet-1k | None | None | Modern 7x7 ConvNet architecture provides intrinsic robustness. |
| **EXP-11** | Stage 4 | ConvNeXt-T | ImageNet-1k | Macenko | Aug-Combined | ConvNeXt-Tiny benefits from the combined defense policy. |
| **EXP-12** | Stage 4 | Phikon (ViT-B) | TCGA (iBOT SSL) | None | None | Large-scale pathology SSL provides intrinsic stain invariance. |
| **EXP-13** | Stage 4 | Phikon (ViT-B) | TCGA (iBOT SSL) | Macenko | Aug-Combined | Frozen foundation encoder benefits from artificial defense policies. |

### 6.3. Setup 3 — Analysis, Robustness, and Ablation Design
To isolate the exact causal factors that govern stain robustness, our matrix enables four targeted ablation comparisons:

1. **Ablation Axis A (Normalization Mechanism):**  
   Comparing `EXP-01` (None) vs. `EXP-02` (Reinhard) vs. `EXP-03` (Macenko) with backbone and augmentation held constant.
2. **Ablation Axis B (Augmentation Policy):**  
   Comparing `EXP-01` (None) vs. `EXP-04` (Aug-Geo) vs. `EXP-05` (Aug-Stain) vs. `EXP-06` (Aug-Combined) with backbone fixed to ResNet-50 and normalization disabled.
3. **Ablation Axis C (Defense Interaction):**  
   Comparing `EXP-03` (Macenko alone) against `EXP-07`, `EXP-08`, and `EXP-09` to test whether data augmentation aids or disrupts physically normalized patches.
4. **Ablation Axis D (Architectural Invariance):**  
   Comparing pairs `EXP-01` vs. `EXP-10` vs. `EXP-12` (undefended) and `EXP-09` vs. `EXP-11` vs. `EXP-13` (defended) across classical CNN, modern CNN, and pathology Vision Transformer architectures.

### 6.4. Implementation and Hardware Environment
All experiments are implemented in Python 3.10 and PyTorch 2.x and executed on Kaggle cloud instances:

| Configuration Parameter | Value |
| :--- | :--- |
| **Compute Hardware** | Nvidia Tesla T4 GPU ($16\,\text{GB}$ VRAM) |
| **Epochs per Experiment** | 8 epochs (sufficient for convergence with cosine decay) |
| **Batch Size** | 64 patches |
| **Precision** | PyTorch Automatic Mixed Precision (`torch.amp.autocast("cuda")`) |
| **Workers** | 4 data loader subprocesses with pinned GPU memory |
| **Model Selection** | Checkpoint achieving highest Macro-F1 on `val_id` split |
| **Total Wall-Clock Runtime** | Approximately 3.5 hours for all 13 experiments |

---
*Sections 7 through 11 (Results and Discussion, Error and Qualitative Analysis, Conclusion, References, and Appendix) will follow in the subsequent report deliverable.*
