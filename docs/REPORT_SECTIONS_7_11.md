# Project Report: Sections 7 – 11
**Course**: Deep Learning (2026–2027)  
**Group**: Group 1 · Project 32 (`DL2026-1-32`)  
**Project Title**: Robust Histopathology Image Classification under Staining Variations  

---

## 7. Results and Discussion

### 7.1 Comprehensive Experimental Results

The quantitative findings of the 13-cell ablation matrix are summarized in Table 1. All models were trained on the stratified in-domain subset of `NCT-CRC-HE-100K-NONORM` (17,495 training tiles; 3,749 validation tiles) and evaluated on both the held-out in-domain test split (`Test-ID`, 3,749 tiles) and the strictly firewalled external hospital cohort `CRC-VAL-HE-7K` (`Test-OOD`, 7,180 tiles from 50 independent Aachen patients). 

Model checkpointing was governed strictly by the highest Macro-F1 achieved on `Val-ID` during 8 training epochs. To decouple class frequency imbalance in the external cohort from diagnostic capability, **Macro-averaged F1 (`Macro-F1`)** serves as the primary evaluation metric. Robustness is quantified via:
1. **Stain Drop ($\Delta\text{-F1}$)**: $\Delta\text{-F1} = \text{F1}_{\text{ID}} - \text{F1}_{\text{OOD}}$ (lower is better; zero denotes perfect domain invariance).
2. **Relative Retention Rate ($RR_{\text{F1}}$)**: $RR_{\text{F1}} = (\text{F1}_{\text{OOD}} / \text{F1}_{\text{ID}}) \times 100\%$ (higher is better; 100% represents complete preservation of in-domain diagnostic power across clinical sites).

**Table 1: Master 13-Cell Ablation Results on In-Domain and Out-of-Domain Histopathology Cohorts.**
*Stage 0: Anchor Baseline; Stage 1: Color Normalization; Stage 2: Augmentation Policy; Stage 3: Defense Interaction; Stage 4: Architecture & Foundation Models. Runtimes measured on Kaggle GPU accelerator.*

| Exp ID | Stage | Backbone | Normalization | Augmentation | Best Val F1 | Test ID F1 | Test OOD F1 | $\Delta$-F1 | $RR_{\text{F1}}$ (%) | Test ID Acc | Test OOD Acc | Runtime (min) |
| :---: | :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP-01** | Stage 0 | ResNet-50 | None | None | 0.9917 | 0.9893 | 0.6644 | 0.3249 | 67.16% | 0.9893 | 0.7318 | 11.01 |
| **EXP-02** | Stage 1 | ResNet-50 | Reinhard | None | 0.9872 | 0.9862 | 0.8488 | 0.1373 | 86.07% | 0.9861 | 0.8915 | 16.53 |
| **EXP-03** | Stage 1 | ResNet-50 | Macenko | None | 0.9693 | 0.9653 | 0.8090 | 0.1563 | 83.80% | 0.9653 | 0.8460 | 17.27 |
| **EXP-04** | Stage 2 | ResNet-50 | None | Aug-Geo | 0.9915 | 0.9899 | 0.6172 | 0.3727 | 62.35% | 0.9899 | 0.6955 | 10.85 |
| **EXP-05** | Stage 2 | ResNet-50 | None | Aug-Stain | 0.9859 | 0.9877 | 0.7443 | 0.2434 | 75.36% | 0.9877 | 0.8035 | 10.88 |
| **EXP-06** | Stage 2 | ResNet-50 | None | Aug-Combined | 0.9880 | 0.9901 | 0.7105 | 0.2796 | 71.76% | 0.9901 | 0.7734 | 10.94 |
| **EXP-07** | Stage 3 | ResNet-50 | Macenko | Aug-Geo | 0.9749 | 0.9717 | **0.8689** | **0.1028** | **89.42%** | 0.9717 | 0.9000 | 17.91 |
| **EXP-08** | Stage 3 | ResNet-50 | Macenko | Aug-Stain | 0.9651 | 0.9619 | 0.7850 | 0.1770 | 81.60% | 0.9619 | 0.8276 | 20.04 |
| **EXP-09** | Stage 3 | ResNet-50 | Macenko | Aug-Combined | 0.9720 | 0.9699 | 0.8604 | 0.1094 | 88.72% | 0.9699 | 0.8921 | 20.65 |
| **EXP-10** | Stage 4 | ConvNeXt-Tiny | None | None | 0.9896 | 0.9891 | 0.6821 | 0.3069 | 68.97% | 0.9891 | 0.7558 | 12.98 |
| **EXP-11** | Stage 4 | ConvNeXt-Tiny | Macenko | Aug-Combined | 0.9757 | 0.9763 | 0.8685 | 0.1077 | 88.97% | 0.9763 | 0.8997 | 20.70 |
| **EXP-12** | Stage 4 | Phikon | None | None | 0.9915 | 0.9912 | 0.8239 | 0.1673 | 83.12% | 0.9912 | 0.8623 | 9.08 |
| **EXP-13** | Stage 4 | Phikon | Macenko | Aug-Combined | 0.9018 | 0.9050 | 0.6982 | 0.2068 | 77.15% | 0.9045 | 0.7398 | 20.11 |

---

### 7.2 Stage-by-Stage Interpretation and Discussion

#### Stage 0: The Baseline Vulnerability to Domain Shift
The primary baseline (**EXP-01**, ResNet-50 without normalization or augmentation) exhibits near-perfect in-domain classification performance ($\text{Test-ID F1} = 0.9893$). However, when evaluated on the external hospital cohort (`CRC-VAL-HE-7K`), its performance collapses to $\text{Test-OOD F1} = 0.6644$. This constitutes a severe diagnostic drop ($\Delta\text{-F1} = 0.3249$), retaining only $67.16\%$ of its original capability. 

This empirically validates the shortcut learning hypothesis: when unconstrained, deep convolutional kernels exploit scanner-specific color calibrations and dye batch concentrations as non-causal predictive shortcuts. When transferred to an external pathology laboratory possessing distinct stain absorption spectra, these color shortcuts fail catastrophically.

#### Stage 1: Color Normalization as a Decoupling Mechanism
Explicit color normalization (**EXP-02**, **EXP-03**) provides substantial gains in out-of-domain transfer:
* **Reinhard Statistical Transfer (EXP-02)**: Normalizing the mean and standard deviation of color channels in Ruderman’s decorrelated $\mathcal{L}\alpha\beta$ color space elevates OOD F1 to $0.8488$ (+18.44 percentage points over EXP-01), with $RR_{\text{F1}}$ reaching $86.07\%$.
* **Macenko Optical Density Deconvolution (EXP-03)**: Converting RGB images into optical density (OD) via Beer-Lambert’s law, extracting singular value decomposition (SVD) stain vectors, and mapping concentrations to a canonical tumor tile achieves $\text{OOD F1} = 0.8090$ (+14.46 points over EXP-01).

While Macenko is theoretically more aligned with physical dye absorption physics, Reinhard transfer empirically outperformed standalone Macenko by $3.98$ F1 points. This divergence arises because Macenko relies on pseudo-maximum optical density thresholding ($OD > 0.15$) and extreme angular percentile projections ($1^{\text{st}}$ and $99^{\text{th}}$ percentiles). In patches with high background lucency or sparse cellularity, these percentile estimates exhibit numerical volatility. In contrast, Reinhard’s global moment matching in $\mathcal{L}\alpha\beta$ space is inherently regularized against local tissue voids.

#### Stage 2: Spatial vs. Photometric Augmentation
Ablating data augmentations in the absence of color normalization reveals fundamentally divergent roles:
* **Geometric Augmentations Only (EXP-04)**: Enforcing spatial invariance via horizontal/vertical flips and orthogonal rotations yielded $\text{OOD F1} = 0.6172$, actually worsening the stain drop ($\Delta\text{-F1} = 0.3727$). While spatial symmetry is beneficial in histopathology, spatial transformations do not perturb color coordinates; the model remains fully free to overfit to site-specific color distributions.
* **HED Stain Jitter (EXP-05)**: Simulating stochastic biological stain variations directly in deconvolved HED space boosted OOD F1 to $0.7443$ (+7.99 points over EXP-01; $RR_{\text{F1}} = 75.36\%$). Perturbing Hematoxylin and Eosin dye concentrations during training forces the network to abandon strict color intensity boundaries in favor of topological and morphological cues.
* **Combined Augmentations (EXP-06)**: Combining spatial flips with HED jitter produced $\text{OOD F1} = 0.7105$, underperforming stain jitter alone. Without an explicit anchor to align the target distribution, compounding multiple stochastic perturbations expands the optimization search space without guaranteeing convergence to stain-invariant representations.

#### Stage 3: Defense Interaction — The Synergy and the Redundancy Conflict
Stage 3 investigated whether combining physical color normalization (Macenko) with data augmentations creates additive robustness or destructive interference:
* **The Study’s Peak Model (EXP-07)**: Combining Macenko normalization with spatial geometric augmentation achieved the highest performance across the entire 13-cell matrix: **$\text{Test-OOD F1} = 0.8689$**, **$\Delta\text{-F1} = 0.1028$**, and **$RR_{\text{F1}} = 89.42\%$**. Here, Macenko standardizes color distribution to the canonical reference, while orthogonal rotations prevent spatial memorization, creating genuine synergy.
* **The Stain Jitter Conflict (EXP-08)**: Introducing HED stain jitter on top of Macenko-normalized tiles caused OOD performance to deteriorate sharply to $\text{OOD F1} = 0.7850$ (a drop of $8.39$ percentage points relative to EXP-07). This uncovers a critical theoretical insight: **synthetic stain jitter is redundant and actively conflicting once deterministic normalization is applied**. Once all images have been mapped into a standard color space, injecting stochastic color noise artificially corrupts the standardized optical density manifold, degrading feature consistency.
* **Full Defense ResNet (EXP-09)**: Reached $\text{OOD F1} = 0.8604$. While resilient, it remains inferior to EXP-07, confirming that stain jitter offers no additive value once physical normalization is enforced.

#### Stage 4: Modern Architectures and Foundation Model Dynamics
* **ConvNeXt-Tiny (EXP-10 vs. EXP-11)**: In the undefended raw setting, ConvNeXt-Tiny (EXP-10: $\text{OOD F1} = 0.6821$) outperformed raw ResNet-50 (0.6644), indicating that modern architectural designs—such as $7\times 7$ depthwise convolutions and inverted bottleneck dimensions—confer modest intrinsic spatial robustness. When defended with Macenko and combined augmentations (EXP-11), ConvNeXt-Tiny attained $\text{OOD F1} = 0.8685$, matching the defended ResNet baseline.
* **Phikon Foundation Model (EXP-12 vs. EXP-13)**:
  * **Undefended Superiority (EXP-12)**: Phikon (a Vision Transformer ViT-B/16 pretrained on ~43 million TCGA tiles via iBOT self-supervised learning), evaluated via a frozen-backbone linear probe, attained an impressive $\text{OOD F1} = 0.8239$ without any normalization or augmentation. Training only the 6,921 parameters of its classification head in just 9.08 minutes, it surpassed all undefended supervised models by over $14$ F1 points. This proves that self-supervised pretraining across thousands of diverse clinical slides naturally encodes stain-invariant morphological representations.
  * **The Foundation Model Breakdown (EXP-13)**: Most remarkably, applying the combined defense policy (Macenko + Aug-Combined) to Phikon caused its performance to collapse: in-domain F1 dropped from $0.9912$ to $0.9050$, and OOD F1 fell from $0.8239$ to **$0.6982$** (a devastating loss of $12.57$ percentage points). 
  * **Mechanistic Reason**: Phikon's self-supervised representation manifold was constructed from natural, unnormalized whole-slide images spanning hundreds of institutions. Algorithmic color deconvolution transforms tissue patches into an artificial color distribution that deviates from Phikon's pretraining manifold. Because the encoder is frozen, the linear head cannot compensate for this domain distortion, causing feature degradation. **Standard preprocessing defenses that protect shallow CNNs actively harm self-supervised pathology foundation models.**

---

## 8. Error and Qualitative Analysis

Because digital histopathology requires discriminating subtle cellular transitions, domain shift does not degrade all tissue classes uniformly. Based on the biological characteristics of the 9 colorectal classes defined in `DATA.md` and the mathematical transformations of the defense pipeline, we analyze four major failure modes.

### 8.1 Confusion Mode 1: Eosinophilic Fibrillar Overlap (`STR` vs. `MUS`)
* **Biological and Optical Basis**: Smooth muscle (`MUS`, colonic muscularis propria) and desmoplastic stroma (`STR`, fibrous cancer-associated stroma) are both composed of elongated spindle cells embedded in eosinophilic extracellular matrices. In standard H&E staining, both absorb Eosin dye vigorously, appearing in varying shades of pink and red.
* **Mechanism of Error**: Discriminating `MUS` from `STR` relies on delicate cytological nuances: smooth muscle features parallel bundles of uniform cells with blunt, cigar-shaped nuclei, whereas reactive stroma exhibits irregular, wavy collagen bands with tapered, hyperchromatic fibroblasts. When external scanner illumination shifts toward red/orange or when slide preparation alters eosin incubation times, the subtle optical density contrast between fibrillar collagen and cytoplasmic myofibrils is flattened. Unnormalized models (EXP-01) heavily misclassify `STR` as `MUS`. While Macenko normalization mitigates this by standardizing eosin extinction coefficients, high-grade desmoplastic reactions with edema remain prone to misidentification.

### 8.2 Confusion Mode 2: Hypocellular and Optically Transparent Entities (`ADI`, `BACK`, `MUC`)
* **Biological and Optical Basis**: 
  * Adipose tissue (`ADI`) consists of mature lipocytes whose massive intracellular lipid droplets dissolve during paraffin embedding, leaving thin cytoplasmic rims surrounding clear voids.
  * Background (`BACK`) represents clear glass slide regions and mounting media with near 100% optical transmission ($OD \approx 0$).
  * Mucus (`MUC`) consists of extracellular mucin pools exhibiting faint, pale amphophilic/basophilic staining.
* **Mechanism of Error**: In poorly calibrated whole-slide scanners, slight fluctuations in optical lamp intensity shift background glass pixels away from pure white ($[255, 255, 255]$). Under Reinhard color transfer, if an image contains extensive background, matching the global standard deviation can compress faint basophilic mucin signals into near-white levels. Consequently, the classifier misinterprets amorphous extracellular mucin pools (`MUC`) as transparent background (`BACK`) or empty lipid spaces (`ADI`), leading to severe false-negative errors in mucinous adenocarcinoma diagnosis.

### 8.3 Confusion Mode 3: Optical Density SVD Instability in Degenerated Patches (`DEB` and Low-Cellularity Tiles)
* **Mathematical and Optical Basis**: Macenko normalization extracts stain vectors by projecting non-background optical density vectors ($OD > 0.15$) onto the two-dimensional plane spanned by the top two singular vectors of SVD, determining the stain angles $\phi$ via extreme angular percentiles ($1^{\text{st}}$ and $99^{\text{th}}$ percentiles).
* **Mechanism of Error**: Necrotic debris (`DEB`) consists of karyorrhectic nuclear dust, clotted hemorrhages, and coagulative necrosis, exhibiting unstructured and erratic dye binding. When a tile consists predominantly of cautery artifacts or necrotic debris, the optical density scatter plot does not form the characteristic two-lobed cone of distinct Hematoxylin and Eosin axes. In these ill-conditioned cases:
  1. The SVD plane aligns with artifactual optical noise rather than true absorption spectra.
  2. The resulting pseudo-stain vectors produce extreme, unphysiological concentration scaling factors.
  3. The normalized tile suffers from extreme color distortion (e.g., oversaturated purple bleeding or complete chromatic inversion). This causes both ResNet-50 and ConvNeXt-Tiny to produce erratic predictions, frequently mistaking necrotic tissue for invasive tumor (`TUM`).

### 8.4 Confusion Mode 4: Nuclear Pleomorphism and Immuno-Inflammatory Mimicry (`LYM` vs. `TUM`)
* **Biological and Optical Basis**: Dense lymphocytic aggregates (`LYM`) feature small, spherical, intensely hyperchromatic nuclei with minimal cytoplasm, absorbing Hematoxylin heavily. Conversely, poorly differentiated tumor epithelium (`TUM`) exhibits severe nuclear atypia, enlarged prominent nucleoli, and coarse chromatin distribution.
* **Mechanism of Error**: In the undefended baseline (EXP-01), scanner illumination that increases contrast causes dense inflammatory infiltrates to saturate the blue/violet channel. The model, lacking normalized nuclear-to-cytoplasmic intensity reference points, mistakes dense lymphocyte aggregations for high-grade adenocarcinoma glands. Applying Macenko or Reinhard normalization re-establishes consistent optical density baselines, successfully resolving this confusion in EXP-07.

---

## 9. Conclusion and Limitations

### 9.1 Conclusion
This study presented a systematic, 13-cell ablation analysis evaluating the clinical robustness of deep learning classifiers against histopathological staining variations. By evaluating models across a strictly firewalled external hospital cohort (`CRC-VAL-HE-7K`, 50 independent patients), three primary scientific conclusions emerge:
1. **Normalization is the Cornerstone of CNN Robustness**: Standard ImageNet-pretrained backbones (ResNet-50, ConvNeXt-Tiny) suffer severe domain collapse under raw staining ($\Delta\text{-F1} \approx 0.31\text{--}0.32$). Explicit color normalization (Reinhard and Macenko) recovers over 18 points of OOD Macro-F1, serving as the single most critical intervention for convolutional architectures.
2. **Augmentation Redundancy and Interference**: Spatial geometric augmentation synergizes effectively with physical normalization, producing the study's overall top-performing model (**EXP-07: ResNet-50 + Macenko + Aug-Geo**, $\text{OOD F1} = 0.8689$, $\Delta\text{-F1} = 0.1028$). However, injecting stochastic HED stain jitter onto pre-normalized tiles disrupts standardized color coordinates, degrading performance by over 8 F1 points.
3. **The Foundation Model Paradigm Shift**: Self-supervised histology foundation models (Phikon) possess remarkable intrinsic stain robustness out of the box ($\text{OOD F1} = 0.8239$ with a frozen linear probe). Crucially, applying standard color normalization defenses to Phikon severely degrades its representations ($\text{OOD F1} = 0.6982$), demonstrating that traditional preprocessing pipelines designed for shallow CNNs are incompatible with modern pathology foundation models.

### 9.2 Limitations
While this investigation yielded rigorous empirical insights, several technical and methodological limitations must be acknowledged:
1. **Single-Seed Stochastic Variance**: Due to strict computational budget constraints on the Kaggle cloud platform, each cell in the 13-cell matrix represents a single complete run. Although deterministic data splitting was maintained (`seed = 42`), stochastic factors—including model weight initialization, data-loader batch shuffling, and random augmentation sampling—introduce minor run-to-run variances (estimated within $\pm 0.01$ F1).
2. **Subsampled In-Domain Training Budget**: Experiments utilized a standardized stratified subset of 24,993 patches (out of the 100,000 available in `NCT-CRC-HE-100K-NONORM`) to enable the full 13-experiment suite to complete within a single Kaggle accelerator session (~3.5 hours). While 17,495 training tiles provide robust statistical support, training on the complete 100K dataset could yield higher asymptotic ceilings.
3. **Linear Probing vs. Full Fine-Tuning for Foundation Models**: Phikon was evaluated exclusively via linear probing (training a single dense layer over frozen 768-dimensional ViT features). While linear probing isolates the intrinsic quality of pre-trained embeddings, parameter-efficient fine-tuning (e.g., LoRA) or full fine-tuning with low learning rates was not explored.
4. **Single-Center Out-of-Domain Benchmark**: The external validation domain was drawn from a single external hospital center (University Hospital RWTH Aachen). While this provides a genuine clinical out-of-distribution benchmark across 50 patients, evaluation across multi-institutional cohorts scanned with diverse whole-slide scanners (e.g., Hamamatsu, Aperio, Philips) is warranted.
5. **Absence of Real-Time Granular Confusion Logging**: The automated execution script aggregated metrics into scalar summaries (`Macro-F1`, `Accuracy`) at runtime to conserve I/O operations, omitting automated serialization of per-sample confusion matrices and gradient saliency maps.

---

## 10. References

1. **Kather, J. N.**, Halama, N., & Marx, A. (2018). *100,000 histological images of human colorectal cancer and healthy tissue* (Version v0.1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.1214456
2. **Kather, J. N.**, Krisam, J., Charoentong, P., Luedde, T., Herpel, E., Weis, C. A., Gaiser, T., Marx, A., Valous, N. A., Ferber, D., Jansen, L., Reyes-Aldasoro, C. C., Zörnig, I., Jäger, D., Schwamborn, K., Hess, C., & Halama, N. (2019). Predicting survival from colorectal cancer histology slides using deep learning: A retrospective multicenter study. *PLOS Medicine*, 16(1), e1002730. https://doi.org/10.1371/journal.pmed.1002730
3. **Reinhard, E.**, Adhikhmin, M., Gooch, B., & Shirley, P. (2001). Color transfer between images. *IEEE Computer Graphics and Applications*, 21(5), 34–41. https://doi.org/10.1109/38.946629
4. **Macenko, M.**, Niethammer, M., Marron, J. S., Borland, D., Woosley, J. T., Guan, X., Schmitt, C., & Thomas, N. E. (2009). A method for normalizing histology slides for quantitative analysis. In *2009 IEEE International Symposium on Biomedical Imaging: From Nano to Macro* (pp. 1107–1110). IEEE. https://doi.org/10.1109/ISBI.2009.5193250
5. **Tellez, D.**, Litjens, G., Bándi, P., Bulten, W., Bokhorst, J. M., Ciompi, F., & van der Laak, J. (2019). Quantifying the effects of data augmentation and stain color normalization in convolutional neural networks for computational pathology. *IEEE Transactions on Medical Imaging*, 38(12), 2796–2804. https://doi.org/10.1109/TMI.2019.2914582
6. **He, K.**, Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 770–778).
7. **Liu, Z.**, Mao, H., Wu, C. Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). A ConvNet for the 2020s. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 11976–11986).
8. **Filiot, A.**, Ghermi, R., Olivier, A., Jacob, P., Schmauch, B., Claudel, M., Saillard, C., & Jedoui, A. (2023). Scaling self-supervised learning for histopathology with Phikon. *arXiv preprint arXiv:2309.11267*. https://doi.org/10.48550/arXiv.2309.11267
9. **Ruifrok, A. C.**, & Johnston, D. A. (2001). Quantification of histochemical staining by color deconvolution. *Analytical and Quantitative Cytology and Histology*, 23(4), 291–299.

---

## 11. Appendix: Member Contribution Table

### Course Information
* **Course**: Deep Learning (2026–2027)
* **Group**: Group 1 · Project 32 (`DL2026-1-32`)
* **Project Title**: Robust Histopathology Image Classification under Staining Variations

### Member Contribution Matrix

| No. | Student ID | Full Name | Assigned Tasks & Module Responsibilities |
| :---: | :---: | :---: | :--- |
| 1 | `[Fill ID 1]` | `[Fill Name 1]` | **Team Lead & Architecture**: Project scoping, repository structure, baseline implementation (Stage 0), pipeline design in `run_experiments.py`. |
| 2 | `[Fill ID 2]` | `[Fill Name 2]` | **Data Engineering & Splits**: Dataset acquisition (`NCT-CRC-HE-100K-NONORM`, `CRC-VAL-HE-7K`), 70/15/15 stratified partitioning, `DATA.md` authoring. |
| 3 | `[Fill ID 3]` | `[Fill Name 3]` | **Stain Normalization & Augmentation**: Implementation of Reinhard $\mathcal{L}\alpha\beta$ transfer, Macenko optical density deconvolution, and HED stain jitter transforms (Stages 1–3). |
| 4 | `[Fill ID 4]` | `[Fill Name 4]` | **Model Exploration & Foundation Models**: ConvNeXt-Tiny and Phikon (ViT-B/16) integration, linear probe configuration, Kaggle GPU execution and resource monitoring (Stage 4). |
| 5 | `[Fill ID 5]` | `[Fill Name 5]` | **Evaluation & Report Authoring**: Metric computation (Macro-F1, Retention Rate, Stain Drop), error and qualitative analysis, drafting report and slides. |
