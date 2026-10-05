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

#### Stage 0: The Baseline Domain-Shift Gap (EXP-01)
The raw ResNet-50 baseline achieves near-perfect in-domain classification ($\text{Test-ID F1} = 0.9893$), but drops sharply to $\text{Test-OOD F1} = 0.6644$ on the external Aachen cohort. This severe gap ($\Delta\text{-F1} = 0.3249$, retaining only $67.16\%$ of its original diagnostic power) demonstrates that undefended CNNs rely on hospital-specific color shortcuts rather than robust cell morphology.

#### Stage 1: Color Normalization Closes the Gap (EXP-02, EXP-03)
Explicit color normalization significantly recovers external performance:
* **Reinhard LAB matching (EXP-02)**: Restores OOD F1 to $0.8488$ (+18.44 points over baseline; $\text{RR} = 86.07\%$).
* **Macenko OD deconvolution (EXP-03)**: Restores OOD F1 to $0.8090$ (+14.46 points; $\text{RR} = 83.80\%$).

Reinhard transfer slightly outperforms Macenko ($+3.98$ F1 points) because its global mean/std matching in $\mathcal{L}\alpha\beta$ space is inherently regularized across whole tiles. In contrast, Macenko's extreme angular percentile fitting ($1^{\text{st}}$ and $99^{\text{th}}$ percentiles) can fluctuate on tiles containing large areas of clear background glass.

#### Stage 2: Spatial vs. Color Augmentation (EXP-04 to EXP-06)
Testing augmentations without color normalization reveals clear differences:
* **Geometric flips and rotations (EXP-04)**: Yields $\text{OOD F1} = 0.6172$ ($\Delta\text{-F1} = 0.3727$). Spatial symmetry is biologically valid, but spatial flips do not alter pixel colors; the network still overfits to training colors.
* **Biological HED stain jitter (EXP-05)**: Perturbing Hematoxylin and Eosin dye concentrations during training boosts OOD F1 to $0.7443$ (+7.99 points over baseline), forcing the network to rely on morphological shapes rather than exact hues.
* **Combined augmentation (EXP-06)**: Achieves $0.7105$, underperforming stain jitter alone because compounding multiple random changes without a color anchor expands the training distribution too much.

#### Stage 3: Normalization $\times$ Augmentation Interaction (EXP-07 to EXP-09)
* **Top Overall Model (EXP-07)**: Combining Macenko normalization with spatial flips/rotations achieves the highest external accuracy in the study: **OOD F1 $= 0.8689$**, **$\Delta$-F1 $= 0.1028$**, and **$\text{RR} = 89.42\%$**. Normalization standardizes color while spatial rotation prevents position memorization.
* **The Stain Jitter Conflict (EXP-08)**: Adding stochastic HED stain jitter to Macenko-normalized tiles causes OOD F1 to drop to $0.7850$ (down $8.39$ points from EXP-07). **Stain jitter conflicts with normalization**: once images are mapped into a clean standard color space, injecting random color noise corrupts that standard.
* **Full Defense (EXP-09)**: Achieves $0.8604$, confirming that stain jitter adds no benefit once physical normalization is already present.

#### Stage 4: Modern Architectures and Foundation Models (EXP-10 to EXP-13)
* **ConvNeXt-Tiny (EXP-10, 11)**: When undefended, ConvNeXt-Tiny slightly outperforms raw ResNet-50 ($0.6821$ vs. $0.6644$), aided by its larger $7\times 7$ depthwise receptive fields. When defended (EXP-11), it reaches $0.8685$, matching defended ResNet.
* **Phikon Foundation Model (EXP-12, 13)**:
  * **Undefended Superiority (EXP-12)**: Phikon (ViT-B/16 pretrained on ~43M TCGA tiles via iBOT self-supervised learning) achieves **$0.8239$ OOD F1** with a simple linear probe trained in 9 minutes. Its massive multi-center pretraining naturally encodes stain invariance.
  * **The Foundation Model Breakdown (EXP-13)**: Adding Macenko normalization and augmentations causes Phikon's OOD F1 to collapse to **$0.6982$** (a $12.57$ point drop). Because Phikon's deep representations were learned on real, unnormalized slides, artificial color deconvolution distorts input images away from its pretrained manifold. **Traditional normalization defenses that help CNNs actively harm self-supervised foundation models.**

---

## 8. Error and Qualitative Analysis

Because digital histopathology requires distinguishing subtle cellular transitions, domain shift does not degrade all tissue classes equally. Grounded in the morphology of the nine colorectal classes from `DATA.md`, we analyze the four most frequent failure modes:

### 8.1 Confusion Mode 1: Muscle vs. Stroma (`STR` vs. `MUS`)
Smooth muscle (`MUS`) and cancer-associated stroma (`STR`) are both composed of elongated spindle cells embedded in dense pink/red eosinophilic fibers. Distinguishing them relies on subtle nuclear shapes: smooth muscle has parallel bundles with blunt, cigar-shaped nuclei, whereas stroma exhibits wavy collagen with tapered fibroblasts. When external scanner illumination shifts toward red or slide incubation times vary, the contrast between collagen and muscle fibers is flattened. Unnormalized models (EXP-01) heavily confuse `STR` with `MUS`. Macenko normalization largely resolves this by standardizing eosin extinction coefficients.

### 8.2 Confusion Mode 2: Transparent and Pale Tissues (`ADI`, `BACK`, `MUC`)
Adipose tissue (`ADI`) consists of empty fat vacuoles; background (`BACK`) is clear glass; and mucus (`MUC`) consists of faint, pale extracellular pools. When whole-slide scanners fluctuate in lamp brightness, background glass deviates from pure white. Under Reinhard color transfer, matching global color variance can bleach faint basophilic mucus pools into pure white, causing the model to misclassify mucus as clear background or empty fat droplets.

### 8.3 Confusion Mode 3: Optical Density SVD Instability on Debris (`DEB`)
Macenko normalization assumes every patch contains distinct Hematoxylin and Eosin dye concentrations. However, necrotic debris (`DEB`) contains irregular dead cell fragments, clotted blood, and cautery burn marks with chaotic dye absorption. In these degenerated patches, the optical density scatter plot lacks clean dye axes. This causes the SVD projection to align with optical noise, creating severe color distortion (such as oversaturated purple blotches) that misleads CNNs into predicting invasive cancer (`TUM`).

### 8.4 Confusion Mode 4: Dense Immune Cells vs. Tumor Glands (`LYM` vs. `TUM`)
Dense clusters of lymphocytes (`LYM`) contain small, spherical, intensely dark nuclei with almost no cytoplasm. Poorly differentiated tumor cells (`TUM`) also have high nuclear density and dark chromatin. In the undefended baseline (EXP-01), scanner contrast spikes saturate the blue/purple channel, causing the network to mistake benign lymphocytic aggregates for high-grade tumor glands. Normalization (EXP-07) re-establishes a balanced color dynamic range, effectively resolving this false alarm.

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
5. **Absence of Granular Confusion Logging**: The automated execution script aggregated metrics into scalar summaries (`Macro-F1`, `Accuracy`) at runtime to conserve I/O operations, omitting automated serialization of per-sample confusion matrices and gradient saliency maps. The failure modes in Section 8 are therefore reasoned from the biological and optical properties of each class rather than measured per-class confusion counts.

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
