# Final Ablation Results - seed 42 - subset = 25.000

| exp_id   | stage   | backbone      | norm     | aug          |   best_val_f1 |   test_id_f1 |   test_ood_f1 |   delta_f1 |   rr_f1 |   test_id_acc |   test_ood_acc |   minutes |
|:---------|:--------|:--------------|:---------|:-------------|--------------:|-------------:|--------------:|-----------:|--------:|--------------:|---------------:|----------:|
| EXP-01   | Stage 0 | resnet50      | none     | none         |        0.9917 |       0.9893 |        0.6644 |     0.3249 |   67.16 |        0.9893 |         0.7318 |     11.01 |
| EXP-02   | Stage 1 | resnet50      | reinhard | none         |        0.9872 |       0.9862 |        0.8488 |     0.1373 |   86.07 |        0.9861 |         0.8915 |     16.53 |
| EXP-03   | Stage 1 | resnet50      | macenko  | none         |        0.9693 |       0.9653 |        0.809  |     0.1563 |   83.8  |        0.9653 |         0.846  |     17.27 |
| EXP-04   | Stage 2 | resnet50      | none     | aug_geo      |        0.9915 |       0.9899 |        0.6172 |     0.3727 |   62.35 |        0.9899 |         0.6955 |     10.85 |
| EXP-05   | Stage 2 | resnet50      | none     | aug_stain    |        0.9859 |       0.9877 |        0.7443 |     0.2434 |   75.36 |        0.9877 |         0.8035 |     10.88 |
| EXP-06   | Stage 2 | resnet50      | none     | aug_combined |        0.988  |       0.9901 |        0.7105 |     0.2796 |   71.76 |        0.9901 |         0.7734 |     10.94 |
| EXP-07   | Stage 3 | resnet50      | macenko  | aug_geo      |        0.9749 |       0.9717 |        0.8689 |     0.1028 |   89.42 |        0.9717 |         0.9    |     17.91 |
| EXP-08   | Stage 3 | resnet50      | macenko  | aug_stain    |        0.9651 |       0.9619 |        0.785  |     0.177  |   81.6  |        0.9619 |         0.8276 |     20.04 |
| EXP-09   | Stage 3 | resnet50      | macenko  | aug_combined |        0.972  |       0.9699 |        0.8604 |     0.1094 |   88.72 |        0.9699 |         0.8921 |     20.65 |
| EXP-10   | Stage 4 | convnext_tiny | none     | none         |        0.9896 |       0.9891 |        0.6821 |     0.3069 |   68.97 |        0.9891 |         0.7558 |     12.98 |
| EXP-11   | Stage 4 | convnext_tiny | macenko  | aug_combined |        0.9757 |       0.9763 |        0.8685 |     0.1077 |   88.97 |        0.9763 |         0.8997 |     20.7  |
| EXP-12   | Stage 4 | phikon        | none     | none         |        0.9915 |       0.9912 |        0.8239 |     0.1673 |   83.12 |        0.9912 |         0.8623 |      9.08 |
| EXP-13   | Stage 4 | phikon        | macenko  | aug_combined |        0.9018 |       0.905  |        0.6982 |     0.2068 |   77.15 |        0.9045 |         0.7398 |     20.11 |

