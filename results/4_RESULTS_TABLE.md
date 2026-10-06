# Final Ablation Results - seed 100 - subset = 50.000

| exp_id   | stage   | backbone      | norm     | aug          |   best_val_f1 |   test_id_f1 |   test_ood_f1 |   delta_f1 |   rr_f1 |   test_id_acc |   test_ood_acc |   minutes |
|:---------|:--------|:--------------|:---------|:-------------|--------------:|-------------:|--------------:|-----------:|--------:|--------------:|---------------:|----------:|
| EXP-01   | Stage 0 | resnet50      | none     | none         |        0.9937 |       0.9929 |        0.6351 |     0.3578 |   63.96 |        0.9929 |         0.7007 |     20.98 |
| EXP-02   | Stage 1 | resnet50      | reinhard | none         |        0.9923 |       0.9929 |        0.8068 |     0.1861 |   81.26 |        0.9929 |         0.8631 |     34.43 |
| EXP-03   | Stage 1 | resnet50      | macenko  | none         |        0.9776 |       0.9788 |        0.8111 |     0.1678 |   82.86 |        0.9788 |         0.8492 |     34.51 |
| EXP-04   | Stage 2 | resnet50      | none     | aug_geo      |        0.9957 |       0.9959 |        0.5704 |     0.4255 |   57.27 |        0.9959 |         0.6155 |     20.06 |
| EXP-05   | Stage 2 | resnet50      | none     | aug_stain    |        0.99   |       0.9919 |        0.7006 |     0.2912 |   70.64 |        0.9919 |         0.7701 |     20.28 |
| EXP-06   | Stage 2 | resnet50      | none     | aug_combined |        0.994  |       0.9943 |        0.7334 |     0.2609 |   73.76 |        0.9943 |         0.783  |     20.27 |
| EXP-07   | Stage 3 | resnet50      | macenko  | aug_geo      |        0.9835 |       0.9835 |        0.8605 |     0.123  |   87.49 |        0.9835 |         0.8915 |     35.11 |
| EXP-08   | Stage 3 | resnet50      | macenko  | aug_stain    |        0.9752 |       0.9762 |        0.8334 |     0.1427 |   85.38 |        0.9761 |         0.8635 |     39.13 |
| EXP-09   | Stage 3 | resnet50      | macenko  | aug_combined |        0.9812 |       0.9825 |        0.8595 |     0.1231 |   87.48 |        0.9825 |         0.8872 |     40.69 |
| EXP-10   | Stage 4 | convnext_tiny | none     | none         |        0.9933 |       0.9935 |        0.794  |     0.1995 |   79.92 |        0.9935 |         0.8412 |     24.04 |
| EXP-11   | Stage 4 | convnext_tiny | macenko  | aug_combined |        0.9808 |       0.9836 |        0.8752 |     0.1084 |   88.98 |        0.9836 |         0.9015 |     44.18 |
| EXP-12   | Stage 4 | phikon        | none     | none         |        0.9939 |       0.9945 |        0.8124 |     0.1821 |   81.69 |        0.9945 |         0.855  |     16.44 |
| EXP-13   | Stage 4 | phikon        | macenko  | aug_combined |        0.9221 |       0.9183 |        0.6933 |     0.225  |   75.5  |        0.9184 |         0.7419 |     40.31 |
