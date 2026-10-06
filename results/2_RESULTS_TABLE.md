# Final Ablation Results - seed 100 - subset = 25.000

| exp_id   | stage   | backbone      | norm     | aug          |   best_val_f1 |   test_id_f1 |   test_ood_f1 |   delta_f1 |   rr_f1 |   test_id_acc |   test_ood_acc |   minutes |
|:---------|:--------|:--------------|:---------|:-------------|--------------:|-------------:|--------------:|-----------:|--------:|--------------:|---------------:|----------:|
| EXP-01   | Stage 0 | resnet50      | none     | none         |        0.9912 |       0.9899 |        0.7234 |     0.2664 |   73.08 |        0.9899 |         0.7868 |     23.97 |
| EXP-02   | Stage 1 | resnet50      | reinhard | none         |        0.9883 |       0.9886 |        0.8244 |     0.1642 |   83.39 |        0.9885 |         0.8776 |     19.39 |
| EXP-03   | Stage 1 | resnet50      | macenko  | none         |        0.9698 |       0.9661 |        0.8393 |     0.1268 |   86.87 |        0.9661 |         0.8721 |     23.19 |
| EXP-04   | Stage 2 | resnet50      | none     | aug_geo      |        0.9915 |       0.9893 |        0.6685 |     0.3208 |   67.57 |        0.9893 |         0.7255 |     11.73 |
| EXP-05   | Stage 2 | resnet50      | none     | aug_stain    |        0.9885 |       0.9885 |        0.7194 |     0.2692 |   72.77 |        0.9885 |         0.784  |     11.2  |
| EXP-06   | Stage 2 | resnet50      | none     | aug_combined |        0.9904 |       0.9899 |        0.761  |     0.2289 |   76.88 |        0.9899 |         0.8134 |     11.31 |
| EXP-07   | Stage 3 | resnet50      | macenko  | aug_geo      |        0.9753 |       0.9721 |        0.8414 |     0.1307 |   86.56 |        0.972  |         0.8773 |     20.77 |
| EXP-08   | Stage 3 | resnet50      | macenko  | aug_stain    |        0.9669 |       0.9618 |        0.8019 |     0.1599 |   83.38 |        0.9619 |         0.8398 |     22.26 |
| EXP-09   | Stage 3 | resnet50      | macenko  | aug_combined |        0.9712 |       0.968  |        0.8307 |     0.1373 |   85.82 |        0.968  |         0.8702 |     25.21 |
| EXP-10   | Stage 4 | convnext_tiny | none     | none         |        0.9896 |       0.9869 |        0.6727 |     0.3142 |   68.16 |        0.9869 |         0.7586 |     13.39 |
| EXP-11   | Stage 4 | convnext_tiny | macenko  | aug_combined |        0.9768 |       0.9776 |        0.8613 |     0.1163 |   88.1  |        0.9776 |         0.8919 |     25.95 |
| EXP-12   | Stage 4 | phikon        | none     | none         |        0.9912 |       0.9917 |        0.8668 |     0.1249 |   87.4  |        0.9917 |         0.9031 |      9.14 |
| EXP-13   | Stage 4 | phikon        | macenko  | aug_combined |        0.9053 |       0.9053 |        0.6903 |     0.215  |   76.26 |        0.905  |         0.7336 |     24.62 |
