# Final Ablation Results - seed 42 - subset = 50.000

| exp_id   | stage   | backbone      | norm     | aug          |   best_val_f1 |   test_id_f1 |   test_ood_f1 |   delta_f1 |   rr_f1 |   test_id_acc |   test_ood_acc |   minutes |
|:---------|:--------|:--------------|:---------|:-------------|--------------:|-------------:|--------------:|-----------:|--------:|--------------:|---------------:|----------:|
| EXP-01   | Stage 0 | resnet50      | none     | none         |        0.994  |       0.9932 |        0.6104 |     0.3828 |   61.45 |        0.9932 |         0.6724 |     20.67 |
| EXP-02   | Stage 1 | resnet50      | reinhard | none         |        0.9917 |       0.9913 |        0.8236 |     0.1677 |   83.09 |        0.9913 |         0.8787 |     33.02 |
| EXP-03   | Stage 1 | resnet50      | macenko  | none         |        0.9737 |       0.9777 |        0.8211 |     0.1566 |   83.98 |        0.9777 |         0.8538 |     33.36 |
| EXP-04   | Stage 2 | resnet50      | none     | aug_geo      |        0.9952 |       0.9951 |        0.5574 |     0.4377 |   56.01 |        0.9951 |         0.6049 |     20.51 |
| EXP-05   | Stage 2 | resnet50      | none     | aug_stain    |        0.9904 |       0.9903 |        0.7487 |     0.2416 |   75.6  |        0.9903 |         0.8001 |     20.61 |
| EXP-06   | Stage 2 | resnet50      | none     | aug_combined |        0.9948 |       0.9931 |        0.8002 |     0.1928 |   80.58 |        0.9931 |         0.8345 |     20.76 |
| EXP-07   | Stage 3 | resnet50      | macenko  | aug_geo      |        0.9816 |       0.9828 |        0.8319 |     0.1509 |   84.65 |        0.9828 |         0.8593 |     35.66 |
| EXP-08   | Stage 3 | resnet50      | macenko  | aug_stain    |        0.9742 |       0.9734 |        0.8266 |     0.1467 |   84.93 |        0.9733 |         0.8584 |     40.41 |
| EXP-09   | Stage 3 | resnet50      | macenko  | aug_combined |        0.9792 |       0.9778 |        0.831  |     0.1468 |   84.99 |        0.9777 |         0.8586 |     40.54 |
| EXP-10   | Stage 4 | convnext_tiny | none     | none         |        0.9913 |       0.9907 |        0.7607 |     0.23   |   76.78 |        0.9907 |         0.818  |     24.33 |
| EXP-11   | Stage 4 | convnext_tiny | macenko  | aug_combined |        0.9817 |       0.9785 |        0.8753 |     0.1032 |   89.45 |        0.9785 |         0.9039 |     41.72 |
| EXP-12   | Stage 4 | phikon        | none     | none         |        0.9937 |       0.994  |        0.8319 |     0.1621 |   83.69 |        0.994  |         0.8701 |     17.03 |
| EXP-13   | Stage 4 | phikon        | macenko  | aug_combined |        0.9132 |       0.915  |        0.679  |     0.236  |   74.21 |        0.9151 |         0.7253 |     40.03 |
