# Complete-sequence end-to-end results

Admission: PASS; 6 cells, 72 formal processes including warmups, 12 untimed audit processes.

Median of three timed processes per cell/method; lower milliseconds is better. Process wall is the primary end-to-end measure. API excludes PNG decode, startup and output.

| Sequence | Clones | Method | Process ms | Replay ms | API ms | Process speedup | ATE m | Admission |
|---|---:|---|---:|---:|---:|---:|---:|---|
| V1_01_easy | 11 | original | 18041.08 | 17979.01 | 11366.40 | 1.000x | 0.064744 | PASS |
| V1_01_easy | 11 | root_messages | 16125.18 | 16054.73 | 9438.73 | 1.119x | 0.064744 | PASS |
| V1_01_easy | 11 | enhanced_ekf | 16028.40 | 15961.15 | 9350.63 | 1.126x | 0.064744 | PASS |
| V1_01_easy | 21 | original | 19705.52 | 19641.80 | 13037.45 | 1.000x | 0.054243 | PASS |
| V1_01_easy | 21 | root_messages | 16629.45 | 16556.34 | 9926.23 | 1.185x | 0.054243 | PASS |
| V1_01_easy | 21 | enhanced_ekf | 16753.66 | 16685.72 | 10072.57 | 1.176x | 0.054243 | PASS |
| V1_02_medium | 11 | original | 10200.14 | 10149.31 | 6428.02 | 1.000x | 0.068339 | PASS |
| V1_02_medium | 11 | root_messages | 9467.19 | 9408.73 | 5686.14 | 1.077x | 0.068339 | PASS |
| V1_02_medium | 11 | enhanced_ekf | 9757.60 | 9702.48 | 5976.14 | 1.045x | 0.068339 | PASS |
| V1_02_medium | 21 | original | 11239.97 | 11188.66 | 7467.14 | 1.000x | 0.063005 | PASS |
| V1_02_medium | 21 | root_messages | 9879.38 | 9821.10 | 6071.30 | 1.138x | 0.063005 | PASS |
| V1_02_medium | 21 | enhanced_ekf | 9994.06 | 9938.85 | 6206.30 | 1.125x | 0.063005 | PASS |
| V1_03_difficult | 11 | original | 11918.31 | 11863.70 | 8123.14 | 1.000x | 0.043227 | PASS |
| V1_03_difficult | 11 | root_messages | 11434.51 | 11372.18 | 7619.13 | 1.042x | 0.043227 | PASS |
| V1_03_difficult | 11 | enhanced_ekf | 11436.11 | 11376.02 | 7630.97 | 1.042x | 0.043227 | PASS |
| V1_03_difficult | 21 | original | 13155.76 | 13100.61 | 9357.07 | 1.000x | 0.064236 | PASS |
| V1_03_difficult | 21 | root_messages | 11934.51 | 11871.60 | 8109.71 | 1.102x | 0.064236 | PASS |
| V1_03_difficult | 21 | enhanced_ekf | 11974.74 | 11914.77 | 8164.47 | 1.099x | 0.064236 | PASS |

The enhanced EKF is our supplementary control. This experiment does not establish a hierarchy-specific benefit or GPU performance. See analysis.json for all admission checks, exact trajectory discrepancies and every trial; trials.csv contains the compact measurements.

Candidate speedup versus official ranges from 1.042x to 1.185x for the complete process, and 1.066x to 1.313x for camera+IMU API compute.

These are measurements of the specified ROS-free folder replay and configurations, not a claim about every OpenVINS application or device. The three processes per cell do not establish population confidence intervals.
