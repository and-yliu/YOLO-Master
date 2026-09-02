# p1coco run summary

| run | epochs | mAP50-95 | mAP50 | task_ratio | kd_raw | kd_weighted | moe_aux | box | cls | seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| a-s17 | 50 | 0.18409 | 0.2881 | 0.11271 | 0.048362 | 9.2818 | 0.99998 | 1.8988 | 2.0104 | 41420.5 |
| off-s17 | 50 | 0.19009 | 0.29511 | - | - | - | 0.99999 | 1.8863 | 1.9837 | 28030.3 |

## 事后无混杂核查

各 run 的 resolved args 仅在声明的对照轴内不同。

## 归档

- `a-s17` → `results/p1coco_a-s17/`
- `off-s17` → `results/p1coco_off-s17/`
