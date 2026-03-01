# highlevel_260301.json 修改说明

依据 **正常** 训练日志对齐配置：
- 高级规划器：`trained_models_highlevel/test/20260301130248/logs/log.txt`
- 低级规划器：`trained_models_lowlevel/test/20260301143126/logs/log.txt`

## 正常训练中的关键量级

| 指标 | 高级规划器 (highlevel) | 低级规划器 (lowlevel) |
|------|------------------------|------------------------|
| Policy_Grad_Norms | ~1.1e6 ~ 2.7e6 | ~3.8e4 ~ 3e5 |
| 学习率 | 0.0001 | 0.0001 |
| 数据 | num_demos=33, num_sequences=425811 | 同上 |
| playdata.goal_image_range | [100, 200] | (N/A) |
| playdata.eval_goal_gap | 150 | (N/A) |
| train | pad_seq_length=true, frame_stack=1, pad_frame_stack=true | seq_length=10, 其余同上 |

## 已对 highlevel_260301.json 做的修改

1. **gradient_clip_norm**：`1.0` → **`2000000.0`**  
   - 正常 highlevel 梯度范数约 1e6~2.7e6，裁剪到 1 会过度压制。  
   - 改为 2e6，只拦断爆炸（如 auto_demo200 的 4e7），不压制正常量级。

2. **learning_rate.initial**：`1e-5` → **`1e-4`**  
   - 与正常高级规划器一致（0.0001）。

3. **playdata**  
   - `goal_image_range`: `[80, 120]` → **`[100, 200]`**  
   - `eval_goal_gap`: `100` → **`150`**  
   - 与正常 highlevel 一致，目标采样更宽、更稳定。

4. **train**  
   - 增加 **pad_seq_length: true, frame_stack: 1, pad_frame_stack: true**  
   - 与正常 highlevel 的 dataset 设置一致。

5. **experiment**  
   - **on_best_rollout_success_rate**: `false` → **`true`**（与正常一致，便于按 success 存 checkpoint）  
   - **render_video**: `false` → **`true`**（与正常一致，可录 rollout 视频）

## 使用与微调建议

- **数据仍为 auto_demo200 等易爆炸数据时**  
  - 若训练中仍出现梯度爆炸，可把 `gradient_clip_norm` 调小，例如 **1e6** 或 **5e5**。  
  - 若 2e6 下训练很稳，可尝试适当调大（如 3e6）或保持 2e6。

- **数据与正常 playdata 相近时**  
  - 可尝试关闭裁剪：在配置中删掉 `gradient_clip_norm` 或设为 `null`，与正常 highlevel 完全一致。

- **学习率**  
  - 当前已与正常一致为 1e-4；若某数据集上仍不稳定，可再试 **5e-5**。

- **运行时**  
  - 训练时通过命令行覆盖数据路径，例如：  
    `--dataset /path/to/your/image_demo_local.hdf5`
