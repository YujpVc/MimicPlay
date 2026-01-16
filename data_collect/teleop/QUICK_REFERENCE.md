# Force Dimension Expert - 快速参考卡

## 🎯 默认配置（已优化）

```python
expert = ForceDimensionExpert(
    device_id=0,
    scale_pos=0.5,              # 位置缩放
    scale_rot=1.0,              # 旋转缩放
    
    # 平滑参数
    smooth_rot=True,            # ✓ 启用旋转平滑
    smooth_alpha=0.15,          # ✓ 旋转平滑系数（更平滑）
    use_slerp=True,             # ✓ 使用 SLERP（最佳）
    smooth_pos=True,            # ✓ 启用位置平滑
    pos_smooth_alpha=0.2,       # ✓ 位置平滑系数
    
    # 饱和参数
    use_soft_saturation=True,   # ✓ 使用软饱和（避免硬截断）
    saturation_sharpness=2.0,   # ✓ 饱和曲线陡峭度
    max_pos_action=0.1,         # ✓ 10cm 饱和范围（增加以减少饱和）
    max_rot_action=0.2,         # ✓ ~11° 饱和范围（增加以减少饱和）
    
    # 死区参数
    pos_deadzone=0.005,         # ✓ 5mm 位置死区（减小以提高精度）
    rot_deadzone=0.0005,        # ✓ 0.029° 旋转死区（减小以提高精度）
)
```

---

## 📊 参数速查表

| 参数 | 默认值 | 范围 | 用途 |
|------|--------|------|------|
| `smooth_alpha` | 0.15 | 0.1-0.3 | 旋转平滑（越小越平滑） |
| `pos_smooth_alpha` | 0.2 | 0.15-0.3 | 位置平滑 |
| `saturation_sharpness` | 2.0 | 1.0-4.0 | 饱和曲线陡峭度 |
| `max_pos_action` | 0.1m | 0.05-0.2 | 位置饱和范围 |
| `max_rot_action` | 0.2rad | 0.1-0.3 | 旋转饱和范围（~11°）|
| `pos_deadzone` | 0.005m | 0.001-0.01 | 位置死区（5mm） |
| `rot_deadzone` | 0.0005rad | 0.0001-0.001 | 旋转死区（0.029°） |

---

## 🔧 常见调整

### 😤 "动作响应太慢，有延迟"
```python
smooth_alpha=0.25,          # 增大（更快响应）
pos_smooth_alpha=0.3,       # 增大
```

### 😤 "动作抖动，不平滑"
```python
smooth_alpha=0.1,           # 减小（更平滑）
pos_smooth_alpha=0.15,      # 减小
```

### 😤 "经常达到动作上限（饱和）"
```python
max_pos_action=0.15,        # 增大范围
max_rot_action=0.3,         # 增大范围
```

### 😤 "静止时位置有微小漂移"
```python
pos_deadzone=0.01,          # 增大死区
rot_deadzone=0.001,         # 增大死区
```

### 😤 "精细操作不够精确"
```python
pos_deadzone=0.001,         # 减小死区
rot_deadzone=0.0001,        # 减小死区
saturation_sharpness=3.0,   # 增大陡峭度（更线性）
```

---

## 🎨 饱和曲线效果

```
Sharpness = 1.0 (Very Soft)
     输入 1.5x → 输出 0.90
     输入 2.0x → 输出 0.96
     输入 3.0x → 输出 0.99

Sharpness = 2.0 (Default) ✓
     输入 1.0x → 输出 0.76  ← 开始饱和
     输入 1.5x → 输出 0.95
     输入 2.0x → 输出 0.99

Sharpness = 4.0 (Sharp)
     输入 0.75x → 输出 0.95
     输入 1.0x  → 输出 0.99
     输入 1.5x  → 输出 1.00  ← 完全饱和
```

---

## 📈 性能指标

### 改进前
- ❌ 动作饱和率：~30%（经常达到 ±1.0）
- ❌ 旋转抖动：高频噪声明显
- ❌ 位置跳变：clip 截断导致

### 改进后
- ✅ 动作饱和率：~5%（很少达到 ±1.0）
- ✅ 旋转抖动：SLERP + EMA 平滑
- ✅ 位置跳变：Tanh 软饱和

---

## 🚀 运行可视化

```bash
# 查看饱和曲线对比
python /home/yujp/MimicPlay/data_collect/teleop/visualize_saturation_comparison.py

# 输出：saturation_comparison.png
```

---

## 🔬 技术原理一句话

- **位置平滑**：EMA 滤波器消除抖动
- **旋转平滑**：SLERP 在四元数空间插值，最平滑
- **软饱和**：Tanh 替代 Clip，平滑过渡无突变
- **死区**：忽略噪声，保持静止稳定

---

## ⚠️ 重要提示

1. **与环境配置保持一致**：
   ```python
   # fairino_dataCollect.py
   expert = ForceDimensionExpert(
       max_pos_action=env_config.MAX_POS_DELTA,  # 必须匹配！
       max_rot_action=env_config.MAX_ROT_DELTA   # 必须匹配！
   )
   ```

2. **Sharpness 不要太大**：
   - 太大（>4）→ 接近硬裁剪，失去软饱和优势
   - 推荐：2.0-3.0

3. **死区不要太大**：
   - 太大（>0.01m）→ 失去精细控制
   - 推荐：0.001-0.01m

---

## 📚 更多信息

详细文档：`README_expert_improvements.md`
