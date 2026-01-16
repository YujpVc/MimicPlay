import re
import matplotlib.pyplot as plt
import numpy as np

def parse_training_log(file_path):
    """
    解析训练日志文件以提取指标
    """
    train_epochs = []
    train_losses = []
    train_log_likelihoods = []
    val_epochs = []
    val_losses = []
    val_log_likelihoods = []

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        return {k: [] for k in ['train_epochs', 'train_losses', 'train_log_likelihoods', 'val_epochs', 'val_losses', 'val_log_likelihoods']}

    train_pattern = r'Train Epoch (\d+)\s*\{[^}]*"Log_Likelihood": ([-\d.]+),\s*"Loss": ([-\d.]+)'
    val_pattern = r'Validation Epoch (\d+)\s*\{[^}]*"Log_Likelihood": ([-\d.]+),\s*"Loss": ([-\d.]+)'

    train_matches = re.findall(train_pattern, content)
    for match in train_matches:
        epoch, log_likelihood, loss = match
        train_epochs.append(int(epoch))
        train_log_likelihoods.append(float(log_likelihood))
        # 【修改1】去掉 abs()，保留真实的符号
        train_losses.append(float(loss))

    val_matches = re.findall(val_pattern, content)
    for match in val_matches:
        epoch, log_likelihood, loss = match
        val_epochs.append(int(epoch))
        val_log_likelihoods.append(float(log_likelihood))
        # 【修改1】去掉 abs()，保留真实的符号
        val_losses.append(float(loss))

    return {
        'train_epochs': train_epochs,
        'train_losses': train_losses,
        'train_log_likelihoods': train_log_likelihoods,
        'val_epochs': val_epochs,
        'val_losses': val_losses,
        'val_log_likelihoods': val_log_likelihoods
    }

def create_clean_visualization(log_file_path):
    """
    创建可视化，自动过滤异常值以显示真实趋势
    """
    data = parse_training_log(log_file_path)

    if not data['train_epochs']:
        print("没有找到训练数据")
        return

    plt.style.use('default')
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    train_color = '#0066CC'
    val_color = '#CC3333'
    best_marker_color = '#FFD700'

    # ==========================================
    # 1. 训练与验证损失 (Real Loss)
    # ==========================================
    ax1 = axes[0]
    ax1.plot(data['train_epochs'], data['train_losses'], color=train_color, linewidth=0.8, alpha=0.8, label='Training Loss')
    ax1.plot(data['val_epochs'], data['val_losses'], color=val_color, linewidth=0.8, alpha=0.8, label='Validation Loss')

    # ---【修改2：适应负数值的 Y 轴缩放】---
    all_losses = data['train_losses'] + data['val_losses']
    if all_losses:
        # 因为现在不仅有向上的尖刺(爆炸)，Loss本身也是负的
        # 我们取中间的 90% 数据 (5% 到 95%)，切掉两头的极端值
        y_min = np.percentile(all_losses, 5) 
        y_max = np.percentile(all_losses, 95)
        
        # 稍微加一点边距让图好看点
        padding = (y_max - y_min) * 0.1 if (y_max - y_min) > 0 else 1.0
        
        # 只有当计算出的范围合理时才应用限制
        if y_max > y_min:
            ax1.set_ylim(y_min - padding, y_max + padding)
            print(f"图1: Y轴范围限制在 [{y_min - padding:.2f}, {y_max + padding:.2f}] (显示真实数值)")

    # 标记最佳点 (Loss 越小越好，哪怕是负数，-20 比 -5 小)
    if data['val_losses']:
        best_val_idx = np.argmin(data['val_losses'])
        ax1.plot(data['val_epochs'][best_val_idx], data['val_losses'][best_val_idx], '*', 
                 color=best_marker_color, markersize=15, markeredgecolor='black', zorder=5)

    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Real Loss (Negative is Good)', fontweight='bold', labelpad=-40)
    ax1.set_title('Training vs Validation Loss', fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend()

    # ==========================================
    # 2. 训练与验证对数似然 (Log-Likelihood)
    # ==========================================
    ax2 = axes[1]
    ax2.plot(data['train_epochs'], data['train_log_likelihoods'], color='#2E8B57', linewidth=0.8, alpha=0.8, label='Train LL')
    ax2.plot(data['val_epochs'], data['val_log_likelihoods'], color='#FF8C00', linewidth=0.8, alpha=0.8, label='Val LL')

    all_logs = data['train_log_likelihoods'] + data['val_log_likelihoods']
    if all_logs:
        # 这里的异常值通常是极小的负数(崩塌)，所以重点切下面
        y_min = np.percentile(all_logs, 5) 
        y_max = np.max(all_logs) + (abs(np.max(all_logs)) * 0.1 if np.max(all_logs) != 0 else 1.0)
        
        if y_min < y_max:
            ax2.set_ylim(y_min, y_max)

    if data['val_log_likelihoods']:
        best_val_log_idx = np.argmax(data['val_log_likelihoods'])
        ax2.plot(data['val_epochs'][best_val_log_idx], data['val_log_likelihoods'][best_val_log_idx], '*', 
                 color=best_marker_color, markersize=15, markeredgecolor='black', zorder=5)

    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Log-Likelihood', fontweight='bold', labelpad=-30)
    ax2.set_title('Log-Likelihood', fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend()

    # ==========================================
    # 3. Epoch-to-Epoch 变化率
    # ==========================================
    ax3 = axes[2]
    if len(data['train_losses']) > 1:
        train_loss = np.array(data['train_losses'])
        # 【修改3】计算百分比变化时，分母要取绝对值，否则负负得正方向就反了
        # 公式: (新 - 旧) / |旧| * 100
        loss_reduction = [(train_loss[i] - train_loss[i - 1]) / abs(train_loss[i - 1]) * 100 for i in range(1, len(train_loss))]
        
        ax3.bar(range(1, len(train_loss)), loss_reduction, color='gray', alpha=0.6)
        ax3.axhline(0, color='black', linewidth=0.5)

        if loss_reduction:
            q_low = np.percentile(loss_reduction, 5)
            q_high = np.percentile(loss_reduction, 95)
            limit_range = max(abs(q_low), abs(q_high), 50) 
            ax3.set_ylim(-limit_range, limit_range)

    ax3.set_xlabel('Epoch', fontweight='bold')
    ax3.set_ylabel('Change %', fontweight='bold', labelpad=-45)
    ax3.set_title('Loss Reduction', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    plt.suptitle('Training Analysis (Real Values)', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('/home/yujp/MimicPlay/training_result/real_training_analysis_lowlevel_20260101115220.png', dpi=100)
    plt.show()

# 运行

create_clean_visualization("/home/yujp/Genesis/scripts/XBOX_control/dataCollect/demo/demo_multitask_cube/lowlevel/test/20260101115220/logs/log.txt")