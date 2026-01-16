"""
计算MimicPlay模型的参数量
直接从checkpoint的state_dict统计参数
"""
import torch
import argparse


def count_params_from_checkpoint(checkpoint_path):
    """从checkpoint直接统计参数量"""
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    # 统计模型参数
    total_params = sum(p.numel() for p in ckpt['model'].values() if hasattr(p, 'numel'))
    
    # 按模块统计
    module_stats = {}
    for name, param in ckpt['model'].items():
        if hasattr(param, 'numel'):
            # 提取主模块名称 (例如: "policy.nets.encoder.obs_nets.agentview_image.0.weight" -> "policy")
            parts = name.split('.')
            if len(parts) > 0:
                module_name = parts[0]
                if module_name not in module_stats:
                    module_stats[module_name] = 0
                module_stats[module_name] += param.numel()
    
    return total_params, module_stats


def format_number(num):
    """格式化数字显示"""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(num)


def main():
    parser = argparse.ArgumentParser(description='计算MimicPlay模型参数量')
    parser.add_argument('--highlevel_checkpoint', type=str, 
                       default='/home/yujp/MimicPlay/trained_models_highlevel/highlevel_model_epoch_357_best_validation_-65.05948219299316.pth',
                       help='Highlevel模型checkpoint路径')
    parser.add_argument('--lowlevel_checkpoint', type=str,
                       default='/home/yujp/MimicPlay/trained_models_lowlevel/test/20251212213957/models/model_epoch_400.pth',
                       help='Lowlevel模型checkpoint路径')
    parser.add_argument('--model_type', type=str, choices=['highlevel', 'lowlevel', 'both'],
                       default='both', help='要分析的模型类型')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("MimicPlay 模型参数量分析")
    print("=" * 80)
    
    highlevel_total = 0
    lowlevel_total = 0
    
    try:
        if args.model_type in ['highlevel', 'both']:
            print("\n" + "=" * 80)
            print("Highlevel 模型 (GMM轨迹规划器)")
            print("Checkpoint:", args.highlevel_checkpoint)
            print("=" * 80)
            
            highlevel_total, hl_modules = count_params_from_checkpoint(args.highlevel_checkpoint)
            
            print(f"\n总参数量: {format_number(highlevel_total)} ({highlevel_total:,})")
            
            print("\n按模块统计:")
            for module, count in sorted(hl_modules.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / highlevel_total * 100) if highlevel_total > 0 else 0
                print(f"  {module}: {format_number(count)} ({count:,}) - {percentage:.1f}%")
        
        if args.model_type in ['lowlevel', 'both']:
            print("\n" + "=" * 80)
            print("Lowlevel 模型 (GPT机器人控制器)")
            print("Checkpoint:", args.lowlevel_checkpoint)
            print("=" * 80)
            
            lowlevel_total, ll_modules = count_params_from_checkpoint(args.lowlevel_checkpoint)
            
            print(f"\n总参数量: {format_number(lowlevel_total)} ({lowlevel_total:,})")
            
            print("\n按模块统计:")
            for module, count in sorted(ll_modules.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / lowlevel_total * 100) if lowlevel_total > 0 else 0
                print(f"  {module}: {format_number(count)} ({count:,}) - {percentage:.1f}%")
        
        if args.model_type == 'both':
            print("\n" + "=" * 80)
            print("总体统计 (Highlevel + Lowlevel)")
            print("=" * 80)
            total_params = highlevel_total + lowlevel_total
            
            print(f"\nHighlevel 参数: {format_number(highlevel_total)} ({highlevel_total:,})")
            print(f"Lowlevel 参数:  {format_number(lowlevel_total)} ({lowlevel_total:,})")
            print(f"{'-' * 60}")
            print(f"整体模型总参数量: {format_number(total_params)} ({total_params:,})")
            
            if highlevel_total > 0 and lowlevel_total > 0:
                hl_percent = (highlevel_total / total_params * 100)
                ll_percent = (lowlevel_total / total_params * 100)
                print(f"\n参数占比:")
                print(f"  Highlevel: {hl_percent:.1f}%")
                print(f"  Lowlevel:  {ll_percent:.1f}%")
    
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
