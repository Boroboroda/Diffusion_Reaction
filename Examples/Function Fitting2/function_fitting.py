import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import *
from Networks.DNN import PINN
from Networks.IA_DNN import Attention_PINN
from Networks.Fourier_DNN import Fourier_PINN
from Networks.AF_PINN import AF_PINN
from Networks.Efficient_KAN import KAN
from Networks.Cheby_KAN import ChebyKAN
from Networks.My_ChebyKAN import ChebyKAN as CP_KAN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Current device:", device)
torch.manual_seed(42)


# Define target function
def target_function(x):
    y = np.zeros_like(x)

    mask1 = x < 1.5
    y[mask1] = np.sin(2.5 * np.pi * x[mask1]) + x[mask1] ** 2

    mask2 = (1.5 <= x) & (x < 3.0)
    y[mask2] = 0.5 * x[mask2] * np.exp(-x[mask2]) + np.abs(np.sin(5 * np.pi * x[mask2]))

    mask3 = (3.0 <= x) & (x < 4.5)
    y[mask3] = np.log(x[mask3] - 1) / np.log(2) - np.cos(2 * np.pi * x[mask3])

    mask4 = (4.5 <= x) & (x <= 6)

    # 创建阶梯函数的值
    steps = np.arange(4.5, 6, 0.3)  # 阶梯位置
    step_values = np.arange(len(steps))  # 每个阶梯的高度

    for i, step in enumerate(steps):
        if i == len(steps) - 1:
            mask_step = (step <= x) & (x <= 6)
        else:
            mask_step = (step <= x) & (x < steps[i + 1])
        y[mask_step & mask4] = step_values[i]

    return y


# 定义x的范围和步长
x = np.linspace(0.0, 6.0, 2500).reshape(-1, 1)
y = target_function(x)

# 转换为 Tensor
x_train_tensor = torch.tensor(x, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y, dtype=torch.float32).to(device)

criterion = nn.MSELoss()
# 训练模型
num_epochs = 100000
loss_record = {}
error_record = {}

for mode in ['PINN', 'IA_PINN']:  # ['PINN', 'IA_PINN', 'Fourier_PINN', 'AF_PINN', 'KAN', 'ChebyKAN']
    print("Training {}:".format(mode))
    layer = [1] + [32] * 3 + [1]
    torch.cuda.empty_cache()
    if mode == 'PINN':
        model = PINN(layer,activation=torch.nn.functional.silu, non_negative=False).to(device)

    elif mode == 'IA_PINN':
        model = Attention_PINN(layer,activation=torch.nn.functional.silu, non_negative=False).to(device)

    elif mode == 'Fourier_PINN':
        model = Fourier_PINN(layer, non_negative=False, use_rff=True, rff_num_features=64).to(device)

    elif mode == 'AF_PINN':
        model = AF_PINN(layer, non_negative=False, use_rff=True, rff_num_features=64).to(device)

    elif mode == 'KAN':
        layer = [1] + [32] * 3 + [1]
        model = KAN(layer, modified_output=False,
                    grid_size=8,
                    base_activation=nn.SiLU).to(device)
    elif mode == 'ChebyKAN':
        layer = [1] + [32] * 3 + [1]
        model = ChebyKAN(layer, degree=8, non_negative=False,
                         use_layer_norm=True).to(device)
        # model = CP_KAN(layer_sizes=layer,degree=5).to(device)
    else:
        print("Man! What can I say! Wrong Mode!!!")
        break

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs,
                                                     eta_min=1e-6)

    loss_record[mode] = []
    error_record[mode] = []
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        output = model(x_train_tensor)
        loss = criterion(output, y_train_tensor)

        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if torch.isnan(loss):
            print(f'Man What Can I say? {mode} out!!!')
            break

        # Save the loss for plotting
        loss_record[mode].append(loss.item())
        current_lr = optimizer.param_groups[0]['lr']
        if (epoch + 1) % 500 == 0:
            pbar.set_postfix(loss='{0:.3e}'.format(loss.item()),
                             lr='{0:.3e}'.format(current_lr))

        model.eval()
        with torch.no_grad():
            y_pred = model(x_train_tensor).cpu().numpy()
            y_true_tensor = torch.FloatTensor(y)
            y_pred_tensor = torch.FloatTensor(y_pred)
            relative_error = torch.norm(y_true_tensor - y_pred_tensor) / torch.norm(y_true_tensor)
            error_record[mode].append(relative_error)

    # 测试模型
    # 设置模型为评估模式
    model.eval()
    with torch.no_grad():
        y_pred = model(x_train_tensor).cpu().numpy()
        y_true_tensor = torch.FloatTensor(y)
        y_pred_tensor = torch.FloatTensor(y_pred)
        relative_error = torch.norm(y_true_tensor - y_pred_tensor) / torch.norm(y_true_tensor)
        print(f'Relative Error: {relative_error.item():.4f}')

    # 绘制结果
    plt.figure(figsize=(15, 6))
    plt.plot(x, y, 'ro', markersize=0.8, label='Target Function')
    plt.plot(x, y_pred, label=f'Fitted with {mode}', color='blue', linestyle='--')
    plt.axvline(x=1.5, color='r', linestyle='--', linewidth=0.8, label='Segment Boundaries')
    plt.axvline(x=3.0, color='r', linestyle='--', linewidth=0.8)
    plt.axvline(x=4.5, color='r', linestyle='--', linewidth=0.8)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Piecewise Target Functions')
    plt.legend()
    plt.grid(True)

    plt.savefig(f'{mode}_results.pdf', format="pdf", dpi=300, bbox_inches='tight')
    plt.close()


# Loss Curve
def plot_loss_curves(loss_record):
    """
    绘制多个损失曲线

    Args:
        loss_record: 字典，键为损失名称，值为损失值列表
    """
    # 创建图表
    plt.figure(figsize=(10, 8))

    # 颜色映射，可以根据需要添加更多颜色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    linestyles = ['-', '-.', '--']

    # 为每个损失创建平滑的曲线
    for idx, (key, values) in enumerate(loss_record.items()):
        # 获取x轴值（迭代次数）
        x = np.arange(len(values))

        # 选择颜色，如果颜色用完就循环使用
        color = colors[idx % len(colors)]
        linestyle = linestyles[idx % len(linestyles)]

        # 绘制原始数据点（小点）
        plt.plot(x, values, label=key, color=color, linestyle=linestyle)

    # 设置图表样式
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.yscale("log")
    plt.title('Training Loss Curves', fontsize=14)

    # 添加图例
    plt.legend(loc='upper right')

    # 调整布局以确保图例完全可见
    plt.tight_layout()

    return plt


fig = plot_loss_curves(loss_record)
plt.savefig(f'loss_curve.pdf', format="pdf", dpi=300, bbox_inches='tight')
plt.close()

# Loss Curve
def plot_error_curves(error_record):
    """
    绘制多个损失曲线

    Args:
        error_record: 字典，键为损失名称，值为损失值列表
    """
    # 创建图表
    plt.figure(figsize=(10, 8))

    # 颜色映射，可以根据需要添加更多颜色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    linestyles = ['-', '-.', '--']

    # 为每个损失创建平滑的曲线
    for idx, (key, values) in enumerate(error_record.items()):
        # 获取x轴值（迭代次数）
        x = np.arange(len(values))

        # 选择颜色，如果颜色用完就循环使用
        color = colors[idx % len(colors)]
        linestyle = linestyles[idx % len(linestyles)]

        # 绘制原始数据点（小点）
        plt.plot(x, values, label=key, color=color, linestyle=linestyle)

    # 设置图表样式
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.yscale("log")
    plt.title('Relative Error $R^2$', fontsize=14)

    # 添加图例
    plt.legend(loc='upper right')

    # 调整布局以确保图例完全可见
    plt.tight_layout()

    return plt


fig = plot_error_curves(error_record)
plt.savefig(f'relative_error_curve.pdf', format="pdf", dpi=300, bbox_inches='tight')
plt.close()
