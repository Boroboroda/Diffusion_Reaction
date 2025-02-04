import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import *
from Networks.DNN import PINN
from Networks.IA_DNN import Attention_PINN
from Networks.RES_DNN import RES_PINN
from Networks.RES_IA_PINN import RES_IA_PINN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Current device:", device)
torch.manual_seed(42)

# 创建数据集
x_train = np.linspace(-2 * np.pi, 2 * np.pi, 1000).reshape(-1, 1)
y_train = np.sin(x_train)

# 转换为 Tensor
x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

layer = [1] + [50] * 3 + [1]

criterion = nn.MSELoss()
# 训练模型
num_epochs = 10000

for mode in ['PINN', 'IA_PINN', 'RES_PINN', 'RES_IA_PINN']:
    print("Training {}:".format(mode))
    torch.cuda.empty_cache()
    if mode == 'PINN':
        model = PINN(layer, non_negative=False).to(device)

    elif mode == 'IA_PINN':
        model = Attention_PINN(layer, non_negative=False).to(device)

    elif mode == 'RES_PINN':
        model = RES_PINN(layer, non_negative=False).to(device)

    elif mode == 'RES_IA_PINN':
        model = RES_IA_PINN(layer, non_negative=False).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs,
                                                     eta_min=1e-5)

    loss_history = []
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        optimizer.zero_grad()
        output = model(x_train_tensor)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Save the loss for plotting
        loss_history.append(loss.item())

        if (epoch + 1) % 500 == 0:
            pbar.set_postfix(loss='{0:.3e}'.format(loss.item()))

    # 测试模型
    # 设置模型为评估模式
    model.eval()
    with torch.no_grad():
        y_pred = model(x_train_tensor).cpu().numpy()
        y_true_tensor = torch.FloatTensor(y_train)
        y_pred_tensor = torch.FloatTensor(y_pred)
        relative_error = torch.norm(y_true_tensor - y_pred_tensor) / torch.norm(y_true_tensor)
        print(f'Relative Error: {relative_error.item():.4f}')

    # Plotting the results and loss curve
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the fitted curve
    axs[0].plot(x_train, y_train, label='True sin(x)', color='blue')
    axs[0].plot(x_train, y_pred, label=f'Fitted with {mode}', color='red', linestyle='--')
    axs[0].legend()
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('sin(x)')
    axs[0].set_title(f'Sine Function Fitting using {mode}')
    axs[0].grid(True)

    # Plot the loss curve
    axs[1].plot(range(num_epochs), loss_history, color='green')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].set_title(f'Loss Curve for {mode}')
    axs[1].set_yscale('log')
    axs[1].grid(True)

    # Save and close the figure
    plt.savefig(f'{mode}_results.png')
    plt.close(fig)
