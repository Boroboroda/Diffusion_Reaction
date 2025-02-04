import os
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
import numpy as np
import pandas as pd

def plot_loss_curve(layer_mode:str,loss_total:list,loss_pde: list, loss_bc: list, loss_ic: list):
    fig, ax = plt.subplots(figsize=(8, 6))

    # 绘制四条损失曲线
    ax.plot(loss_total, label="Total Loss", color="purple", linestyle="-")
    ax.plot(loss_pde, label="PDE Loss", color="blue", linestyle="--")
    ax.plot(loss_bc, label="Boundary Condition Loss", color="orange", linestyle="-.")
    ax.plot(loss_ic, label="Initial Condition Loss", color="green", linestyle=":")

    # 添加图例
    ax.legend(loc="upper right")

    # 添加标题和标签
    ax.set_title("Loss Curve for PDE, Boundary Condition, and Initial Condition")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")

    save_path = f'./Results/{layer_mode}/'
    file_name = f'{layer_mode}_loss.png'

    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 显示图形
    plt.grid(True)
    plt.savefig(os.path.join(save_path, file_name))
    plt.close()


def plot_relative_concen(layer_mode, data_set, data_save:bool,annealing_time):
    N_SA,N_SB,N_SC,N_SAB,N_SABB,N_SAAB = 90,48,90,90,90,90

    result = data_set
    c_a,c_bc,c_c,c_ab,c_abb,c_aab = result[:,0],result[:,1],result[:,2],result[:,3],result[:,4],result[:,5]
    c_a,c_bc,c_c,c_ab,c_abb,c_aab = [np.where((tensor < 0),np.array(0),tensor) for tensor in [c_a,c_bc,c_c,c_ab,c_abb,c_aab]]

    c_a,c_bc,c_c,c_ab,c_abb,c_aab = c_a*N_SA, c_bc*N_SB, c_c*N_SC, c_ab*N_SAB, c_abb*N_SABB, c_aab*0

    c_a,c_bc,c_c,c_ab,c_abb,c_aab = [concen.reshape(101,101) for concen in [c_a,c_bc,c_c,c_ab,c_abb,c_aab]]

    sum = c_a[:,-1] + c_bc[:,-1] + c_c[:,-1] + c_ab[:,-1] + c_abb[:,-1] + c_aab[:,-1]

    re_a = c_a[:,-1] / sum * 100
    re_bc = c_bc[:,-1] / sum * 100 /2
    re_c = c_c[:,-1] / sum * 100
    re_ab = c_ab[:,-1] / sum * 100
    re_abb = c_abb[:,-1] / sum * 100
    re_aab = c_aab[:,-1] / sum * 100

    component_x = np.linspace(0, 600, 101)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(component_x, re_a, label=r'Concentration of Ni', color=(97/255,108/255,140/255))
    ax.plot(component_x, re_bc, label=r'Concentration of SiC', color=(86/140,140/255,135/255))
    ax.plot(component_x, re_c, label=r'Concentration of C', color=(178/255,213/255,155/255))
    ax.plot(component_x, re_ab, label=r'Concentration of NiSi', color=(242/255,222/255,121/255))
    ax.plot(component_x, re_abb, label=r'Concentration of NiSi2', color=(217/255,95/255,24/255))
    ax.set_title(f'Relative Concentration,time = {annealing_time}')
    ax.set_xlabel('x')
    ax.set_ylabel('Concentration, %')
    ax.set_xlim([0, 600])
    ax.set_ylim([0, 100])
    ax.legend()
    plt.tight_layout()
    save_path = f'./Results/{layer_mode}/'
    file_name = f'{layer_mode}_relative concentration.png'

    if not os.path.exists(save_path):
        os.makedirs(save_path)


    plt.grid(True)
    plt.savefig(os.path.join(save_path, file_name))
    plt.close()

    if data_save:
        data_dict = {
            "c_a": c_a,
            "c_bc": c_bc,
            "c_c": c_c,
            "c_ab": c_ab,
            "c_abb": c_abb,
            "c_aab": c_aab
            }
        for name, concentration in data_dict.items():
            df = pd.DataFrame(concentration)
            # print(df)
            # df = df.transpose() 

            new_df = pd.DataFrame(np.nan, index=range(102), columns=range(102))
            new_df.iloc[0, 1:] = np.linspace(0,60,101)
            new_df.iloc[1:,0] = np.linspace(0,600,101)

            new_df.iloc[1:, 1:] = df.values

            folder_path = f'./Results/{layer_mode}/csv/'
            os.makedirs(folder_path, exist_ok=True)
            new_df.to_csv(os.path.join(folder_path, f"{name}.csv"), index=False, header=False)

def plot_3d_concen(layer_mode:str,data_net,data_true):
    N_SA,N_SB,N_SC,N_SAB,N_SABB,N_SAAB = 90,48,90,90,90,90
    component = ['Ni', 'SiC', 'C', 'NiSi', 'NiSi2', 'NiSi2']

    result = data_net
    result_fdm = data_true

    c_a,c_bc,c_c,c_ab,c_abb,c_aab = result[:,0],result[:,1],result[:,2],result[:,3],result[:,4],result[:,5]
    c_a,c_bc,c_c,c_ab,c_abb,c_aab = [np.where((tensor < 0),np.array(0),tensor) for tensor in [c_a,c_bc,c_c,c_ab,c_abb,c_aab]]

    c_a,c_bc,c_c,c_ab,c_abb,c_aab = c_a*N_SA, c_bc*N_SB, c_c*N_SC, c_ab*N_SAB, c_abb*N_SABB, c_aab*0
    net_date = [concen.reshape(101,101) for concen in [c_a,c_bc,c_c,c_ab,c_abb,c_aab]]

    fdm_date = [fdm.iloc[1:,1:].iloc[::10,::10].values for fdm in result_fdm]


    x = np.linspace(0, 600, 101)
    t = np.linspace(0, 60, 101)
    T, X = np.meshgrid(t, x)

    R2_Error, Inf_Error = {},{}
    for i,(net, fdm) in enumerate(zip(net_date,fdm_date)):
        if i == 5:
            continue

        abs_error = np.abs(net - fdm)

        relative_R2 = np.linalg.norm(net - fdm, 2)/np.linalg.norm(fdm, 2)
        relative_Inf = np.linalg.norm(net - fdm, np.Inf)/np.linalg.norm(fdm, np.Inf)

        fig = plt.figure(figsize=(24, 6))
        gs = gridspec.GridSpec(1,3,width_ratios=[1, 1, 1])  # 最右边的列较窄

        ax_net = fig.add_subplot(gs[0], projection='3d')
        surf_net = ax_net.plot_surface(X, T, net, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0003, antialiased=True)
        ax_net.set_title(f'Net: {component[i]}')
        ax_net.set_xlabel('Space')
        ax_net.set_ylabel('Time')
        ax_net.set_zlabel('Value')
        ax_net.set_zlim([0, 100])

        ax_fdm = fig.add_subplot(gs[1], projection='3d')
        surf_fdm = ax_fdm.plot_surface(X, T, fdm, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0003, antialiased=True)
        ax_fdm.set_title(f'FDM: {component[i]}')
        ax_fdm.set_xlabel('Space')
        ax_fdm.set_ylabel('Time')
        ax_fdm.set_zlabel('Value')
        ax_fdm.set_zlim([0, 100])

        ax_abs = fig.add_subplot(gs[2])
        cax_abs = ax_abs.imshow(abs_error, extent=(t.min(), t.max(), x.max(), x.min()), cmap=cm.RdYlBu_r, aspect='auto')
        ax_abs.set_title(f'Error : {component[i]},$R^2$ = {relative_R2:.4f}, Inf = {relative_Inf:.4f}')
        ax_abs.set_xlabel('Time')
        ax_abs.set_ylabel('Space')

        fig.colorbar(surf_net, ax=ax_net, shrink=0.6, aspect=10)
        fig.colorbar(surf_fdm, ax=ax_fdm, shrink=0.6, aspect=10)
        fig.colorbar(cax_abs, ax=ax_abs, shrink=0.6, aspect=10)

            # 调整布局
        plt.tight_layout()
        # plt.savefig(f'H:/paper_Graphics/Diffusion_Reaction_Results/{component[i]}')  # 保存图形为 PNG 文件
        save_path = f'./Results/{layer_mode}/Concentrations'
        file_name = f'{layer_mode}_{component[i]}_Concentration.png'

        if not os.path.exists(save_path):
            os.makedirs(save_path)


        plt.grid(True)
        plt.savefig(os.path.join(save_path, file_name))
        plt.close()

        R2_Error[component[i]] = round(relative_R2, 3)
        Inf_Error[component[i]] = round(relative_Inf, 3)

    # print(f'R2 Error: {R2_Error}\nInf Error: {Inf_Error}')