from Utilitiy.gradients_func import *
import numpy as np
import torch

def compute_pde(model, coords, anneal_time):
    D_A = 360  #6nm^2/s 360nm^2/min
    L, h = 600,100

    k11,k12,k21 = 0.0,0.0,0.0
    N_SA,N_SB,N_SC,N_SAB,N_SABB,N_SAAB = 90,48,90,90,90,90

    coords = coords.clone().detach().requires_grad_(True)

    u = model(coords)

    c_a,c_bc,c_c,c_ab,c_abb,c_aab = u[:,0], u[:,1], u[:,2], u[:,3], u[:,4], u[:,5]

    ##Derivative Terms
    gradients_t = [gradients(c, coords)[:,1] for c in [c_a,c_bc,c_c,c_ab,c_abb,c_aab]]
    dca_dt,dcbc_dt,dcc_dt,dcab_dt,dcabb_dt,dcaab_dt = gradients_t

    gradients_x = [gradients(c, coords)[:,0] for c in [c_a,c_bc,c_c]]  #c_ab,c_abb,c_aab NOT USED
    dca_dx, dcbc_dx, dcc_dx = gradients_x #,dcab_dx, dcabb_dx, dcaab_dx also not used

    ##Construcion of compouned Diffusivity
    # sig_numerator = c_bc*48 + c_c*90 + c_ab*90 + c_abb*90 + c_aab*90
    # sig_denominator = c_a*90 + c_bc*48 + c_c*90 + c_ab*90 + c_abb*90 + c_aab*90

    sig_numerator = c_bc + c_c + c_ab + c_abb + c_aab
    sig_denominator = c_a + c_bc + c_c + c_ab + c_abb + c_aab

    # sig_numerator = c_bc + c_c + c_ab + c_abb + c_aab
    # sig_denominator = c_a + c_bc + c_c + c_ab + c_abb + c_aab
    # partial_part = dcbc_dx + dcc_dx + dcab_dx + dcabb_dx + dcaab_dx    

    sig_denominator = torch.where((sig_denominator == 0), 1e-10, sig_denominator)
    
    sig_D = sig_numerator /sig_denominator
    D_A = D_A* (anneal_time/(L**2))  #With Non-Dimensionalization
    D_star = D_A * sig_D

    #Residuals of Governing Equations with Non-Dimensionalization
    eq_1 =  dca_dt - gradients(D_star*dca_dx,coords)[:,0]  +\
            k11 *(N_SA * N_SB* anneal_time) / N_SA* c_a * c_bc + k21*(N_SA * N_SAB *anneal_time)/N_SA * c_a * c_ab
    
    eq_2 = dcbc_dt -gradients(D_star*dcbc_dx,coords)[:,0]+\
            k11*(N_SA * N_SB*anneal_time)/ N_SB * c_a * c_bc + k12*(N_SAB * N_SB*anneal_time) /N_SB * c_ab * c_bc
        
    eq_3 = dcc_dt - gradients(D_star*dcc_dx,coords)[:,0] -\
            k11*(N_SA* N_SB *anneal_time) /N_SC * c_a * c_bc - k12 *(N_SAB * N_SB*anneal_time) / N_SC * c_ab * c_bc    
            
    eq_4 = dcab_dt - k11*(N_SA * N_SB *anneal_time) / N_SAB *c_a *c_bc + k21 *(N_SA * N_SAB *anneal_time)/N_SAB* c_a * c_ab +\
            k12*(N_SAB * N_SB*anneal_time) /N_SAB * c_ab * c_bc
        
    eq_5 = dcabb_dt - k12 *(N_SAB * N_SB *anneal_time) / N_SABB * c_ab * c_bc
        
    eq_6 = dcaab_dt - k21 *(N_SA * N_SAB * anneal_time)/ N_SAAB * c_a * c_ab     

    loss_eq1 = torch.mean(eq_1**2)
    loss_eq2 = torch.mean(eq_2**2)
    loss_eq3 = torch.mean(eq_3**2)
    loss_eq4 = torch.mean(eq_4**2)
    loss_eq5 = torch.mean(eq_5**2)
    loss_eq6 = torch.mean(eq_6**2)

    # loss = torch.mean(eq_1**2) + torch.mean(eq_2**2) + torch.mean(eq_3**2) +\
    #        torch.mean(eq_4**2) + torch.mean(eq_5**2) + torch.mean(eq_6**2) 
    
    return loss_eq1, loss_eq2, loss_eq3, loss_eq4, loss_eq5, loss_eq6, D_star

"""Since Left and Right Boundary Conditions are same, So we only need to call it twice and pass in different coords"""
def compute_bc(model,coords):
    coords = coords.clone().detach().requires_grad_(True)
    u = model(coords)
    bc_ca,bc_cbc,bc_cc = u[:,0], u[:,1], u[:,2]

    ##Neumann Boundary Condition
    # bc_ca_dx = gradients(bc_ca, coords)[:, 0]
    # bc_cbc_dx = gradients(bc_cbc, coords)[:, 0]
    # bc_cc_dx = gradients(bc_cc, coords)[:, 0]
    # bc_cab_dx = gradients(bc_cab, coords)[:, 0]
    # bc_cabb_dx = gradients(bc_cabb, coords)[:, 0]
    # bc_caab_dx = gradients(bc_caab, coords)[:, 0]

    gradients_x = [gradients(c, coords)[:,0] for c in [bc_ca,bc_cbc,bc_cc]]
    bc_ca_dx,bc_cbc_dx,bc_cc_dx = gradients_x


    ##Gradient Enhance
    # bc_ca_dxx = gradients(bc_ca_dx, coords)[:, 0]
    # bc_cbc_dxx = gradients(bc_cbc_dx, coords)[:, 0]
    # bc_cc_dxx = gradients(bc_cc_dx, coords)[:, 0]    

    loss = torch.mean(bc_ca_dx**2) + torch.mean(bc_cbc_dx**2) + torch.mean(bc_cc_dx**2)
    # loss += 0.1 * (torch.mean(bc_ca_dxx**2) + torch.mean(bc_cbc_dxx**2) + torch.mean(bc_cc_dxx**2))  # 可调权重
    return loss

def compute_ic(model,coords):
    L, h = 600.0,100.0
    N_SA,N_SB = 90.0, 48.0
    
    coords = coords.clone().detach().requires_grad_(True)

    mask_h = (coords[:, 0] <= h/L) & (coords[:, 0] >= 0.0)
    mask_L = (coords[:, 0] >= h/L) & (coords[:, 0] <= L/L)

    tensor_h = coords[mask_h].requires_grad_(True)
    tensor_L = coords[mask_L].requires_grad_(True)

    ca_in_h,cbc_in_h = model(tensor_h)[:,0],model(tensor_h)[:,1]
    ca_in_L,cbc_in_L = model(tensor_L)[:,0],model(tensor_L)[:,1]
    
    cc_init,cab_init,cabb_init,caab_init = model(coords)[:,2],model(coords)[:,3],model(coords)[:,4],model(coords)[:,5]

    loss_ic1 = torch.mean((ca_in_h - N_SA/N_SA)**2) + torch.mean((cbc_in_L - N_SB/N_SB)**2) + torch.mean(ca_in_L**2) + torch.mean(cbc_in_h**2)
    
    loss_ic2 = torch.mean(cc_init **2) + torch.mean(cab_init **2) + torch.mean(cabb_init **2) + torch.mean(caab_init **2)

    return loss_ic1, loss_ic2

def relative_error(data_net,data_true):
    N_SA,N_SB,N_SC,N_SAB,N_SABB,N_SAAB = 90.0,48.0,90.0,90.0,90.0,90.0
    component = ['Ni', 'SiC', 'C', 'NiSi', 'NiSi2', 'NiSi2']

    result = data_net
    result_fdm = data_true

    c_a,c_bc,c_c,c_ab,c_abb,c_aab = result[:,0],result[:,1],result[:,2],result[:,3],result[:,4],result[:,5]
    c_a,c_bc,c_c,c_ab,c_abb,c_aab = [np.where((tensor < 0),np.array(0),tensor) for tensor in [c_a,c_bc,c_c,c_ab,c_abb,c_aab]]

    c_a,c_bc,c_c,c_ab,c_abb,c_aab = c_a*N_SA, c_bc*N_SB, c_c*N_SC, c_ab*N_SAB, c_abb*N_SABB, c_aab*0
    net_date = [concen.reshape(101,101) for concen in [c_a,c_bc,c_c,c_ab,c_abb,c_aab]]

    fdm_date = [fdm.iloc[1:,1:].iloc[::10,::10].values for fdm in result_fdm]

    R2_Error, Inf_Error = {},{}
    r2_log, inf_log = [],[]

    for i,(net, fdm) in enumerate(zip(net_date,fdm_date)):
        if i == 5:
            continue
        relative_R2 = np.linalg.norm(net - fdm, 2)/np.linalg.norm(fdm, 2)
        relative_Inf = np.linalg.norm(net - fdm, np.Inf)/np.linalg.norm(fdm, np.Inf)

        r2_log.append(relative_R2)
        inf_log.append(relative_Inf)

        
        R2_Error[component[i]] = round(relative_R2, 3)
        Inf_Error[component[i]] = round(relative_Inf, 3)
    print(f'R^2 Error: {R2_Error}\nInf Error: {Inf_Error}')
    print(f'Mean Error: {np.mean(r2_log), np.mean(inf_log)}')