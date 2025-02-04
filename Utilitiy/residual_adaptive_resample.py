import torch
import numpy as np
from Main_Diffusion_Reaction.pde_system import compute_pde,compute_bc, compute_ic

def compute_point_wise_loss(model, coords, loss_type='pde'):
    """Caculate single point loss
    Args:
        model: Model
        coords: 
        loss_type: ('pde', 'left_bc', 'right_bc', 'initial')
    Returns:
        point_losses: 
    """
    coords.requires_grad_(True)
    point_losses = np.zeros(len(coords))

    # model.eval() #swtich to evaluation mode
    
    for i,subtensor in enumerate(coords):
        #use unsqueeze to make the tensor 2D
        subtensor = subtensor.unsqueeze(0)
    
        if loss_type == 'pde':
            loss_eq1, loss_eq2, loss_eq3, loss_eq4, loss_eq5, loss_eq6, D_star = compute_pde(model, subtensor)
            loss =  loss_eq4 + loss_eq5 

        elif loss_type == 'left_bc':
            loss = compute_bc(model, subtensor)
        
        elif loss_type == 'right_bc':
            loss = compute_bc(model, subtensor)
        
        elif loss_type == 'initial':
            loss_ic1,loss_ic2 = compute_ic(model, subtensor)
            loss = loss_ic1 + loss_ic2
        
        point_losses[i] = loss.item()
    
    
    return point_losses

def select_high_loss_points(model, resample_points, n_select=200, loss_type='pde'):
    """Select high loss points
    Args:
        model: /
        resample_points: points
        n_select: number of selected points
        loss_type: /
    Returns:
        selected_points: /
    """
    # caculation
    point_losses = compute_point_wise_loss(model, resample_points, loss_type)
        
    high_loss_indices = np.argsort(point_losses)[-n_select:]
    
    selected_points = resample_points[high_loss_indices]
    
    return selected_points