"""
@Project: Diffusion Reaction PDE
@File: DataGenerator.py
@Author: Cheng
"""
import torch
import numpy as np
import random
from scipy.stats import qmc
from torch.utils.data import Dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataGenerator(Dataset):
    """Use the to generate samples
    Args:
        geom: Geometry of the problem
        time: Time interval of the problem
        name: Name of the sample points
        seed: Seed for the generator
    """
    def __init__(self, geom, time, name = None, seed = 42):
        self.geom = geom
        self.time = time
        self.name = name
        self.seed = seed

        # Set seed on initialization
        self.set_seed(self.seed)


    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def grid_generator(self, N, points_type = 'domain', nondim = False):
        """
        Args:
            grid: (x,t) -> x: length of x; t: length of t
            nondim(bool): If true, then return non-dimensionalized coordinates
            points_type(str): Type of the points
                ['domain', 'left_bc', 'right_bc', 'initial']
        
        Returns:
            (torch.tensor)coords: Coordinates in shape [N, 2]
            Grid sampled points
        """
        dict_type ={
            'domain':[[self.geom[0],self.time[0]],[self.geom[1],self.time[1]]],
            'left_bc':[[self.geom[0],self.time[0]],[self.geom[0],self.time[1]]],
            'right_bc':[[self.geom[1],self.time[0]],[self.geom[1],self.time[1]]],
            'initial':[[self.geom[0],self.time[0]],[self.geom[1],self.time[0]]],
        }
        grid_size = np.ceil(np.sqrt(N)).astype(int)
        
        if points_type == 'domain':
            x = np.linspace(dict_type[points_type][0][0], dict_type[points_type][1][0], grid_size)
            t = np.linspace(dict_type[points_type][0][1], dict_type[points_type][1][1], grid_size)
        elif points_type == 'left_bc':
            x = np.linspace(dict_type[points_type][0][0], dict_type[points_type][1][0], grid_size)
            t = np.linspace(dict_type[points_type][0][1], dict_type[points_type][1][1], grid_size)
        elif points_type == 'right_bc':
            x = np.linspace(dict_type[points_type][0][0], dict_type[points_type][1][0], grid_size)
            t = np.linspace(dict_type[points_type][0][1], dict_type[points_type][1][1], grid_size)
        elif points_type == 'initial':
            x = np.linspace(dict_type[points_type][0][0], dict_type[points_type][1][0], grid_size)
            t = np.linspace(dict_type[points_type][0][1], dict_type[points_type][1][1], grid_size)
        else:
            raise ValueError('Type must be one of {}'.format(dict_type.keys()))

        #Grid sample
        X,T = np.meshgrid(x,t,indexing='ij')
        output = np.stack([X.flatten(),T.flatten()],axis=1)
        
        if nondim:
            output = output / np.array([self.geom[1], self.time[1]])
        return torch.tensor(output, requires_grad= True).float().to(device)
    
    def random_generator(self, N = 10000, points_type = 'domain', nondim = False):
        """
        Args:
            N(int): Number of sample points
            nondim(bool): If true, then return non-dimensionalized coordinates
            points_type(str): Type of the points
                ['domain', 'left_bc', 'right_bc', 'initial']
        
        Returns:
            (torch.tensor)coords: Coordinates in shape [N, 2]
            Uniformly distributed random numbers
        """
        dict_type ={
            'domain':[[self.geom[0],self.time[0]],[self.geom[1],self.time[1]]],
            'left_bc':[[self.geom[0],self.time[0]],[self.geom[0],self.time[1]]],
            'right_bc':[[self.geom[1],self.time[0]],[self.geom[1],self.time[1]]],
            'initial':[[self.geom[0],self.time[0]],[self.geom[1],self.time[0]]],
        }

        if points_type == 'domain':
            output = np.array(dict_type[points_type])[0:1,:] + (np.array(dict_type[points_type])[1:2,:] - np.array(dict_type[points_type])[0:1,:]) * np.random.rand(N,2)
        elif points_type == 'left_bc':
            output = np.array(dict_type[points_type])[0:1,:] + (np.array(dict_type[points_type])[1:2,:] - np.array(dict_type[points_type])[0:1,:]) * np.random.rand(N,2)
        elif points_type == 'right_bc':
            output = np.array(dict_type[points_type])[0:1,:] + (np.array(dict_type[points_type])[1:2,:] - np.array(dict_type[points_type])[0:1,:]) * np.random.rand(N,2)
        elif points_type == 'initial':
            output = np.array(dict_type[points_type])[0:1,:] + (np.array(dict_type[points_type])[1:2,:] - np.array(dict_type[points_type])[0:1,:]) * np.random.rand(N,2)                    
        else:
            raise ValueError('Type must be one of {}'.format(dict_type.keys()))
        
        if nondim:
            output = output / np.array([self.geom[1], self.time[1]])
        
        return torch.tensor(output, requires_grad= True).float().to(device)
    
    def LHS_generator(self, N = 10000, points_type = 'domain', nondim = False):
        """
        Args:
            N(int): Number of sample points
            nondim(bool): If true, then return non-dimensionalized coordinates
            points_type(str): Type of the points
                ['domain', 'left_bc', 'right_bc', 'initial']
        
        Returns:
            (torch.tensor)coords: Coordinates in shape [N, 2]
            Latin Hypercube Sampled points
        """
        dict_type ={
            'domain':[[self.geom[0],self.time[0]],[self.geom[1],self.time[1]]],
            'left_bc':[[self.geom[0],self.time[0]],[self.geom[0],self.time[1]]],
            'right_bc':[[self.geom[1],self.time[0]],[self.geom[1],self.time[1]]],
            'initial':[[self.geom[0],self.time[0]],[self.geom[1],self.time[0]]],
        }
        lhs_sampler = qmc.LatinHypercube(d=1, seed = self.seed)

        if points_type == 'domain':
            x = lhs_sampler.random(N)
            x = qmc.scale(x, l_bounds = dict_type[points_type][0][0], u_bounds = dict_type[points_type][1][0])
            t = lhs_sampler.random(N)
            t = qmc.scale(t, l_bounds = dict_type[points_type][0][1], u_bounds = dict_type[points_type][1][1])

        elif points_type == 'left_bc':
            t = lhs_sampler.random(N)
            t = qmc.scale(t, l_bounds = dict_type[points_type][0][1], u_bounds = dict_type[points_type][1][1])
            x = np.ones(N) * dict_type[points_type][0][0]

        elif points_type == 'right_bc':
            t = lhs_sampler.random(N)
            t = qmc.scale(t, l_bounds = dict_type[points_type][0][1], u_bounds = dict_type[points_type][1][1])
            x = np.ones(N)*dict_type[points_type][1][0]

        elif points_type == 'initial':
            x = lhs_sampler.random(N)
            x = qmc.scale(x, l_bounds = dict_type[points_type][0][0], u_bounds = dict_type[points_type][1][0])
            t = np.zeros(N)          
        else:
            raise ValueError('Type must be one of {}'.format(dict_type.keys()))
        
        output = np.stack([x.flatten(),t.flatten()],axis=1)
        if nondim:
            output = output / np.array([self.geom[1], self.time[1]])
        
        return torch.tensor(output, requires_grad= True).float().to(device)
    
    def LHS_local_enhance(self, N = 10000, points_type = 'domain', enhance_area = [[],[]], nondim = False):
        """
        Args:
            N(int): Number of sample points
            enhance_area(list): Area to be enhanced [[x1,x2],[t1,t2]]
            nondim(bool): If true, then return non-dimensionalized coordinates
            points_type(str): Type of the points
                ['domain', 'left_bc', 'right_bc', 'initial']
        
        Returns:
            (torch.tensor)coords: Coordinates in shape [N, 2]
            Latin Hypercube Sampled points
        """
        dict_type ={
            'domain':[[self.geom[0],self.time[0]],[self.geom[1],self.time[1]]],
            'left_bc':[[self.geom[0],self.time[0]],[self.geom[0],self.time[1]]],
            'right_bc':[[self.geom[1],self.time[0]],[self.geom[1],self.time[1]]],
            'initial':[[self.geom[0],self.time[0]],[self.geom[1],self.time[0]]],
        }
        lhs_sampler = qmc.LatinHypercube(d=1, seed = self.seed)

        if points_type == 'domain':
            x = lhs_sampler.random(N)
            x = qmc.scale(x, l_bounds = enhance_area[0][0], u_bounds = enhance_area[0][1])
            t = lhs_sampler.random(N)
            t = qmc.scale(t, l_bounds = enhance_area[1][0], u_bounds = enhance_area[1][1])
        elif points_type == 'left_bc':
            t = lhs_sampler.random(N)
            t = qmc.scale(t, l_bounds = enhance_area[1][0], u_bounds = enhance_area[1][1])
            x = np.ones(N) * dict_type[points_type][0][0]
        elif points_type == 'right_bc':
            t = lhs_sampler.random(N)
            t = qmc.scale(t, l_bounds = enhance_area[1][0], u_bounds = enhance_area[1][1])
            x = np.ones(N)*dict_type[points_type][1][0]
        elif points_type == 'initial':
            x = lhs_sampler.random(N)
            x = qmc.scale(x, l_bounds = enhance_area[0][0], u_bounds = enhance_area[0][1])
            t = np.zeros(N)     

        else:
            raise ValueError('We just enhance the domain area')

        output = np.stack([x.flatten(),t.flatten()],axis=1)

        if nondim:
            output = output / np.array([self.geom[1], self.time[1]])

        return torch.tensor(output, requires_grad= True).float().to(device)












