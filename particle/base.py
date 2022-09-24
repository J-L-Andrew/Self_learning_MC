import numpy as np
import torch
import random

from copy import deepcopy
from pytorch3d import transforms


class Particle(object):
    def __init__(self):
        self.name = ''
        # spatial dimensions
        self.dim = None

        self.centroid = None
        self.centroid_L = None
        
        self.orientation = None # quaternion
        
        # Retain policy
        self.centroid_old = None # for cell deform
        self.centroid_L_old = None # for particle move
        self.orientation_old = None

        # color
        self.color = None

    def scaled_centroid(self, lattice):
        """
        Convert absolute centroid to the one in scaled coordinate frame.
        # Note: lattice = [v1, v2, v3] (Column-based Storage)
        """
        temp = np.linalg.pinv(lattice)
        new_pos = np.matmul(temp, self.centroid.T).T
        return new_pos
    
    def update_centroid(self, lattice):
        """
        Collective rearrangement.
        """
        self.centroid = np.matmul(lattice, self.centroid_L.T).T

    def periodic_image(self, vector):
        """
        Translate the target particle by the vector.
        """
        image = deepcopy(self)
        image.centroid += vector
        return image
    
    def periodic_check(self, lattice):
        """
        Check whether the centroid is in the unit cell (legal),
        otherwise return its legal image.
        # Note: lattice = [v1, v2, v3] (Column-based Storage)
        """
        scaled_pos = self.scaled_centroid(lattice)

        for i in range(self.dim):
            while (scaled_pos[i] >= 1):
                scaled_pos[i] -= 1
            while (scaled_pos[i] < 0):
                scaled_pos[i] += 1
  
        self.centroid = np.matmul(lattice, scaled_pos.T).T
    
    #----------------------------------------------------------
    # Particle Movement
    #----------------------------------------------------------
    
    def rotate(self, vector, angle):
        re = vector * np.sin(angle)
        im = np.array([np.cos(angle)])
        qua = torch.as_tensor(np.concatenate([im] + [re]), dtype=torch.double)
        
        self.orientation = transforms.quaternion_multiply(qua, self.orientation)
    
    
    def randomize(self, lattice):
        """
        Randomly generate particles.
        # Note: lattice = [v1, v2, v3] (Column-based Storage)
        """
        # random centroid
        self.centroid_L = np.random.rand(self.dim)
        self.centroid = np.matmul(lattice, self.centroid_L.T).T
        
        # random orientation
        self.orientation = transforms.random_quaternions(n=1, dtype=torch.double)[0]
        
    def random_translate(self, lattice, mod):
        self.centroid_L_old = self.centroid_L
        
        vec = np.random.rand(self.dim) - 0.5
        self.centroid_L += vec * mod
        for i in range(self.dim):
            while (self.centroid_L[i] >= 1):
                self.centroid_L[i] -= 1
            while (self.centroid_L[i] < 0):
                self.centroid_L[i] += 1
        
        self.update_centroid(lattice)
    
    def retain_translate(self, lattice):
        self.centroid_L = self.centroid_L_old
        self.update_centroid(lattice)
        
    def random_rotate(self, mod):
        """
        https://www.zhihu.com/question/26579222/answer/48625038
        """
        self.orientation_old = self.orientation
        angle = mod*random.random()*2.*np.pi
        
        u = random.random()
        v = random.random()
        theta = 2.*np.pi*u
        phi = np.arccos(2.*v-1.)
        vector = np.array([np.sin(theta)*np.sin(phi), np.cos(theta)*np.sin(phi), np.cos(phi)])
        
        self.rotate(vector, angle)
        
    def retain_rotate(self):
        self.orientation = self.orientation_old
        
        
        
        
            
    
    
        
        
        


    def translate(self, vector, mod):
        self.centroid += vector * mod


    
