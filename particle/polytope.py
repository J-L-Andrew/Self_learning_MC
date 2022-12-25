import numpy as np
from particle.base import Particle

# number of vertice & volume 
# radius of outscribed sphere & radius of inscribed sphere
hedron = {}
hedron["tetra"] = [4, np.sqrt(2)/12, np.sqrt(6)/4, np.sqrt(6)/12]
hedron["hexa"] = [8, 1, np.sqrt(2)/2, np.sqrt(6)/6]
hedron["octa"] = [6, np.sqrt(2)/3, np.sqrt(2)/2, np.sqrt(6)/6]


# reference vertices
vertice = {}
vertice["tetra"] = np.array([[1., 1., 1.],
                             [1., -1., -1.],
                             [-1., -1., 1.],
                             [-1., 1., -1.]]) # sidelength = 2*sqrt(2)
vertice["octa"] = np.array([[1., 0., 0.],
                             [0., 1., 0.],
                             [0., 0., 1.],
                             [-1., 0., 0.],
                             [0., -1., 0.],
                             [0., 0., -1.]]) # sidelength = sqrt(2)


class Polytope(Particle):
    def __init__(self, length, type = "tetra"):
        super().__init__()
        self.dim = 3
        
        self.length = length # edge length
        
        self.rot_mat = np.diag(np.ones(self.dim))
        
        self.type = type
        self.n_vertice = hedron[type][0]
        
        self.vertice = vertice[type]

    @property
    def volume(self):
        vol = hedron[self.type][1]*self.length**3
        return vol

    @property
    def inscribed_d(self):
        """
        diameter of the outscribed sphere
        """
        return 2.*hedron[self.type][3]*self.length
    
    @property
    def outscribed_d(self):
        """
        diameter of the outscribed sphere
        """
        # Hölder Inequality
        return 2.*hedron[self.type][2]*self.length