import numpy as np
from utils import surface_area

class Cell(object):
    def __init__(self, dim):
        self.dim = dim
        # origin of lattice (located in the origin by default)
        self.origin = None
        
        self.lattice = None
        
        # color
        self.color = None

    
    @property
    def volume(self):
        if self.dim == 3:
            v = np.dot(np.cross(self.lattice[0], self.lattice[1]), self.lattice[2])
        elif self.dim == 2:
            v = np.cross(self.lattice[0], self.lattice[1])
        return np.fabs(v)
    
    @property
    def distortion(self):
        """
        Measure the distortion of the simulation cell, for 3D now only 
        """
        norm = 0.
        for i in range(3):
            norm += np.linalg.norm(self.state.lattice[i])
    
        fun = norm * surface_area(self.state.lattice) / self.volume / 18.
        return fun

    def new_combination(self):
        """
        Replace the cell by an equivalent set of basis vectors, 
        which are shorter and more orthogonal
        """
        new_lattice = self.state.lattice.copy()
        is_terminated = True
        for i in range(3):
            for j in range(3):
                if (j == i): continue

                for k in range(2):
                    lattice = self.state.lattice.copy()
                    lattice[i] = self.state.lattice[i] + (-1)**k * self.state.lattice[j]

                    if surface_area(lattice) < surface_area(new_lattice):
                        is_terminated = False
                        new_lattice = lattice.copy()
        
        self.state.lattice = new_lattice
        return is_terminated

    def Diji_reduction(self):
        """
        Repeat the above procedure.
        """
        if self.distortion > 1.5:
            terminal = False
            iter = 0
            while (not terminal):
                terminal = self.new_combination()
                iter += 1

                # it is prudent to impose a cutoff at roughly 10 iterations
                if (iter > 10): break
      
    def LLL_reduction(self):
        pass

    
    
    
    
