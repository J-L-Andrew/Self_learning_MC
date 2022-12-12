import numpy as np
from particle.base import Particle

class Sphere(Particle):
    def __init__(self, radius):
        super().__init__()
        self.dim = 3

        # shape parameters
        self.radius = radius

    @property
    def volume(self):
        return 4./3.*np.pi*self.radius**3
    
    @property
    def inscribed_d(self):
        """
        diameter of the inscribed sphere
        """
        return 2.*self.radius

    @property
    def outscribed_d(self):
        """
        diameter of the outscribed sphere
        """
        return 2.*self.radius
