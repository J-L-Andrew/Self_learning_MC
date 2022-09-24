import numpy as np
from particle.base import Particle
from pytorch3d import transforms

class Ellipsoid(Particle):
    def __init__(self, convention="Donev"):
        """
        Here we note some different notations:
        # Donev: alpha: beta: 1 (beta=1)
        # Jin: S2M: 1.0: 1./S2M
        """
        super().__init__()
        self.dim = 3
        
        self.convention = convention

        # shape parameters (alpha: alpha^beta : 1)
        self.alpha = None
        self.beta = None
        
        # for self-dual ellipsoids
        self.S2M = None

    @property
    def semi_axis(self):
        """
        Here we note some different notations:
        # Donev: alpha: beta: 1 (beta=1)
        """
        if (self.convention == "Donev"):
            return np.array([self.alpha, self.alpha**self.beta, 1.])
        elif (self.convention == "Jin"):
            return np.array([self.S2M, 1., 1./self.S2M])

    @property
    def volume(self):
        return 4./3.*np.pi*np.cumprod(self.semi_axis)[2]

    @property
    def inscribed_d(self):
        """
        diameter of the outscribed sphere
        """
        return 2.*np.min(self.semi_axis)

    @property
    def outscribed_d(self):
        """
        diameter of the outscribed sphere
        """
        return 2.*np.max(self.semi_axis)

    @property
    def char_mat(self):
        """
        characteristic ellipsoid matrix describing the 
        shape and orientation of the ellipsoid
        """
        # note that: rot_mat = Rz*Ry*Rx = Q^T
        rot_mat = transforms.quaternion_to_matrix(self.orientation)

        # O: a diagonal matrix containing the major semi-axes
        O = np.diag(self.semi_axis)
        temp_mat = np.linalg.pinv(O)**2

        matrix = np.matmul(rot_mat, temp_mat)
        matrix = np.matmul(matrix, np.transpose(rot_mat))

        return matrix
