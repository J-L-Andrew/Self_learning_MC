import numpy as np
from base import Particle
from pytorch3d import transforms
from scipy import optimize

class SuperEllipsoid(Particle):
    def __init__(self, a, b, c, p):
        super().__init__()
        self.dim = 3

        # semi-axis length (a: b: c) and shape parameter (by default p1=p2=p)
        self.a = a
        self.b = b
        self.c = c
        self.p = p
        
        self.centroid_new = None
        self.rot_mat_new = None

    def support_func(self, u: np.array):
        """
        Adapated from:
        Yoav Kallus and Veit Elser Dense-packing crystal structures of physical tetrahedra
        """
        # u_ = np.matmul(self.rot_mat, u.T).T
        
        # https://baike.baidu.com/item/球坐标系/8315363
        r_u = np.linalg.norm(u)
        temp = np.sqrt(u[0]**2. + u[1]**2.)
        sin_cita = temp / r_u
        cos_cita = u[2] / r_u
        
        if (temp == 0.):
            sin_fai, cos_fai = 1., 0.
        else:
            sin_fai, cos_fai = u[1] / temp, u[0] / temp
        
        kx, ky, kz = np.fabs(sin_cita*cos_fai/self.a), np.fabs(sin_cita*sin_fai/self.b), np.fabs(cos_cita/self.c)
        r_new = (1./(kx**(2.*self.p) + ky**(2.*self.p) + kz**(2.*self.p)))**(0.5/self.p)
        
        y = r_new * r_u
        return y


    @property
    def semi_axis(self):
        return np.array([self.a, self.b, self.c])

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

def resolve_overlap(x: np.array, p1: SuperEllipsoid, p2: SuperEllipsoid):
    u, v1, v2 = x[0:3], x[3:6], x[6:9]
    temp = v1 - np.matmul(p1.rot_mat, u.T).T
    p1.rot_mat_new = p1.rot_mat + np.matmul(temp.T, u) / (np.linalg.norm(u)**2.)
    temp = v2 - np.matmul(p2.rot_mat, u.T).T
    p2.rot_mat_new = p2.rot_mat + np.matmul(temp.T, u) / (np.linalg.norm(u)**2.)
    
    r12 = p1.centroid - p2.centroid
    delta_h = p1.support_func(v1) + p2.support_func(-v2) + np.dot(r12, u)
    
    p1.centroid_new = p1.centroid - u*delta_h/2./(np.linalg.norm(u)**2.)
    p2.centroid_new = p2.centroid + u*delta_h/2./(np.linalg.norm(u)**2.)


def config_dis(x: np.array, p1: SuperEllipsoid, p2: SuperEllipsoid):
    u, v1, v2 = x[0:3], x[3:6], x[6:9]
    r12 = p1.centroid - p2.centroid
    delta_h = p1.support_func(v1) + p2.support_func(-v2) + np.dot(r12, u)
    
    y = (delta_h**2.)/2. + np.linalg.norm(np.matmul(p1.rot_mat, u.T).T-v1)**2. + np.linalg.norm(np.matmul(p2.rot_mat, u.T).T-v2)**2.
    y /= (np.linalg.norm(u)**2.)
    
    return y

def min_d(p1: SuperEllipsoid, p2: SuperEllipsoid):
    """
    calculate overlap potential (energy) between two particles
    """
    x0 = np.ones(9)
    res = optimize.minimize(lambda x: config_dis(x, p1, p2), x0, method='SLSQP')
    
    resolve_overlap(res, p1, p2)

