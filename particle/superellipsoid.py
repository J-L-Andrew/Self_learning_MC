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
        
        self.rot_mat = np.diag(np.ones(self.dim))
        
        self.centroid_new = None
        self.rot_mat_new = None
    
    @property
    def semi_axis(self):
        return np.array([self.a, self.b, self.c])

    @property
    def volume(self):
        return 4./3.*np.pi*np.cumprod(self.semi_axis)[2]
      
    def shapeFun(self, r_l: np.array):
        """ Superellipsoid function """
        y = 0.
        for i in range(self.dim):
            y += np.fabs(r_l[i] / self.semi_axis[i])**(2.*self.p)
        return y
    
    def shapeFund(self, r_l: np.array):
        dx = np.zeros(self.dim)
        for i in range(self.dim):
            dx[i] = 2.*self.p/self.semi_axis[i]*np.fabs(r_l[i]/self.semi_axis[i])**(2.*self.p-1.)*np.sign(r_l[i])
        return dx
    
    def support_func(self, u: np.array):
        """
        Particle's support function:
        
        located in the origin, without rotation
        """
      
        """
        Find r_c with its normal vector equals u
        We should not consider the orientation of a single particle because
        the rotation is applied to u
        Adapated from:
        Yoav Kallus and Veit Elser Dense-packing crystal structures of physical tetrahedra
        """
        # SuperballFunctionD(p1, rs1, ra_l, dxt); MatrixMult(Aa, dxt, dxa);
        r_u = np.linalg.norm(u)
        # dxt = k*Ru/|u|
        dp = 2.*self.p - 1.
        
        normal = u / r_u
        r_c = np.zeros(self.dim)
        for i in range(self.dim):
            r_c[i] = self.semi_axis[i] * np.fabs(normal[i]*self.semi_axis[i]/2./self.p)**(1./dp) * np.sign(u[i])
        
        k = (1. / self.shapeFun(r_c))**(dp/2./self.p)
        r_c *= k**(1./dp)    
        
        return np.dot(r_c, u)
      
    def supFun(self, u: np.array):
        """
        The support function of a specific particle
        """
        y = self.support_func(np.matmul(self.rot_mat.T, u.T).T)
        y += np.dot(u, self.centroid)
        return y
      
    def check(self, vector):
      
        r_v = np.linalg.norm(vector)
        
        # https://baike.baidu.com/item/球坐标系/8315363
        temp = np.sqrt(vector[0]**2. + vector[1]**2.)
        sin_cita = temp / r_v
        cos_cita = vector[2] / r_v
        
        if (temp == 0.):
            sin_fai, cos_fai = 1., 0.
        else:
            sin_fai, cos_fai = vector[1] / temp, vector[0] / temp
        
        point = np.array([sin_cita*cos_fai, sin_cita*sin_fai, cos_cita])
        r = (1. / self.shapeFun(point))**(0.5/self.p)
        point *= r
        
        u = self.shapeFund(point)
        
        tr = np.dot(point, u)
        
        return (tr-self.support_func(u))

"""
Check whether two particle overlap
"""
def delta_h_normalized(u: np.array, p1: SuperEllipsoid, p2: SuperEllipsoid):
    """ delta_h(u) / ||u|| """
    y = (p1.supFun(u) + p2.supFun(-u)) / np.linalg.norm(u)
    return y

def is_overlap(p1: SuperEllipsoid, p2: SuperEllipsoid):
    """
    According to the separating plane theorem.
    """
    x0 = np.ones(3)*2.
    u_c = optimize.minimize(lambda x: delta_h_normalized(x, p1, p2), x0, method='SLSQP')
    
    y = delta_h_normalized(u_c, p1, p2)
    if (y <= 0.): return 0
    else: return 1
    
    
        

def resolve_overlap_new(x: np.array, p1: SuperEllipsoid, p2: SuperEllipsoid):
    u, v1, v2 = x[0:3], x[3:6], x[6:9]
    col = u.reshape(-1, 1)
    row = (v1 - np.matmul(u, p1.rot_mat)).reshape(1, -1)
    p1.rot_mat_new = p1.rot_mat + np.matmul(col, row) / (np.linalg.norm(u)**2.)
    row = (v2 - np.matmul(u, p2.rot_mat)).reshape(1, -1)
    p2.rot_mat_new = p2.rot_mat + np.matmul(col, row) / (np.linalg.norm(u)**2.)
    
    r12 = p1.centroid - p2.centroid
    delta_h = p1.support_func(v1) + p2.support_func(-v2) + np.dot(r12, u)
    
    p1.centroid_new = p1.centroid - u*delta_h/2./(np.linalg.norm(u)**2.)
    p2.centroid_new = p2.centroid + u*delta_h/2./(np.linalg.norm(u)**2.)

def resolve_overlap(x: np.array, p1: SuperEllipsoid, p2: SuperEllipsoid):
    u, v1, v2 = x[0:3], x[3:6], x[6:9]
    r12 = p1.centroid - p2.centroid
    delta_h = p1.support_func(v1) + p2.support_func(-v2) + np.dot(r12, u)
    
    p1.centroid -= u*delta_h/2./(np.linalg.norm(u)**2.)
    p2.centroid += u*delta_h/2./(np.linalg.norm(u)**2.)
    
    col = u.reshape(-1, 1)
    row = (v1 - np.matmul(u, p1.rot_mat)).reshape(1, -1)
    p1.rot_mat += np.matmul(col, row) / (np.linalg.norm(u)**2.)
    row = (v2 - np.matmul(u, p2.rot_mat)).reshape(1, -1)
    p2.rot_mat += np.matmul(col, row) / (np.linalg.norm(u)**2.)
    
    # sum = p1.support_func(v1) + np.dot(u, p1.centroid) + p2.support_func(-v2) - np.dot(u, p2.centroid)
    
    # print(sum)
    

def config_dis(x: np.array, p1: SuperEllipsoid, p2: SuperEllipsoid):
    u, v1, v2 = x[0:3], x[3:6], x[6:9]
    r12 = p1.centroid - p2.centroid
    delta_h = p1.support_func(v1) + p2.support_func(-v2) + np.dot(r12, u)
    
    y = (delta_h**2.)/2. + np.linalg.norm(np.matmul(p1.rot_mat.T, u)-v1)**2. + np.linalg.norm(np.matmul(p2.rot_mat.T, u)-v2)**2.
    y /= np.linalg.norm(u)**2.
    
    return y

def min_d(p1: SuperEllipsoid, p2: SuperEllipsoid):
    """
    """
    x0 = np.ones(9)*2.
    res = optimize.minimize(lambda x: config_dis(x, p1, p2), x0, method='SLSQP')
    
    resolve_overlap(res.x, p1, p2)

