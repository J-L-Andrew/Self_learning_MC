import numpy as np
from particle.base import Particle

# number of vertice & volume
hedron = {}
hedron["tetra"] = [4, np.sqrt(2)/12]
hedron["hexa"] = [8, 1]
hedron["octa"] = [6, np.sqrt(2)/3]

class Polytope(Particle):
    def __init__(self, length, type = "tetra"):
        super().__init__()
        self.dim = 3
        
        self.length = length # edge length
        
        self.rot_mat = np.diag(np.ones(self.dim))
        
        self.type = type
        self.n_vertice = hedron[type][0]
        
        self.vertice = None

    @property
    def volume(self):
        vol = hedron[self.type][1]*self.length**3
        return vol

    @property
    def inscribed_d(self):
        """
        diameter of the outscribed sphere
        """
        return np.min(self.semi_axis)
    
    @property
    def outscribed_d(self):
        """
        diameter of the outscribed sphere
        """
        # Hölder Inequality
        if (self.p > 1.):
            k = self.p/(self.p-1.)
            temp = self.a**(2.*k) + self.b**(2.*k) + self.c**(2.*k)
            return temp**(1./k)
        else: return np.max(self.semi_axis)
    
    def shapeFun(self, r_l: np.array):
        """ Superellipsoid function """
        y = 0.
        for i in range(self.dim):
            y += np.fabs(r_l[i] / self.semi_axis[i])**(2.*self.p)
        return y
    
    def shapeFund(self, r_l: np.array):
        """ The first-order derivative of superellipsoid function"""
        dx = np.zeros(self.dim)
        for i in range(self.dim):
            dx[i] = 2.*self.p/self.semi_axis[i]*np.fabs(r_l[i]/self.semi_axis[i])**(2.*self.p-1.)*np.sign(r_l[i])
        return dx
    
    def support_func(self, u: np.array):
        """
        Particle's support function: h(u) = max(K) u . x
        
        located in the origin, without rotation by default.
        """

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
      
    def support_funcs(self, u: np.array):
        """
        The support function of a specific particle:
        
        Particle's centroid & rotation matrix must be taken into consideration, i.e., 
        h_i(u) = h(R^T_i*u) + u*r_i
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
