import numpy as np
import sys
sys.path.append(r'/mnt/Edisk/andrew/Self_learning_MC')

from particle.sphere import *
from particle.polytope import *
from particle.superellipsoid import *
from packing import Packing

max_nA = 400000
tau = 1000
maxstep = 1e10

""" sphere """
# dim = 3
# nV = 1
# nP = 1 # number of particles
# nB = nP

# replica = [Sphere(1.) for i in range(2)] # a pair of particles (unit spheres)

""" polytope """
### tetra
dim = 3
nP = 4 # number of particles
nV = 4 # number of vertices
nB = nP*nV

PLANE_TOL = 1e-11

replica = [Polytope(2.*np.sqrt(2), "tetra") for i in range(2)]

### octa
# dim = 3
# nP = 4 # number of particles
# nV = 6 # number of vertices
# nB = nP*nV

# PLANE_TOL = 1e-11

# replica = [Polytope(np.sqrt(2), "octa") for i in range(2)]

""" superellipsoid """
# dim = 3
# nP = 4 # number of particles
# nV = dim+1 # just for convient
# nB = nP*(dim+1)

# replica = [SuperEllipsoid(1.5, 1., 1., 1.) for i in range(2)]

outscribed_d = replica[0].outscribed_d
inscribed_d = replica[0].inscribed_d

packing = Packing()

class pdc(object):
    def __init__(self):

        self.nA = None
        self.V0 = None
        self.V1 = None
        
    def allocate(self):
        self.u = np.empty([dim+nB, dim])
        self.r = np.empty([dim*nB, dim])

        # configuration x
        self.x = np.empty([max_nA, dim])
        self.x1 = np.empty([max_nA, dim]) # divide projection
        self.x2 = np.empty([max_nA, dim]) # concur projection
        self.xt = np.empty([max_nA, dim])
        
        self.ref = np.empty([nV, dim])

        # linear map
        self.Ad = np.empty([max_nA, dim+nB])
        self.Anew = np.empty([max_nA, dim+nB])
        self.Al = np.empty([max_nA, dim+nB])
        self.Alnew = np.empty([max_nA, dim+nB])
        self.atwa = np.empty([dim+nB, dim+nB])
        self.atwainv = np.empty([dim+nB, dim+nB])
        self.Q = np.empty([dim, dim])
        self.Qinv = np.empty([dim, dim])
        self.mw11iw10 = np.empty([nB, dim])

        # metric weights
        self.W = np.empty(max_nA)
        self.Wnew = np.empty(max_nA)

        self.LRr = np.empty([dim+nB,dim+nB])