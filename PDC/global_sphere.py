import numpy as np
import sys
sys.path.append(r'/mnt/Edisk/andrew/Self_learning_MC')

from particle.sphere import *
from packing import Packing

dim = 3
nP = 4 # number of particles
max_nA = 400000
tau = 1000

# a pair of particles (unit spheres)
replica = [Sphere(1.) for i in range(2)]

packing = Packing()

class pdc(object):
    def __init__(self):
        self.nA = None
        
        self.u = np.empty([dim+nP, dim])
        self.r = np.empty([dim*nP, dim])

        # configuration x
        self.x = np.empty([max_nA, dim])
        self.x1 = np.empty([max_nA, dim]) # divide projection
        self.x2 = np.empty([max_nA, dim]) # concur projection
        self.xt = np.empty([max_nA, dim])

        # linear map
        self.Ad = np.empty([max_nA, dim+nP])
        self.Anew = np.empty([max_nA, dim+nP])
        self.Al = np.empty([max_nA, dim+nP])
        self.Alnew = np.empty([max_nA, dim+nP])
        self.atwa = np.empty([dim+nP, dim+nP])
        self.atwainv = np.empty([dim+nP, dim+nP])
        self.Q = np.empty([dim, dim])
        self.Qinv = np.empty([dim, dim])
        self.mw11iw10 = np.empty([nP, dim])

        # metric weights
        self.W = np.empty(max_nA)
        self.Wnew = np.empty(max_nA)

        self.LRr = np.empty([dim+nP,dim+nP])
        
        self.V0 = None
        self.V1 = None