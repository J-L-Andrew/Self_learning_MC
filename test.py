import numpy as np
from environment import ASC
from particle.ellipsoid import Ellipsoid
import sys
sys.path.append(r'/mnt/Edisk/andrew/Self_learning_MC')
import packing

particle = Ellipsoid("Jin")
num_particle = 24
        
# particle shape parameter
particle.S2M = np.sqrt(93./64.)
        
transMod, rotMod = 0.3, 0.3
        
p_trans = 0.5 
        
density = packing.initialize(num_particle, particle.S2M, transMod, rotMod, p_trans, verb=True)

transMod, rotMod = 0.15, 0.15

for i in range(1e3):
    list = packing.sim(transMod, rotMod, p_trans, 1)

# packing.scr(1, 1)