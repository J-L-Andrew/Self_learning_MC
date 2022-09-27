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
        
        # Translation and rotation magnitudes: 
        # In MC simulation, they are adjusted to 
        # maintain the rate of successful trial moves around 0.3.
        # here we initilaize they as 0.3 as well
transMod = 0.3
rotMod = 0.3
        
p_trans = 0.5
        
        
density = packing.initialize(num_particle, particle.S2M, transMod, rotMod, p_trans, verb=False)

list = packing.sim(transMod, rotMod, 1)
p_trans = list[0]
p_rot = list[1]
print(list)

packing.scr(1, 1)