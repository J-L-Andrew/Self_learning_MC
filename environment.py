import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

from particle.ellipsoid import Ellipsoid

# environment for unit cell agent in the packing
class ASC(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, packing):

        self.packing = packing
        self.particle = Ellipsoid("Jin")
        self.num_particle = 24
        
        # particle shape parameter
        self.particle.S2M = np.sqrt(93./64.)
        
        # Translation and rotation magnitudes: 
        # In MC simulation, they are adjusted to 
        # maintain the rate of successful trial moves around 0.3.
        # here we initilaize they as 0.3 as well
        self.transMod = 0.3
        self.rotMod = 0.3
        
        self.p_trans = 0.5
        
        self.num_step = 0
        
        self.density_old = None
        
        self.density = self.packing.initialize(self.num_particle, self.particle.S2M, self.transMod, self.rotMod, self.p_trans, verb=True)
        
        # action space
        ### transmod and rotmod (*action) reasonable?
        self.action_space = spaces.Box(low=0.5, high=2., shape=(2, ), dtype=np.float32)

        # observation space
        ### the shape of particles can vary as initial parameters for different experiments 
        ### to obtain the dependacy of packing density on shape parameter, 
        ### how should it be added in RL?
        self.observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(3, ), dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self):
      
        self.density_old = self.density
        
        # acceptance rate of translation and rotation
        ### To do: both two probabilities can be neither too small or too large
        ### how can this princple be mainfest in RL?
        list = self.packing.sim(self.transMod, self.rotMod, self.num_step)
        p_trans, p_rot = list[0], list[1]
        self.num_step += 1
        
        self.density = self.packing.density()

        # reward
        reward = (self.density - self.density_old) / self.density_old
        
        # observation
        ### 1-densiy: want to make density close to 1
        ### but I don't know how to modify two probabilities
        obs = np.array([1.-self.density, p_trans, p_rot])
        
        # done
        delta_density = self.density - self.density_old
        if (delta_density < 1e-7):
            done = True
        else: done = False

        ### not clear yet
        info = {}

#         info = {
# #             "is_overlap":self.packing.is_overlap,
# #             "overlap_potential":self.packing.potential_energy,
#             "cell_penalty":self.packing.cell_penalty,
#             "packing_fraction":self.packing.fraction
#         }

        return obs, reward, done, info

    def reset(self):
        # reset packing
        self.transMod = 0.3
        self.rotMod = 0.3
        
        self.p_trans = 0.5
        
        self.density_old = None
        
        self.density = self.packing.initialize(self.num_particle, self.particle.S2M, self.transMod, self.rotMod, self.p_trans, verb=True)
        
        # reset renderer
        #self._reset_render()
        
        # record observation
        ### 0.8529 is estimated based on MC simulation
        obs = np.array([1.-self.density, 0.8529, 0.8529])
        return obs

    # def render(self):
    #     print("is_overlap {:d} overlap_potential {:2f} packing_fraction {:2f}".format(self.packing.is_overlap, self.packing.potential_energy,self.packing.fraction))
