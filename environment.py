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

    def __init__(self, packing, alpha, num_particle):

        self.packing = packing
        self.particle = Ellipsoid("Jin")
        
        self.num_particle = num_particle
        
        # particle shape parameter
        self.particle.S2M = alpha
        
        # Translation and rotation magnitudes: 
        # In MC simulation, they are adjusted to 
        # maintain the rate of successful trial moves around 0.3.
        # here we initilaize they as 0.3 as well
        self.transMod, self.rotMod = 0.3, 0.3
        
        # probability of translation trial move
        self.p_trans = 0.5
        
        # acceptance rate of translation and rotation
        self.pa_t, self.pa_r = 0.74, 0.98
         
        self.num_step = 0
        self.total_step = 0
        
        self.density_old = None
        self.density = self.packing.initialize(self.num_particle, self.particle.S2M, self.transMod, self.rotMod, self.p_trans, verb=False)
        
        self.transMod, self.rotMod = 0.15, 0.15
        
        # action space
        ### transmod and rotmod (*action) reasonable?
        self.action_space = spaces.Box(low=0.5, high=2., shape=(3, ), dtype=np.float32)

        # observation space
        ### the shape of particles can vary as initial parameters for different experiments 
        ### to obtain the dependacy of packing density on shape parameter, 
        ### how should it be added in RL?
        self.observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(3, ), dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        self.density_old = self.density
        
        self.transMod = np.clip(self.transMod*action[0], 0, 0.2)
        self.rotMod = np.clip(self.rotMod*action[1], 0, 0.2)
        self.p_trans = np.clip(self.p_trans*action[2], 0, 1)
        
        # acceptance rate of translation and rotation
        ### To do: both two probabilities can be neither too small or too large
        ### how can this princple be mainfest in RL?
        list = self.packing.sim(self.transMod, self.rotMod, self.p_trans, self.total_step)
        self.pa_t, self.pa_r = list[0], list[1]
        self.num_step += 1
        self.total_step += 1
        
        self.density = self.packing.density()

        # reward
        reward = (self.density - self.density_old) / self.density_old
        
        # observation
        ### 1-densiy: want to make density close to 1
        ### but I don't know how to modify two probabilities
        obs = np.array([1.-self.density, self.pa_t, self.pa_r])
        
        # done
        # delta_density = self.density - self.density_old
        if (self.num_step == 300):
            done = True
        else: done = False
              
        # if (delta_density < 1e-7):
        #     done = True
        #     self.packing.scr(1, self.num_step)
        # else: done = False
        
        ### not clear yet
        info = {"packing_fraction":self.density}

#         info = {
# #             "is_overlap":self.packing.is_overlap,
# #             "overlap_potential":self.packing.potential_energy,
#             "cell_penalty":self.packing.cell_penalty,
#             "packing_fraction":self.packing.fraction
#         }

        return obs, reward, done, info

    def reset(self):
        # reset packing
        # self.transMod, self.rotMod = 0.3, 0.3
        # self.p_trans = 0.5
        
        self.num_step = 0
        
        # self.density_old = None
        
        # self.density = self.packing.initialize(self.num_particle, self.particle.S2M, self.transMod, self.rotMod, self.p_trans, verb=False)
        
        # self.transMod, self.rotMod = 0.15, 0.15
        
        # reset renderer
        #self._reset_render()
        
        # record observation
        obs = np.array([1.-self.density, self.pa_t, self.pa_r])
        return obs

    def render(self):
        print("packing_fraction {:2f}".format(self.density))
