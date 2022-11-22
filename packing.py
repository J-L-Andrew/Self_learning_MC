import numpy as np

class Packing(object):
    """
    combination of multi-particles and cell
    """
    def __init__(self):
        # spatial dimension
        self.dim = None

        # particle info
        self.particle_type = None
        self.particles = []
        self.num_particles = None

        # cell info
        self.cell = None

        # color dimensionality
        self.dim_color = 3

        # log the change of fraction for "done"
        self.fraction_delta = 0.01  

    
    @property
    def volume_allp(self):
        """
        volume of all particles
        """
        volume = 0.
        for particle in self.particles:
            volume += particle.volume
        return volume

    @property
    def fraction(self):
        v = 0.
        for particle in self.particles:
            v += particle.volume
        return v / self.cell.volume

    @property
    def max_od(self):
        """
        The diameter of the largest outscribed sphere of all particles
        """
        d = 0.
        for particle in self.particles:
            d = max(particle.outscribed_d, d)
        return d

    @property
    def primitive_particles(self):
        """
        Return particles in all vertices of unit cell (parallelpiiped)
        """
        copy_particles = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    if(i==j==k==0): continue
                    
                    pos = [i, j, k]
                    base = np.matmul(pos, self.cell.lattice)
                    
                    for particle in self.particles:
                        image = particle.periodic_image(base)
                        image.color = np.array([0.992,0.525,0.529])
                        copy_particles.append(image)                 

        return self.particles + copy_particles