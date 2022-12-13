import numpy as np
from utils import Transform, Ham2JPL

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
    def visable_particles(self):
        """
        Return particles in within frist shell of unit cell (3*3)
        """
        replica = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    if (i == j == k == 0): continue

                    pos = [i, j, k]
                    base = np.matmul(pos, self.cell.lattice)

                    for particle in self.particles:
                        image = particle.periodic_image(base)
                        image.color = np.array([0.992, 0.525, 0.529])
                        replica.append(image)

        return self.particles + replica

    def output_xyz(self, filename, repeat=True):
        """
        For visulaization in ovito, only access spheres and ellipsoids.
        
        Note: quaternion: JPL (x, y, z, w)
        """
        if (repeat):
            centroid = [particle.centroid for particle in self.visable_particles]
            color = [particle.color for particle in self.visable_particles]
            n = len(self.visable_particles)
            
            if (self.particle_type == 'ellipsoid'):
                quaternion = []
                for particle in self.visable_particles:
                    orientation = Transform().mat2qua(particle.rot_mat, "JPL").numpy()
                    quaternion.append(orientation)
                semi_axis = [particle.semi_axis for particle in self.visable_particles]
            elif (self.particle_type == 'sphere'):
                radius = [particle.radius for particle in self.visable_particles]

        else:
            centroid = [particle.centroid for particle in self.particles]
            color = [particle.color for particle in self.particles]
            n = len(self.particles)
            
            if (self.particle_type == 'ellipsoid'):
                quaternion = []
                for particle in self.particles:
                    orientation = Transform().mat2qua(particle.rot_mat, "JPL").numpy()
                    quaternion.append(orientation)
                semi_axis = [particle.semi_axis for particle in self.particles]
            elif (self.particle_type == 'sphere'):
                radius = [particle.radius for particle in self.particles]

        with open(filename, 'w') as f:
            # The keys should be strings
            f.write(str(n) + '\n')
            f.write('Lattice="' + ' '.join([str(vector) for vector in self.cell.lattice.flat]) + '" ')
            # f.write('Origin="' + ' '.join(str(index) for index in packing.cell.origin) + '" ')
            
            if (self.particle_type == 'ellipsoid'):
                f.write(
                    'Properties=pos:R:3:orientation:R:4:aspherical_shape:R:3:color:R:3 \n'
                )
                np.savetxt(f, np.column_stack([centroid, quaternion, semi_axis, color]))
            elif (self.particle_type == 'sphere'):
                f.write(
                    'Properties=pos:R:3:radius:R:1:color:R:3 \n'
                )
                np.savetxt(f, np.column_stack([centroid, radius, color]))
