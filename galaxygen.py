from numba import cuda
import numpy as np


def create_galaxy(n, size, G):
    '''
    Generates a spiral galaxy
    '''
    # Generates random masses
    # masses = abs(np.random.normal(0.5, 0.05, n))
    masses = np.ones(n, dtype=np.float32)

    # Sets positions for points
    positions = create_positions(n, size)

    # Finds the COM, and total mass, of all points (radially) interior to each point
    internal_masses, internal_positions = get_internal_masses(
        masses, positions)
    relative_internal_positions = positions - internal_positions

    # Finds the initial velocity of each point
    velocities = create_velocities(
        relative_internal_positions, internal_masses, G)

    return velocities, positions


def create_positions(n, size):
    '''
    Creates galaxy particle positions
    '''
    galaxy_spread = size / 5

    # Generates radial and angular components in 2D
    points_r = np.random.exponential(size, n)
    points_p = np.random.rand(n) * 2 * np.pi

    # Converts to Cartesian
    points_x = points_r * np.cos(points_p)
    points_y = points_r * np.sin(points_p)

    points = np.empty((n, 3), dtype=np.float32)

    # Adds 'noise' to all 3D to generate points
    points[:, 0] = np.random.normal(0, galaxy_spread, n) + points_x
    points[:, 1] = np.random.normal(0, galaxy_spread, n) + points_y
    points[:, 2] = np.random.normal(0, galaxy_spread, n)

    return points


def create_velocities(relative_internal_positions, internal_masses, G):
    '''
    Creates point velocities
    '''
    # Calculates the speed of particle for circular motion (considering interior mass)
    relative_distances = np.linalg.norm(relative_internal_positions, axis=1)
    speeds = np.sqrt((internal_masses * G) / relative_distances) * 0.7

    # Calculates unit vectors in 3D
    radial_vectors = relative_internal_positions[:, 0:2]
    vertical_vectors = np.random.normal(0, 0.1, len(relative_distances))

    vectors = np.zeros_like(relative_internal_positions)
    vectors[:, 0] = - radial_vectors[:, 1]
    vectors[:, 1] = radial_vectors[:, 0]
    vectors[:, 2] = vertical_vectors

    # Adds normalisation for correct speed (due to addition of z-component)
    normalisation = np.linalg.norm(vectors, axis=1)

    # Generates velocities
    return speeds[:, None] * vectors / normalisation[:, None]


def get_internal_masses(masses, positions):
    internal_center_of_mass_positions = np.zeros_like(positions)
    internal_center_of_mass_masses = np.zeros_like(masses)

    # Finds the radial distance for each point from the centre of the galaxy
    distances_from_zero = np.linalg.norm(positions, axis=1)

    threads_per_block = 32
    blocks_per_grid = (len(masses) + (threads_per_block - 1)) // threads_per_block

    cuda_get_internal_masses[blocks_per_grid, threads_per_block](masses, positions, internal_center_of_mass_masses, internal_center_of_mass_positions, distances_from_zero)

    return internal_center_of_mass_masses, internal_center_of_mass_positions


@cuda.jit('void(float32[:], float32[:, :], float32[:], float32[:, :], float32[:])')
def cuda_get_internal_masses(masses, positions, internal_center_of_mass_masses, internal_center_of_mass_positions, distances_from_zero):
    n = len(masses)

    i = cuda.grid(1)

    if i < n:
        for j in range(n):
            if i != j and distances_from_zero[j] < distances_from_zero[i]:
                internal_center_of_mass_masses[i] += masses[j]
                internal_center_of_mass_positions[i, 0] += positions[j, 0] * masses[j]
                internal_center_of_mass_positions[i, 1] += positions[j, 1] * masses[j]
                internal_center_of_mass_positions[i, 2] += positions[j, 2] * masses[j]

    if internal_center_of_mass_masses[i] != 0:
        internal_center_of_mass_positions[i, 0] /= internal_center_of_mass_masses[i]
        internal_center_of_mass_positions[i, 1] /= internal_center_of_mass_masses[i]
        internal_center_of_mass_positions[i, 2] /= internal_center_of_mass_masses[i]
