from numba import njit, prange
import numpy as np


def create_galaxy(n, size, G):
    '''
    Generates a spiral galaxy
    '''
    # Generates random masses
    # masses = abs(np.random.normal(0.5, 0.05, n))
    masses = np.ones(n)

    # Sets positions for points
    positions = create_positions(n, size)

    # Finds the COM, and total mass, of all points (radially) interior to each point
    internal_masses, internal_positions = get_internal_masses(
        masses, positions)
    relative_internal_positions = positions - internal_positions

    # Finds the initial velocity of each point
    velocities = create_velocities(
        n, relative_internal_positions, internal_masses, G)

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

    points = np.zeros((n, 3))

    # Adds 'noise' to all 3D to generate points
    points[:, 0] = np.random.normal(0, galaxy_spread, n) + points_x
    points[:, 1] = np.random.normal(0, galaxy_spread, n) + points_y
    points[:, 2] = np.random.normal(0, galaxy_spread, n)

    return points


def create_velocities(n, relative_internal_positions, internal_masses, G):
    '''
    Creates point velocities
    '''
    # Calculates the speed of particle for circular motion (considering interior mass)
    relative_distances = np.linalg.norm(relative_internal_positions, axis=1)
    speeds = np.sqrt((internal_masses * G) / relative_distances) * 0.7

    # Calculates unit vectors in 3D
    radial_vectors = relative_internal_positions[:, 0:2]
    vertical_vectors = np.random.normal(0, 0.1, n)

    vectors = np.zeros((n, 3))
    vectors[:, 0] = - radial_vectors[:, 1]
    vectors[:, 1] = radial_vectors[:, 0]
    vectors[:, 2] = vertical_vectors

    # Adds normalisation for correct speed (due to addition of z-component)
    normalisation = np.linalg.norm(vectors, axis=1)

    # Generates velocities
    return speeds[:, None] * vectors / normalisation[:, None]


def get_internal_masses(masses, positions):
    n = len(masses)

    internal_center_of_mass_positions = np.empty((n, 3))
    internal_center_of_mass_masses = np.empty((n))

    # Finds the radial distance for each point from the centre of the galaxy
    distances_from_zero = np.linalg.norm(positions, axis=1)

    for i in range(n):
        positions_internal_to_particle = positions[distances_from_zero <
                                                   distances_from_zero[i]]
        masses_internal_to_particle = masses[distances_from_zero <
                                             distances_from_zero[i]]

        internal_center_of_mass_masses[i], internal_center_of_mass_positions[i] = get_center_of_mass(
            masses_internal_to_particle, positions_internal_to_particle)

    return internal_center_of_mass_masses, internal_center_of_mass_positions


# @njit(parallel=True, fastmath=True)
# def get_center_of_mass(masses, positions):
#     com_mass = np.sum(masses)

#     com_position_x = 0.0
#     com_position_y = 0.0
#     com_position_z = 0.0

#     n = len(masses)

#     if com_mass != 0:
#         for i in prange(n):
#             com_position_x += (positions[i, 0] * masses[i]) / com_mass
#             com_position_y += (positions[i, 1] * masses[i]) / com_mass
#             com_position_z += (positions[i, 2] * masses[i]) / com_mass

#     return com_mass, (com_position_x, com_position_y, com_position_z)

def get_center_of_mass(masses, positions):
    com_mass = np.sum(masses)
    com_position = np.zeros(3)

    if com_mass != 0:
        com_position = np.sum(positions * masses[:, None], axis=0) / com_mass

    return com_mass, com_position
