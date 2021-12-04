import math
import numpy as np
from numba import cuda
from cudakernal import get_kernal_parameters


def update_positions(positions, velocities, time_step, softening_factor=0):
    accelerations = get_accelerations(positions, softening_factor)
    velocities += accelerations * time_step
    positions += velocities * time_step
    return positions


def get_accelerations(positions, softening_factor):
    n_particles = len(positions)
    accelerations = np.zeros_like(positions)
    update_acceleration[get_kernal_parameters(n_particles, 2)](positions, accelerations, softening_factor)
    return accelerations


@cuda.jit('void(float64[:, :], float64[:, :], float64)')
def update_acceleration(positions, accelerations, softening_factor):
    n = len(positions)

    i, j = cuda.grid(2)

    if i != j and i < n and j < n:
        dx = positions[j, 0] - positions[i, 0]
        dy = positions[j, 1] - positions[i, 1]
        dz = positions[j, 2] - positions[i, 2]
        drSquared = dx * dx + dy * dy + dz * dz + softening_factor * softening_factor
        drPowerN32 = 1.0 / (math.sqrt(drSquared) * drSquared)
        accelerations[i, 0] += dx * drPowerN32
        accelerations[i, 1] += dy * drPowerN32
        accelerations[i, 2] += dz * drPowerN32
