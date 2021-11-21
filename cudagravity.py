import math
import numpy as np
from numba import cuda
from cudakernal import get_kernal_parameters


def get_accelerations(positions, sf):
    n_particles = len(positions)
    accelerations = np.zeros_like(positions)
    update_acceleration[get_kernal_parameters(n_particles, 2)](positions, accelerations, sf)
    return accelerations


def iterate(positions, velocities, time_step, sf):
    n_particles = len(positions)
    accelerations = np.zeros_like(positions)
    update_acceleration[get_kernal_parameters(n_particles, 2)](positions, accelerations, sf)
    update_velocity[get_kernal_parameters(n_particles, 1)](velocities, accelerations, time_step)
    update_position[get_kernal_parameters(n_particles, 1)](positions, velocities, time_step)
    return positions


@cuda.jit('void(float64[:, :], float64[:, :], float64)')
def update_acceleration(positions, accelerations, sf):
    n = len(positions)

    i, j = cuda.grid(2)

    if i != j and i < n and j < n:
        dx = positions[j, 0] - positions[i, 0]
        dy = positions[j, 1] - positions[i, 1]
        dz = positions[j, 2] - positions[i, 2]
        drSquared = dx * dx + dy * dy + dz * dz + sf * sf
        drPowerN32 = 1.0 / (math.sqrt(drSquared) * drSquared)
        accelerations[i, 0] += dx * drPowerN32
        accelerations[i, 1] += dy * drPowerN32
        accelerations[i, 2] += dz * drPowerN32


@cuda.jit('void(float64[:, :], float64[:, :], float64)')
def update_velocity(velocities, accelerations, time_step):
    n = len(velocities)

    i = cuda.grid(1)

    if i < n:
        velocities[i, 0] += accelerations[i, 0] * time_step
        velocities[i, 1] += accelerations[i, 1] * time_step
        velocities[i, 2] += accelerations[i, 2] * time_step


@cuda.jit('void(float64[:, :], float64[:, :], float64)')
def update_position(positions, velocities, time_step):
    n = len(positions)

    i = cuda.grid(1)

    if i < n:
        positions[i, 0] += velocities[i, 0] * time_step
        positions[i, 1] += velocities[i, 1] * time_step
        positions[i, 2] += velocities[i, 2] * time_step
