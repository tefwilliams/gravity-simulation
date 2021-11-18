import csv
import math
from time import time
import numpy as np
from numba import cuda
from galaxygen import create_galaxy

n_particles = 20000
velocities, positions = create_galaxy(n_particles, 1, 1)
# velocities = positions = np.random.random((n_particles, 3))

n_steps = 1000
time_step = 10.0 ** -4

sf = 0.58 * n_particles ** (-0.26)


def get_kernal_parameters(n_dim: int):
    number_of_threads = 256

    threads_per_dimension = math.floor(number_of_threads ** (1 / n_dim))
    threads_per_block = tuple(threads_per_dimension for _ in range(n_dim))

    blocks_per_grid = tuple((n_particles + (threads_in_dimension - 1)) //
                            threads_in_dimension for threads_in_dimension in threads_per_block)

    return blocks_per_grid, threads_per_block


@cuda.jit('void(float32[:, :], float32[:, :])')
def update_acceleration(positions, accelerations):
    i, j = cuda.grid(2)

    if i != j and i < n_particles and j < n_particles:
        dx = positions[j, 0] - positions[i, 0]
        dy = positions[j, 1] - positions[i, 1]
        dz = positions[j, 2] - positions[i, 2]
        drSquared = dx * dx + dy * dy + dz * dz
        drPowerN2 = 1.0 / (math.sqrt(drSquared) *
                           (drSquared + sf ** 2))
        accelerations[i, 0] += dx * drPowerN2
        accelerations[i, 1] += dy * drPowerN2
        accelerations[i, 2] += dz * drPowerN2


@cuda.jit('void(float32[:, :], float32[:, :])')
def update_velocity(velocities, accelerations):
    i = cuda.grid(1)

    if i < n_particles:
        velocities[i, 0] += time_step * accelerations[i, 0]
        velocities[i, 1] += time_step * accelerations[i, 1]
        velocities[i, 2] += time_step * accelerations[i, 2]


@cuda.jit('void(float32[:, :], float32[:, :])')
def update_position(positions, velocities):
    i = cuda.grid(1)

    if i < n_particles:
        positions[i, 0] += velocities[i, 0] * time_step
        positions[i, 1] += velocities[i, 1] * time_step
        positions[i, 2] += velocities[i, 2] * time_step


points_record = []

start = time()
for i in range(n_steps):
    accelerations = np.zeros_like(positions)
    update_acceleration[get_kernal_parameters(2)](positions, accelerations)
    update_velocity[get_kernal_parameters(1)](velocities, accelerations)
    update_position[get_kernal_parameters(1)](positions, velocities)
    points_record.append(positions.flatten())
print(time() - start)


with open('points_record.csv', 'w', newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(points_record)
