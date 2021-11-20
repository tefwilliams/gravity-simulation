import csv
from time import time
import numpy as np
from cudakernal import get_kernal_parameters
from cudagravity import update_acceleration, update_velocity, update_position
from galaxygen import create_galaxy

n_particles = 10000
velocities, positions = create_galaxy(n_particles, 1, 1)
# velocities = positions = np.random.random((n_particles, 3))

n_steps = 1000
time_step = 10.0 ** -4

sf = 0.58 * n_particles ** (-0.26)

points_record = []

start = time()
for i in range(n_steps):
    accelerations = np.zeros_like(positions)
    update_acceleration[get_kernal_parameters(n_particles, 2)](positions, accelerations, sf)
    update_velocity[get_kernal_parameters(n_particles, 1)](velocities, accelerations, time_step)
    update_position[get_kernal_parameters(n_particles, 1)](positions, velocities, time_step)
    points_record.append(positions.flatten())
print(time() - start)


with open('points_record.csv', 'w', newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(points_record)
