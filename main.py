import csv
import time
import numpy as np
from numba import njit, prange
from numpy import sqrt

from galaxygen import create_galaxy

nParticles = 2000
particlev, particle = create_galaxy(nParticles, 1, 1)


@njit(parallel=True, fastmath=True)
def nbody_nb(particle, particlev):  # NumPy arrays as input
    dt = 10.0 ** -4
    sf = 0.58 * nParticles ** (-0.26)
    for i in prange(nParticles):
        Fx = 0.0
        Fy = 0.0
        Fz = 0.0
        for j in range(nParticles): 
            if j != i:
                dx = particle[j, 0] - particle[i, 0]
                dy = particle[j, 1] - particle[i, 1]
                dz = particle[j, 2] - particle[i, 2]
                drSquared = dx * dx + dy * dy + dz * dz
                drPowerN2 = 1.0 / (sqrt(drSquared) * (drSquared + sf ** 2))
                Fx += dx * drPowerN2
                Fy += dy * drPowerN2
                Fz += dz * drPowerN2
        particlev[i, 0] += dt * Fx
        particlev[i, 1] += dt * Fy
        particlev[i, 2] += dt * Fz
    for i in prange(nParticles):
        particle[i, 0] += particlev[i, 0] * dt
        particle[i, 1] += particlev[i, 1] * dt
        particle[i, 2] += particlev[i, 2] * dt


nSteps = 100
points_record = np.empty((nSteps, nParticles * 3))

t0 = time.time()
for i in range(nSteps):
    nbody_nb(particle, particlev)
    points_record[i] = particle.flatten()
ti_nb = time.time() - t0

print(ti_nb)

with open('points_record.csv', 'w', newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(points_record)
