import csv
import math
from time import time
import numpy as np
from numba import cuda

from galaxygen import create_galaxy

nParticles = 100000
particlev, particle = create_galaxy(nParticles, 1, 1)
# particlev = particle = np.random.random((nParticles, 3))

dt = 10.0 ** -4
sf = 0.58 * nParticles ** (-0.26)


@cuda.jit('void(float64[:, :], float64[:, :])')
def n_body(particle, particlev):
    n = len(particle)

    i = cuda.grid(1)

    Fx = 0.0
    Fy = 0.0
    Fz = 0.0
    for j in range(n):
        if j != i:
            dx = particle[j, 0] - particle[i, 0]
            dy = particle[j, 1] - particle[i, 1]
            dz = particle[j, 2] - particle[i, 2]
            drSquared = dx * dx + dy * dy + dz * dz
            drPowerN2 = 1.0 / (math.sqrt(drSquared) *
                               (drSquared + sf ** 2))
            Fx += dx * drPowerN2
            Fy += dy * drPowerN2
            Fz += dz * drPowerN2

    particlev[i, 0] += dt * Fx
    particlev[i, 1] += dt * Fy
    particlev[i, 2] += dt * Fz

    particle[i, 0] += particlev[i, 0] * dt
    particle[i, 1] += particlev[i, 1] * dt
    particle[i, 2] += particlev[i, 2] * dt


threadsperblock = 32
blockspergrid = (len(particle) + (threadsperblock - 1)) // threadsperblock

nSteps = 100
points_record = np.empty((nSteps, nParticles * 3), dtype=np.float64)

t0 = time()
for i in range(nSteps):
    n_body[blockspergrid, threadsperblock](particle, particlev)
    points_record[i] = particle.flatten()
ti_nb = time() - t0

print(ti_nb)

with open('points_record.csv', 'w', newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(points_record)
