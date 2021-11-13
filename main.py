import time
import random
import numpy as np
from numba import njit, prange
from numpy import sqrt

nParticles = 100000

particle = np.random.standard_normal((nParticles, 3))
particlev = np.zeros_like(particle)


# @vectorize(['float32(float32, float32)'], target='cuda')
# @njit(parallel=True, fastmath=True)
# def nbody_nb(particle, particlev):  # NumPy arrays as input
#     nSteps = 5
#     dt = 0.01
#     for step in range(1, nSteps + 1, 1):
#         for i in range(nParticles):
#             Fx = 0.0
#             Fy = 0.0
#             Fz = 0.0
#             for j in range(nParticles):
#                 if j != i:
#                     dx = particle[j, 0] - particle[i, 0]
#                     dy = particle[j, 1] - particle[i, 1]
#                     dz = particle[j, 2] - particle[i, 2]
#                     drSquared = dx * dx + dy * dy + dz * dz
#                     drPowerN32 = 1.0 / (drSquared + sqrt(drSquared))
#                     Fx += dx * drPowerN32
#                     Fy += dy * drPowerN32
#                     Fz += dz * drPowerN32
#                 particlev[i, 0] += dt * Fx
#                 particlev[i, 1] += dt * Fy
#                 particlev[i, 2] += dt * Fz
#         for i in range(nParticles):
#             particle[i, 0] += particlev[i, 0] * dt
#             particle[i, 1] += particlev[i, 1] * dt
#             particle[i, 2] += particlev[i, 2] * dt

@njit(parallel=True, fastmath=True)
def nbody_nb(particle, particlev):  # NumPy arrays as input
    dt = 0.01
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
                drPowerN32 = 1.0 / (drSquared + sqrt(drSquared))
                Fx += dx * drPowerN32
                Fy += dy * drPowerN32
                Fz += dz * drPowerN32
            particlev[i, 0] += dt * Fx
            particlev[i, 1] += dt * Fy
            particlev[i, 2] += dt * Fz
    for i in prange(nParticles):
        particle[i, 0] += particlev[i, 0] * dt
        particle[i, 1] += particlev[i, 1] * dt
        particle[i, 2] += particlev[i, 2] * dt


nbody_nb(particle, particlev)
t0 = time.time()
# nbody_nb(particle, particlev)
nSteps = 5
for i in range(1, nSteps + 1, 1):
    nbody_nb(particle, particlev)
ti_nb = time.time() - t0

print(ti_nb)
