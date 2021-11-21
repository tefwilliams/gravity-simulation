import csv
from time import time
import numpy as np
from cudagravity import iterate
from galaxygen import create_galaxy


def main():
    n_particles = 10000
    velocities, positions = create_galaxy(n_particles, 1)

    n_steps = 10000
    time_step = 10.0 ** -4

    # Generate softening factor
    # sf = 0.58 * n_particles ** (-0.26)
    sf = 10.0 ** -3

    points_record = np.empty((n_steps, n_particles * 3), dtype=np.float32)

    start = time()
    for i in range(n_steps):
        points_record[i] = iterate(positions, velocities, time_step, sf).flatten()
    print(time() - start)

    with open('points_record.csv', 'w', newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(points_record)


if __name__ == '__main__':
    main()
