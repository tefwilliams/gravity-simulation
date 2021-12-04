import csv
from time import time
import numpy as np
from cudagravity import update_positions
from galaxygen import create_galaxy


def main():
    n_particles = 10000
    n_steps = 1000

    velocities, positions = create_galaxy(n_particles, 1)

    points_record = np.empty((n_steps, positions.size), dtype=np.float32)

    time_step = 10.0 ** -4
    softening_factor = 10.0 ** -4

    start = time()
    for i in range(n_steps):
        points_record[i] = update_positions(positions, velocities, time_step, softening_factor).flatten()
    print(time() - start)

    with open('points_record.csv', 'w', newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(points_record)


if __name__ == '__main__':
    main()
