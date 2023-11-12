import csv
from time import time
import numpy as np
from cudagravity import update_positions
from galaxygen import create_galaxy


def main() -> None:
    n_particles = 2 ** 15
    n_steps = 1000

    velocities, positions = create_galaxy(n_particles, 1)

    points_record = np.empty((n_steps, positions.size), dtype=np.float32)

    time_step = 10.0 ** -4
    softening_factor = 10.0 ** -4

    start = time()
    for i in range(n_steps):
        points_record[i] = update_positions(positions, velocities, time_step, softening_factor).flatten()
        update_estimated_remaining_time(start, i + 1, n_steps)
    runtime = time() - start

    print(f"""
    
Time to run: {runtime:.3g}s
Time per iteration: {runtime / n_steps:.3g}s

Number of particles: {n_particles}
Number of iterations: {n_steps}
    """)

    with open('points_record.csv', 'w', newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(points_record)


def update_estimated_remaining_time(start_time: float, current_step: int, total_steps: int) -> None:
    current_runtime = time() - start_time
    fraction_steps_completed = current_step / total_steps
    estimated_total_runtime = current_runtime / fraction_steps_completed
    estimated_remaining_time = estimated_total_runtime - current_runtime

    print(f"Time until completion: {estimated_remaining_time:.0f}s \033[K", end='\r')


if __name__ == '__main__':
    main()
