import math


def get_kernal_parameters(n: int, n_dim: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    number_of_threads = 256

    threads_per_dimension = math.floor(number_of_threads ** (1 / n_dim))
    threads_per_block = tuple(threads_per_dimension for _ in range(n_dim))

    blocks_per_grid = tuple((n + (threads_in_dimension - 1)) //
                            threads_in_dimension for threads_in_dimension in threads_per_block)

    return blocks_per_grid, threads_per_block
