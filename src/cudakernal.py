def get_kernal_parameters(n: int, n_dim: int, threads_per_dimension: int = 16) -> tuple[tuple[int, ...], tuple[int, ...]]:
    blocks_per_grid = tuple((n + (threads_per_dimension - 1)) // threads_per_dimension for _ in range(n_dim))
    threads_per_block = tuple(threads_per_dimension for _ in range(n_dim))

    return blocks_per_grid, threads_per_block
