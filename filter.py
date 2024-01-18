import numpy as np
from numba import njit, prange

from constants import TANK, SMOOTHING_RADIUS
START = np.array([TANK[0], TANK[1]]).reshape((1, 2))
HASH = np.array([15823, 9737333]).reshape((1, 2))

@njit
def pos_to_coord(positions: np.ndarray, radius: float = SMOOTHING_RADIUS) -> np.ndarray:
    return (positions -START)//radius

@njit
def coords_to_hash(coords: np.ndarray) -> np.ndarray:
    return np.sum(coords * HASH, axis=1)

@njit
def positions_to_hash(positions: np.ndarray, radius: float = SMOOTHING_RADIUS) -> np.ndarray:
    return coords_to_hash(pos_to_coord(positions, radius))


CELL_SPACE = np.array([
    [-1, -1], [0, -1], [1, -1],
    [-1,  0], [0,  0], [1,  0],
    [-1,  1], [0,  1], [1,  1]
])
HASH_SPACE = coords_to_hash(CELL_SPACE)

@njit
def get_lookup_space(hash: float) -> np.ndarray:
    return HASH_SPACE + hash

@njit(parallel=True)
def get_indices(hash: float, ref: np.ndarray) -> np.ndarray:
    filter = np.array([False for _ in range(ref.shape[0])])
    lookup_space = get_lookup_space(hash)

    for i in prange(lookup_space.shape[0]):
        value = lookup_space[i]
        filter = filter | (ref == value)

    return filter