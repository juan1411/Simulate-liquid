from time import time
from constants import *
from liquid import *


def test_density():
    particules = create_particules()
    positions = np.concatenate([p.get_pos() for p in particules], axis=0)

    ref_index = 130 * 5 + 15
    ref_pos = particules[ref_index].get_pos()
    ref_dens = calculate_density(positions, ref_pos)
    print(f"Density of reference: {ref_dens:.4f}")

    i = 1
    while i <= 100:
        pos = particules[ref_index + i].get_pos()
        d = calculate_density(positions, pos)
        print(f"{pos[0]} - density value {d:.4f}")

        i += 1

    return None

# NOTE: v1 - forloop: 10k particules ~ 491s
# NOTE: v2 - numpy: 10k particules ~ 47s
# NOTE: v3 - numba: 10k particules ~ 4s
def time_density():
    num = 10000
    particules = create_particules(num)
    positions = np.array([p.pos for p in particules])

    start = time()
    for i in range(num):
        _ = calculate_density(positions, particules[i].get_pos())

    print(f'Took {time() - start:.0f} segs to calculate all densities.')


# test_density()
time_density()