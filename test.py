from time import time
from constants import *
from liquid import * 


def test_density():
    particules = create_particules()

    ref_index = 130 * 5 + 15
    ref_pos = particules[ref_index].pos
    ref_dens = calculate_density(particules, ref_pos)
    print(f"Density of reference: {ref_dens:.4f}")

    i = 1
    while i <= 100:
        pos = particules[ref_index + i].pos
        d = calculate_density(particules, pos)
        print(f"{pos} - density value {d:.4f}")

        i += 1

# NOTE: v1: 10k particules ~ 491s
# NOTE: v2: 10k particules ~ 47s
def time_density():
    particules = create_particules(10000)

    start = time()
    for p in particules:
        _ = calculate_density(particules, p.pos)
        # _ = calculate_density_np(particules, p.pos)

    print(f'Took {time() - start:.0f} segs to calculate all densities.')


# test_density()
# time_density()