from constants import *
from main import create_particules, calculate_density


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


test_density()