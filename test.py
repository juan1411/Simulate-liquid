from time import time
from constants import *
from liquid import *


def test_density():
    num = 360
    positions = create_particules(num, "grid")
    positions = np.concatenate(positions)

    ref_index = int(300/RADIUS) * 3 + 10
    ref_pos = positions[ref_index]
    ref_dens = calculate_density(positions, ref_pos)
    print(f"Density of reference: \n{ref_pos} - density value {ref_dens:.4f}\n")

    i = 1
    while i < 10:
        pos = positions[ref_index + i]
        d = calculate_density(positions, pos)
        print(f"{pos} - density value {d:.4f}")

        i += 1

    return None

# NOTE: v1 - forloop: 10k particules ~ 491s
# NOTE: v2 - numpy: 10k particules ~ 47s
# NOTE: v3 - numba: 10k particules ~ 4s
def time_density():
    num = 10000
    positions = create_particules(num)
    positions = np.array(positions)

    start = time()
    for i in range(num):
        _ = calculate_density(positions, positions[i])

    print(f'Took {time() - start:.0f} segs to calculate all densities.')


def test_pressure():
    num = 360
    positions = create_particules(num, "grid")
    positions = np.array(positions)

    densities = []
    for p in positions:
        d = calculate_density(positions, p)
        densities.append(d)

    densities = np.array(densities)

    ref_index = int(300/RADIUS) * 0 + 2
    ref_pos = positions[ref_index]
    ref_dens= densities[ref_index]
    ref_pres = calculate_pressure_force(positions, densities, ref_pos, ref_dens)
    print(f"Pressure of reference: \n{ref_pos} - pressure value {ref_pres}\n")

    i = 1
    while i < 10:
        pos = positions[ref_index + i]
        dens= densities[ref_index + i]
        pres = calculate_pressure_force(positions, densities, pos, dens)
        print(f"{pos} - pressure value {pres}")

        i += 1

    return None


# test_density()
# time_density()

test_pressure()