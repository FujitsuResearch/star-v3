"""Resource-estimation helpers used by StarSimulator."""

import numpy as np

from starsim.task.format import StarTaskFormat


ERROR_THRESHOLD = 1e-2
NUM_PATCH_FOR_PREPARATION = 10
MAX_CLOCK = {"compact": 11, "fast": 3}
MEASUREMENT_TIME_FOR_SINGLE_ROUND = 1e-6


def logical_error(p_ph: float, d: int) -> float:
    """Logical error-rate model for the surface code."""

    return 0.1 * (p_ph / ERROR_THRESHOLD) ** ((d + 1) / 2)


def get_num_patch(task: StarTaskFormat, layout: str = "compact") -> int:
    """Estimate the number of patches for the chosen layout."""

    nqubit = task.num_qubit
    if layout == "compact":
        if (nqubit + NUM_PATCH_FOR_PREPARATION) % 2 == 0:
            return int((nqubit + NUM_PATCH_FOR_PREPARATION) * 3 / 2)
        return int((nqubit + NUM_PATCH_FOR_PREPARATION - 1) * 3 / 2 + 2)
    if layout == "fast":
        return int(np.ceil(2 * nqubit + np.sqrt(8 * nqubit) + NUM_PATCH_FOR_PREPARATION + 1))
    raise ValueError(f"Unsupported layout: {layout}")


def calc_average_clock(task: StarTaskFormat, layout: str = "compact") -> int:
    """Return the worst-case clock count per rotation gate."""

    del task
    return MAX_CLOCK[layout]


def estimate_distance(
    task: StarTaskFormat,
    p_ph: float,
    N_repeat: int = 1,
    layout: str = "compact",
) -> int:
    """Determine the surface-code distance from the task size."""

    num_rot_gate = task.num_gate
    num_patch = get_num_patch(task, layout=layout)
    num_clock = calc_average_clock(task, layout=layout)

    assert p_ph < ERROR_THRESHOLD, (
        "ValueError: The physical error rate is higher than the surface-code threshold."
    )

    d = 3
    while 1 / logical_error(p_ph, d) < 100 * d * num_rot_gate * int(N_repeat) * num_clock * num_patch:
        d += 2

    assert 100 * d * num_rot_gate * int(N_repeat) * num_clock * num_patch >= 0, (
        "OverflowError: An overflow has occurred due to the input of type int32."
    )
    return d


def physical_qubit_count(
    task: StarTaskFormat,
    p_ph: float,
    N_repeat: int = 1,
    layout: str = "compact",
) -> int:
    """Estimate the total number of physical qubits."""

    num_patch = get_num_patch(task, layout)
    d = estimate_distance(task, p_ph, N_repeat, layout)
    return 2 * d**2 * num_patch


def estimate_single_shot_time(
    task: StarTaskFormat,
    p_ph: float,
    N_repeat: int = 1,
    layout: str = "compact",
) -> float:
    """Estimate the runtime of one logical shot in seconds."""

    num_rot_gate = task.num_gate
    d = estimate_distance(task, p_ph, N_repeat, layout)
    num_clock = calc_average_clock(task, layout=layout)
    return d * num_clock * num_rot_gate * N_repeat * MEASUREMENT_TIME_FOR_SINGLE_ROUND
