"""Minimal Trotter-circuit helper used by the paper notebooks."""

import numpy as np

from starsim.task.format import StarTaskFormat
from starsim.task.hamiltonian import get_term_list


def generate_1stTrotter_unit_circ(ham, num_step: int) -> StarTaskFormat:
    """Generate a single first-order Trotter step as a `StarTaskFormat`."""

    n_qubit = ham.get_qubit_count()
    all_index_list, all_pauli_ids, all_coefs = get_term_list(ham)
    angle_list = list(2 * np.array(all_coefs).real / num_step)

    return StarTaskFormat(
        n_qubit,
        all_index_list,
        all_pauli_ids,
        angle_list,
        task_type="1st-order Trotter simulation",
    )
