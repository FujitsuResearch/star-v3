"""Minimal task container used by the public paper subset."""

from typing import Optional


class StarTaskFormat:
    """Container for the circuit data needed by preprocessing and resource estimation."""

    def __init__(
        self,
        num_qubit: int,
        target_list: list[list[int]],
        axis_list: list[list[int]],
        angle_list: list[float],
        measured_pauli_strings: Optional[list[list[int]]] = None,
        *,
        task_type: Optional[str] = None,
    ):
        assert len(target_list) == len(axis_list) == len(angle_list), (
            "target_list, axis_list, and angle_list must have the same length."
        )

        self.num_qubit = num_qubit
        self.target_list = target_list
        self.axis_list = axis_list
        self.angle_list = angle_list
        self.measured_pauli_strings = measured_pauli_strings
        self.task_type = task_type
        self.num_gate = len(target_list)
