"""Resource-estimation entry point for the public paper subset."""

import math
from typing import Optional

import numpy as np

from starsim.core.preprocess import preprocessing
from starsim.simulator.resource import (
    ERROR_THRESHOLD,
    estimate_distance,
    estimate_single_shot_time,
    physical_qubit_count,
)
from starsim.task.format import StarTaskFormat


def _mitigation_cost(
    angle_list: list[float],
    alpha_list: list[float],
    p_ph: float,
    num_repeat: int,
) -> float:
    """Return the probabilistic error-cancellation overhead."""

    p_list = np.array(alpha_list) * np.abs(np.array(angle_list)) * p_ph
    gamma_list = 1 / (1 - 2 * p_list)
    return np.prod(gamma_list) ** num_repeat


class StarSimulator:
    """Minimal STAR wrapper kept for resource estimation in the paper notebooks."""

    def __init__(
        self,
        task: StarTaskFormat,
        gate_error_rate: float = 1e-3,
        p_cult: float = 2e-9,
        num_repeat: int = 1,
        cutoff_angle: Optional[float] = None,
        layout: str = "compact",
    ):
        assert gate_error_rate < ERROR_THRESHOLD, (
            "ValueError: `gate_error_rate` is higher than the surface-code threshold."
        )
        assert num_repeat >= 1, "ValueError: `num_repeat` should be a positive integer."

        self.task = task
        self.gate_error_rate = gate_error_rate
        self.p_cult = p_cult
        self.num_repeat = num_repeat
        self.cutoff_angle = cutoff_angle
        self.layout = layout

        self.code_distance = None
        self.k = None
        self.alpha_list = None

        self._set_k_parameter()
        self._load()

    def _set_k_parameter(self) -> None:
        self.code_distance = estimate_distance(
            self.task,
            self.gate_error_rate,
            self.num_repeat,
            self.layout,
        )
        self.k = math.ceil(self.code_distance / 3)

    def _load(self) -> None:
        _, _, _, _, alpha_list = preprocessing(
            self.task,
            self.k,
            self.gate_error_rate,
            self.p_cult,
            self.cutoff_angle,
        )
        self.alpha_list = alpha_list

    def estimate_resource(self, epsilon: float = 1e-2) -> tuple[int, float, float]:
        """Estimate physical qubits, single-shot runtime, and total runtime."""

        num_phys_qubit = physical_qubit_count(
            self.task,
            self.gate_error_rate,
            self.num_repeat,
            self.layout,
        )
        single_shot_runtime = estimate_single_shot_time(
            self.task,
            self.gate_error_rate,
            self.num_repeat,
            self.layout,
        )
        gamma = _mitigation_cost(
            self.task.angle_list,
            self.alpha_list,
            self.gate_error_rate,
            self.num_repeat,
        )
        total_runtime = single_shot_runtime * (gamma / epsilon) ** 2
        return num_phys_qubit, single_shot_runtime, total_runtime
