"""Utilities for building a circuit-level noise model with Stim."""

import stim


def get_qubit_set(circuit: stim.Circuit) -> set[int]:
    """Return the set of qubit labels used in the input circuit."""

    qubit_set = set()
    for i in range(len(circuit)):
        if circuit[i].name == "QUBIT_COORDS":
            qubit_set.add(circuit[i].targets_copy()[0].qubit_value)
    return qubit_set


def decompose_MR(circuit: stim.Circuit) -> stim.Circuit:
    """Replace each ``MR`` instruction with ``M`` + ``R``."""

    new_circuit = stim.Circuit()
    for i in range(len(circuit)):
        if circuit[i].name == "MR":
            target_list = [gate_target.qubit_value for gate_target in circuit[i].targets_copy()]
            new_circuit.append("M", target_list)
            new_circuit.append("TICK")
            new_circuit.append("R", target_list)
        else:
            new_circuit.append(circuit[i])
    return new_circuit


def repeat_block_expansion(circuit: stim.Circuit) -> stim.Circuit:
    """Expand repeat blocks contained in the input circuit."""

    repeat_index = None
    for i in range(len(circuit)):
        if circuit[i].name == "REPEAT":
            repeat_index = i

    circuit_new = stim.Circuit()
    if repeat_index is not None:
        circuit_new += circuit[:repeat_index]
        circuit_rep = circuit[repeat_index].body_copy()
        repeat_count = circuit[repeat_index].repeat_count
        for _ in range(repeat_count):
            circuit_new += circuit_rep
        circuit_new += circuit[repeat_index + 1 :]
    else:
        circuit_new += circuit

    return circuit_new


def layer_decomposition(circuit: stim.Circuit) -> list[stim.Circuit]:
    """Decompose a circuit into a list of layers separated by ``TICK``."""

    layer_list = []
    start = -1

    # Remove a leading TICK if present.
    if circuit[0].name == "TICK":
        circuit_temp = circuit[1:]
    else:
        circuit_temp = circuit

    for i in range(len(circuit_temp)):
        if circuit_temp[i].name == "TICK":
            layer_list.append(circuit_temp[start + 1 : i])
            start = i
    layer_list.append(circuit_temp[start + 1 :])

    return layer_list


def get_target_in_layer(layer: stim.Circuit) -> set[int]:
    """Return the target qubits in one layer after checking for overlap."""

    target_temp = set()
    ignored_ops = {"DETECTOR", "MPAD", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS", "TICK"}

    for i in range(len(layer)):
        if layer[i].name not in ignored_ops:
            target_list = {gate_target.qubit_value for gate_target in layer[i].targets_copy()}
            if len(target_temp & target_list) == 0:
                target_temp |= target_list
            else:
                raise ValueError("ERROR: Input layer includes overlapping target qubits.")

    return target_temp


def connect_layers(layer_list: list[stim.Circuit]) -> stim.Circuit:
    """Reconnect a list of layers into a single circuit with ``TICK`` separators."""

    circuit = stim.Circuit()
    for layer in layer_list:
        circuit += layer
        circuit.append("TICK")

    return circuit[:-1]


def add_noise_to_layer(
    layer: stim.Circuit,
    qubit_set: set[int],
    p_1q=0,
    p_2q=0,
    p_mes=0,
    p_reset=0,
    p_idle=0,
) -> stim.Circuit:
    """Insert circuit-level noise into a single layer."""

    noisy_layer = stim.Circuit()

    # Add idling errors.
    target_set = get_target_in_layer(layer)
    idling_qubit = qubit_set - target_set
    noisy_layer.append("DEPOLARIZE1", idling_qubit, p_idle)

    for i in range(len(layer)):
        if layer[i].name in ["I", "X", "Y", "Z", "H", "S", "S_DAG"]:
            noisy_layer.append(layer[i])
            target_list = [gate_target.qubit_value for gate_target in layer[i].targets_copy()]
            noisy_layer.append("DEPOLARIZE1", target_list, p_1q)

        elif layer[i].name in ["CNOT", "CX", "CY", "CZ", "CXSWAP", "SWAP", "ISWAP"]:
            noisy_layer.append(layer[i])
            target_list = [gate_target.qubit_value for gate_target in layer[i].targets_copy()]
            noisy_layer.append("DEPOLARIZE2", target_list, p_2q)

        elif layer[i].name in ["M", "MZ"]:
            target_list = [gate_target.qubit_value for gate_target in layer[i].targets_copy()]
            noisy_layer.append("X_ERROR", target_list, p_mes)
            noisy_layer.append(layer[i])

        elif layer[i].name in ["MR"]:
            target_list = [gate_target.qubit_value for gate_target in layer[i].targets_copy()]
            noisy_layer.append("ERROR_X", target_list, p_mes)
            noisy_layer.append(layer[i])
            noisy_layer.append("X_ERROR", target_list, p_reset)

        elif layer[i].name in ["R", "RZ"]:
            target_list = [gate_target.qubit_value for gate_target in layer[i].targets_copy()]
            noisy_layer.append(layer[i])
            noisy_layer.append("X_ERROR", target_list, p_reset)

        else:
            noisy_layer.append(layer[i])

    return noisy_layer


def generate_circuit_level_noise_model(
    circuit: stim.Circuit,
    qubit_set=None,
    p_1q=0.0,
    p_2q=0.0,
    p_mes=0.0,
    p_reset=0.0,
    p_idle=0.0,
) -> stim.Circuit:
    """Convert a circuit into its circuit-level noisy counterpart."""

    circuit_ideal = circuit.without_noise()
    expanded_circuit = circuit_ideal.flattened()
    expanded_circuit = decompose_MR(expanded_circuit)

    if qubit_set is None:
        qubit_set = get_qubit_set(expanded_circuit)

    layer_list = layer_decomposition(expanded_circuit)
    noisy_layer_list = []
    for layer in layer_list:
        noisy_layer_list.append(
            add_noise_to_layer(layer, qubit_set, p_1q, p_2q, p_mes, p_reset, p_idle)
        )

    return connect_layers(noisy_layer_list)
