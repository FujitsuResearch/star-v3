"""Injection-protocol utilities used by the paper notebooks.

In this file we ignore the sign of the target rotation and always work with the
positive angle. The sign of the resource-state rotation can be absorbed by the
freedom in gate teleportation, so only the relative sign between the target
angle and the error angle matters. In actual hardware implementations the sign
still has to be handled explicitly.

Rotation angles are defined by ``R(theta) = exp(-i theta Z / 2)``, so the
factor of one half must be treated consistently.
"""


import numpy as np
import stim
import scipy.optimize as optimize
from starsim.core.parameter import ERROR_COEF_V2
from starsim.core.circuit_level import generate_circuit_level_noise_model, get_qubit_set
from typing import List


def physical_to_logical_angle_conversion(physical_angle:float, k:int):
    temp = np.sin(physical_angle/2)**k / np.sqrt( np.cos(physical_angle/2)**(2*k) + np.sin(physical_angle/2)**(2*k) )
    return np.abs(2*np.arcsin(temp))


def logical_to_physical_angle_conversion(target_angle:float, k:int):
    obj_func = lambda theta: np.abs(physical_to_logical_angle_conversion(theta,k) - np.abs(target_angle))

    res = optimize.minimize_scalar(obj_func, bounds=(0, np.pi/2))

    return res.x


# Probability of sampling one Pauli string with weight ``N_flip`` or ``k - N_flip``.
def sample_probability(k:int, theta, N_flip):   
    u_b = np.multiply(np.power(np.cos(theta/2),k-N_flip), np.power(np.sin(theta/2),N_flip))
    v_b = np.multiply(np.power(np.cos(theta/2),N_flip), np.power(np.sin(theta/2),k-N_flip))
    return u_b**2+v_b**2  



# Rotation angle of the resource state when a Pauli string with weight ``N_flip`` is sampled.
# The sign must always be adjusted so that it is opposite to the target angle.
def magic_angle(theta_L: float, k:int, N_flip: list):
    theta_ph = logical_to_physical_angle_conversion(theta_L, k)
    u_b = np.multiply((np.cos(theta_ph/2))**(k-N_flip), (np.sin(theta_ph/2))**(N_flip))
    v_b = np.multiply((np.cos(theta_ph/2))**(N_flip), (np.sin(theta_ph/2))**(k-N_flip))
    norm = np.sqrt(u_b**2 + v_b**2)
    return (-1)**N_flip * np.sign(theta_L) * 2 * np.abs(np.arcsin(v_b/norm))


#Calculate Δ_θ in the original paper (Toshio2024)
def angle_gap(theta_ph: float, k:int):
    return magic_angle(theta_ph, k, 1) - magic_angle(theta_ph, k, 0)


# Calculate the proportionality coefficient in P_L against p_ph in Eq.(16) : P_L ∝ p_ph (NOTE:neglect O(p_ph) term in p_suc, which means that p_suc ~ p_ideal)
def rotation_error_coef(theta_ph:float, k:int):
    p_ideal = sample_probability(k, theta_ph, 0) 
    p_error = sample_probability(k, theta_ph, 1)

    return ERROR_COEF_V2 * k * p_error / p_ideal


# Calculate the proportionality coefficient in the diamond norm in Eq.(22): ε_♢ ∝ p_ph
def effective_error_coef(theta_L, k:int):
    theta_ph = logical_to_physical_angle_conversion(theta_L, k)
    error_coef = rotation_error_coef(theta_ph, k)
    error_angle = magic_angle(theta_L, k, 1)
    delta = error_angle - theta_L
    #NOTE: The definition of angle differs from original paper.
    effective_error_coef = 2 * error_coef * np.square(np.sin(delta/2))

    return effective_error_coef



def Find_MR(circuit):
    """Find the "MR" in the circuit and return the array of qubit values after MR."""
    for i in range(len(circuit)):
        if "MR" == circuit[i].name:
            N_rep = i
    array = [circuit[N_rep].targets_copy()[i].qubit_value for i in range(len(circuit[N_rep].targets_copy()))]
    return array



def find_repeat_block(circuit):
    for i in range(len(circuit)):
        if "REPEAT" == circuit[i].name:
            N_rep = i
    return N_rep



def generate_multirotation_rotated_circuit(d:int, rounds:int, pauli_string: list, list_multirotation: list, p_1q=0.0, p_2q=0.0, p_mes=0.0, p_reset=0.0, p_idle=0.0):

    """
    arg "d" can be setted in odd numbers.  
    list_multirotation (new arg) : list of m. 
    ex.) list_multirotation = [3,3,3,2] (assume d=11)
    """
    # assertion 
    k = len(list_multirotation)
    if not len(pauli_string) == k:
        print("The length of pauli-string is incorrect.")
        return None
    
    if not sum(list_multirotation) == d:
        print("incorrect input of list_multirotation.")
        return None

    # Generate the surface-code circuit.
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        rounds=rounds,
        distance=d)

    # Obtain the qubit domain.
    qubit_set = get_qubit_set(circuit)

    # Record the repeat-block location.
    N_rep = find_repeat_block(circuit)

    # First half of the circuit.
    circuit1 = circuit[:N_rep]
    circuit1 = generate_circuit_level_noise_model(circuit1, qubit_set, p_1q, p_2q, p_mes, p_reset, p_idle)

    #--------------------Transversal rotation--------------------
    
    # specify data qubits and grouping with m
    # list of qubit groups applied by multi-Pauli rotation
    target_data_qubits: List[List[int]] = []
    for i in range(len(list_multirotation)):
        offset = sum(list_multirotation[:i])
        target_data_qubits.append([2*offset + 2*j + 1  for j in range(list_multirotation[i])])
    #print("multi Pauli decomposition:")
    #print(target_data_qubits)
    #print(qubit_set)

    circuit2 = stim.Circuit()
    circuit2.append("TICK")

    #step1 SWAP
    swap_pair = []
    for i in range(len(target_data_qubits)):
        if(len(target_data_qubits[i]) == 2):
            pair = [target_data_qubits[i][0], target_data_qubits[i][0]+1] 
            swap_pair.append(pair)
        elif(len(target_data_qubits[i]) == 3):
            pair1 = [target_data_qubits[i][0], target_data_qubits[i][0]+1] 
            pair2 = [target_data_qubits[i][-1]-1, target_data_qubits[i][-1]] 
            swap_pair.append(pair1)
            swap_pair.append(pair2)
    #print(swap_pair)
    #print(sum(swap_pair,[]))
    circuit2.append("DEPOLARIZE1", qubit_set - set(sum(swap_pair,[])), p_idle)
    circuit2.append("SWAP", set(sum(swap_pair,[])))
    circuit2.append("DEPOLARIZE2", set(sum(swap_pair,[])), p_2q)
    circuit2.append("TICK")
    
    #step2 CNOT
    cnot_pair = []
    for i in range(len(target_data_qubits)):
        if(len(target_data_qubits[i]) == 2):
            pair = [target_data_qubits[i][0]+1, target_data_qubits[i][1]] 
            cnot_pair.append(pair)
            circuit2.append("CNOT", pair)
            circuit2.append("DEPOLARIZE2", pair, p_2q)
        elif(len(target_data_qubits[i]) == 3):
            pair = [target_data_qubits[i][0]+1, target_data_qubits[i][1]] 
            cnot_pair.append(pair)
            circuit2.append("CNOT", pair)
            circuit2.append("DEPOLARIZE2", pair, p_2q)
    #print(cnot_pair)
    #print(sum(cnot_pair,[]))
    circuit2.append("DEPOLARIZE1", qubit_set - set(sum(cnot_pair,[])), p_idle)
    circuit2.append("TICK")
    
    #step3 CNOT (m = 3) or Rz (m = 2)
    cnot_pair = []
    for i in range(len(target_data_qubits)):
        if(len(target_data_qubits[i]) == 2): # not pair, just a single qubit (Rz)
            pair = [target_data_qubits[i][1]] 
            cnot_pair.append(pair)
            circuit2.append("Z_ERROR", target_data_qubits[i][1], pauli_string.astype(np.int32)[i])
            circuit2.append("DEPOLARIZE1", target_data_qubits[i][1], p_1q)
        elif(len(target_data_qubits[i]) == 3): # CNOT
            pair = [target_data_qubits[i][1], target_data_qubits[i][1]+1] 
            cnot_pair.append(pair)
            circuit2.append("CNOT", pair)
            circuit2.append("DEPOLARIZE2", pair, p_2q)
    #print(cnot_pair)
    #print(sum(cnot_pair,[]))
    circuit2.append("DEPOLARIZE1", qubit_set - set(sum(cnot_pair,[])), p_idle)
    circuit2.append("TICK")

    #step4 Rz (m = 3) or CNOT (m = 2)
    cnot_pair = []
    for i in range(len(target_data_qubits)):
        if(len(target_data_qubits[i]) == 2): # CNOT
            pair = [target_data_qubits[i][0]+1, target_data_qubits[i][1]] 
            cnot_pair.append(pair)
            circuit2.append("CNOT", pair)
            circuit2.append("DEPOLARIZE2", pair, p_2q)
        elif(len(target_data_qubits[i]) == 3): # not pair, just a single qubit (Rz)
            pair = [target_data_qubits[i][1]+1] 
            cnot_pair.append(pair)
            circuit2.append("Z_ERROR", target_data_qubits[i][1]+1, pauli_string.astype(np.int32)[i])
            circuit2.append("DEPOLARIZE1", target_data_qubits[i][1]+1, p_1q)
    #print(cnot_pair)
    #print(sum(cnot_pair,[]))
    circuit2.append("DEPOLARIZE1", qubit_set - set(sum(cnot_pair,[])), p_idle)
    circuit2.append("TICK")

    #step5 CNOT (m = 3) or SWAP (m = 2)
    cnot_pair = []
    for i in range(len(target_data_qubits)):
        if(len(target_data_qubits[i]) == 2): # SWAP
            pair = [target_data_qubits[i][0], target_data_qubits[i][0]+1] 
            cnot_pair.append(pair)
            circuit2.append("SWAP", pair)
            circuit2.append("DEPOLARIZE2", pair, p_2q)
        elif(len(target_data_qubits[i]) == 3): # CNOT
            pair = [target_data_qubits[i][1],  target_data_qubits[i][1]+1] 
            cnot_pair.append(pair)
            circuit2.append("CNOT", pair)
            circuit2.append("DEPOLARIZE2", pair, p_2q)
    #print(cnot_pair)
    #print(sum(cnot_pair,[]))
    circuit2.append("DEPOLARIZE1", qubit_set - set(sum(cnot_pair,[])), p_idle)
    circuit2.append("TICK")

    if(np.max(list_multirotation) > 2.5):
        #step6 CNOT (m = 3) 
        cnot_pair = []
        for i in range(len(target_data_qubits)):
            if(len(target_data_qubits[i]) == 3): 
                pair = [target_data_qubits[i][0]+1,  target_data_qubits[i][0]+2] 
                cnot_pair.append(pair)
                circuit2.append("CNOT", pair)
                circuit2.append("DEPOLARIZE2", pair, p_2q)
        #print(cnot_pair)
        #print(sum(cnot_pair,[]))
        circuit2.append("DEPOLARIZE1", qubit_set - set(sum(cnot_pair,[])), p_idle)
        circuit2.append("TICK")

        #step7 SWAP (m = 3)
        swap_pair = []
        for i in range(len(target_data_qubits)): 
            if(len(target_data_qubits[i]) == 3): 
                pair1 = [target_data_qubits[i][0], target_data_qubits[i][0]+1] 
                pair2 = [target_data_qubits[i][-1]-1, target_data_qubits[i][-1]] 
                swap_pair.append(pair1)
                swap_pair.append(pair2)
                circuit2.append("SWAP", pair1)
                circuit2.append("DEPOLARIZE2", pair1, p_2q)
                circuit2.append("SWAP", pair2)
                circuit2.append("DEPOLARIZE2", pair2, p_2q)
        #print(swap_pair)
        #print(sum(swap_pair,[]))
        circuit2.append("DEPOLARIZE1", qubit_set - set(sum(swap_pair,[])), p_idle)
        circuit2.append("TICK")
    
    #-------------------------------------------------------



    # Second half of the circuit.
    # maybe it's better to insert reset operations on measurement qubits here?
    circuit3 = stim.Circuit()
    reset_qubits = Find_MR(circuit)
    circuit3.append("R", reset_qubits)
#     circuit3.append("X_ERROR", reset_qubits)
#     circuit3.append("DEPOLARIZE1", qubit_set - set(reset_qubits), p_idle)
    circuit3 += circuit[N_rep:]
    circuit3 = generate_circuit_level_noise_model(circuit3, qubit_set, p_1q, p_2q, p_mes, p_reset, p_idle)

    # Combine all circuit pieces.
    injection_circuit = circuit1 + circuit2 + circuit3

    return injection_circuit, circuit2




def calc_success_rate_with_EC(circuit: stim.Circuit, d:int, num_shots: int) -> int:
    """Calculate the success rate of post-selection in TMR protocol within the error-correction mode.
    NOTE: The total success rate of TMR protocol is roughly given by "the success rate of the post-selection" * "sampling probability of II...I or ZZ...Z".
    
    When d is odd, the total number of detectors within one round is (d**2-1).
    In other words, in each round, there are N_syn = (d**2-1)//2 stabilizers for X and Z errors, respectively.
    However, at the first round, only X-stabilizer contributes to the detectors.

    Args:
        circuit (stim.Circuit): input circuit for TMR protocol
        d (int): odd code distance
        num_shots (int): number of shots for sampling

    """

    assert d % 2 == 1, "d must be odd number."

    # Sample the circuit.
    sampler = circuit.compile_detector_sampler()
    detection_events, _ = sampler.sample(num_shots, separate_observables=True)

    #--------------------Post-selected syndrome set---------------------------
    N_syn = (d**2-1)//2  # Number of X- or Z-stabilizers per round

    # post-selected syndrome set for 1st-round
    ps_syndrome1 = set(np.arange(0,N_syn,d+1)) | set(np.arange(1,N_syn,d+1)) | set(np.arange((d+1)//2,N_syn,d+1))

    # post-selected syndrome set for 2nd-round
    ps_syndrome2 = set(np.arange(N_syn,N_syn+2*d+(d-1)//2))

    # post-selected syndrome set for 3rd-round
    ps_syndrome3 = set(np.arange(3*N_syn,3*N_syn+2*d+(d-1)//2))

    # total post-selected syndrome set
    ps_syndrome = ps_syndrome1 | ps_syndrome2 | ps_syndrome3

    # Count the number of successful shots
    N_success = 0
    for shot in range(num_shots):
        #post-selection
        if not any(detection_events[shot][i] for i in ps_syndrome):
            N_success += 1

    return N_success/num_shots



def calc_supply_rate_per_clock(d:int, num_shots:int, p_ph:float, theta_list: np.ndarray):

    # set a list to store success rates
    p_suc_list = np.zeros_like(theta_list)

    # Set error parameters
    p_dict = {"p_1q": p_ph, 
            "p_2q": p_ph, 
            "p_mes": p_ph, 
            "p_reset": p_ph, 
            "p_idle": p_ph}

    # Determine the form of multi-rotation from code distance d
    if d%3 ==0:
        list_multirot = [3 for _ in range(d//3)]
    elif d%3 ==1:
        list_multirot = [2, 2] + [3 for _ in range((d-4)//3)]
    elif d%3 ==2:
        list_multirot = [2] + [3 for _ in range((d-2)//3)]
    k = len(list_multirot)

    injection_circuit, _ = generate_multirotation_rotated_circuit(d, rounds=3, pauli_string=np.array([False for _ in range(k)]), list_multirotation=list_multirot, **p_dict)
    p_suc = calc_success_rate_with_EC(injection_circuit, d, num_shots) 

    for i, theta in enumerate(theta_list):
        theta_ph = logical_to_physical_angle_conversion(theta, k)
        p_sample = sample_probability(k,theta_ph,0)
        p_suc_list[i] = p_sample * p_suc * d/4

    return p_suc_list
