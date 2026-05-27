"""
NOTE: In this file, I have redefined the rotation angle to conform to the notation of the paper.
NOTE: Therefore, when you integrate this file to STAR simulator, please modify the definition of angles, and check the consistency. 
"""


import sys
import os
import pathlib
import numpy as np

__file__ = os.path.abspath('')    #`__file__` is not defined on jupyter file (not needed in python file)
src_path = pathlib.Path(__file__).parent / 'src'
assert src_path.exists()
sys.path.append(str(src_path))
from starsim.core.preprocess import generate_error_parameters
from starsim.core.injection_module import calc_supply_rate_per_clock


# Number of shots for estimating the supply rate.
NUM_SHOTS = 1000  #100_000


def calc_tradeoff_for_mutation(d:int, theta:float, theta_th:float, p_ph:float = 1e-3, p_cult:float = 2e-9, t_cult:float = 10.):
    """Calculate the execution time and error rate of the STAR-magic mutation. Assume the fast block.

    Args:
        d (int): Code distance.
        theta (float): Target rotation angle. Note that this angle is defined as Rz(θ)=exp(iθZ).
        theta_th (float): Threshold angle for switching to T-gate synthesis. Note that this angle is defined as Rz(θ)=exp(iθZ).
        p_ph (float, optional): Physical error rate. Defaults to 1e-3
        p_cult (float, optional): Error rate of magic state cultivation. Defaults to 2e-9.
        t_cult (float, optional): Average execution time of magic state cultivation [clock]. Defaults to 10.
    Returns:
        Tuple[float, float]: Total error rate and total execution time.
    
    """

    # Determine the parameter k for the TMR protocol
    k = np.ceil(d/3)

    # Threshold of RUS trials
    N_rus = int(np.ceil(np.log2(theta_th/theta)))

    # calculate the accumulated error rate in the analog stage
    _, _, alpha_RUS = generate_error_parameters(2*theta, N_rus = 30, k = k, p_ph=p_ph, p_cult=0, theta_th = 2*theta_th)   # Factor "2" is needed for conforming the definition of angles.
    p_analog = 2 * alpha_RUS * p_ph * theta    # Factor "2" is needed for conforming the definition of angles.

    # Set an accuracy for T-gate synthesis 
    epsilon = max([0.1 * p_analog * (2**N_rus), p_cult])

    # calculate the number of T gates required in the digital stage
    N_t = np.ceil(3 * np.log2(1/epsilon))

    # Evaluate the total error rate
    p_total = p_analog + ((1/2)**N_rus) * (N_t * p_cult + epsilon)

    # Evaluate the total runtime
    t_total = 0
    num_shots = NUM_SHOTS   # Number of shots for estimating the supply rate.
    theta_list = np.array([theta * (2**i) for i in range(N_rus)])    # List of rotation angles for each RUS trial. Factor "2" is needed for conforming the definition of angles.
    p_tmr_list = calc_supply_rate_per_clock(d, num_shots, p_ph, 2*theta_list)    # Execution time of TMR protocol [clock]
    for i in range(N_rus):
        t_temp = 0
        for j in range(i+1):
            t_temp += 1/p_tmr_list[j]/2 + 1  #The factor "1/2" is needed since we utilize two patches to prepare a resource state.
        t_total += ((1/2)**(i+1)) * t_temp    # Prepare a resource state with t_tmr [clock], then perform an analog rotation with 1 [clock]. The factor "2" is needed since we utilize two patches to prepare a resource state.
    t_total += ((1/2)**N_rus) * (t_cult/2 + 1) * N_t    # Prepare a magic state with t_cult [clock], then perform Z-rotation with 1 [clock]. The factor "1/2" is needed since we utilize two patches to prepare a magic state.

    return p_total, t_total




def calc_tradeoff_for_mutation_without_preparation(d:int, theta:float, theta_th:float, p_ph:float = 1e-3, p_cult:float = 2e-9, t_cult:float = 10.):
    """Calculate the execution time and error rate of the STAR-magic mutation. Assume the fast block.

    Args:
        d (int): Code distance.
        theta (float): Target rotation angle. Note that this angle is defined as Rz(θ)=exp(iθZ).
        theta_th (float): Threshold angle for switching to T-gate synthesis. Note that this angle is defined as Rz(θ)=exp(iθZ).
        p_ph (float, optional): Physical error rate. Defaults to 1e-3
        p_cult (float, optional): Error rate of magic state cultivation. Defaults to 2e-9.
        t_cult (float, optional): Average execution time of magic state cultivation [clock]. Defaults to 10.
    Returns:
        Tuple[float, float]: Total error rate and total execution time.
    
    """

    # Determine the parameter k for the TMR protocol
    k = np.ceil(d/3)

    # Threshold of RUS trials
    N_rus = int(np.ceil(np.log2(theta_th/theta)))

    # calculate the accumulated error rate in the analog stage
    _, _, alpha_RUS = generate_error_parameters(2*theta, N_rus = 30, k = k, p_ph=p_ph, p_cult=0, theta_th = 2*theta_th)   # Factor "2" is needed for conforming the definition of angles.
    p_analog = 2 * alpha_RUS * p_ph * theta    # Factor "2" is needed for conforming the definition of angles.

    # Set an accuracy for T-gate synthesis 
    epsilon = 0.1 * p_analog * (2**N_rus)

    # calculate the number of T gates required in the digital stage
    N_t = np.ceil(3 * np.log2(1/epsilon))

    # Evaluate the total error rate
    p_total = p_analog + ((1/2)**N_rus) * (N_t * p_cult + epsilon)

    # Evaluate the total runtime
    t_total = 0
    for i in range(N_rus):
        t_temp = 0
        for j in range(i+1):
            t_temp += 1 
        t_total += ((1/2)**(i+1)) * t_temp    #Perform an analog rotation with 1 [clock].
    t_total += ((1/2)**N_rus) * N_t    # Perform Z-rotation with 1 [clock]. 

    return p_total, t_total





def calc_tradeoff_for_cultivation(epsilon:float, p_ph:float = 1e-3, p_cult:float = 2e-9, t_cult:float = 10.):

    # calculate the number of T gates required for T-gate synthesis
    N_t = np.ceil(3 * np.log2(1/epsilon))

    # Evaluate the total error rate
    p_total = N_t * p_cult + epsilon

    # Evaluate the total runtime.
    # The factor "2" is needed since we utilize two patches to prepare a magic state.
    t_total = (t_cult/2 + 1) * N_t

    return p_total, t_total
