"""

NOTE: The definition of alpha_RUS differs from the one in original paper (Toshio2024).
Our alpha_RUS has half value of the one in the paper.
THis is because our definition of angle is half of the one in the original paper, which also changes the difinition of alpha_RUS:
    Definition of angles: R_z(θ)=e^{iθZ}  →  R_z(θ)=e^{-iθZ/2}
"""

from __future__ import annotations
from starsim.core.injection_module import logical_to_physical_angle_conversion, magic_angle, rotation_error_coef, effective_error_coef
from starsim.core.parameter import ERROR_COEF_V1, NUM_RUS
import numpy as np
from typing import Optional, Tuple, List

# To avoid circular import issues for StarTaskFormat
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from starsim.task.format import StarTaskFormat



def generate_error_parameters(theta_L:float, N_rus:int, k:int, p_ph:float, p_cult:float, theta_th: Optional[float] = None):
    """Given a single rotation angle, calculate error parameters up to N-th RUS processes.
    The parameter `p_th` controls the threshold of switching to T-gate decomposition.
    STAR ver.2 corresponds to the case of `theta_th = None`, and
    STAR ver.3 corresponds to the case of `theta_th != None`.

    Args:
        theta_L (float): Logical-level rotation angle. Note that this angle is defined as Rz(θ)=exp(-iθZ/2).
        N_rus (int): Maximum number of RUS trials.
        k (int): A parameter to describe transversal multi-rotation protocol.
        p_ph (float): Physical error rate.
        p_cult (float): Cultivation error rate.
        theta_th (Optional[float]): Threshold angle for switching to T-gate decomposition.

    Returns:
        Tuple[List[float], List[float], float]: 
            - List of error angles at each RUS trial. shape = (NUM_RUS,)
            - List of error coefficients at each RUS trial. shape = (NUM_RUS,)
            - α_RUS parameter that determines the strength of errors in the entire RUS process.
    """

    if theta_th != None and theta_th < np.pi:
        # Coefficient of error rate of single RUS trial with TMR protocol 
        coef_th = effective_error_coef(theta_th, k)
    else:
        # Set coef_th to make "coef_th > ERROR_COEF_V1" always true
        coef_th = ERROR_COEF_V1 + 1

    ###------------------Calculate P_L, θ_error------------------###
    error_angle_list = []
    error_coef_list = []
    theta_RUS = theta_L
    error_sum = 0  # Accumulated error during RUS processes
    alpha_RUS = 0  # α_RUS×θ: NOTE: Finally devide this parameter by theta_L to obtain α_RUS

    #If coef_th > ERROR_COEF_V1, we use STAR ver.2
    if coef_th > ERROR_COEF_V1:
        switching_to_v1 = False
        for i in range(N_rus):
            if not switching_to_v1:
                theta_ph = logical_to_physical_angle_conversion(theta_RUS, k)
                error_coef = rotation_error_coef(theta_ph, k)
                error_angle = magic_angle(theta_RUS, k, 1)
                delta = error_angle - theta_RUS
                #NOTE: The definition of angle differs from original paper.
                eff_error_coef = 2 * error_coef * np.square(np.sin(delta/2))

                if eff_error_coef > ERROR_COEF_V1:           
                    # Switch to Akahoshi protocol
                    switching_to_v1 = True
                    error_angle_list.append(0)
                    error_coef_list.append(ERROR_COEF_V1)
                    error_sum += ERROR_COEF_V1
                else:
                    # Use transversal multi-rotation (TMR) protocol
                    error_angle_list.append(error_angle)
                    error_coef_list.append(error_coef)
                    error_sum += eff_error_coef
            else:
                # Use Akahoshi protocol
                error_angle_list.append(np.pi)
                error_coef_list.append(ERROR_COEF_V1)
                error_sum += ERROR_COEF_V1

            alpha_RUS += np.power(0.5, i+1) * error_sum
            theta_RUS *= 2


    #If coef_th <= ERROR_COEF_V1, we use STAR ver.3
    else:
        for i in range(N_rus):  
            theta_ph = logical_to_physical_angle_conversion(theta_RUS, k)
            error_coef = rotation_error_coef(theta_ph, k)
            error_angle = magic_angle(theta_RUS, k, 1)
            delta = error_angle - theta_RUS
            #NOTE: The definition of angle differs from original paper.
            eff_error_coef = 2 * error_coef * np.square(np.sin(delta/2))

            if eff_error_coef >= coef_th:
                # Break for switching to T-gate decomposition
                error_angle_list.append(0)
                error_coef_list.append(0)
                N_th = i
                break
            else:
                # Use transversal multi-rotation protocol
                error_angle_list.append(error_angle)
                error_coef_list.append(error_coef)
                error_sum += eff_error_coef

            alpha_RUS += np.power(0.5, i+1) * error_sum
            theta_RUS *= 2
        
        ### Finally, add logical errors due to cultivation
        # Set an accuracy for T-gate synthesis
        p_analog = alpha_RUS * p_ph   #NOTE: Here alpha_RUS includes theta_L.
        epsilon = max([0.1 * p_analog * (2**N_th), p_cult])    
        
        # If p_analog=0, set epsilon to be a near-optimal value
        if epsilon == 0:
            if p_cult == 0:
                epsilon = 1e-100
            else:
                epsilon = p_cult * 0.1
        

        # calculate the number of T gates required in the digital stage
        N_t = np.ceil(3 * np.log2(1/epsilon))

        # Evaluate the total error rate
        # NOTE: Devide by p_ph to normalize with respect to p_ph
        p_digital = N_t * p_cult + epsilon
        alpha_RUS += ((1/2)**N_th) * p_digital / p_ph

    return error_angle_list, error_coef_list, alpha_RUS/np.abs(theta_L)


# Extract independent angle set, and then, give a mapping from angle_list to angle_set
def _extract_angle_set(task: StarTaskFormat):

    # independent angle set
    angle_set = []
    # list with same length as task.angle_list
    index_list = []

    for i, angle in enumerate(task.angle_list):
        if angle in angle_set:
            index_list.append(angle_set.index(angle))
        else:
            angle_set.append(angle)
            index_list.append(angle_set.index(angle))
    
    return angle_set, index_list


def preprocessing(task: StarTaskFormat, 
                  k:int, 
                  p_ph:float, 
                  p_cult:float,
                  theta_th: Optional[float] = None
                  ) -> Tuple[List[float], List[int], List[list[float]], List[list[float]], List[float]]:
    """Given a StarTaskFormat instance, generate some parameters to denote the strength or frequency of errors. 
    These paramters are needed for resource estimate and noisy circuit simulation. 

    Args:
        task (StarTaskFormat): StarTaskFormat instance to describe ideal quantum circuit.
        k (int): A parameter to describe transversal multi-rotation protocol.
        theta_th (Optional[float]): Threshold value of RUS angle for STAR ver.3. `None` means that we use the injection protocol of STAR ver.2. Defaults to None.

    Returns:
        angle_set (List[float]): lenth = # of independent angles = N_angle.
        angle_index_list (List[int]): lenth = (task.num_gate).
        error_angle_series_list (List[list[float]]): shape = (N_angle, NUM_RUS)
        error_coef_series_list (List[list[float]]): shape = (N_angle, NUM_RUS)
        alpha_list (List[float]): lenth = (task.num_gate).
    """
    angle_set, angle_index_list = _extract_angle_set(task)
    error_angle_series_list = []
    error_coef_series_list = []
    alpha_set = []

    for angle in angle_set:
        error_angle_series, error_coef_series, alpha_RUS = generate_error_parameters(angle, NUM_RUS, k, p_ph, p_cult, theta_th = theta_th)
        error_angle_series_list.append(error_angle_series)
        error_coef_series_list.append(error_coef_series)
        alpha_set.append(alpha_RUS)
    
    alpha_list = []
    for index in angle_index_list:
        alpha_list.append(alpha_set[index])

    return angle_set, angle_index_list, error_angle_series_list, error_coef_series_list, alpha_list