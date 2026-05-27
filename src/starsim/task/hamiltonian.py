"""Minimal Hamiltonian helpers needed by the paper notebooks."""


def get_term_list(ham) -> tuple[list[list[int]], list[list[int]], list[float]]:
    """Extract targets, Pauli IDs, and coefficients from a Qulacs observable."""

    num_term = ham.get_term_count()
    term = [ham.get_term(i) for i in range(num_term)]
    all_index_list = [term[i].get_index_list() for i in range(num_term)]
    all_pauli_ids = [term[i].get_pauli_id_list() for i in range(num_term)]
    all_coefs = [term[i].get_coef() for i in range(num_term)]
    return all_index_list, all_pauli_ids, all_coefs
