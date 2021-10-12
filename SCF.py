"""
SCF.py is a module that contains all of the functions
for the HF SCF Procedure
"""

import numpy as np
import scipy.linalg as sp_la


def calc_nuclear_repulsion_energy(mol_):
    """
    calc_nuclear_repulsion_energy - calculates the n-e repulsion energy of
    molecule

    Arguments:
        mol_: the PySCF molecule data structure created from Input

    Returns:
        Enuc: The n-e repulsion energy
    """
    charges = mol_.atom_charges()
    coords = mol_.atom_coords()
    Enuc = 0.0
    distance_matrix = np.zeros((3, 3), dtype=np.double)

    for i in range(3):
        for j in range(3):
            distance_matrix[i][j] = np.linalg.norm(coords[i] - coords[j])

    for i in range(2):
        for j in range(i + 1, 3):
            Enuc += (charges[i] * charges[j]) / distance_matrix[i][j]

    return Enuc


def calc_initial_density(mol_):
    """
    calc_initial_density - Function to calculate the initial guess density

    Arguments
        mol_: the PySCF molecule data structure created from Input

    Returns:
        Duv: the (mol.nao x mol.nao) Guess Density Matrix
    """

    num_aos = mol_.nao  # Number of atomic orbitals, dimensions of the mats
    Duv = np.zeros((num_aos, num_aos), dtype=np.double)

    return Duv


def calc_hcore_matrix(Tuv_, Vuv_):
    """
    calc_hcore_matrix - Computes the 1 electron core matrix

    Arguments:
        Tuv_: The Kinetic Energy 1e integral matrix
        Vuv_: The Nuclear Repulsion 1e integrals matrix

    Returns:
        h_core: The one electron hamiltonian matrix
    """

    h_core = Tuv_ + Vuv_
    return h_core


def calc_fock_matrix(mol_, h_core_, er_ints_, Duv_):
    """
    calc_fock_matrix - Calculates the Fock Matrix of the molecule

    Arguments:
        mol_: the PySCF molecule data structure created from Input
        h_core_: the one electron hamiltonian matrix
        er_ints_: the 2e electron repulsion integrals
        Duv_: the density matrix

    Returns:
        Fuv: The fock matrix
    """

    Fuv = h_core_.copy()
    num_aos = mol_.nao
    for mu in range(mol_.nao):
        for nu in range(mol_.nao):
            Fuv[mu, nu] += ((Duv_*er_ints_[mu, nu]).sum()) \
                - 0.5 * (Duv_*er_ints_[mu, :, nu]).sum()
    return Fuv


def solve_Roothan_equations(Fuv_, Suv_):
    """
    solve_Roothan_equations - Solves the matrix equations to determine
                              the MO coefficients

    Arguments:
        Fuv_: The Fock matrix
        Suv_: The overlap matrix

    Returns:
        mo_energies: an array that contains eigenvalues of the solution
        mo_coefficients: a matrix of the eigenvectors of the solution
    """

    mo_energies, mo_coeffs = sp_la.eigh(Fuv_, Suv_)
    return mo_energies.real, mo_coeffs.real


def form_density_matrix(mol_, mo_coeffs_):
    """
    form_dentsity_matrix - forms the density matrix from the eigenvectors

    Note: the loops are over the number of electrons / 2, not all of the
    atomic orbitals

    Arguments:
        mol_: the PySCF molecule data structure created from Input
        mo_coefficients: a matrix of the eigenvectors of the solution

    Returns:
        Duv: the density matrix
    """

    nelec = mol_.nelec[0]  # Number of occupied orbitals
    num_aos = mol_.nao  # Number of atomic orbitals, dimensions of the mats
    Duv = np.zeros((mol_.nao, mol_.nao), dtype=np.double)

    for mu in range(num_aos):
        for nu in range(num_aos):
            Duv[mu, nu] = 2 * (mo_coeffs_[mu, 0:nelec] *
                               mo_coeffs_[nu, 0:nelec]).sum()
    return Duv


def calc_tot_energy(Fuv_, Huv_, Duv_, Enuc_):
    """
    calc_total_energy - This function calculates the total energy of the
    molecular system

    Arguments:
        Fuv_: the current Fock Matrix
        Huv_: the core Hamiltonian Matrix
        Duv_: the Density Matrix that corresponds to Fuv_
        Enuc: the Nuclear Repulsion Energy

    Returns:
        Etot: the total energy of the molecule
    """
    Etot = 0.5 * (Duv_ * (Huv_ + Fuv_)).sum() + Enuc_

    return Etot
