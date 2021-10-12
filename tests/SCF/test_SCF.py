import pytest
import SCF
import pickle
from mol import mol


Suv = pickle.load(open("suv.pkl", "rb"))
Tuv = pickle.load(open("real_tuv.pkl", "rb"))
eri = pickle.load(open("real_eri.pkl", "rb"))
Vuv = pickle.load(open("real_vuv.pkl", "rb"))


def test_calc_nuclear_repulsion_energy(mol_h2o):
    assert SCF.calc_nuclear_repulsion_energy(mol_h2o) == 8.00236706181077
    "Nuclear Repulsion Energy Test (h2o) Failed"


def test_calc_initial_density(mol_h2o):
    Duv = SCF.calc_initial_density(mol_h2o)
    assert Duv.shape == (mol_h2o.nao, mol_h2o.nao)
    assert Duv.sum() == 0.0
    "Calculate Initial Density Test Failed"


def test_calc_hcore_matrix(mol_h2o):
    assert SCF.calc_hcore_matrix(Tuv, Vuv)[0, 0] == -32.57739541261037
    assert SCF.calc_hcore_matrix(Tuv, Vuv)[3, 4] == 0.0
    assert SCF.calc_hcore_matrix(Tuv, Vuv)[4, 3] == 0.0
    "Calculate h_core Matrix Test Failed"


def test_calc_fock_matrix(mol_h2o):
    Duv = SCF.calc_initial_density(mol_h2o)
    h_core = SCF.calc_hcore_matrix(Tuv, Vuv)

    assert SCF.calc_fock_matrix(mol_h2o, h_core, eri, Duv)[0, 0] == \
        pytest.approx(-32.57739541261037)

    assert SCF.calc_fock_matrix(mol_h2o, h_core, eri, Duv)[2, 5] == \
        pytest.approx(-1.6751501447185015)

    assert SCF.calc_fock_matrix(mol_h2o, h_core, eri, Duv)[5, 2] == \
        pytest.approx(-1.6751501447185015)
    "Calculate Fock Matrix Test Failed"


def test_solve_Roothan_equations(mol_h2o):
    Duv = SCF.calc_initial_density(mol_h2o)
    h_core = SCF.calc_hcore_matrix(Tuv, Vuv)
    Fuv = SCF.calc_fock_matrix(mol_h2o, h_core, eri, Duv)
    mo_energies, mo_coeffs = SCF.solve_Roothan_equations(Fuv, Suv)

    assert mo_energies == pytest.approx([-32.57830292, -8.08153571,
                                        -7.55008599, -7.36396923,
                                        -7.34714487, -4.00229867, -3.98111115])

    assert mo_coeffs[0, 0] == pytest.approx(-1.00154358e+00)
    assert mo_coeffs[0, 1] == pytest.approx(abs(-2.33624458e-01))
    assert mo_coeffs[0, 2] == pytest.approx(-4.97111543e-16)
    assert mo_coeffs[0, 3] == pytest.approx(-8.56842145e-02)
    assert mo_coeffs[0, 4] == pytest.approx(2.02299681e-29)
    assert mo_coeffs[0, 5] == pytest.approx(-4.82226067e-02)
    assert mo_coeffs[0, 6] == pytest.approx(-4.99600361e-16)
    "Solve Roothan Equations Test Failed"


def test_form_density_matrix(mol_h2o):
    Duv = SCF.calc_initial_density(mol_h2o)
    h_core = SCF.calc_hcore_matrix(Tuv, Vuv)
    Fuv = SCF.calc_fock_matrix(mol_h2o, h_core, eri, Duv)
    mo_energies, mo_coeffs = SCF.solve_Roothan_equations(Fuv, Suv)
    Duv_new = SCF.form_density_matrix(mol_h2o, mo_coeffs)

    assert Duv_new[0, 0] == pytest.approx(2.130023428655504)
    assert Duv_new[2, 5] == Duv_new[5, 2] == pytest.approx(-0.29226330209653156)
    "Form Density Matrix Test Failed"


def test_calc_tot_energy(mol_h2o):
    Duv = SCF.calc_initial_density(mol_h2o)
    h_core = SCF.calc_hcore_matrix(Tuv, Vuv)
    Fuv = SCF.calc_fock_matrix(mol_h2o, h_core, eri, Duv)
    mo_energies, mo_coeffs = SCF.solve_Roothan_equations(Fuv, Suv)
    Enuc = SCF.calc_nuclear_repulsion_energy(mol_h2o)

    Etot = SCF.calc_tot_energy(Fuv, h_core, Duv, Enuc)
    assert Etot == pytest.approx(8.0023670618)
    "Calculate Total Energy Test Failed"
