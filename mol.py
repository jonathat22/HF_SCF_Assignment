"""
Mock mol class
"""

import numpy as np
import pickle


class mol:
    """
    class mol

    Contains the information about the molecule
    """

    def __init__(self, atom_string_, nao_, nelect_, nmo_):
        """
        mol Creator - creates an instance of the mol class

        Arguments:
          atoms_string_: A string for defining the atoms in the molecule
          nao_: Number of atomic orbitals
          nelec_: Number of electrons / 2
          nmo_: Number of molecular orbitals

        Returns:
            instance of mol
        """
        self.nao = nao_
        self.nelec = [nelect_, nelect_]
        self.nmo = nmo_
        self.charges = []
        coords_list = []
        atom_list = atom_string_.split(";")
        for a in atom_list:
            print(a.split()[2])
            self.charges.append(int(a.split()[0]))
            coords_list.append([float(a.split()[1]),
                                float(a.split()[2]),
                                float(a.split()[3])])
        self.coords = np.array(coords_list)

    def atom_charges(self):
        return self.charges

    def atom_coords(self):
        return self.coords

    def intor(self, int_type_str_=None):
        if int_type_str_ == "int1e_ovlp":
            return pickle.load(open("suv.pkl", "rb"))
        elif int_type_str_ == "int1e_kin":
            return pickle.load(open("tuv.pkl", "rb"))
        elif int_type_str_ == "int1e_nuc":
            return pickle.load(open("vuv.pkl", "rb"))
        elif int_type_str_ == "int2e":
            return pickle.load(open("eri.pkl", "rb"))
        else:
            raise Exception("Unrecognized integral type")
