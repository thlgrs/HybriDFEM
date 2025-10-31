import time

import h5py

from Theo.Structures.Structure_2D import *


class Modal:

    def __init__(self, modes=None, no_inertia=False, filename="Results_Modal", dir_name="", save=True, initial=False):
        self.modes = modes
        self.no_inertia = no_inertia
        self.filename = filename
        self.dir_name = dir_name
        self.save = save
        self.initial = initial

    def modal(self, structure):
        time_start = time.time()

        structure.get_P_r()
        structure.get_M_str(no_inertia=self.no_inertia)

        if not self.initial:
            if not hasattr(structure, "K"):
                structure.get_K_str()

            if self.modes is None:
                # print('HEllo')
                # structure.K = np.around(structure.K,6)
                # structure.M = np.around(structure.M,8)
                omega, phi = sc.linalg.eig(
                    structure.K[np.ix_(structure.dof_free, structure.dof_free)],
                    structure.M[np.ix_(structure.dof_free, structure.dof_free)]
                )

            elif isinstance(self.modes, int):
                if np.linalg.det(structure.M) == 0:
                    warnings.warn(
                        "Might need to use linalg.eig if the matrix M is non-invertible"
                    )
                omega, phi = sc.sparse.linalg.eigsh(
                    structure.K[np.ix_(structure.dof_free, structure.dof_free)],
                    self.modes,
                    structure.M[np.ix_(structure.dof_free, structure.dof_free)],
                    which="SM",
                )

            else:
                warnings.warn("Required self.modes should be either int or None")
        else:
            structure.get_K_str0()
            if self.modes is None:
                omega, phi = sc.linalg.eigh(
                    structure.K0[np.ix_(structure.dof_free, structure.dof_free)],
                    structure.M[np.ix_(structure.dof_free, structure.dof_free)],
                )
            elif isinstance(self.modes, int):
                if np.linalg.det(structure.M) == 0:
                    warnings.warn(
                        "Might need to use linalg.eig if the matrix M is non-invertible"
                    )
                omega, phi = sc.sparse.linalg.eigsh(
                    structure.K0[np.ix_(structure.dof_free, structure.dof_free)],
                    self.modes,
                    structure.M[np.ix_(structure.dof_free, structure.dof_free)],
                    which="SM",
                )

            else:
                warnings.warn("Required self.modes should be either int or None")
        # print(omega)
        # for i in range(len(omega)):
        #     if omega[i] < 0: omega[i] = 0
        structure.eig_vals = np.sort(np.real(np.sqrt(omega))).copy()
        structure.eig_modes = (np.real(phi).T)[np.argsort((np.sqrt(omega)))].T.copy()
        # print(structure.eig_vals)

        if self.save:
            time_end = time.time()
            total_time = time_end - time_start
            print("Simulation done... writing results to file")

            self.filename = self.filename + ".h5"
            file_path = os.path.join(self.dir_name, self.filename)

            with h5py.File(file_path, "w") as hf:
                hf.create_dataset("eig_vals", data=structure.eig_vals)
                hf.create_dataset("eig_modes", data=structure.eig_modes)

                hf.attrs["Simulation_Time"] = total_time
        return structure

    @staticmethod
    def solve_modal(structure, modes=None, no_inertia=False, filename="Results_Modal",
                    dir_name="", save=True, initial=False):
        solver = Modal(modes, no_inertia, filename, dir_name, save, initial)
        return solver.modal(structure)
