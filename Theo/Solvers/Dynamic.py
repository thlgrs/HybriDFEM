import time
from copy import deepcopy

import h5py

from Theo.Structures.Structure_2D import *


class Dynamic:
    SOLVER_TYPE = "Dynamic solver"
    STRUCTURE_TYPE = Union[Structure_block, Hybrid]

    def __init__(self, T, dt, U0=None, V0=None, lmbda=None, Meth=None, filename="", dir_name=""):
        self.T = T
        self.dt = dt
        self.U0 = U0
        self.V0 = V0
        self.lmbda = lmbda
        self.Meth = Meth
        self.filename = filename
        self.dir_name = dir_name

    @staticmethod
    def impose_dyn_excitation(structure: Union[Structure_block, Hybrid], node, dof, U_app, dt):
        if 3 * node + dof not in structure.dof_fix:
            warnings.warn("Excited DoF should be a fixed one")

        if not hasattr(structure, "dof_moving"):
            structure.dof_moving = []
            structure.disp_histories = []
            structure.times = []

        structure.dof_moving.append(3 * node + dof)
        structure.disp_histories.append(U_app)
        structure.times.append(dt)

        # Later, add a function to interpolate when different timesteps are used.

    def linear(self, structure: Union[Structure_block, Hybrid]):
        time_start = time.time()

        structure.get_K_str0()
        structure.get_M_str()
        structure.get_C_str()

        if self.U0 is None:
            if np.linalg.norm(structure.U) == 0:
                self.U0 = np.zeros(structure.nb_dofs)
            else:
                self.U0 = deepcopy(structure.U)

        if self.V0 is None:
            V0 = np.zeros(structure.nb_dofs)

        if hasattr(structure, "times"):
            for timestep in structure.times:
                if timestep != self.dt:
                    warnings.warn(
                        "Unmatching timesteps between excitation and simulation"
                    )
            for i, disp in enumerate(structure.disp_histories):
                if self.U0[structure.dof_moving[i]] != disp[0]:
                    warnings.warn("Unmatching initial displacements")
        else:
            structure.dof_moving = []
            structure.disp_histories = []
            structure.times = []

        Time = np.arange(0, self.T, self.dt, dtype=float)
        Time = np.append(Time, self.T)
        nb_steps = len(Time)

        loading = np.zeros(nb_steps)

        if callable(self.lmbda):
            for i, t in enumerate(Time):
                loading[i] = self.lmbda(t)
        elif isinstance(self.lmbda, list):
            pass

        U_conv = np.zeros((structure.nb_dofs, nb_steps))
        V_conv = np.zeros((structure.nb_dofs, nb_steps))
        A_conv = np.zeros((structure.nb_dofs, nb_steps))
        P_conv = np.zeros((structure.nb_dofs, nb_steps))

        U_conv[:, 0] = self.U0.copy()
        V_conv[:, 0] = V0.copy()
        A_conv[:, 0] = sc.linalg.solve(
            structure.M, loading[0] * structure.P - structure.C @ V_conv[:, 0] - structure.K0 @ U_conv[:, 0]
        )

        self.Meth, P = structure.ask_method(self.Meth)

        if self.Meth == "CDM":
            U_conv[:, -1] = U_conv[:, 0] - self.dt * V_conv[:, 0] + self.dt ** 2 * A_conv[:, 0] / 2

            K_h = structure.M / self.dt ** 2 + structure.C / (2 * self.dt)
            a = structure.M / self.dt ** 2 - structure.C / (2 * self.dt)
            b = structure.K0 - 2 * structure.M / self.dt ** 2

            a_ff = a[np.ix_(structure.dof_free, structure.dof_free)]
            a_fd = a[np.ix_(structure.dof_free, structure.dof_moving)]
            a_df = a[np.ix_(structure.dof_moving, structure.dof_free)]
            a_dd = a[np.ix_(structure.dof_moving, structure.dof_moving)]

            b_ff = b[np.ix_(structure.dof_free, structure.dof_free)]
            b_fd = b[np.ix_(structure.dof_free, structure.dof_moving)]
            b_df = b[np.ix_(structure.dof_moving, structure.dof_free)]
            b_dd = b[np.ix_(structure.dof_moving, structure.dof_moving)]

            k_ff = K_h[np.ix_(structure.dof_free, structure.dof_free)]
            k_fd = K_h[np.ix_(structure.dof_free, structure.dof_moving)]
            k_df = K_h[np.ix_(structure.dof_moving, structure.dof_free)]
            k_dd = K_h[np.ix_(structure.dof_moving, structure.dof_moving)]

            for i in np.arange(1, nb_steps):
                P_h_f = (
                        loading[i - 1] * structure.P[structure.dof_free]
                        - a_ff @ U_conv[structure.dof_free, i - 2]
                        - a_fd @ U_conv[structure.dof_moving, i - 2]
                        - b_ff @ U_conv[structure.dof_free, i - 1]
                        - b_fd @ U_conv[structure.dof_moving, i - 1]
                )

                U_d = np.zeros(len(structure.disp_histories))

                for j, disp in enumerate(structure.disp_histories):
                    U_d[j] = disp[i]

                U_conv[structure.dof_free, i] = np.linalg.solve(k_ff, P_h_f - k_fd @ U_d)
                U_conv[structure.dof_moving, i] = U_d

                P_h_d = (
                        k_df @ U_conv[structure.dof_free, i]
                        + k_dd @ U_d
                        + a_df @ U_conv[structure.dof_free, i - 2]
                        + a_dd @ U_conv[structure.dof_moving, i - 2]
                        + b_df @ U_conv[structure.dof_free, i - 1]
                        + b_dd @ U_conv[structure.dof_moving, i - 1]
                )

                V_conv[structure.dof_free, i] = (
                                                        U_conv[structure.dof_free, i] - U_conv[
                                                    structure.dof_free, i - 1]
                                                ) / (2 * self.dt)
                V_conv[structure.dof_moving, i] = (
                                                          U_conv[structure.dof_moving, i] - U_conv[
                                                      structure.dof_moving, i - 1]
                                                  ) / (2 * self.dt)

                A_conv[structure.dof_free, i] = (
                                                        U_conv[structure.dof_free, i]
                                                        - 2 * U_conv[structure.dof_free, i - 1]
                                                        + U_conv[structure.dof_free, i - 2]
                                                ) / (self.dt ** 2)
                A_conv[structure.dof_moving, i] = (
                                                          U_conv[structure.dof_moving, i]
                                                          - 2 * U_conv[structure.dof_moving, i - 1]
                                                          + U_conv[structure.dof_moving, i - 2]
                                                  ) / (self.dt ** 2)

                P_conv[structure.dof_free, i] = P_h_f.copy()
                P_conv[structure.dof_moving, i] = P_h_d.copy()

        elif self.Meth == "NWK":
            A1 = structure.M / (P["b"] * self.dt ** 2) + P["g"] * structure.C / (P["b"] * self.dt)
            A2 = structure.M / (P["b"] * self.dt) + (P["g"] / P["b"] - 1) * structure.C
            A3 = (1 / (2 * P["b"]) - 1) * structure.M + self.dt * (
                    P["g"] / (2 * P["b"]) - 1
            ) * structure.C

            a1_ff = A1[np.ix_(structure.dof_free, structure.dof_free)]
            a1_fd = A1[np.ix_(structure.dof_free, structure.dof_moving)]

            a2_ff = A2[np.ix_(structure.dof_free, structure.dof_free)]
            a2_fd = A2[np.ix_(structure.dof_free, structure.dof_moving)]

            a3_ff = A3[np.ix_(structure.dof_free, structure.dof_free)]
            a3_fd = A3[np.ix_(structure.dof_free, structure.dof_moving)]

            K_h = structure.K0 + A1

            k_ff = K_h[np.ix_(structure.dof_free, structure.dof_free)]
            k_fd = K_h[np.ix_(structure.dof_free, structure.dof_moving)]
            k_df = K_h[np.ix_(structure.dof_moving, structure.dof_free)]
            k_dd = K_h[np.ix_(structure.dof_moving, structure.dof_moving)]

            for i in np.arange(1, nb_steps):
                P_h_f = (
                        loading[i] * structure.P[structure.dof_free]
                        + structure.P_fixed[structure.dof_free]
                        + a1_ff @ U_conv[structure.dof_free, i - 1]
                        + a2_ff @ V_conv[structure.dof_free, i - 1]
                        + a3_ff @ A_conv[structure.dof_free, i - 1]
                        + a1_fd @ U_conv[structure.dof_moving, i - 1]
                        + a2_fd @ V_conv[structure.dof_moving, i - 1]
                        + a3_fd @ A_conv[structure.dof_moving, i - 1]
                )

                for j, disp in enumerate(structure.disp_histories):
                    U_conv[structure.dof_moving[j], i] = disp[i]
                    V_conv[structure.dof_moving[j], i] = (
                                                                 U_conv[structure.dof_moving[j], i]
                                                                 - U_conv[structure.dof_moving[j], i - 1]
                                                         ) / self.dt
                    A_conv[structure.dof_moving[j], i] = (
                                                                 V_conv[structure.dof_moving[j], i]
                                                                 - V_conv[structure.dof_moving[j], i - 1]
                                                         ) / self.dt

                U_conv[structure.dof_free, i] = sc.linalg.solve(
                    k_ff, P_h_f - k_fd @ U_conv[structure.dof_moving, i]
                )

                V_conv[structure.dof_free, i] = (
                        (P["g"] / (P["b"] * self.dt))
                        * (U_conv[structure.dof_free, i] - U_conv[structure.dof_free, i - 1])
                        + (1 - P["g"] / P["b"]) * V_conv[structure.dof_free, i - 1]
                        + self.dt * (1 - P["g"] / (2 * P["b"])) * A_conv[structure.dof_free, i - 1]
                )
                A_conv[structure.dof_free, i] = (
                        (1 / (P["b"] * self.dt ** 2))
                        * (U_conv[structure.dof_free, i] - U_conv[structure.dof_free, i - 1])
                        - V_conv[structure.dof_free, i - 1] / (P["b"] * self.dt)
                        - (1 / (2 * P["b"]) - 1) * A_conv[structure.dof_free, i - 1]
                )

                P_conv[structure.dof_free, i] = P_h_f.copy()
                P_conv[structure.dof_moving, i] = (
                        k_df @ U_conv[structure.dof_free, i] + k_dd @ U_conv[structure.dof_moving, i]
                )

        elif self.Meth == "WIL":
            A1 = 6 / (P["t"] * self.dt) * structure.M + 3 * structure.C
            A2 = 3 * structure.M + P["t"] * self.dt / 2 * structure.C

            K_h = structure.K0 + 6 / (P["t"] * self.dt) ** 2 * structure.M + 3 / (P["t"] * self.dt) * structure.C

            loading = np.append(loading, loading[-1])

            for i in np.arange(1, nb_steps):
                dp_h = (
                               (P["t"] - 1) * (loading[i + 1] - loading[i])
                               + loading[i]
                               - loading[i - 1]
                       ) * structure.P

                dp_h += A1 @ V_conv[:, i - 1] + A2 @ A_conv[:, i - 1]

                d_Uh = sc.linalg.solve(K_h, dp_h)

                d_A = (
                              6 / (P["t"] * self.dt) ** 2 * d_Uh
                              - 6 / (P["t"] * self.dt) * V_conv[:, i - 1]
                              - 3 * A_conv[:, i - 1]
                      ) / (P["t"])

                d_V = self.dt * A_conv[:, i - 1] + self.dt / 2 * d_A
                d_U = (
                        self.dt * V_conv[:, i - 1]
                        + (self.dt ** 2) / 2 * A_conv[:, i - 1]
                        + (self.dt ** 2) / 6 * d_A
                )

                U_conv[structure.dof_free, i] = (U_conv[:, i - 1] + d_U)[structure.dof_free]
                V_conv[structure.dof_free, i] = (V_conv[:, i - 1] + d_V)[structure.dof_free]
                A_conv[structure.dof_free, i] = (A_conv[:, i - 1] + d_A)[structure.dof_free]

        elif self.Meth == "GEN":
            am = 0
            b = P["b"]
            g = P["g"]
            af = P["af"]

            A1 = (1 - am) / (b * self.dt ** 2) * structure.M + g * (1 - af) / (b * self.dt) * structure.C
            A2 = (1 - am) / (b * self.dt) * structure.M + (g * (1 - af) / b - 1) * structure.C
            A3 = ((1 - am) / (2 * b) - 1) * structure.M + self.dt * (1 - af) * (
                    g / (2 * b) - 1
            ) * structure.C

            a1_ff = A1[np.ix_(structure.dof_free, structure.dof_free)]
            a1_fd = A1[np.ix_(structure.dof_free, structure.dof_moving)]

            a2_ff = A2[np.ix_(structure.dof_free, structure.dof_free)]
            a2_fd = A2[np.ix_(structure.dof_free, structure.dof_moving)]

            a3_ff = A3[np.ix_(structure.dof_free, structure.dof_free)]
            a3_fd = A3[np.ix_(structure.dof_free, structure.dof_moving)]

            K_h = structure.K0 * (1 - af) + A1

            k_ff = K_h[np.ix_(structure.dof_free, structure.dof_free)]
            k_fd = K_h[np.ix_(structure.dof_free, structure.dof_moving)]
            k_df = K_h[np.ix_(structure.dof_moving, structure.dof_free)]
            k_dd = K_h[np.ix_(structure.dof_moving, structure.dof_moving)]

            for i in np.arange(1, nb_steps):
                P_h_f = (
                        loading[i] * structure.P[structure.dof_free]
                        + structure.P_fixed[structure.dof_free]
                        + a1_ff @ U_conv[structure.dof_free, i - 1]
                        + a2_ff @ V_conv[structure.dof_free, i - 1]
                        + a3_ff @ A_conv[structure.dof_free, i - 1]
                        + a1_fd @ U_conv[structure.dof_moving, i - 1]
                        + a2_fd @ V_conv[structure.dof_moving, i - 1]
                        + a3_fd @ A_conv[structure.dof_moving, i - 1]
                        - af
                        * (
                                structure.K0[np.ix_(structure.dof_free, structure.dof_free)]
                                @ U_conv[structure.dof_free, i - 1]
                                + structure.K0[np.ix_(structure.dof_free, structure.dof_moving)]
                                @ U_conv[structure.dof_moving, i - 1]
                        )
                )

                for j, disp in enumerate(structure.disp_histories):
                    U_conv[structure.dof_moving[j], i] = disp[i]
                    # V_conv[structure.dof_moving[j],i] = (U_conv[structure.dof_moving[j],i] - U_conv[structure.dof_moving[j],i-1]) / self.dt
                    # A_conv[structure.dof_moving[j],i] = (V_conv[structure.dof_moving[j],i] - V_conv[structure.dof_moving[j],i-1]) / self.dt

                U_conv[structure.dof_free, i] = sc.linalg.solve(
                    k_ff, P_h_f - k_fd @ U_conv[structure.dof_moving, i]
                )

                V_conv[:, i][structure.dof_free] = (
                        P["g"] / (P["b"] * self.dt) * (U_conv[:, i] - U_conv[:, i - 1])
                        + (1 - P["g"] / P["b"]) * V_conv[:, i - 1]
                        + self.dt * (1 - P["g"] / (2 * P["b"])) * A_conv[:, i - 1]
                )[structure.dof_free]
                A_conv[:, i][structure.dof_free] = (
                        1 / (P["b"] * self.dt ** 2) * (U_conv[:, i] - U_conv[:, i - 1])
                        - 1 / (self.dt * P["b"]) * V_conv[:, i - 1]
                        - (1 / (2 * P["b"]) - 1) * A_conv[:, i - 1]
                )[structure.dof_free]
                P_conv[structure.dof_free, i] = P_h_f.copy()
                P_conv[structure.dof_moving, i] = (
                        k_df @ U_conv[structure.dof_free, i] + k_dd @ U_conv[structure.dof_moving, i]
                )

        elif self.Meth is None:
            print("Method does not exist")

        time_end = time.time()
        total_time = time_end - time_start
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        print(
            f"Simulation done in {int(hours)} h {int(minutes)} m {int(seconds)} s... writing results to file"
        )

        Params = []
        for key, value in P.items():
            Params.append(f"{key}={np.around(value, 2)}")

        Params = "_".join(Params)

        self.filename = self.filename + "_" + self.Meth + "_" + Params + ".h5"
        file_path = os.path.join(self.dir_name, self.filename)

        with h5py.File(file_path, "w") as hf:
            hf.create_dataset("U_conv", data=U_conv)
            hf.create_dataset("V_conv", data=V_conv)
            hf.create_dataset("A_conv", data=A_conv)
            hf.create_dataset("P_ref", data=structure.P)
            hf.create_dataset("P_conv", data=P_conv)
            hf.create_dataset("Load_Multiplier", data=loading)
            hf.create_dataset("Time", data=Time)
            hf.create_dataset("Last_conv", data=nb_steps - 1)

            hf.attrs["Descr"] = "Results of the" + self.Meth + "simulation"
            hf.attrs["Method"] = self.Meth
        return structure

    def nonlinear(self, structure: Union[Structure_block, Hybrid]):
        time_start = time.time()

        if self.U0 is None:
            if np.linalg.norm(structure.U) == 0:
                self.U0 = np.zeros(structure.nb_dofs)
            else:
                self.U0 = deepcopy(structure.U)

        if self.V0 is None:
            self.V0 = np.zeros(structure.nb_dofs)

        Time = np.arange(0, self.T, self.dt, dtype=float)
        Time = np.append(Time, self.T)
        nb_steps = len(Time)

        loading = np.zeros(nb_steps)

        if callable(self.lmbda):
            for i, t in enumerate(Time):
                loading[i] = self.lmbda(t)
        elif isinstance(self.lmbda, list):
            loading = self.lmbda
            if len(loading) > nb_steps:
                print("Truncate")
                loading = loading[:nb_steps]
            elif len(loading) < nb_steps:
                print("Add 0")
                missing = nb_steps - len(loading)
                for i in range(missing):
                    loading.append(0)

        structure.get_P_r()
        structure.get_K_str0()
        structure.get_M_str()
        structure.get_C_str()

        if hasattr(structure, "times"):
            for timestep in structure.times:
                if timestep != self.dt:
                    warnings.warn(
                        "Unmatching timesteps between excitation and simulation"
                    )
            for i, disp in enumerate(structure.disp_histories):
                if self.U0[structure.dof_moving[i]] != disp[0]:
                    warnings.warn("Unmatching initial displacements")

        else:
            structure.dof_moving = []
            structure.disp_histories = []
            structure.times = []

        U_conv = np.zeros((structure.nb_dofs, nb_steps), dtype=float)
        V_conv = np.zeros((structure.nb_dofs, nb_steps), dtype=float)
        A_conv = np.zeros((structure.nb_dofs, nb_steps), dtype=float)
        F_conv = np.zeros((structure.nb_dofs, nb_steps), dtype=float)

        U_conv[:, 0] = deepcopy(self.U0)
        V_conv[:, 0] = deepcopy(self.V0)
        F_conv[:, 0] = deepcopy(structure.P_r)

        structure.commit()

        last_sec = 0

        self.Meth, P = structure.ask_method(self.Meth)

        if self.Meth == "CDM":
            structure.U = U_conv[:, 0].copy()
            structure.get_P_r()
            F_conv[:, 0] = structure.P_r.copy()

            A_conv[:, 0] = sc.linalg.solve(
                structure.M,
                loading[0] * structure.P
                + structure.P_fixed
                - structure.C @ V_conv[:, 0]
                - F_conv[:, 0],
            )

            U_conv[:, -1] = U_conv[:, 0] - self.dt * V_conv[:, 0] + self.dt ** 2 / 2 * A_conv[:, 0]

            K_h = 1 / (self.dt ** 2) * structure.M + 1 / (2 * self.dt) * structure.C
            A = 1 / (self.dt ** 2) * structure.M - 1 / (2 * self.dt) * structure.C
            B = -2 / (self.dt ** 2) * structure.M

            a_ff = A[np.ix_(structure.dof_free, structure.dof_free)]
            a_fd = A[np.ix_(structure.dof_free, structure.dof_moving)]
            a_df = A[np.ix_(structure.dof_moving, structure.dof_free)]
            a_dd = A[np.ix_(structure.dof_moving, structure.dof_moving)]

            b_ff = B[np.ix_(structure.dof_free, structure.dof_free)]
            b_fd = B[np.ix_(structure.dof_free, structure.dof_moving)]
            b_df = B[np.ix_(structure.dof_moving, structure.dof_free)]
            b_dd = B[np.ix_(structure.dof_moving, structure.dof_moving)]

            k_ff = K_h[np.ix_(structure.dof_free, structure.dof_free)]
            k_fd = K_h[np.ix_(structure.dof_free, structure.dof_moving)]
            k_df = K_h[np.ix_(structure.dof_moving, structure.dof_free)]
            k_dd = K_h[np.ix_(structure.dof_moving, structure.dof_moving)]

            for i in np.arange(1, nb_steps):
                if structure.stiff_type[:3] == "TAN":
                    structure.get_C_str()

                    K_h = 1 / (self.dt ** 2) * structure.M + 1 / (2 * self.dt) * structure.C
                    A = 1 / (self.dt ** 2) * structure.M - 1 / (2 * self.dt) * structure.C

                    a_ff = A[np.ix_(structure.dof_free, structure.dof_free)]
                    a_fd = A[np.ix_(structure.dof_free, structure.dof_moving)]
                    a_df = A[np.ix_(structure.dof_moving, structure.dof_free)]
                    a_dd = A[np.ix_(structure.dof_moving, structure.dof_moving)]

                    k_ff = K_h[np.ix_(structure.dof_free, structure.dof_free)]
                    k_fd = K_h[np.ix_(structure.dof_free, structure.dof_moving)]
                    k_df = K_h[np.ix_(structure.dof_moving, structure.dof_free)]
                    k_dd = K_h[np.ix_(structure.dof_moving, structure.dof_moving)]

                structure.U = U_conv[:, i - 1].copy()
                try:
                    structure.get_P_r()
                except Exception as e:
                    print(e)
                    break

                F_conv[:, i - 1] = deepcopy(structure.P_r)

                P_h_f = (
                        loading[i] * structure.P[structure.dof_free]
                        + structure.P_fixed[structure.dof_free]
                        - a_ff @ U_conv[structure.dof_free, i - 2]
                        - a_fd @ U_conv[structure.dof_moving, i - 2]
                        - b_ff @ U_conv[structure.dof_free, i - 1]
                        - b_fd @ U_conv[structure.dof_moving, i - 1]
                        - F_conv[structure.dof_free, i - 1]
                )

                U_d = np.zeros(len(structure.disp_histories))

                for j, disp in enumerate(structure.disp_histories):
                    U_d[j] = disp[i]

                U_conv[structure.dof_free, i] = np.linalg.solve(k_ff, P_h_f - k_fd @ U_d)
                U_conv[structure.dof_moving, i] = U_d

                P_h_d = (
                        k_df @ U_conv[structure.dof_free, i]
                        + k_dd @ U_d
                        + a_df @ U_conv[structure.dof_free, i - 2]
                        + a_dd @ U_conv[structure.dof_moving, i - 2]
                        + b_df @ U_conv[structure.dof_free, i - 1]
                        + b_dd @ U_conv[structure.dof_moving, i - 1]
                )

                V_conv[structure.dof_free, i] = (
                                                        U_conv[structure.dof_free, i] - U_conv[
                                                    structure.dof_free, i - 1]
                                                ) / (2 * self.dt)
                V_conv[structure.dof_moving, i] = (
                                                          U_conv[structure.dof_moving, i] - U_conv[
                                                      structure.dof_moving, i - 1]
                                                  ) / (2 * self.dt)

                A_conv[structure.dof_free, i] = (
                                                        U_conv[structure.dof_free, i]
                                                        - 2 * U_conv[structure.dof_free, i - 1]
                                                        + U_conv[structure.dof_free, i - 2]
                                                ) / (self.dt ** 2)
                A_conv[structure.dof_moving, i] = (
                                                          U_conv[structure.dof_moving, i]
                                                          - 2 * U_conv[structure.dof_moving, i - 1]
                                                          + U_conv[structure.dof_moving, i - 2]
                                                  ) / (self.dt ** 2)

                if i * self.dt >= last_sec:
                    print(
                        f"reached {np.around(last_sec, 3)} seconds out of {int(Time[-1])} seconds"
                    )
                    last_sec += 0.1

                last_conv = i

                structure.commit()

        elif self.Meth == "NWK":
            tol = 1
            singular_steps = []
            # tol = np.max(structure.M) / np.max(structure.K) * 10
            print(f"Tolerance is {tol}")

            structure.U = deepcopy(U_conv[:, 0])

            g = P["g"]
            b = P["b"]

            print(g)
            print(b)
            A_conv[:, 0] = sc.linalg.solve(
                structure.M, loading[0] * structure.P + structure.P_fixed - structure.C @ self.V0 - F_conv[:, 0]
            )

            A1 = (1 / (b * self.dt ** 2)) * structure.M + (g / (b * self.dt)) * structure.C
            A2 = (1 / (b * self.dt)) * structure.M + (g / b - 1) * structure.C
            A3 = (1 / (2 * b) - 1) * structure.M + self.dt * (g / (2 * b) - 1) * structure.C

            no_conv = 0

            a1 = 1 / (b * self.dt ** 2)
            a2 = 1 / (b * self.dt)
            a3 = 1 / (2 * b) - 1

            a4 = g / (b * self.dt)
            a5 = 1 - g / b
            a6 = self.dt * (1 - g / (2 * b))

            for i in np.arange(1, nb_steps):
                structure.U = U_conv[:, i - 1].copy()

                try:
                    structure.get_P_r()
                except Exception as e:
                    print(e)
                    break

                for j, disp in enumerate(structure.disp_histories):
                    structure.U[structure.dof_moving[j]] = disp[i]
                    U_conv[structure.dof_moving[j], i] = disp[i]
                    V_conv[structure.dof_moving[j], i] = (
                            a4
                            * (
                                    U_conv[structure.dof_moving[j], i]
                                    - U_conv[structure.dof_moving[j], i - 1]
                            )
                            + a5 * V_conv[structure.dof_moving[j], i - 1]
                            + a6 * A_conv[structure.dof_moving[j], i - 1]
                    )
                    A_conv[structure.dof_moving[j], i] = (
                            a1
                            * (
                                    U_conv[structure.dof_moving[j], i]
                                    - U_conv[structure.dof_moving[j], i - 1]
                            )
                            - a2 * V_conv[structure.dof_moving[j], i - 1]
                            - a3 * A_conv[structure.dof_moving[j], i - 1]
                    )

                P_h_f = (
                        loading[i] * structure.P[structure.dof_free]
                        + structure.P_fixed[structure.dof_free]
                        + A1[np.ix_(structure.dof_free, structure.dof_free)]
                        @ U_conv[structure.dof_free, i - 1]
                        + A1[np.ix_(structure.dof_free, structure.dof_moving)]
                        @ U_conv[structure.dof_moving, i - 1]
                        + A2[np.ix_(structure.dof_free, structure.dof_free)]
                        @ V_conv[structure.dof_free, i - 1]
                        + A2[np.ix_(structure.dof_free, structure.dof_moving)]
                        @ V_conv[structure.dof_moving, i - 1]
                        + A3[np.ix_(structure.dof_free, structure.dof_free)]
                        @ A_conv[structure.dof_free, i - 1]
                        + A3[np.ix_(structure.dof_free, structure.dof_moving)]
                        @ A_conv[structure.dof_moving, i - 1]
                )
                counter = 0
                conv = False

                while not conv:
                    # structure.revert_commit()

                    try:
                        structure.get_P_r()
                    except Exception as e:
                        print(e)
                        break

                    structure.get_K_str()

                    counter += 1
                    if counter > 100:
                        no_conv = i
                        break

                    R = (
                            P_h_f
                            - structure.P_r[structure.dof_free]
                            - A1[np.ix_(structure.dof_free, structure.dof_free)]
                            @ structure.U[structure.dof_free]
                            - A1[np.ix_(structure.dof_free, structure.dof_moving)]
                            @ structure.U[structure.dof_moving]
                    )
                    if np.linalg.norm(R) < tol:
                        structure.commit()
                        U_conv[:, i] = deepcopy(structure.U)
                        F_conv[:, i] = deepcopy(structure.P_r)
                        conv = True
                        # print(counter)
                        last_conv = i

                    Kt_p = structure.K + A1

                    dU = np.linalg.solve(Kt_p[np.ix_(structure.dof_free, structure.dof_free)], R)
                    structure.U[structure.dof_free] += dU
                    # structure.U[structure.dof_moving] += dU_d

                if no_conv > 0:
                    print(f"Step {no_conv} did not converge")
                    break

                dU_step = U_conv[structure.dof_free, i] - U_conv[structure.dof_free, i - 1]
                V_conv[structure.dof_free, i] = (
                        a4 * dU_step
                        + a5 * V_conv[structure.dof_free, i - 1]
                        + a6 * A_conv[structure.dof_free, i - 1]
                )
                A_conv[structure.dof_free, i] = (
                        a1 * dU_step
                        - a2 * V_conv[structure.dof_free, i - 1]
                        - a3 * A_conv[structure.dof_free, i - 1]
                )

                if structure.stiff_type[:3] == "TAN":
                    structure.get_C_str()
                    A1 = (1 / (b * self.dt ** 2)) * structure.M + (g / (b * self.dt)) * structure.C
                    A2 = (1 / (b * self.dt)) * structure.M + (g / b - 1) * structure.C
                    A3 = (1 / (2 * b) - 1) * structure.M + self.dt * (g / (2 * b) - 1) * structure.C

                # if np.min(np.real(np.sqrt(eigvals))) <=  1:
                #     singular_steps.append(i)
                #     print(f'Step{i} is singular - {np.linalg.cond(structure.C[np.ix_(structure.dof_free, structure.dof_free)])}')
                #     # print(f'Step{i} is singular - {np.linalg.cond(structure.K[np.ix_(structure.dof_free, structure.dof_free)])}')

                if i * self.dt >= last_sec:
                    print(
                        f"reached {np.around(last_sec, 3)} seconds out of {int(Time[-1])} seconds"
                    )
                    structure.plot_structure(
                        scale=1,
                        plot_forces=False,
                        plot_cf=False,
                        plot_supp=False,
                        lims=[[-6.0, 6.0], [-1.2, 6.5]],
                    )
                    last_sec += 0.1

        elif self.Meth == "WIL":
            pass

        elif self.Meth == "GEN":
            tol = 1e-3
            singular_steps = []
            # tol = np.max(structure.M) / np.max(structure.K) * 10
            print(f"Tolerance is {tol}")

            structure.U = deepcopy(U_conv[:, 0])

            g = P["g"]
            b = P["b"]
            af = P["af"]
            am = P["am"]

            A_conv[:, 0] = sc.linalg.solve(
                structure.M, loading[0] * structure.P + structure.P_fixed - structure.C @ self.V0 - F_conv[:, 0]
            )

            A1 = ((1 - am) / (b * self.dt ** 2)) * structure.M + (g * (1 - af) / (b * self.dt)) * structure.C
            A2 = ((1 - am) / (b * self.dt)) * structure.M + (g * (1 - af) / b - 1) * structure.C
            A3 = ((1 - am) / (2 * b) - 1) * structure.M + self.dt * (1 - af) * (
                    g / (2 * b) - 1
            ) * structure.C

            no_conv = 0

            a1 = 1 / (b * self.dt ** 2)
            a2 = 1 / (b * self.dt)
            a3 = 1 / (2 * b) - 1

            a4 = g / (b * self.dt)
            a5 = 1 - g / b
            a6 = self.dt * (1 - g / (2 * b))

            for i in np.arange(1, nb_steps):
                structure.U = U_conv[:, i - 1].copy()

                try:
                    structure.get_P_r()
                except Exception as e:
                    print(e)
                    break

                for j, disp in enumerate(structure.disp_histories):
                    structure.U[structure.dof_moving[j]] = disp[i]
                    U_conv[structure.dof_moving[j], i] = disp[i]
                    # V_conv[structure.dof_moving[j],i] = a4 * (U_conv[structure.dof_moving[j],i] - U_conv[structure.dof_moving[j],i-1]) + a5*V_conv[structure.dof_moving[j],i-1] + a6 * A_conv[structure.dof_moving[j],i-1]
                    # A_conv[structure.dof_moving[j],i] = a1 * (U_conv[structure.dof_moving[j],i] - U_conv[structure.dof_moving[j],i-1]) - a2*V_conv[structure.dof_moving[j],i-1] - a3 * A_conv[structure.dof_moving[j],i-1]

                P_h_f = (
                        loading[i] * structure.P[structure.dof_free]
                        + structure.P_fixed[structure.dof_free]
                        + A1[np.ix_(structure.dof_free, structure.dof_free)]
                        @ U_conv[structure.dof_free, i - 1]
                        + A1[np.ix_(structure.dof_free, structure.dof_moving)]
                        @ U_conv[structure.dof_moving, i - 1]
                        + A2[np.ix_(structure.dof_free, structure.dof_free)]
                        @ V_conv[structure.dof_free, i - 1]
                        + A2[np.ix_(structure.dof_free, structure.dof_moving)]
                        @ V_conv[structure.dof_moving, i - 1]
                        + A3[np.ix_(structure.dof_free, structure.dof_free)]
                        @ A_conv[structure.dof_free, i - 1]
                        + A3[np.ix_(structure.dof_free, structure.dof_moving)]
                        @ A_conv[structure.dof_moving, i - 1]
                        - af * F_conv[structure.dof_free, i - 1]
                )

                counter = 0
                conv = False

                while not conv:
                    # structure.revert_commit()

                    try:
                        structure.get_P_r()
                    except Exception as e:
                        print(e)
                        break

                    structure.get_K_str()

                    counter += 1
                    if counter > 100:
                        no_conv = i
                        break

                    R = (
                            P_h_f
                            - structure.P_r[structure.dof_free]
                            - A1[np.ix_(structure.dof_free, structure.dof_free)]
                            @ structure.U[structure.dof_free]
                            - A1[np.ix_(structure.dof_free, structure.dof_moving)]
                            @ structure.U[structure.dof_moving]
                    )
                    if np.linalg.norm(R) < tol:
                        structure.commit()
                        U_conv[:, i] = deepcopy(structure.U)
                        F_conv[:, i] = deepcopy(structure.P_r)
                        conv = True
                        # print(counter)
                        last_conv = i

                    Kt_p = structure.K + A1

                    dU = np.linalg.solve(Kt_p[np.ix_(structure.dof_free, structure.dof_free)], R)
                    structure.U[structure.dof_free] += dU
                    # structure.U[structure.dof_moving] += dU_d

                if no_conv > 0:
                    print(f"Step {no_conv} did not converge")
                    break

                dU_step = U_conv[structure.dof_free, i] - U_conv[structure.dof_free, i - 1]
                V_conv[structure.dof_free, i] = (
                        a4 * dU_step
                        + a5 * V_conv[structure.dof_free, i - 1]
                        + a6 * A_conv[structure.dof_free, i - 1]
                )
                A_conv[structure.dof_free, i] = (
                        a1 * dU_step
                        - a2 * V_conv[structure.dof_free, i - 1]
                        - a3 * A_conv[structure.dof_free, i - 1]
                )

                if structure.stiff_type[:3] == "TAN":
                    structure.get_C_str()
                    A1 = ((1 - am) / (b * self.dt ** 2)) * structure.M + (
                            g * (1 - af) / (b * self.dt)
                    ) * structure.C
                    A2 = ((1 - am) / (b * self.dt)) * structure.M + (
                            g * (1 - af) / b - 1
                    ) * structure.C
                    A3 = ((1 - am) / (2 * b) - 1) * structure.M + self.dt * (1 - af) * (
                            g / (2 * b) - 1
                    ) * structure.C

                # if np.min(np.real(np.sqrt(eigvals))) <=  1:
                #     singular_steps.append(i)
                #     print(f'Step{i} is singular - {np.linalg.cond(structure.C[np.ix_(structure.dof_free, structure.dof_free)])}')
                #     # print(f'Step{i} is singular - {np.linalg.cond(structure.K[np.ix_(structure.dof_free, structure.dof_free)])}')

                if i * self.dt >= last_sec:
                    print(
                        f"reached {np.around(last_sec, 3)} seconds out of {int(Time[-1])} seconds"
                    )
                    structure.plot_structure(
                        scale=1,
                        plot_forces=False,
                        plot_cf=False,
                        plot_supp=False,
                        lims=[[-6.0, 6.0], [-1.2, 6.5]],
                    )
                    last_sec += 0.1

        elif self.Meth is None:
            print("Method does not exist")

        time_end = time.time()
        total_time = time_end - time_start
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        print(
            f"Simulation done in {int(hours)} h {int(minutes)} m {int(seconds)} s... writing results to file"
        )

        Params = []
        for key, value in P.items():
            Params.append(f"{key}={np.around(value, 2)}")

        Params = "_".join(Params)

        self.filename = self.filename + "_" + self.Meth + "_" + Params + ".h5"
        file_path = os.path.join(self.dir_name, self.filename)

        with h5py.File(file_path, "w") as hf:
            hf.create_dataset("U_conv", data=U_conv)
            hf.create_dataset("V_conv", data=V_conv)
            hf.create_dataset("A_conv", data=A_conv)
            hf.create_dataset("F_conv", data=F_conv)
            hf.create_dataset("P_ref", data=structure.P)
            hf.create_dataset("Load_Multiplier", data=loading)
            hf.create_dataset("Time", data=Time)
            hf.create_dataset("Last_conv", data=last_conv)
            # hf.create_dataset('Singular_steps', data=singular_steps)

            hf.attrs["Descr"] = "Results of the" + self.Meth + "simulation"
            hf.attrs["Method"] = self.Meth
        return structure

    @staticmethod
    def solve_dyn_linear(structure: Union[Structure_block, Hybrid], T, dt, U0=None, V0=None, lmbda=None, Meth=None,
                         filename="", dir_name=""):
        solver = Dynamic(T, dt, U0, V0, lmbda, Meth, filename, dir_name)
        return solver.linear(structure)

    @staticmethod
    def solve_dyn_nonlinear(structure: Union[Structure_block, Hybrid], T, dt, U0=None, V0=None, lmbda=None, Meth=None,
                            filename="", dir_name=""):
        solver = Dynamic(T, dt, U0, V0, lmbda, Meth, filename, dir_name)
        return solver.nonlinear(structure)
