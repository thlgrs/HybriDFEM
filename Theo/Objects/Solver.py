import time
from copy import deepcopy

import h5py
import matplotlib.pyplot as plt

from .Structure_2D import *


class Static:

    @staticmethod
    def solve_linear(structure: Union[Structure_block, Structure_FEM, Hybrid]):
        structure.get_P_r()
        structure.get_K_str0()

        K_ff = structure.K0[np.ix_(structure.dof_free, structure.dof_free)]
        K_fr = structure.K0[np.ix_(structure.dof_free, structure.dof_fix)]
        K_rf = structure.K0[np.ix_(structure.dof_fix, structure.dof_free)]
        K_rr = structure.K0[np.ix_(structure.dof_fix, structure.dof_fix)]

        structure.U[structure.dof_free] = sc.linalg.solve(
            K_ff,
            structure.P[structure.dof_free]
            + structure.P_fixed[structure.dof_free]
            - K_fr @ structure.U[structure.dof_fix],
        )
        # structure.P[structure.dof_fix] = K_rf @ structure.U[structure.dof_free] + K_rr @ structure.U[structure.dof_fix]
        structure.get_P_r()
        return structure

    @staticmethod
    def solve_forcecontrol(structure: Union[Structure_block, Hybrid], steps, tol=1, stiff="tan", max_iter=25,
                           filename="Results_ForceControl", dir_name=""):
        time_start = time.time()

        if isinstance(steps, list):
            nb_steps = len(steps) - 1
            lam = steps

        elif isinstance(steps, int):
            nb_steps = steps
            lam = np.linspace(0, 1, nb_steps + 1)

        else:
            warnings.warn(
                "Steps of the simulation should be either a list or a number of steps (int)"
            )

        # Displacements, forces and stiffness
        U_conv = np.zeros((structure.nb_dofs, nb_steps + 1), dtype=float)
        P_r_conv = np.zeros((structure.nb_dofs, nb_steps + 1), dtype=float)
        save_k = False
        if save_k:
            K_conv = np.zeros((structure.nb_dofs, structure.nb_dofs, nb_steps + 1), dtype=float)

        structure.get_P_r()
        structure.get_K_str()
        structure.get_K_str0()
        U_conv[:, 0] = deepcopy(structure.U)
        P_r_conv[:, 0] = deepcopy(structure.P_r)
        if save_k:
            K_conv[:, :, 0] = deepcopy(structure.K)

        # Parameters of the simulation
        iter_counter = np.zeros(nb_steps)
        res_counter = np.zeros(nb_steps)

        non_conv = False

        for i in range(1, nb_steps + 1):
            converged = False
            iteration = 0

            P_target = lam[i] * structure.P + structure.P_fixed
            R = P_target[structure.dof_free] - structure.P_r[structure.dof_free]

            while not converged:
                # print(structure.K[np.ix_(structure.dof_free, structure.dof_free)])

                try:
                    if (
                            np.linalg.cond(structure.K[np.ix_(structure.dof_free, structure.dof_free)])
                            < 1e12
                    ):
                        dU = sc.linalg.solve(
                            structure.K[np.ix_(structure.dof_free, structure.dof_free)], R
                        )
                    else:
                        try:
                            dU = sc.linalg.solve(
                                K_conv[:, :, i - 1][
                                    np.ix_(structure.dof_free, structure.dof_free)
                                ],
                                R,
                            )
                        except Exception:
                            dU = sc.linalg.solve(
                                structure.K0[np.ix_(structure.dof_free, structure.dof_free)], R
                            )

                except np.linalg.LinAlgError:
                    warnings.warn("The tangent and initial stiffnesses are singular")

                structure.U[structure.dof_free] += dU

                try:
                    structure.get_P_r()
                except Exception as e:
                    non_conv = True
                    iteration = max_iter + 1
                    print(e)
                    break
                structure.get_K_str()
                # print(structure.P_r[structure.dof_free])

                R = P_target[structure.dof_free] - structure.P_r[structure.dof_free]
                res = np.linalg.norm(R)

                # print(res)
                if res < tol:
                    converged = True
                    # structure.plot_structure(scale=20, plot_cf=True, plot_forces=False)
                else:
                    # structure.revert_commit()
                    iteration += 1

                if iteration > max_iter:
                    non_conv = True
                    print(f"Method did not converge at step {i}")
                    break

            if non_conv:
                break

            else:
                structure.commit()
                res_counter[i - 1] = res
                iter_counter[i - 1] = iteration
                last_conv = i

                U_conv[:, i] = deepcopy(structure.U)
                P_r_conv[:, i] = deepcopy(structure.P_r)
                if save_k:
                    K_conv[:, :, i] = deepcopy(structure.K)

                print(f"Force increment {i} converged after {iteration + 1} iterations")

        time_end = time.time()
        total_time = time_end - time_start
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        print(
            f"Simulation done in {int(hours)} h {int(minutes)} m {int(seconds)} s... writing results to file"
        )

        filename = filename + ".h5"
        file_path = os.path.join(dir_name, filename)

        with h5py.File(file_path, "w") as hf:
            hf.create_dataset("U_conv", data=U_conv)
            hf.create_dataset("P_r_conv", data=P_r_conv)
            if save_k:
                hf.create_dataset("K_conv", data=K_conv)
            hf.create_dataset("Residuals", data=res_counter)
            hf.create_dataset("Iterations", data=iter_counter)
            hf.create_dataset("Last_conv", data=last_conv)
            hf.create_dataset("Lambda", data=lam)

            hf.attrs["Descr"] = "Results of the force_control simulation"
            hf.attrs["Tolerance"] = tol
            # hf.attrs['Lambda'] = lam
            hf.attrs["Simulation_Time"] = total_time
        return structure

    @staticmethod
    def solve_dispcontrol(structure: Union[Structure_block, Hybrid], steps, disp, node, dof, tol=1, stiff="tan",
                          max_iter=25, filename="Results_DispControl", dir_name=""):
        time_start = time.time()

        if isinstance(steps, list):
            nb_steps = len(steps) - 1
            lam = [step / max(steps, key=abs) for step in steps]
            d_c = steps

        elif isinstance(steps, int):
            nb_steps = steps
            lam = np.linspace(0, 1, nb_steps + 1)
            # print(lam)
            d_c = lam * disp

        else:
            warnings.warn(
                "Steps of the simulation should be either a list or a number of steps (int)"
            )
        # Displacements, forces and stiffness

        U_conv = np.zeros((structure.nb_dofs, nb_steps + 1), dtype=float)
        P_r_conv = np.zeros((structure.nb_dofs, nb_steps + 1), dtype=float)
        save_k = False
        if save_k:
            K_conv = np.zeros((structure.nb_dofs, structure.nb_dofs, nb_steps + 1), dtype=float)

        structure.get_P_r()
        structure.get_K_str()
        structure.get_K_str0()
        # print('K', structure.K[np.ix_(structure.dof_free, structure.dof_free)])

        U_conv[:, 0] = deepcopy(structure.U)
        P_r_conv[:, 0] = deepcopy(structure.P_r)
        if save_k:
            K_conv[:, :, 0] = deepcopy(structure.K0)

        # Parameters of the simulation
        iter_counter = np.zeros(nb_steps)
        res_counter = np.zeros(nb_steps)
        last_conv = 0
        if isinstance(node, int):
            control_dof = [3 * node + dof]
        elif isinstance(node, list):
            control_dof = []
            for n in node:
                control_dof.append(3 * n + dof)
        other_dofs = structure.dof_free[structure.dof_free != control_dof]

        structure.list_norm_res = [[] for _ in range(nb_steps)]
        structure.list_residual = [[] for _ in range(nb_steps)]

        P_f = structure.P[other_dofs].reshape(len(other_dofs), 1)
        P_c = structure.P[control_dof]
        K_ff_conv = structure.K0[np.ix_(other_dofs, other_dofs)]
        K_cf_conv = structure.K0[control_dof, other_dofs]
        K_fc_conv = structure.K0[other_dofs, control_dof]
        K_cc_conv = structure.K0[control_dof, control_dof]

        for i in range(1, nb_steps + 1):
            converged = False
            iteration = 0
            non_conv = False

            lam[i] = lam[i - 1]
            dU_c = d_c[i] - d_c[i - 1]

            R = -structure.P_r + lam[i] * structure.P + structure.P_fixed

            Rf = R[other_dofs]
            Rc = R[control_dof]

            # print('R0', R[structure.dof_free])

            while not converged:
                K_ff = structure.K[np.ix_(other_dofs, other_dofs)]
                K_cf = structure.K[control_dof, other_dofs]
                K_fc = structure.K[other_dofs, control_dof]
                K_cc = structure.K[control_dof, control_dof]

                # if i >= 40:
                #     ratio = .5
                #     K_ff = ratio * structure.K0[np.ix_(other_dofs, other_dofs)] + (1-ratio) * structure.K[np.ix_(other_dofs, other_dofs)]
                #     K_cf = ratio * structure.K0[control_dof, other_dofs] + (1-ratio)  * structure.K[control_dof, other_dofs]
                #     K_fc = ratio * structure.K0[other_dofs, control_dof] + (1-ratio)  * structure.K[other_dofs, control_dof]
                #     K_cc = ratio * structure.K0[control_dof, control_dof] + (1-ratio)  * structure.K[control_dof, control_dof]

                # if i >= 20:

                #     structure.get_K_str_LG()
                #     K_ff = structure.K_LG[np.ix_(other_dofs, other_dofs)]
                #     K_cf = structure.K_LG[control_dof, other_dofs]
                #     K_fc = structure.K_LG[other_dofs, control_dof]
                #     K_cc = structure.K_LG[control_dof, control_dof]

                # print('K', np.around(structure.K[np.ix_(structure.dof_free, structure.dof_free)],5))
                #
                solver = np.block([[K_ff, -P_f], [K_cf, -P_c]])
                solution = np.append(Rf - dU_c * K_fc, Rc - dU_c * K_cc)

                # print(np.around(solver, 10))

                try:
                    if np.linalg.cond(solver) < 1e10:
                        dU_dl = np.linalg.solve(solver, solution)

                    else:
                        solver = np.block([[K_ff_conv, -P_f], [K_cf_conv, -P_c]])
                        solution = np.append(
                            Rf - dU_c * K_fc_conv, Rc - dU_c * K_cc_conv
                        )

                        dU_dl = np.linalg.solve(solver, solution)

                except Exception as e:
                    non_conv = True
                    iteration = max_iter + 1
                    print(e)
                    break

                    # warnings.warn(f'Iteration {iteration} {i} - Tangent stiffness is singular. Trying with initial stiffness')

                # Update solution and state determination
                lam[i] += dU_dl[-1]
                structure.U[other_dofs] += dU_dl[:-1]
                structure.U[control_dof] += dU_c

                try:
                    structure.get_P_r()
                    structure.get_K_str()
                except Exception as e:
                    non_conv = True
                    iteration = max_iter + 1
                    print(e)
                    break

                R = -structure.P_r + lam[i] * structure.P + structure.P_fixed
                Rf = R[other_dofs]
                Rc = R[control_dof]

                res = np.linalg.norm(R[structure.dof_free])

                if res < tol:
                    converged = True
                    structure.commit()

                    list_blocks_yielded = []
                    for cf in structure.list_cfs:
                        for cp in cf.cps:
                            if cp.sp1.law.tag == "STC" and cp.sp2.law.tag == "STC":
                                if cp.sp1.law.yielded or cp.sp2.law.yielded:
                                    list_blocks_yielded.append(cf.bl_A.connect)
                                    list_blocks_yielded.append(cf.bl_B.connect)

                    for cf in structure.list_cfs:
                        for cp in cf.cps:
                            if cp.sp1.law.tag == "BSTC" and cp.sp2.law.tag == "BSTC":
                                if (
                                        cf.bl_A.connect in list_blocks_yielded
                                        or cf.bl_B.connect in list_blocks_yielded
                                ):
                                    # print('Reducing')
                                    cp.sp1.law.reduced = True
                                    cp.sp2.law.reduced = True

                else:
                    # structure.revert_commit()
                    iteration += 1
                    dU_c = 0

                if iteration > max_iter and not converged:
                    non_conv = True
                    print(f"Method did not converge at Increment {i}")
                    break

            if non_conv:
                structure.U = U_conv[:, last_conv]
                break
                # structure.U = U_conv[:,last_conv]

            if converged:
                # if i < 9:
                K_ff_conv = K_ff.copy()
                K_cf_conv = K_cf.copy()
                K_fc_conv = K_fc.copy()
                K_cc_conv = K_cc.copy()
                # structure.plot_structure(scale=20, plot_cf=True, plot_forces=False)
                # else:
                # print('Vertical disp', np.around(structure.U[-2],15))
                # structure.commit()
                # structure.plot_structure(scale=1, plot_cf=True, plot_supp=False, plot_forces=False)
                res_counter[i - 1] = res
                iter_counter[i - 1] = iteration
                last_conv = i

                U_conv[:, i] = deepcopy(structure.U)
                P_r_conv[:, i] = deepcopy(structure.P_r)
                if save_k:
                    K_conv[:, :, i] = deepcopy(structure.K)

                print(f"Disp. Increment {i} converged after {iteration + 1} iterations")

        time_end = time.time()
        total_time = time_end - time_start
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        print(
            f"Simulation done in {int(hours)} h {int(minutes)} m {int(seconds)} s... writing results to file"
        )

        filename = filename + ".h5"
        file_path = os.path.join(dir_name, filename)

        with h5py.File(file_path, "w") as hf:
            hf.create_dataset("U_conv", data=U_conv)
            hf.create_dataset("P_r_conv", data=P_r_conv)
            if save_k:
                hf.create_dataset("K_conv", data=K_conv)
            hf.create_dataset("Residuals", data=res_counter)
            hf.create_dataset("Iterations", data=iter_counter)
            hf.create_dataset("Last_conv", data=last_conv)
            hf.create_dataset("Control_Disp", data=d_c)
            hf.create_dataset("Lambda", data=lam)

            hf.attrs["Descr"] = "Results of the force_control simulation"
            hf.attrs["Tolerance"] = tol
            hf.attrs["Simulation_Time"] = total_time
        return structure


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


class Modal:

    def __init__(self, modes=None, no_inertia=False, filename="Results_Modal", dir_name="", save=True, initial=False):
        self.modes = modes
        self.no_inertia = no_inertia
        self.filename = filename
        self.dir_name = dir_name
        self.save = save
        self.initial = initial

    def modal(self, structure: Union[Structure_block, Hybrid]):
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
    def solve_modal(structure: Union[Structure_block, Hybrid], modes=None, no_inertia=False, filename="Results_Modal",
                    dir_name="", save=True, initial=False):
        solver = Modal(modes, no_inertia, filename, dir_name, save, initial)
        return solver.modal(structure)


class Plotter:
    def __init__(self, save=None, angle=None, tag=None, cf_index=0, scale=0, plot_cf=True,
                 plot_forces=True, plot_supp=True, lighter=False, modes=None, lims=None, folder=None, show=True):
        self.save = save
        self.angle = angle
        self.tag = tag
        self.cf_index = cf_index
        self.scale = scale
        self.plot_cf = plot_cf
        self.plot_forces = plot_forces
        self.plot_supp = plot_supp
        self.lighter = lighter
        self.modes = modes
        self.lims = lims
        self.folder = folder
        self.show = show

    def stiffness(self, structure: Union[Structure_block, Hybrid]):
        E = []
        vertices = []

        for j, CF in enumerate(structure.list_cfs):
            for i, CP in enumerate(CF.cps):
                E.append(np.around(CP.sp1.law.stiff["E"], 3))
                E.append(np.around(CP.sp2.law.stiff["E"], 3))
                vertices.append(CP.vertices_fibA)
                vertices.append(CP.vertices_fibB)

        from matplotlib.colors import Normalize
        from matplotlib import cm

        def normalize(smax, smin):
            if (smax - smin) == 0 and smax < 0:
                return Normalize(
                    vmin=1.1 * smin / 1e9, vmax=0.9 * smax / 1e9, clip=False
                )
            elif (smax - smin) == 0 and smax == 0:
                return Normalize(vmin=-1e-6, vmax=1e-6, clip=False)
            elif (smax - smin) == 0:
                return Normalize(
                    vmin=0.9 * smin / 1e9, vmax=1.1 * smax / 1e9, clip=False
                )
            else:
                return Normalize(vmin=smin / 1e9, vmax=smax / 1e9, clip=False)

        def plot(stiff, vertex):
            smax = np.max(stiff)
            smin = np.min(stiff)

            plt.axis("equal")
            plt.axis("off")
            plt.title("Axial stiffness [GPa]")

            norm = normalize(smax, smin)
            cmap = cm.get_cmap("coolwarm", 200)

            for i in range(len(stiff)):
                if smax - smin == 0:
                    index = norm(np.around(stiff[i], 6) / 1e9)
                else:
                    index = norm(np.around(stiff[i], 6) / 1e9)
                color = cmap(index)
                vertices_x = np.append(vertex[i][:, 0], vertex[i][0, 0])
                vertices_y = np.append(vertex[i][:, 1], vertex[i][0, 1])
                plt.fill(vertices_x, vertices_y, color=color)

            from mpl_toolkits.axes_grid1 import make_axes_locatable

            divider = make_axes_locatable(plt.gca())

            cax = divider.append_axes("right", size="10%", pad=0.2)
            plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)

        plt.figure()

        plot(E, vertices)

        if self.save is not None:
            plt.savefig(self.save)

    def get_stresses(self, structure: Union[Structure_block, Hybrid]):
        # Compute maximal stress and minimal stress:

        eps = np.array([])
        sigma = np.array([])
        x_s = np.array([])

        for j, CF in enumerate(structure.list_cfs):
            if (self.angle is None) or (abs(CF.self.angle - self.angle) < 1e-6):
                for i, CP in enumerate(CF.cps):
                    # print(CF.bl_B.disps[0])
                    # print

                    if not CP.to_ommit():
                        if self.tag is None or CP.sp1.law.tag == self.tag:
                            eps = np.append(eps, np.around(CP.sp1.law.strain["e"], 12))
                            # print(np.around(CP.sp1.law.strain['e'],12))
                            sigma = np.append(
                                sigma, np.around(CP.sp1.law.stress["s"], 12)
                            )
                            x_s = np.append(x_s, CP.x_cp[0])
        return sigma, eps, x_s

    def stresses(self, structure: Union[Structure_block, Hybrid]):
        # Compute maximal stress and minimal stress:

        tau = []
        sigma = []
        vertices = []

        for j, CF in enumerate(structure.list_cfs):
            if (self.angle is None) or (abs(CF.self.angle - self.angle) < 1e-6):
                for i, CP in enumerate(CF.cps):
                    if not CP.to_ommit():
                        if self.tag is None or CP.sp1.law.tag == self.tag:
                            tau.append(np.around(CP.sp1.law.stress["t"], 12))
                            tau.append(np.around(CP.sp2.law.stress["t"], 12))
                            sigma.append(np.around(CP.sp1.law.stress["s"], 12))
                            sigma.append(np.around(CP.sp2.law.stress["s"], 12))
                            vertices.append(CP.vertices_fibA)
                            vertices.append(CP.vertices_fibB)

        from matplotlib.colors import Normalize
        from matplotlib import cm

        def normalize(smax, smin):
            if (smax - smin) == 0 and smax < 0:
                return Normalize(
                    vmin=1.1 * smin / 1e6, vmax=0.9 * smax / 1e6, clip=False
                )
            elif (smax - smin) == 0 and smax == 0:
                return Normalize(vmin=-1e-6, vmax=1e-6, clip=False)
            elif (smax - smin) == 0:
                return Normalize(
                    vmin=0.9 * smin / 1e6, vmax=1.1 * smax / 1e6, clip=False
                )
            else:
                return Normalize(vmin=smin / 1e6, vmax=smax / 1e6, clip=False)

        def plot(stress, vertex, name_stress=None):
            smax = np.max(stress)
            smin = np.min(stress)

            print(
                f"Maximal {'axial' if name_stress == 'sigma' else 'shear'} stress is {np.around(smax / 1e6, 3)} MPa"
            )
            print(
                f"Minimum {'axial' if name_stress == 'sigma' else 'shear'} stress is {np.around(smin / 1e6, 3)} MPa"
            )
            # Plot sigmas

            plt.axis("equal")
            plt.axis("off")
            plt.title(
                f"{'Axial' if name_stress == 'sigma' else 'Shear'} stresses [MPa]"
            )

            norm = normalize(smax, smin)
            cmap = cm.get_cmap("viridis", 200)

            for i in range(len(sigma)):
                if smax - smin == 0:
                    index = norm(np.around(stress[i], 6) / 1e6)
                else:
                    index = norm(np.around(stress[i], 6) / 1e6)
                color = cmap(index)
                vertices_x = np.append(vertex[i][:, 0], vertex[i][0, 0])
                vertices_y = np.append(vertex[i][:, 1], vertex[i][0, 1])
                plt.fill(vertices_x, vertices_y, color=color)

            from mpl_toolkits.axes_grid1 import make_axes_locatable

            divider = make_axes_locatable(plt.gca())

            cax = divider.append_axes("right", size="10%", pad=0.2)
            plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)

        plt.figure()

        plt.subplot(2, 1, 1)
        plot(sigma, vertices, name_stress="sigma")
        plt.subplot(2, 1, 2)
        plot(tau, vertices, name_stress="tau")

        if self.save is not None:
            plt.savefig(self.save)

    def stress_profile(self, structure: Union[Structure_block, Hybrid]):
        stresses = []
        x = []
        counter = 0
        for cp in structure.list_cfs[self.cf_index].cps:
            counter += 1
            if not cp.to_ommit():
                stresses.append(cp.sp1.law.stress["s"] / 1e6)
                x.append(cp.x_cp[1] * 100)

        offset = 0.5 / (2 * len(stresses))
        # x = np.linspace(-.25+offset,0.25-offset,len(stresses))

        # x2 = np.linspace(-.25,0.25,100)
        # y_sigma = np.linspace(-36,36,100)
        # y_tau = 6 * 100e3 * (0.25**2 - (x2)**2) / (0.5**3 * 0.2)
        # # y_tau = - (5*100e3) / (0.5**2 * 0.5 * 0.2) * x2**2 + 5 * 100e3 / (4*0.5*0.2)
        # y_tau = 100e3 * (1 - 6*x2/0.5**2 + 4*x2**3/0.5**3) / (0.5*0.2)

        # print(max(stresses))

        plt.figure(None, figsize=(5, 5), dpi=600)
        # plt.scatter(x*100, stresses, label='HybriDFEM', marker='.', color='blue')
        plt.bar(
            x,
            stresses,
            label="HybriDFEM",
            facecolor="white",
            edgecolor="blue",
            linewidth=1,
            width=50 / counter,
        )
        # print(x)
        # print(stresses)
        # plt.plot(str,y_sigma,label='Analytical',color='red')
        # elif stress=='tau':
        #     plt.plot(x2*100,y_tau/1e6,label='Analytical',color='red')
        plt.legend(fontsize=12)
        plt.ylabel(r"Stress [MPa]")
        plt.xlabel(r"Height [cm]")
        plt.grid(True, linestyle="--", linewidth=0.3)

        if self.save:
            plt.savefig(self.save)

    def def_structure(self, structure: Union[Structure_block, Hybrid]):
        # structure.get_P_r()

        for block in structure.list_blocks:
            block.disps = structure.U[block.dofs]
            block.plot_block(scale=self.scale, lighter=self.lighter)

        for fe in structure.list_fes:
            if self.scale == 0:
                fe.PlotUndefShapeElem()
            else:
                defs = structure.U[fe.dofs]
                fe.PlotDefShapeElem(defs, scale=self.scale)

        # for cf in structure.list_cfs:
        #     if cf.cps[0].sp1.law.tag == 'CTC':
        #         if cf.cps[0].sp1.law.cracked:
        #             disp1 = structure.U[cf.bl_A.dofs[0]]
        #             disp2 = structure.U[cf.bl_B.dofs[0]]
        #             cf.plot_cf(scale, disp1, disp2)

        if self.plot_cf:
            for cf in structure.list_cfs:
                cf.plot_cf(self.scale)

        if self.plot_forces:
            for i in structure.dof_free:
                if structure.P[i] != 0:
                    node_id = int(i / 3)
                    dof = i % 3

                    start = (
                            structure.list_nodes[node_id]
                            + self.scale
                            * structure.U[
                                3 * node_id * np.ones(2, dtype=int)
                                + np.array([0, 1], dtype=int)
                                ]
                    )
                    arr_len = 0.3

                    if dof == 0:
                        end = arr_len * np.array([1, 0]) * np.sign(structure.P[i])
                        plt.arrow(
                            start[0],
                            start[1],
                            end[0],
                            end[1],
                            head_width=0.05,
                            head_length=0.075,
                            fc="green",
                            ec="green",
                        )
                    elif dof == 1:
                        end = arr_len * np.array([0, 1]) * np.sign(structure.P[i])
                        plt.arrow(
                            start[0],
                            start[1],
                            end[0],
                            end[1],
                            head_width=0.05,
                            head_length=0.075,
                            fc="green",
                            ec="green",
                        )
                    else:
                        if np.sign(structure.P[i]) == 1:
                            plt.plot(
                                start[0],
                                start[1],
                                marker="o",
                                markerfacecolor="None",
                                markeredgecolor="green",
                                markersize=10,
                            )
                            plt.plot(
                                start[0],
                                start[1],
                                marker=".",
                                markerfacecolor="green",
                                markeredgecolor="green",
                                markersize=5,
                            )
                        else:
                            plt.plot(
                                start[0],
                                start[1],
                                marker="o",
                                markerfacecolor="None",
                                markeredgecolor="green",
                                markersize=10,
                            )
                            plt.plot(
                                start[0],
                                start[1],
                                marker="x",
                                markerfacecolor="None",
                                markeredgecolor="green",
                                markersize=10,
                            )

                if structure.P_fixed[i] != 0:
                    node_id = int(i / 3)
                    dof = i % 3

                    start = (
                            structure.list_nodes[node_id]
                            + self.scale
                            * structure.U[
                                3 * node_id * np.ones(2, dtype=int)
                                + np.array([0, 1], dtype=int)
                                ]
                    )
                    arr_len = 0.3

                    if dof == 0:
                        end = arr_len * np.array([1, 0]) * np.sign(structure.P_fixed[i])
                        plt.arrow(
                            start[0],
                            start[1],
                            end[0],
                            end[1],
                            head_width=0.05,
                            head_length=0.075,
                            fc="red",
                            ec="red",
                        )
                    elif dof == 1:
                        end = arr_len * np.array([0, 1]) * np.sign(structure.P_fixed[i])
                        plt.arrow(
                            start[0],
                            start[1],
                            end[0],
                            end[1],
                            head_width=0.05,
                            head_length=0.075,
                            fc="red",
                            ec="red",
                        )
                    else:
                        if np.sign(structure.P_fixed[i]) == 1:
                            plt.plot(
                                start[0],
                                start[1],
                                marker="o",
                                markerfacecolor="None",
                                markeredgecolor="red",
                                markersize=10,
                            )
                            plt.plot(
                                start[0],
                                start[1],
                                marker=".",
                                markerfacecolor="red",
                                markeredgecolor="red",
                                markersize=5,
                            )
                        else:
                            plt.plot(
                                start[0],
                                start[1],
                                marker="o",
                                markerfacecolor="None",
                                markeredgecolor="red",
                                markersize=10,
                            )
                            plt.plot(
                                start[0],
                                start[1],
                                marker="x",
                                markerfacecolor="None",
                                markeredgecolor="red",
                                markersize=10,
                            )

        if self.plot_supp:
            for fix in structure.dof_fix:
                node_id = int(fix / 3)
                dof = fix % 3

                node = (
                        structure.list_nodes[node_id]
                        + self.scale
                        * structure.U[
                            3 * node_id * np.ones(2, dtype=int)
                            + np.array([0, 1], dtype=int)
                            ]
                )

                import matplotlib as mpl

                if dof == 0:
                    mark = mpl.markers.MarkerStyle(marker=5)
                elif dof == 1:
                    mark = mpl.markers.MarkerStyle(marker=6)
                else:
                    mark = mpl.markers.MarkerStyle(marker="x")

                plt.plot(node[0], node[1], marker=mark, color="blue", markersize=8)

    def modes(self, structure: Union[Structure_block, Hybrid]):
        if not hasattr(structure, "eig_modes"):
            warnings.warn("Eigen modes were not determined yet")

        if self.modes is None:
            modes = structure.nb_dof_free

        if len(structure.eig_vals) < modes:
            warnings.warn("Asking for too many modes, fewer were computed")

        for i in range(modes):
            structure.U[structure.dof_free] = structure.eig_modes.T[i]

            if self.lims is None:
                plt.figure(None, dpi=400, figsize=(6, 6))
            else:
                x_len = self.lims[0][1] - self.lims[0][0]
                y_len = self.lims[1][1] - self.lims[1][0]
                if x_len > y_len:
                    plt.figure(None, dpi=400, figsize=(6, 6 * y_len / x_len))
                else:
                    plt.figure(None, dpi=400, figsize=(6 * x_len / y_len, 6))

            plt.axis("equal")
            plt.axis("off")

            self.def_structure(structure)

            if self.lims is not None:
                plt.xlim(self.lims[0][0], self.lims[0][1])
                plt.ylim(self.lims[1][0], self.lims[1][1])

            w = np.around(structure.eig_vals[i], 3)
            f = np.around(structure.eig_vals[i] / (2 * np.pi), 3)
            if not w == 0:
                T = np.around(2 * np.pi / w, 3)
            else:
                T = float("inf")
            plt.title(
                rf"$\omega_{{{i + 1}}} = {w}$ rad/s - $T_{{{i + 1}}} = {T}$ s - $f_{{{i + 1}}} = {f}$ "
            )
            if self.save:
                if self.folder is not None:
                    if not os.path.exists(self.folder):
                        os.makedirs(self.folder)

                    plt.savefig(self.folder + f"/Mode_{i + 1}.eps")
                else:
                    plt.savefig(f"Mode_{i + 1}.eps")

            if not self.show:
                # print('Closing figure...')
                plt.close()
            else:
                plt.show()

    def structure(self, structure: Union[Structure_block, Hybrid]):
        desired_aspect = 1.0

        if self.lims is not None:
            x0, x1 = self.lims[0][0], self.lims[0][1]
            xrange = x1 - x0
            y0, y1 = self.lims[1][0], self.lims[1][1]
            yrange = y1 - y0
            aspect = xrange / yrange

            if aspect > desired_aspect:
                center_y = (y0 + y1) / 2
                yrange_new = xrange
                y0 = center_y - yrange_new / 2
                y1 = center_y + yrange_new / 2
            else:
                center_x = (x0 + x1) / 2
                xrange_new = yrange
                x0 = center_x - xrange_new / 2
                x1 = center_x + xrange_new / 2

        plt.figure(None, dpi=400, figsize=(6, 6))

        # plt.axis('equal')
        plt.axis("off")

        self.def_structure(structure)

        if self.lims is not None:
            plt.xlim((x0, x1))
            plt.ylim((y0, y1))

        if self.save is not None:
            plt.savefig(self.save)

        if not self.show:
            # print('Closing figure...')
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_stiffness(structure: Union[Structure_block, Hybrid], save=None):
        plotter = Plotter(save=save)
        return plotter.stiffness(structure)

    @staticmethod
    def get_stresses(structure: Union[Structure_block, Hybrid], angle=None, tag=None):
        plotter = Plotter(angle=angle, tag=tag)
        return plotter.get_stresses(structure)

    @staticmethod
    def plot_stresses(structure: Union[Structure_block, Hybrid], angle=None, save=None, tag=None):
        plotter = Plotter(angle=angle, tag=tag)
        return plotter.stresses(structure)

    @staticmethod
    def plot_stress_profile(structure: Union[Structure_block, Hybrid], cf_index=0, save=None):
        plotter = Plotter(cf_index=cf_index, save=save)
        return plotter.stress_profile(structure)

    @staticmethod
    def plot_def_structure(structure: Union[Structure_block, Hybrid], scale=0, plot_cf=True, plot_forces=True,
                           plot_supp=True, lighter=False):
        plotter = Plotter(scale=scale, plot_cf=plot_cf, plot_forces=plot_forces, plot_supp=plot_supp, lighter=lighter)
        return plotter.def_structure(structure)

    @staticmethod
    def plot_modes(structure: Union[Structure_block, Hybrid], modes=None, scale=1, save=False, lims=None, folder=None,
                   show=True):
        plotter = Plotter(modes=modes, scale=scale, save=save, lims=lims, folder=folder, show=show)
        return plotter.modes(structure)

    @staticmethod
    def plot_structure(structure: Union[Structure_block, Hybrid], scale=0, plot_cf=True, plot_forces=True,
                       plot_supp=True, show=True, save=None, lims=None):
        plotter = Plotter(scale=scale, plot_cf=plot_cf, plot_forces=plot_forces, show=show)
        return plotter.structure(structure)
