import time
from copy import deepcopy

import h5py
import scipy as sc

from Theo.Structures.Structure_2D import *


class Static:

    @staticmethod
    def solve_linear(structure):
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
        structure.P[structure.dof_fix] = K_rf @ structure.U[structure.dof_free] + K_rr @ structure.U[structure.dof_fix]
        structure.get_P_r()
        return structure

    @staticmethod
    def solve_forcecontrol(structure, steps, tol=1, stiff="tan", max_iter=25, filename="Results_ForceControl",
                           dir_name=""):
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
    def solve_dispcontrol(structure, steps, disp, node, dof, tol=1, stiff="tan", max_iter=25,
                          filename="Results_DispControl", dir_name=""):
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
            # Use variable DOF system: node_dof_offsets instead of 3*node
            control_dof = [structure._global_dof(node, dof)]
        elif isinstance(node, list):
            control_dof = []
            for n in node:
                control_dof.append(structure._global_dof(n, dof))
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
