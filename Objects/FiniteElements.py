from abc import ABC, abstractmethod

import numpy as np
from test_pycosat import nvars1


class Element_2D():
    def __init__(self):
        pass

    @abstractmethod
    def get_k_bsc(self):
        pass

    @abstractmethod
    def get_p_bsc(self):
        pass

    @abstractmethod
    def get_k_loc(self):
        pass

    @abstractmethod
    def get_p_loc(self):
        pass

    @abstractmethod
    def get_k_glob(self):
        pass

    @abstractmethod
    def get_p_glob(self, q_glob):
        pass

    @abstractmethod
    def solve(self, method):
        """
        def solve_linear(self)
        def solve_forcecontrol(self, steps, tol=1, stiff='tan', max_iter=15, filename='Results_ForceControl', dir_name='')

        """

        pass

    @abstractmethod
    def plot(self):
        pass

class TriangularElement(Element_2D):
    def __init__(self,n1,n2,n3,connect, E:float, nu:float, b:float, h:float, lin_geom=True, rho=0.):
        super().__init__()
        self.nodes = np.array([n1,n2,n3])
        self.connect = connect

class QuadilateralElement(Element_2D):
    def __init__(self,n1,n2,n3,n4):
        super().__init__()
        self.nodes = np.array([n1,n2,n3,n4])

    def shape_functions(self, xi, eta):
        """Fonctions de forme pour un quadrilatère bilinéaire."""
        return [
            0.25 * (1 - xi) * (1 - eta),
            0.25 * (1 + xi) * (1 - eta),
            0.25 * (1 + xi) * (1 + eta),
            0.25 * (1 - xi) * (1 + eta)
        ]

    def shape_function_derivatives(self, xi, eta):
        """Dérivées des fonctions de forme par rapport à xi et eta."""
        return np.array([
            [-0.25 * (1 - eta), 0.25 * (1 - eta), 0.25 * (1 + eta), -0.25 * (1 + eta)],
            [-0.25 * (1 - xi), -0.25 * (1 + xi), 0.25 * (1 + xi), 0.25 * (1 - xi)]
        ])

    def jacobian(self, dN_dxi):
        """Calcul du jacobien basé sur les coordonnées des nœuds."""
        coords = np.array([[node[0], node[1]] for node in self.nodes]).T
        return dN_dxi @ coords

    def get_K_elem(self):
        """Calcul de la matrice de rigidité élémentaire."""
        gauss_points = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
        gauss_weights = [1, 1]
        K_elem = np.zeros((8, 8))  # 4 nœuds * 2 DOF par nœud

        for xi in gauss_points:
            for eta in gauss_points:
                # Dérivées des fonctions de forme
                dN_dxi = self.shape_function_derivatives(xi, eta)

                # Jacobien et son déterminant
                jacobian = self.jacobian(dN_dxi)
                det_jacobian = np.linalg.det(jacobian)

                # Inversion du jacobien pour calculer les dérivées dans l'espace global
                dN_dx = np.linalg.inv(jacobian) @ dN_dxi

                # Matrice B
                B = np.zeros((3, 8))
                for i in range(4):  # 4 nœuds
                    B[0, 2 * i] = dN_dx[0, i]
                    B[1, 2 * i + 1] = dN_dx[1, i]
                    B[2, 2 * i] = dN_dx[1, i]
                    B[2, 2 * i + 1] = dN_dx[0, i]

                # Contribution à la matrice de rigidité
                K_elem += B.T @ self.D @ B * det_jacobian * self.thickness

        self.K_elem = K_elem
        return K_elem


