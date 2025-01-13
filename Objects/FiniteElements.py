from abc import ABC, abstractmethod

class FiniteElements:
    def __init__(self,dimension:int):
        self.dimension = dimension

    @abstractmethod
    def get_mass(self, no_inertia=False):
        pass

class Element_2D(FiniteElements):
    def __init__(self):
        super().__init__(2)
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

class TriangularElement(Element_2D):
    def __init__(self,n1,n2,n3,connect, E:float, nu:float, b:float, h:float, lin_geom=True, rho=0.):
        super().__init__()
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.connect = connect


