from numpy import ndarray


class Geometry:
    def __init__(self, name, h):
        self.name = name
        self.h = h

class Rectangle(Geometry):
    def __init__(self,n1:ndarray[2], n2:ndarray[2], n3:ndarray[2], n4:ndarray[2]):
        pass


class FE_2D:
    def __init__(self, E:float, nu:float, lin_geom=True):
        self.E = E
        self.nu = nu
        self.G = E / (2 * (1 + nu))
        self.chi = (6 + 5 * self.nu) / (5 * (1 + self.nu))
        self.lin_geom = lin_geom

class TriElement(FE_2D):
    def __init__(self, node1:ndarray, node2:ndarray, node3:ndarray, E:float, nu:float, lin_geom=True):
        super.__init__(E, nu, lin_geom)
        self.N1 = node1
        self.N2 = node2
        self.N3 = node3
        if not lin_geom:
            self.N4 = (node1 + node2)/2
            self.N5 = (node2 + node3)/2
            self.N6 = (node3 + node1)/2

class QuadElement(FE_2D):
    def __init__(self, node1:ndarray, node2:ndarray, node3:ndarray, node4:ndarray, E:float, nu:float, lin_geom=True):
        super.__init__(E, nu, lin_geom)
        self.N1 = node1
        self.N2 = node2
        self.N3 = node3
        self.N4 = node4
        if not lin_geom:
            self.N4 = (node1 + node2)/2
            self.N5 = (node2 + node3)/2
            self.N6 = (node3 + node4)/2
            self.N7 = (node4 + node1)/2
