from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson
from sfepy.discrete import Material


class Material2D:
    """Handles material properties and constitutive relations for 2D problems"""

    def __init__(self, young_modulus=210e9, poisson_ratio=0.3, plane="stress"):
        """
        Initialize material properties

        Parameters:
        -----------
        young_modulus : float
            Young's modulus in Pa (default: 210 GPa for steel)
        poisson_ratio : float
            Poisson's ratio (default: 0.3 for steel)
        plane : str
            'stress' for plane stress or 'strain' for plane strain
        """
        self.E = young_modulus
        self.nu = poisson_ratio
        self.plane = plane
        self._D_matrix = None
        self._material = None

    @property
    def D_matrix(self):
        """Get stiffness matrix"""
        if self._D_matrix is None:
            self._D_matrix = stiffness_from_youngpoisson(
                2, self.E, self.nu, plane=self.plane
            )
        return self._D_matrix

    def create_material(self, name="m"):
        """Create SfePy material object"""
        self._material = Material(name, D=self.D_matrix)
        return self._material

    def create_load_material(self, force_vector, name="load"):
        """
        Create material for load definition

        Parameters:
        -----------
        force_vector : list
            Force components [Fx, Fy]
        name : str
            Name for the load material
        """
        # Convert force vector to column format expected by SfePy
        force = [[comp] for comp in force_vector]
        return Material(name, val=force)
