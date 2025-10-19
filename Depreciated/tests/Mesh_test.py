from Depreciated.old.FE_Mesh import Mesh

# Example usage of the Mesh class
if __name__ == '__main__':
    initial_points = [(0.0, 0.0), (10.0, 0.0), (10.0, 1.0), (0.0, 1.0)]
    with Mesh(points=initial_points, element_type='tri', element_size=0.2, name='triMesh') as mesh:
        mesh.generate_mesh()
        mesh.plot()
