import numpy as np
import os
import h5py
import sys
import pathlib
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union, polygonize
from scipy.spatial import Voronoi, voronoi_plot_2d

folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

points = [(0, 0), (0, 1), (1, 1), (1, 0), (.5, .5)]

vor = Voronoi(points)
voronoi_plot_2d(vor)

plt.figure(None)

regions = []

nb_regions = len(vor.regions) - 1

print(f'Detected {nb_regions} Voronoi regions')
print(vor.regions[1:])

for i in range(nb_regions):
    vertices_index = vor.regions[i + 1]
    vertices_index = [j for j in vertices_index if j != -1]
    vertices = vor.vertices[vertices_index]
    regions.append(np.array(vertices))

for i in range(len(regions)):
    # print('Hello')
    print(regions[i])
    plt.fill(regions[i][:, 0], regions[i][:, 1], edgecolor='black')

for i in range(len(points)):
    plt.plot(points[i][0], points[i][1], marker='o', color='black')

plt.axis('equal')
