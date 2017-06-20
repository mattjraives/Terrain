import numpy as np
from scipy.spatial import Voronoi as spv
from geometry import *

class Voronoi:
  
  def __init__(self,n=1024,bbox=[0,1,0,1],relax=5,**kwargs):
    self.n = n
    self.bbox = np.array(bbox)
    self.relax = relax
    self.build_grid()

  def build_grid(self):
    towers = np.random.rand(self.n, 2)
    vor = bound(towers, self.bbox)
    i = 0
    while i < self.relax:
      i = i+1
      # Compute centroids
      centroids = []
      for region in vor.filtered_regions:
        vertices = vor.vertices[region + [region[0]], :]
        centroid = poly_centroid(vertices)
        centroids.append(centroid)
      towers = np.array(centroids)
      vor = bound(towers, self.bbox)
    self.grid = vor
    
  def plot_grid(self,ax):
    # Plot initial points
    ax.plot(np.array(self.grid.filtered_points)[:, 0],
            np.array(self.grid.filtered_points)[:, 1], 'b.',markersize=2)
    # Plot ridges points
    for region in self.grid.filtered_regions:
      vertices = self.grid.vertices[region, :]
      ax.plot(vertices[:, 0], vertices[:, 1], 'g.')
    # Plot ridges
    for region in self.grid.filtered_regions:
      vertices = self.grid.vertices[region + [region[0]], :]
      ax.plot(vertices[:, 0], vertices[:, 1], color="gray",ls=":")
