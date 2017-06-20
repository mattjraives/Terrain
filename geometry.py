import numpy as np
from scipy.spatial import Voronoi as spv
from scipy.spatial import voronoi_plot_2d as plv
import sys
from matplotlib import pyplot as plt

eps = sys.float_info.epsilon
#~ eps = 0

def conic(theta,l=1,e=1,f=(0.5,0.5),alpha=0,n=1024,bbox=[0,1,0,1],**kwargs):
  base = np.random.randint(n)*0
  if e<0:
    x = np.linspace(0,1,n)
    y = np.tan(alpha)*(x-f[0]) + f[1] 
    return -1,x,y,-1
  else:
    if e>1:
      lim = np.arccos((-1 + l/np.sqrt(2))/e)
    else:
      lim = np.pi
    r = l/(1+e*np.cos(theta))
    x = r*np.cos(theta-alpha) + f[0] 
    y = r*np.sin(theta-alpha) + f[1]
    mask = (x<bbox[1])*(x>bbox[0])*(y<bbox[3])*(y>bbox[2])
    return r,x,y,mask

def poly_area(vertices):
    x = vertices[:,0]
    y = vertices[:,1]
    xp = np.pad(x,1,"wrap")[2:]
    yp = np.pad(y,1,"wrap")[2:]
    area = 0.5 * np.sum(x*yp - y*xp)
    return area

#~ def poly_centroid(vertices):
    #~ x = vertices[:,0]
    #~ y = vertices[:,1]
    #~ if len(x)==1:
        #~ return vertices
    #~ xp = np.pad(x,1,"wrap")[2:]
    #~ yp = np.pad(y,1,"wrap")[2:]
    #~ area = poly_area(vertices)
    #~ cx = np.sum((x+xp)*(x*yp - y*xp))
    #~ cy = np.sum((y+yp)*(x*yp - y*xp))
    #~ if cx>6*area:
        #~ cx=6*area
    #~ elif cx<0:
        #~ cx=0
    #~ if cy>6*area:
        #~ cy=6*area
    #~ elif cy<0:
        #~ cy=0
    #~ centroid = (1./(6*area))*np.array([cx,cy])
    #~ return centroid

def poly_centroid(vertices):
    A = 0 # Polygon's signed area
    C_x = 0 # Centroid's x
    C_y = 0 # Centroid's y
    for i in range(0, len(vertices) - 1):
        s = (vertices[i, 0] * vertices[i + 1, 1] -\
             vertices[i + 1, 0] * vertices[i, 1])
        A = A + s
        C_x = C_x + (vertices[i, 0] + vertices[i + 1, 0]) * s
        C_y = C_y + (vertices[i, 1] + vertices[i + 1, 1]) * s
    A = 0.5 * A
    C_x = (1.0 / (6.0 * A)) * C_x
    C_y = (1.0 / (6.0 * A)) * C_y
    return [C_x, C_y]

def bound(towers,bbox):
  # Mirror points
  mirror_l = np.array([2*bbox[0] - towers[:,0],towers[:,1]]).T
  mirror_r = np.array([2*bbox[1] - towers[:,0],towers[:,1]]).T
  mirror_d = np.array([towers[:,0],2*bbox[2] - towers[:,1]]).T
  mirror_u = np.array([towers[:,0],2*bbox[3] - towers[:,1]]).T
  vor = spv(np.concatenate([towers,mirror_l,mirror_r,mirror_d,mirror_u]))
  # Filter regions
  regions = []
  for region in vor.regions:
    flag = True
    for index in region:
      if index == -1:
        flag = False
        break
      else:
        x = vor.vertices[index, 0]
        y = vor.vertices[index, 1]
        if not (bbox[0] - eps <= x and x <= bbox[1] + eps and\
                bbox[2] - eps <= y and y <= bbox[3] + eps):
          flag = False
          break
    if region != [] and flag:
      regions.append(region)
  points = []
  for i,j in enumerate(vor.point_region):
    region = vor.regions[j]
    if region in regions:
      points.append(vor.points[i])
  vor.filtered_points = points
  vor.filtered_regions = regions
  return vor

def in_box(x,y,bbox):
    verdict = (x>=bbox[0])*(x<=bbox[1])*(y>=bbox[2])*(y<=bbox[3])
    return verdict
