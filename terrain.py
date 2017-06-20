import numpy as np
from matplotlib import pyplot as plt
from geometry import *
from voronoi import *
from scipy.spatial import Voronoi as spv
from datetime import datetime

def mirror_vor(towers,bbox):
  mirror_l = np.array([2*bbox[0] - towers[:,0],towers[:,1]]).T
  mirror_r = np.array([2*bbox[1] - towers[:,0],towers[:,1]]).T
  mirror_d = np.array([towers[:,0],2*bbox[2] - towers[:,1]]).T
  mirror_u = np.array([towers[:,0],2*bbox[3] - towers[:,1]]).T
  vor = spv(np.concatenate([towers,mirror_l,mirror_r,mirror_d,mirror_u]))
  return vor

class Terrain:
  def __init__(self,**kwargs):
    st0 = datetime.now()
    self.build_grid(**kwargs)
    et = datetime.now()
    print("{0}".format(et-st0))
    #
    st = datetime.now()
    self.neighbors(**kwargs)
    et = datetime.now()
    print("{0} | {1}".format(et-st,et-st0))
    #
    st = datetime.now()
    self.boundary_regions(**kwargs)
    et = datetime.now()
    print("{0} | {1}".format(et-st,et-st0))
    #
    st = datetime.now()
    self.build_heightmap(**kwargs)
    et = datetime.now()
    print("{0} | {1}".format(et-st,et-st0))
    #
    st = datetime.now()
    self.erode(**kwargs)
    et = datetime.now()
    print("{0} | {1}".format(et-st,et-st0))
    #
    st = datetime.now()
    self.biomes(**kwargs)
    et = datetime.now()
    print("{0} | {1}".format(et-st,et-st0))
    #
    st = datetime.now()
    self.cities(**kwargs)
    et = datetime.now()
    print("{0} | {1}".format(et-st,et-st0))
    #
    st = datetime.now()
    self.make_shoreline(**kwargs)
    et = datetime.now()
    print("{0} | {1}".format(et-st,et-st0))
    print("Done!")
  
  def build_grid(self,n=1024,bbox=[0,1,0,1],relax=5,**kwargs):
    print("Initializing grid.....")
    self.size = n
    self.bbox = bbox
    towers = np.random.rand(self.size, 2)
    vor = mirror_vor(towers, self.bbox)
    i = 0
    print("  Relaxing Voronoi diagram.....")
    while i < relax:
      #~ print (len(vor.regions))
      i = i+1
      centroids = []
      for region in vor.regions:
        if not region or -1 in region:
          continue
        vertices = vor.vertices[region + [region[0]]]
        centroid = poly_centroid(vertices)
        centroids.append(centroid)
      towers = np.array(centroids)
      bound = in_box(towers[:,0],towers[:,1],bbox)
      vor = mirror_vor(towers[bound], self.bbox)
    print("  Filtering Voronoi diagram.....")
    regions_filter = []
    points_filter = []
    ridges_filter = []
    verts_filter = []
    print("    Regions/Vertices.....")
    for i,region in enumerate(vor.regions):
      flag = False
      for j in region:
        if j==-1:
          break
        else:
          x,y = vor.vertices[j]
          flag = in_box(x,y,self.bbox)
      if flag:
        regions_filter.append(i)
        verts_filter += region
    verts_filter = set(verts_filter)
    print("    Points.....")
    for i,j in enumerate(vor.point_region):
      if j in regions_filter:
        points_filter.append(i)
    points_filter = set(points_filter)
    print("    Ridges.....")
    for i,ridge in enumerate(vor.ridge_points):
      if ridge[0] in points_filter or ridge[1] in points_filter:
        ridges_filter.append(i)
    ridges_filter = set(ridges_filter)
    #~ self.points = np.array([vor.points[i] for i in points_filter])
    #~ self.vertices = np.array([vor.vertices[i] for i in verts_filter])
    #~ self.ridge_points = [vor.ridge_points[i] for i in ridges_filter]
    #~ self.ridge_vertices = [vor.ridge_vertices[i] for i in ridges_filter]
    #~ self.regions = [vor.regions[i] for i in regions_filter]
    #~ self.point_region = [vor.point_region[i] for i in points_filter]
    self.points = vor.points
    self.vertices = vor.vertices
    self.ridge_points = vor.ridge_points
    self.ridge_vertices = vor.ridge_vertices
    self.regions = vor.regions
    self.point_region = vor.point_region
    self.filters = {"regions" : regions_filter,
                    "points"  : points_filter,
                    "ridges"  : ridges_filter,
                    "verts"   : verts_filter}
    
  def build_heightmap(self,nblobs=[5,5,5],res=10000,**kwargs):
    print ("Elevating map.....")
    theta_arr = np.linspace(-np.pi,np.pi,res)
    r_arr,x_arr,y_arr,mask = conic(theta_arr,**kwargs)
    self.height = np.zeros_like(self.vertices[:,0])
    self.heightmap_conic(r_arr,x_arr,y_arr,theta_arr,mask,**kwargs)
    self.heightmap_inner_blobs(r_arr,x_arr,y_arr,theta_arr,mask,\
                               nb=nblobs[0],**kwargs)
    self.heightmap_coast_blobs(r_arr,x_arr,y_arr,theta_arr,mask,\
                               nb=nblobs[1],**kwargs)
    self.heightmap_outer_blobs(r_arr,x_arr,y_arr,theta_arr,mask,\
                               nb=nblobs[2],**kwargs)
    self.normalize()
    self.region_height = np.zeros_like(self.points[:,0])
    for i in self.filters["points"]:
      h = 0
      r = self.regions[self.point_region[i]]
      for j in r:
        h += self.height[j]
      h = (1.0*h)/len(r)
      self.region_height[i] = h
    
  def heightmap_conic(self,r_arr,x_arr,y_arr,theta_arr,mask,**kwargs):
    print ("  Base Conic.....")
    theta_arr = np.linspace(-np.pi,np.pi,10000)
    r_arr,x_arr,y_arr,mask = conic(theta_arr,**kwargs)
    self.height = np.zeros_like(self.vertices[:,0])
    for j in self.filters["verts"]:
      x,y = self.vertices[j]
      h = np.min(np.sqrt((x_arr - x)**2 + (y_arr - y)**2))**0.5
      i = np.argmin(np.sqrt((x_arr - x)**2 + (y_arr - y)**2))
      if y < y_arr[i]:
        h = -h
      self.height[j] = h
    self.normalize(**kwargs)
  
  def heightmap_coast_blobs(self,r_arr,x_arr,y_arr,theta_arr,mask,nb=5,\
                            **kwargs):
    print ("  Coastal Blobs.....")
    new_height = np.zeros_like(self.height)
    for i in range(nb):
      j = np.random.randint(len(x_arr[mask]))
      fx = x_arr[mask][j] + (np.random.rand()-0.5)/self.size
      fy = y_arr[mask][j] + (np.random.rand()-0.5)/self.size
      #~ print (fx,fy)
      rb,xb,yb,maskb = conic(theta_arr,f=(fx,fy),e=0.5*np.random.rand(),\
                             l=0.07*(1 + (np.random.rand() - 0.5)),\
                             alpha=np.pi*np.random.rand())
      for k in self.filters["verts"]:
        x,y = self.vertices[k]
        i = np.argmin(np.sqrt((xb - x)**2 + (yb - y)**2))
        h = np.min(np.sqrt((xb - x)**2 + (yb - y)**2))
        if np.min(np.sqrt((xb[i] - fx)**2 + (yb[i] - fy)**2)) < \
          np.min(np.sqrt((x - fx)**2 + (y - fy)**2)):
          h = 1e-5*h
        new_height[k] += h
      if np.max(new_height) >= 0.1:
        new_height *= 0.1/new_height.max()
      self.height += new_height
    
  def heightmap_inner_blobs(self,r_arr,x_arr,y_arr,theta_arr,mask,nb=5,\
                            **kwargs):
    print ("  Inland Blobs.....")
    new_height = np.zeros_like(self.height)
    for i in range(nb):
      m = np.median(self.height)
      j = np.random.randint(len(self.height[self.height>m]))
      fx = self.vertices[:,0][self.height>m][j] + \
           (np.random.rand()-0.5)/self.size
      fy = self.vertices[:,1][self.height>m][j] + \
           (np.random.rand()-0.5)/self.size
      #~ print (fx,fy)
      rb,xb,yb,maskb = conic(theta_arr,f=(fx,fy),e=0.5*np.random.rand(),\
                             l=0.07*(1 + (np.random.rand() - 0.5)),\
                             alpha=np.pi*np.random.rand())
      for k in self.filters["verts"]:
        x,y = self.vertices[k]
        i = np.argmin(np.sqrt((xb - x)**2 + (yb - y)**2))
        h = np.min(np.sqrt((xb - x)**2 + (yb - y)**2))
        if np.min(np.sqrt((xb[i] - fx)**2 + (yb[i] - fy)**2)) < \
          np.min(np.sqrt((x - fx)**2 + (y - fy)**2)):
          h = 1e-5*h
        new_height[k] += h
      if np.max(new_height) >= 0.1:
        new_height *= 0.1/new_height.max()
      self.height += new_height
    
  def heightmap_outer_blobs(self,r_arr,x_arr,y_arr,theta_arr,mask,nb=5,\
                            **kwargs):
    print ("  Island Blobs.....")
    m = np.median(self.height)
    new_height = np.zeros_like(self.height)
    for i in range(nb):
      j = np.random.randint(len(self.height[self.height<m]))
      fx = self.vertices[:,0][self.height<m][j] + \
           (np.random.rand()-0.5)/self.size
      fy = self.vertices[:,1][self.height<m][j] + \
           (np.random.rand()-0.5)/self.size
      #~ print (fx,fy)
      rb,xb,yb,maskb = conic(theta_arr,f=(fx,fy),e=0.5*np.random.rand(),\
                             l=0.1*(1 + (np.random.rand() - 0.5)),\
                             alpha=np.pi*np.random.rand())
      for k in self.filters["verts"]:
        x,y = self.vertices[k]
        i = np.argmin(np.sqrt((xb - x)**2 + (yb - y)**2))
        h = np.min(np.sqrt((xb - x)**2 + (yb - y)**2))
        if np.min(np.sqrt((xb[i] - fx)**2 + (yb[i] - fy)**2)) < \
          np.min(np.sqrt((x - fx)**2 + (y - fy)**2)):
          h = -0.01*h*h
        new_height[k] += h
      if np.max(new_height) >= 0.3:
        new_height *= 0.3/new_height.max()
      elif np.max(new_height) <= 0.1:
        new_height *= 0.1/new_height.max()
      self.height += new_height
    
  def heightmap_shelf(self,**kwargs):
    print ("  Continental Shelf.....")
    for k in self.filters["verts"]:
      x,y = self.vertices[k]
      self.height[k] += (1-(x-0.5)**2)*(y**3)*1e-1
    
  def erode(self,**kwargs):
    print("Eroding terrain.....")
    self.watersheds(**kwargs)
    self.flow(**kwargs)
    self.slope(**kwargs)
    for i,ridge in enumerate(self.water_ridges):
      j = self.ridge_vertices[i]
      self.height[j] -= 0.1*self.water_flow[i]*self.slope[i]
    self.normalize(**kwargs)
    
  def watersheds(self,eps=0.01,**kwargs):
    W = 100*np.ones_like(self.points[:,0])
    for i in self.filters["points"]:
      if self.regions_boundary[i]:
        W[i] = self.region_height[i]
    loop = True
    while loop:
      loop = False
      for i in self.filters["points"]:
        if self.regions_boundary[i]:
          continue
        elif W[i] > self.region_height[i]:
          for j in self.neighbors[i]:
            if self.region_height[i] > W[j] + eps:
              W[i] = self.region_height[i]
              loop = True
              break
            elif W[i] > W[j] + eps:
              W[i] = W[j] + eps
              loop = True
              break
    self.water_height = W
    self.water_direction = -1*np.ones_like(self.points[:,0],dtype=int)
    self.water_ridges = []
    rlist = self.ridge_points.tolist()
    for i in self.filters["points"]:
      n = self.neighbors[i]
      j = np.argmin(self.water_height[n])
      if self.water_height[i] > n[j]:
        self.water_direction[i] = j
        try:
          self.water_ridges.append(rlist.index([i,j]))
        except ValueError:
          try: 
            self.water_ridges.append(rlist.index([j,i]))
          except ValueError:
            continue
  
  def flow(self,**kwargs):
    self.water_flow = np.ones_like(self.points[:,0])
    fwh = self.water_height[list(self.filters["points"])]
    srt = np.argsort(fwh)[::-1]
    for i in np.array(list(self.filters["points"]))[srt]:
      if 1+np.sign(self.water_direction[i]):
        j = self.water_direction[i]
        self.water_flow[j] += self.water_flow[i]
    self.water_flow = (self.water_flow - self.water_flow.min())/\
                      (self.water_flow.max() - self.water_flow.min())
    self.water_flow = np.sqrt(self.water_flow)
  
  def slope(self,**kwargs):
    self.slope = np.zeros_like(self.points[:,0])
    for i,j in enumerate(self.water_direction):
      if 1+np.sign(j):
        xi,yi = self.points[i]
        xj,yj = self.points[j]
        hi,hj = self.region_height[i],self.region_height[j]
        r = np.sqrt((xi-xj)**2 + (yi-yj)**2)
        dh = np.abs(hi-hj)
        self.slope[i] = r/dh
    
  def biomes(self,**kwargs):
    print("Determining biomes.....")
    pass
  
  def cities(self,**kwargs):
    print("Placing cities.....")
    pass
  
  def normalize(self,shape=1,**kwargs):
    hmax = np.max(self.height)
    hmin = np.min(self.height)
    self.height = (self.height - hmin)/(hmax - hmin)
    self.height = self.height**shape
  
  def make_shoreline(self,sealevel=1,**kwargs):
    print("Drawing Shoreline.....")
    sealevel = sealevel*np.median(self.height[self.height!=0])
    shoreline = []
    for j in self.filters["ridges"]:
      rv = self.ridge_vertices[j]
      if np.sum(np.sign(self.height[rv]-sealevel))==0:
        shoreline.append(self.points[self.ridge_points[j]])
    self.shoreline = shoreline
  
  def make_rivers(self,**kwargs):
    pass
    
  def boundary_regions(self,bbox=[0,1,0,1],**kwargs):
    print("Determining Boundary Cells.....")
    self.regions_boundary = np.zeros_like(self.points[:,0],dtype=bool)
    for i in self.filters["points"]:
      r = self.point_region[i]
      for j in self.regions[r]:
        x,y = self.vertices[j]
        if x==0 or x==1 or y==0 or y==1:
          self.regions_boundary[i] = True
          break
  
  def neighbors(self,**kwargs):
    print("Determining Neighbors.....")
    self.neighbors = [[]]*len(self.points[:,0])
    for i in self.filters["points"]:
      #~ n = [self.ridge_points[j].tolist() for j in self.filters["ridges"] \
           #~ if i in self.ridge_points[j]]
      #~ n = list(set([p for ridge in n for p in ridge]))
      #~ n.remove(i)
      w = np.where(self.ridge_points==[i,i])
      w1 = (~w[1].astype(bool)).astype(int)
      ww = np.vstack((w[0],w1)).T
      n = list(set([self.ridge_points[j][k] for j,k in ww]))
      self.neighbors[i] = n
      
      
