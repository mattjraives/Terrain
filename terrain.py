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
    self.heightmap(**kwargs)
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
   
  def heightmap(self,l=1,e=1,f=(0,0),alpha=0,residual=-0.1,res=10000,\
                nb=[5,8,3],thresh=[0.95,0.9,0.9],**kwargs):
    print("Building Heightmap.....")
    self.height = np.zeros_like(self.vertices[:,0])
    print("  Central Conic.....")
    self.heightmap_add_conic(l=l,e=e,f=f,alpha=alpha,res=res,residual=residual)
    self.normalize(shape=0.75,filt="verts")
    print("  Coastal Blobs.....")
    self.heightmap_coastal_blobs(nb=nb[0],thresh=thresh[0],mult=0.5)
    print("  Inland Blobs.....")
    self.heightmap_inland_blobs(nb=nb[1],thresh=thresh[1],mult=0.75)
    print("  Island Blobs.....")
    self.heightmap_island_blobs(nb=nb[2],thresh=thresh[2],mult=4)
    #~ print("  Random Noise.....")
    #~ for j in self.filters["verts"]:
      #~ self.height[j] += 5e-3*(np.random.rand() - 0.5)
    self.region_height = np.zeros_like(self.points[:,0])
    for i in self.filters["points"]:
      h = 0
      r = self.regions[self.point_region[i]]
      for j in r:
        h += self.height[j]
      h = h/len(r)
      self.region_height[i] = h
   
  def heightmap_add_conic(self,l=1,e=1,f=(0,0),alpha=0,residual=-1,mult=1,\
                          res=10000):
    theta = np.linspace(-np.pi,np.pi,res)
    r,x,y,mask = conic(theta,l=l,e=e,f=f,alpha=alpha,n=theta.size)
    h = np.zeros_like(self.vertices[:,0])
    for j in self.filters["verts"]:
      vx,vy = self.vertices[j]
      h[j] = np.min(np.sqrt((x - vx)**2 + (y - vy)**2))
      i = np.argmin(np.sqrt((x - vx)**2 + (y - vy)**2))
      if e < 1:
        if np.sqrt((vx - f[0])**2 + (vy - f[1])**2) > r[i]:
          h[j] *= residual
      else:
        if vy < y[i]:
          h[j] *= residual
    self.height += h*mult
  
  def heightmap_inland_blobs(self,nb=5,thresh=0.8,residual=-0.05,mult=0.3):
    blob_locs = []
    sealevel = np.median(self.height[list(self.filters["verts"])])
    for i in range(nb):
      l = 0.1*(1 + (np.random.rand() - 0.5))
      e = 0.5*np.random.rand()
      a = np.pi*(np.random.rand() - 0.5)
      score = np.zeros_like(self.vertices[:,0])
      for j in self.filters["verts"]:
        score[j] = 0.1*self.height[j]
        vx,vy = self.vertices[j]
        for x,y in blob_locs:
          score[j] += 0.5*np.sqrt((x - vx)**2 + (y - vy)**2)
      score *= np.sign(self.height - sealevel) + 1
      score = score[list(self.filters["verts"])] - score.min()
      bj = np.random.choice(np.array(list(self.filters["verts"]))\
           [score>thresh*score.max()])
      f = self.vertices[bj] + (np.random.rand(2) - 0.5)/self.size
      blob_locs.append(f)
      self.heightmap_add_conic(l=l,e=e,f=f,alpha=a,residual=residual,\
                               mult=mult)
    self.normalize(filt="verts")
    self.lblobs=blob_locs
  
  def heightmap_coastal_blobs(self,nb=5,thresh=0.8,residual=-0.01,mult=0.1):
    blob_locs = []
    sealevel = np.median(self.height[list(self.filters["verts"])])
    for i in range(nb):
      l = 0.1*(1 + (np.random.rand() - 0.5))
      e = 0.25*np.random.rand()
      a = np.pi*(np.random.rand() - 0.5)
      score = np.zeros_like(self.vertices[:,0])
      for j in self.filters["verts"]:
        score[j] -= np.abs(self.height[j] - sealevel) 
        vx,vy = self.vertices[j]
        for x,y in blob_locs:
          score[j] += 0.25*np.sqrt((x - vx)**2 + np.abs(y - vy)**2)
      score = score[list(self.filters["verts"])] - score.min()
      bj = np.random.choice(np.array(list(self.filters["verts"]))\
           [score>thresh*score.max()])
      f = self.vertices[bj] + (np.random.rand(2) - 0.5)/self.size
      blob_locs.append(f)
      self.heightmap_add_conic(l=l,e=e,f=f,alpha=a,residual=residual,\
                               mult=mult)
    self.normalize(filt="verts")
    self.cblobs=blob_locs
  
  def heightmap_island_blobs(self,nb=5,thresh=0.8,residual=0.01,mult=1):
    blob_locs = []
    sealevel = np.median(self.height[list(self.filters["verts"])])
    for i in range(nb):
      l = 0.05*(1 + (np.random.rand() - 0.5))
      e = 0.5*np.random.rand()
      a = np.pi*(np.random.rand() - 0.5)
      score = np.zeros_like(self.vertices[:,0])
      for j in self.filters["verts"]:
        score[j] -= 0.5*self.height[j]
        vx,vy = self.vertices[j]
        score[j] += vx + vy + (1 - vx) + (1 - vy)
        score[j] += 0.25*np.abs(self.height[j] - sealevel)
        for x,y in blob_locs:
          score[j] += 0.125*np.sqrt((x - vx)**2 + (y - vy)**2)
      score *= np.sign(sealevel - self.height) + 1
      score = score[list(self.filters["verts"])] - score.min()
      bj = np.random.choice(np.array(list(self.filters["verts"]))\
           [score>thresh*score.max()])
      f = self.vertices[bj] + (np.random.rand(2) - 0.5)/self.size
      blob_locs.append(f)
      self.heightmap_add_conic(l=l,e=e,f=f,alpha=a,residual=residual,\
                               mult=mult)
    self.normalize(filt="verts")
    self.sblobs=blob_locs
    
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
  
  def normalize(self,shape=1,filt=None,**kwargs):
    if filt:
      filt = list(self.filters[filt])
    else:
      filt = True
    hmax = np.max(self.height[filt])
    hmin = np.min(self.height[filt])
    self.height[filt] = (self.height[filt] - hmin)/(hmax - hmin)
    self.height[filt] = self.height[filt]**shape
  
  def make_shoreline(self,sealevel=1,**kwargs):
    print("Drawing Shoreline.....")
    sealevel = sealevel*np.median(self.height[list(self.filters["verts"])])
    shoreline = []
    for j in self.filters["ridges"]:
      rv = self.ridge_vertices[j]
      if np.sum(np.sign(self.height[rv]-sealevel))==0:
        shoreline.append(self.points[self.ridge_points[j]])
    self.shoreline = shoreline
    self.sealevel = sealevel
  
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
      
      
