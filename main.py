import numpy as np
from matplotlib import pyplot as plt
from matplotlib import collections as mc
from geometry import *
from voronoi import *
from terrain import *

np.random.seed(0)

fig,ax = plt.subplots(1,figsize=(12,12))
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
fig.subplots_adjust(top=0.99,bottom=0.01,left=0.01,right=0.99)

#~ V = Voronoi(relax=5)
#~ V.plot_grid(ax)
#~ plt.show()

l=0.1
f = [0.25,0.25]
alpha = 0.575*np.pi

T = Terrain(n=4096,e=1,l=l,f=f,alpha=alpha,shape=0.9,nblobs=[10,5,5],\
            res=10000)
#~ ax.tripcolor(T.vertices[:,0],T.vertices[:,1],T.height,cmap="terrain")
ax.tricontour(T.vertices[:,0],T.vertices[:,1],T.height,\
              np.linspace(T.sealevel,1,16)[1:],cmap="copper")
ax.tricontour(T.vertices[:,0],T.vertices[:,1],T.height,\
              np.linspace(0,T.sealevel,16)[:-1],cmap="Blues")
#~ ax.plot(T.vertices[:,0],T.vertices[:,1],"ko",markersize=2)
ax.add_collection(mc.LineCollection(T.shoreline,color="k"))
for x,y in T.cblobs:
  ax.plot(x,y,marker="x",color="k")
for x,y in T.lblobs:
  ax.plot(x,y,marker="o",color="k")
for x,y in T.sblobs:
  ax.plot(x,y,marker="^",color="k")
fig.savefig("heightmaptest.png")
