#!/usr/bin/python
# -*- coding: utf-8 -*-


import numpy as np
import cairo,Image
from time import time as time
import sys
from scipy.sparse import coo_matrix
from scipy.spatial import Delaunay

def timeit(method):
  def timed(*args, **kw):
    ts = time()
    result = method(*args, **kw)
    te = time()
    print '\n{:s} took {:2.2f} sec\n'\
            .format(method.__name__,te-ts)
    return result
  return timed


@timeit
def main():
  """
  time to load up the ponies
  """

  ## numpy functions

  cos       = np.cos
  sin       = np.sin
  arctan2   = np.arctan2
  sqrt      = np.sqrt
  random    = np.random.random
  pi        = np.pi
  ft        = np.float64
  bigint    = np.int64
  ones      = np.ones
  zeros     = np.zeros
  array     = np.array
  bool      = np.bool
  tile      = np.tile
  maximum   = np.maximum
  colstack  = np.column_stack
  triag     = Delaunay
  unique    = np.unique
  positive  = lambda a: a[a>-1]
  vectorize = np.vectorize
  logicNot  = np.logical_not
  

  ## GLOBAL-ISH CONSTANTS (SYSTEM RELATED)
  
  ## pixel size of canvas
  SIZE   = 1500
  ## background color (white)
  BACK   = 1.
  ## foreground color (black)
  FRONT  = 0.
  ## filename of image
  OUT    = './q.img'
  ## size of pixels on canvas
  STP    = 1./SIZE
  ## center of canvas
  C      = 0.5
  ## radius of circle with source nodes
  RAD    = 0.4
  ## number of grains used to sand paint each vein node
  GRAINS = 10
  ## because five is right out
  FOUR = 4

  ## GLOBAL-ISH CONSTANTS (PHYSICAL PROPERTIES)
  
  ## minimum distance between source nodes
  sourceDist  = 10.*STP
  ## vein nodes die when they get this close to a source node
  killzone    = 5.*STP
  ## radius of vein nodes when rendered
  veinNodeRad = 4.*STP
  ## maximum number of vein nodes
  vmax        = 2*1e6
  ## maximum number of source nodes
  smax        = 1500
  ## widht of widest vein node when rendered
  rootWidth   = 13.*STP

  @timeit
  def ctxInit():
    """
    make the drawing board
    """
    sur = cairo.ImageSurface(cairo.FORMAT_ARGB32,SIZE,SIZE)
    ctx = cairo.Context(sur)
    ctx.scale(SIZE,SIZE)
    ctx.set_source_rgb(BACK,BACK,BACK)
    ctx.rectangle(0,0,1,1)
    ctx.fill()
    return sur,ctx
  sur,ctx = ctxInit()

  def stroke(x,y):
    """
    draw dot for each (x,y)
    """
    ctx.rectangle(x,y,1./SIZE,1./SIZE)
    ctx.fill()
    return
  vstroke = vectorize(stroke)

  def circ(x,y,cr):
    """
    draw circle for each (x,y) with radius cr
    """
    ctx.arc(x,y,cr,0,2.*pi)
    ctx.fill()
    return
  vcirc = vectorize(circ)


  @timeit
  def darts(xx,yy,rr,n):
    """
    get at most n random, uniformly distributed, points in a circle.
    centered at (xx,yy), with radius rr.
    """
    t        = 2.*pi*random(n)
    u        = random(n)+random(n)
    r        = zeros(n,dtype=ft)
    mask     = u>1.
    xmask    = logicNot(mask)
    r[mask]  = 2.-u[mask]
    r[xmask] = u[xmask]
    xp       = rr*r*cos(t)
    yp       = rr*r*sin(t)
    gridx    = xx+xp
    gridy    = yy+yp

    o = []
    for i in xrange(n-1):
      dx = gridx[i] - gridx[i+1:]
      dy = gridy[i] - gridy[i+1:]
      dd = sqrt(dx*dx+dy*dy)
      if (dd > sourceDist).all():
        o.append(i)

    o = array(o,dtype=bigint)
    return gridx[o],gridy[o]

 
  def makeNodemap(oo,snum,distVS,distVV,tri,sX,sY):
    """
    nodemap[i,j] == True if vein node i is relative neighbour of 
    source node j
    u_i is relative neighbour of s if for all u_i:
      ||v-s|| < max{ ||u_i-s||, ||v-u_i|| }

      we save time by only checking the vein nodes that belong to the
      surounding simplices.
    """

    row = []
    col = []
    def listapp(i,j):
      row.append(i)
      col.append(j)

    neigh      = tri.neighbors
    simplices  = tri.simplices
    getSimplex = tri.find_simplex
    sxy        = colstack((sX,sY))

    for j in xrange(snum):
      simplex    = getSimplex(sxy[j,:])
      nsimplices = positive(neigh[simplex].flatten())
      vertices   = positive(simplices[nsimplices,:].flatten())
      ii         = unique(positive(vertices-FOUR)) # 4 initial nodes in tri

      for i in ii:
        ma = maximum(distVV[i,ii],distVS[ii,j])
        jj = ( distVS[i,j]<ma ).sum() == ma.shape[0]-1
        if jj:
          listapp(i,j)
    
    nodemap = coo_matrix( ( [True]*len(col),(row,col) ),\
                shape=(oo,snum),dtype=bool ).tocsr()

    return nodemap

  ### INITIALIZE

  ## arrays

  X      = zeros(vmax,dtype=ft)
  Y      = zeros(vmax,dtype=ft)
  PARENT = zeros(vmax,dtype=bigint)-1
  WIDTH  = zeros(vmax,dtype=float)

  sX,sY = darts(C,C,RAD,smax)
  snum = sX.shape[0]

  ## (START) VEIN NODES

  ## triangulation needs at least four initial points
  ## in addition we need the initial triangulation 
  ## to contain all source nodes
  ## remember that ndoes in tri will be four indices higher than in X,Y

  ## number of root (vein) nodes
  initialPoints = 5

  oo = initialPoints
  xinit = zeros((initialPoints+FOUR,1))
  yinit = zeros((initialPoints+FOUR,1))

  ## don't change
  xinit[0] = 0.; yinit[0] = 0.
  xinit[1] = 1.; yinit[1] = 0.
  xinit[2] = 1.; yinit[2] = 1.
  xinit[3] = 0.; yinit[3] = 1.
  
  for i in xrange(initialPoints):
    the = random()*2.*pi
    s = 0.1 + random()*0.7
    x = C   + cos(the)*RAD*s
    y = C   + sin(the)*RAD*s

    xinit[i+FOUR] = x
    yinit[i+FOUR] = y
    Y[i]          = y
    X[i]          = x

  tri    = triag(colstack((xinit,yinit)),incremental=True)
  triadd = lambda x,y: tri.add_points(colstack((x,y)))

  ### MAIN LOOP

  itt = 0
  iti = time()
  try:
    while True:
      itt += 1

      ## distance: vein nodes -> source nodes
      distVS = zeros((oo,snum),dtype=ft)
      for i in xrange(oo):
        vsx = (X[i] - sX)**2
        vsy = (Y[i] - sY)**2
        distVS[i,:] = vsx+vsy
      distVS = sqrt(distVS)
      
      ## distance: vein nodes -> vein nodes
      distVV = zeros((oo,oo),dtype=ft)
      for i in range(oo):
        vvx = (X[:oo,None] - X[i])**2
        vvy = (Y[:oo,None] - Y[i])**2
        distVV[:,i]  = (vvx+vvy)[:,0]
      distVV = sqrt(distVV)
     
      ## this is where the magic might happen
      nodemap = makeNodemap(oo,snum,distVS,distVV,tri,sX,sY)

      ## grow new vein nodes
      cont = False
      ooo = oo
      for i in xrange(oo):
        mask = nodemap[i,:].nonzero()[1]
        if mask.any():
          cont = True
          tx    = ( X[i]-sX[mask] ).sum()
          ty    = ( Y[i]-sY[mask] ).sum()
          a     = arctan2(ty,tx)
          X[oo] = X[i] - cos(a)*veinNodeRad
          Y[oo] = Y[i] - sin(a)*veinNodeRad
          PARENT[oo] = i
          oo += 1
        
      ## add new points to triangulation
      triadd(X[ooo:oo],Y[ooo:oo])

      ## terminate if nothing happened.
      if not cont:
        break

      ## mask out dead source nodes
      sourcemask = ones(snum,dtype=bool)
      for j in xrange(snum):
        vinds = nodemap[:,j].nonzero()[0]
        if (distVS[vinds,j]<killzone).all():
          sourcemask[j] = False

      ## map out dead source nodes
      sX = sX[sourcemask].copy()
      sY = sY[sourcemask].copy()
      snum = sX.shape[0]

      if not itt % 50:
        print('it: {:3d}\tvein nodes: {:5d}\ttime since last: {:2.2f} sec'\
               .format(itt,oo,time()-iti))
        sys.stdout.flush()
        iti = time()

  except KeyboardInterrupt:
    pass

  finally:
    
    ## set line width if using lines
    #ctx.set_line_width(2./SIZE)

    ## set color
    ctx.set_source_rgb(FRONT,FRONT,FRONT)

    ## simple vein width
    ## TODO: rewrite this
    for i in reversed(xrange(initialPoints,oo)):
      ii = PARENT[i]
      while ii>1:
        WIDTH[ii]+=1.
        ii = PARENT[ii]

    wmax = WIDTH.max()
    WIDTH = sqrt(WIDTH/wmax)*rootWidth
    WIDTH[WIDTH<STP] = STP

    ## show vein nodes
    ## TODO: fix overshooting
    i = oo-1
    while i>1:
      dx = -X[i] + X[PARENT[i]]
      dy = -Y[i] + Y[PARENT[i]]
      a  = arctan2(dy,dx)
      s  = random(GRAINS)*veinNodeRad*2.
      xp = X[PARENT[i]] - s*cos(a)
      yp = Y[PARENT[i]] - s*sin(a)

      vcirc(xp,yp,[WIDTH[i]/2.]*GRAINS)

      i-=1
    
    ## draw nodes as circles
    #vcirc(X[:oo],Y[:oo],[veinNodeRad/2.]*oo)

    ## show source nodes
    #ctx.set_source_rgb(1,0,0)
    #vcirc(sX,sY,[sourceDist/2.]*len(sX))
    #ctx.set_source_rgb(FRONT,FRONT,FRONT)

    ## save to file
    sur.write_to_png('{:s}.png'.format(OUT))

  return


if __name__ == '__main__' : main()

