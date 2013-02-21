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
  square    = np.square
  

  ## GLOBAL-ISH CONSTANTS (SYSTEM RELATED)
  
  ## pixel size of canvas
  SIZE   = 2000
  ## background color (white)
  BACK   = 1.
  ## foreground color (black)
  FRONT  = 0.
  ## filename of image
  OUT    = './z.img'
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
  smax        = 2000
  ## widht of widest vein node when rendered
  rootW       = 13.*STP
  ## number of root (vein) nodes
  rootNodes   = 3


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


  def draw(P,W,oo):
    """
    draws the veins
    """

    ## simple vein W
    for i in reversed(xrange(rootNodes,oo)):
      ii = P[i]
      while ii>1:
        W[ii]+=1.
        ii = P[ii]

    wmax = W.max()
    W = sqrt(W/wmax)*rootW
    W[W<STP] = STP

    ## show vein nodes
    i = oo-1
    while i>1:
      dx = X[P[i]]-X[i]
      dy = Y[P[i]]-Y[i]
      a  = arctan2(dy,dx)
      s  = random(GRAINS)*veinNodeRad*2.
      xp = X[P[i]] - s*cos(a)
      yp = Y[P[i]] - s*sin(a)

      vcirc(xp,yp,[W[i]/2.]*GRAINS)

      i-=1


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

 
  def makeNodemap(snum,ldistVS,ltri,lX,lY,lsX,lsY):
    """
    nodemap[i,j] == True if vein node i is relative neighbor of 
    source node j
    u_i is relative neighbor of s if for all u_i:
      ||v-s|| < max{ ||u_i-s||, ||v-u_i|| }

      we save time by only checking the vein nodes that belong to the
      surounding simplices.
    """

    row = []
    col = []
    def listext(i,j):
      row.extend(i)
      col.extend(j)

    #neigh      = ltri.neighbors
    #simplices  = ltri.simplices
    getSimplex = ltri.find_simplex
    sxy        = colstack( (lsX,lsY) )

    for j in xrange(snum):
      simplex    = getSimplex(sxy[j,:])
      nsimplices = positive(ltri.neighbors[simplex].flatten())
      vertices   = positive(ltri.simplices[nsimplices,:].flatten())
      ii         = unique(positive(vertices-FOUR)) # 4 initial nodes in tri
      iin        = ii.shape[0]

      if iin:

        ## distance: vein nodes -> vein nodes
        idistVV = zeros((iin,iin),dtype=ft)
        for k,i in enumerate(ii):
          vvx = square(X[ii] - X[i])
          vvy = square(Y[ii] - Y[i])
          idistVV[:,k]  = (vvx+vvy)
        sqrt(idistVV,idistVV)
        
        idistVS  = tile( ldistVS[ii,j],(iin,1) ).T
        mas      = maximum( idistVV,ldistVS[ii,j] )
        compare  = idistVS < mas
        count    = compare.sum(axis=1)
        mask     = count==(iin-1)
        maskn    = mask.sum() 

        if maskn > 0:
          listext( list(ii[mask]),[j]*maskn )
    
    nodemap = coo_matrix( ( [True]*len(col),(row,col) ),\
                shape=(oo,snum),dtype=bool ).tocsr()

    return nodemap


  ### INITIALIZE

  ## arrays

  X = zeros(vmax,dtype=ft)
  Y = zeros(vmax,dtype=ft)
  P = zeros(vmax,dtype=bigint)-1
  W = zeros(vmax,dtype=float)

  sX,sY = darts(C,C,RAD,smax)
  snum  = sX.shape[0]

  distVS  = None
  nodemap = None

  ## (START) VEIN NODES

  ## triangulation needs at least four initial points
  ## in addition we need the initial triangulation 
  ## to contain all source nodes
  ## remember that ndoes in tri will be four indices higher than in X,Y

  oo = rootNodes
  xinit = zeros( (rootNodes+FOUR,1) )
  yinit = zeros( (rootNodes+FOUR,1) )

  ## don't change
  xinit[0] = 0.; yinit[0] = 0.
  xinit[1] = 1.; yinit[1] = 0.
  xinit[2] = 1.; yinit[2] = 1.
  xinit[3] = 0.; yinit[3] = 1.
  
  for i in xrange(rootNodes):
    t = random()*2.*pi
    s = .1 + random()*0.7
    x = C  + cos(t)*RAD*s
    y = C  + sin(t)*RAD*s

    xinit[i+FOUR] = x
    yinit[i+FOUR] = y
    X[i]          = x
    Y[i]          = y

  tri = triag(colstack( (xinit,yinit) ),incremental=True)

  ### MAIN LOOP

  itt  = 0
  aggt = 0
  iti = time()
  try:
    while True:
      itt += 1

      ## distance: vein nodes -> source nodes
      del(distVS)
      distVS = zeros((oo,snum),dtype=ft)
      for i in xrange(oo):
        vsx = (X[i] - sX)**2
        vsy = (Y[i] - sY)**2
        distVS[i,:] = vsx+vsy
      sqrt(distVS,distVS)
      
      ## this is where the magic might happen
      del(nodemap)
      nodemap = makeNodemap(snum,distVS,tri,X,Y,sX,sY)

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
          P[oo] = i
          oo += 1
        
      ## add new points to triangulation
      tri.add_points(colstack((X[ooo:oo],Y[ooo:oo])))

      ## terminate if nothing happened.
      if not cont:
        break

      ## mask out dead source nodes
      sourcemask = ones(snum,dtype=bool)
      for j in xrange(snum):
        vinds = nodemap[:,j].nonzero()[0]
        if (distVS[vinds,j]<killzone).all():
          sourcemask[j] = False

      ## remove dead soure nodes
      sX   = sX[sourcemask]
      sY   = sY[sourcemask]
      snum = sX.shape[0]

      if not itt % 50:
        aggt += time()-iti
        print("""#i: {:6d} | #s: {:7.2f} | """\
              """#vn: {:6d} | #sn: {:6d}"""\
               .format(itt,aggt,oo,snum))
        sys.stdout.flush()
        iti = time()

  except KeyboardInterrupt:
    pass

  finally:

    ## set line W if using lines
    #ctx.set_line_W(2./SIZE)

    ## set color
    ctx.set_source_rgb(FRONT,FRONT,FRONT)

    ## draws all vein nodes in
    draw(P,W,oo)

    ## save to file
    sur.write_to_png('{:s}.veins.png'.format(OUT))
    
    ## draw nodes as circles
    #vcirc(X[:oo],Y[:oo],[veinNodeRad/2.]*oo)

    ## show source nodes
    #ctx.set_source_rgb(1,0,0)
    #vcirc(sX,sY,[sourceDist/2.]*len(sX))
    #ctx.set_source_rgb(FRONT,FRONT,FRONT)

  return


if __name__ == '__main__' : main()

