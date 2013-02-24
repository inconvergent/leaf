#!/usr/bin/python
# -*- coding: utf-8 -*-


import numpy as np
import cairo
from time import time as time
from time import strftime
import sys
from scipy.spatial import Delaunay,distance
from collections import defaultdict
import cProfile
import pstats

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
  linspace  = np.linspace
  cdist     = distance.cdist
  eye       = np.eye
  transpose = np.transpose
  ceil      = np.ceil

  ## GLOBAL-ISH CONSTANTS (SYSTEM RELATED)
  
  ## pixel size of canvas
  SIZE   = 512
  ## background color (white)
  BACK   = 1.
  ## foreground color (black)
  FRONT  = 0.
  ## filename of image
  OUT    = './e.img'
  ## size of pixels on canvas
  STP    = 1./SIZE
  ## center of canvas
  C      = 0.5
  ## radius of circle with source nodes
  RAD    = 0.4
  ## number of grains used to sand paint each vein node
  GRAINS = 1
  ## alpha channel when rendering
  ALPHA  = 1.
  ## because five is right out
  FOUR = 4

  ## GLOBAL-ISH CONSTANTS (PHYSICAL PROPERTIES)
  
  ## minimum distance between source nodes
  sourceDist  = 10.*STP
  ## a source node dies when all approaching vein nodes are close than
  ## killzone.
  killzone    = STP
  ## radius of vein nodes when rendered
  veinNodeRad = STP
  ## maximum number of vein nodes
  vmax        = 1*1e7
  ## maximum number of source nodes
  smax        = 100
  ## widht of widest vein node when rendered
  rootW       = 20.*STP
  ## number of root (vein) nodes
  rootNodes   = 1


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


  def draw(P,W,oo,XY):
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
    W[W<2.*STP] = 2.*STP

    ## show vein nodes
    for i in reversed(range(rootNodes,oo)):
      dxy = XY[P[i],:]-XY[i,:]
      a   = arctan2(dxy[1],dxy[0])
      s   = linspace(0,1,GRAINS)*veinNodeRad
      xyp = XY[P[i],:] - array( cos(a),sin(a) )*s

      vcirc(xyp[0],xyp[1],[W[i]/2.]*GRAINS)


  def tesselation(tri):
    """
    show triangulation of all vein nodes
    """

    for s in tri.simplices:
      ## ignore the four "container vertices" in the corners
      if all(s>FOUR-1):
        xy = tri.points[s,:]
        ctx.move_to(xy[0,0],xy[0,1])
        for i in xrange(2):
          ctx.line_to(xy[i,0],xy[i,1])
        ctx.stroke()


  def darts(xx,yy,rr,n):
    """
    get at most n random, uniformly distributed, points in a circle.
    centered at (xx,yy), with radius rr.
    """
    ## random uniform points in a circle
    t        = 2.*pi*random(n)
    u        = random(n)+random(n)
    r        = zeros(n,dtype=ft)
    mask     = u>1.
    xmask    = logicNot(mask)
    r[mask]  = 2.-u[mask]
    r[xmask] = u[xmask]
    xyp      = colstack(( rr*r*cos(t),rr*r*sin(t) ))
    gridxy   = xyp + array( [xx,yy] )

    o = []
    ## we only want points that have no neghbors 
    ## within radius sourceDist
    for i in xrange(n-1):
      dxy = gridxy[i,:] - gridxy[i+1:,:]
      dd  = sqrt(dxy[:,0]*dxy[:,0]+dxy[:,1]*dxy[:,1])
      if all(dd > sourceDist):
        o.append(i)

    return gridxy[array(o,dtype=bigint),:]

 
  def makeNodemap(snum,ldistVS,ltri,lXY,lsXY):
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

    SVdict = {}
    VSdict = defaultdict(list)

    getSimplex = ltri.find_simplex

    for j in xrange(snum):
      simplex    = getSimplex(lsXY[j,:])
      nsimplices = positive( ltri.neighbors[simplex].flatten() )
      vertices   = positive( ltri.simplices[nsimplices,:].flatten() )
      ii         = unique(positive(vertices-FOUR)) # 4 initial nodes in tri
      iin        = ii.shape[0]

      if iin:

        ## distance: vein nodes -> vein nodes
        idistVV = cdist(lXY[ii,:],lXY[ii,:],'euclidean')
        
        idistVS = transpose(tile( ldistVS[ii,j],(iin,1) ))
        mas     = maximum( idistVV,ldistVS[ii,j] )
        compare = ( idistVS<mas ) == ( 1-eye(iin,dtype=bool) )
        count   = compare.sum(axis=1)
        mask    = count==iin
        maskn   = mask.sum() 

        if maskn > 0:
          SVdict[j]=ii[mask]
          [VSdict[i].append(j) for i in ii[mask]]


    return VSdict,SVdict

  ### INITIALIZE

  ## arrays

  XY      = zeros((vmax,2),dtype=ft)
  TAG     = zeros(vmax,dtype=bigint)-1
  P       = zeros(vmax,dtype=bigint)-1
  W       = zeros(vmax,dtype=float)
  sXY     = darts(C,C,RAD,smax)
  snum    = sXY.shape[0]

  distVS  = None
  nodemap = None

  ## (START) VEIN NODES

  ## triangulation needs at least four initial points
  ## in addition we need the initial triangulation 
  ## to contain all source nodes
  ## remember that ndoes in tri will be four indices higher than in X,Y

  oo = rootNodes
  xyinit          = zeros( (rootNodes+FOUR,2) )
  xyinit[:FOUR,:] = array( [[0.,0.],[1.,0.],[1.,1.],[0.,1.]] )

  for i in xrange(rootNodes):
    t = random()*2.*pi
    s = .1 + random()*0.7
    xy = C  + array( [cos(t),sin(t)] )*RAD*s
    #xy = array([C,C])

    xyinit[i+FOUR,:] = xy
    XY[i,:]          = xy


  tri    = triag(xyinit,incremental=True)
  triadd = tri.add_points

  ### MAIN LOOP

  itt  = 0
  aggt = 0
  iti = time()
  try:
    while True:

      ## distance: vein nodes -> source nodes
      distVS = cdist(XY[:oo,:],sXY,'euclidean')
      
      ## this is where the magic might happen
      VSdict,SVdict = makeNodemap(snum,distVS,tri,XY,sXY)

      ## grow new vein nodes
      cont = False
      ooo  = oo
      for i,jj in VSdict.iteritems():
        killmask = distVS[i,jj]<=killzone
        if jj and not any(killmask):
          cont     = True
          txy      = ( XY[i,:] -sXY[jj,:] ).sum(axis=0)
          a        = arctan2( txy[1],txy[0] )
          XY[oo,:] = XY[i,:] - array( [cos(a),sin(a)] )*veinNodeRad
          P[oo]    = i
          oo      += 1
        
      ## mask out dead source nodes
      sourcemask = ones(snum,dtype=bool)
      for j,ii in SVdict.iteritems():
        if all(distVS[ii,j]<=killzone):
          sourcemask[j] = False
          mergenum = ii.shape[0]
          XY[oo:oo+mergenum,:] = sXY[j,:]
          P[oo:oo+mergenum] = ii
          oo += mergenum

      ## add new points to triangulation
      triadd(XY[ooo:oo,:])


      ## remove dead soure nodes
      sXY  = sXY[sourcemask,:]
      snum = sXY.shape[0]

      if snum<ceil(smax*0.025):
        break

      if not itt % 50:
        aggt += time()-iti
        print("""{} | #i: {:6d} | #s: {:7.2f} | """\
              """#vn: {:6d} | #sn: {:6d}"""\
              .format(strftime("%Y-%m-%d %H:%M:%S"),\
                               itt,aggt,oo,snum))
        sys.stdout.flush()
        iti = time()

      itt += 1

  except KeyboardInterrupt:
    pass

  finally:

    ## set color
    ctx.set_source_rgb(FRONT,FRONT,FRONT)

    ## draws all vein nodes in
    draw(P,W,oo,XY)

    ## save to file
    sur.write_to_png('{:s}.veins.png'.format(OUT))

  return


if __name__ == '__main__' :
  OUT = 'profile'
  ## profile code
  pfilename = '{:s}.txt'.format(OUT)
  cProfile.run('main()',pfilename)
  p = pstats.Stats(pfilename)
  p.strip_dirs().sort_stats('cumulative').print_stats()

  ## run code regularly
  #main()

