#!/usr/bin/python
# -*- coding: utf-8 -*-


## when simulation is running you can abort by pressing C^c. The current
## geometry will then be drawn. Press C^c again to exit right away.


import numpy as np
import cairo
from time import time as time
from time import strftime
import sys
from scipy.spatial import Delaunay,distance
from collections import defaultdict

np.random.seed(1)


#@profile
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
  rowstack  = np.row_stack
  hstack    = np.hstack
  vstack    = np.vstack
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
  reshape   = np.reshape
  npsum     = np.sum
  npall     = np.all


  ## GLOBAL-ISH CONSTANTS (SYSTEM RELATED)
  
  ## pixel size of canvas
  SIZE = 1000
  ## background color (white)
  BACK = 1.
  ## foreground color (black)
  FRONT  = 0.
  ## filename of image
  OUT = './img.leaf'
  ## size of pixels on canvas
  STP = 1./SIZE
  ## center of canvas
  C = 0.5
  ## radius of circle with source nodes
  RAD = 0.4
  ## number of grains used to sand paint each vein node
  GRAINS = 1
  ## alpha channel when rendering
  ALPHA = 1.
  ## because five is right out
  FOUR = 4

  ## GLOBAL-ISH CONSTANTS (PHYSICAL PROPERTIES)
  
  ## minimum distance between source nodes
  sourceDist = 7.*STP
  ## a source node dies when all approaching vein nodes are closer than this
  ## only killzone == veinNode == STP will cause consistently visible merging
  ## of branches in rendering.
  killzone = STP
  ## radius of vein nodes when rendered
  veinNode = STP
  ## maximum number of vein nodes
  vmax = 1*1e7
  # number of inital source nodes
  sinit = 10000
  ## width of widest vein nodes when rendered
  rootW = 0.01*SIZE*STP
  ## width of smallest vein nodes when rendered
  leafW = 1.3*STP
  ## number of root (vein) nodes
  rootNodes = 2


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


  #@profile
  def dist2hd(x,y):
    d = zeros((x.shape[0],y.shape[0]),dtype=x.dtype)
    for i in xrange(x.shape[1]):
      diff2 = x[:,i,None]-y[:,i]
      diff2 **= 2.
      d += diff2
    sqrt(d,d)
    return d


  def draw(P,W,o,XY):
    """
    draws the veins
    """

    ## simple vein width calculation
    for i in reversed(xrange(rootNodes,o)):
      ii = P[i]
      while ii>1:
        W[ii]+=1.
        ii = P[ii]

    wmax = W.max()
    W = sqrt(W/wmax)*rootW

    W[W<leafW] = leafW
    #W[:] = leafW

    ## show vein nodes
    for i in reversed(range(rootNodes,o)):
      dxy = XY[P[i],:]-XY[i,:]
      a = arctan2(dxy[1],dxy[0])
      s = linspace(0,1,GRAINS)*veinNode
      xyp = XY[P[i],:] - array( cos(a),sin(a) )*s

      vcirc(xyp[0],xyp[1],[W[i]]*GRAINS)


  def randomPointsInCircle(n,xx=C,yy=C,rr=RAD):
    """
    get n random points in a circle.
    """

    ## random uniform points in a circle
    t = 2.*pi*random(n)
    u = random(n)+random(n)
    r = zeros(n,dtype=ft)
    mask = u>1.
    xmask = logicNot(mask)
    r[mask] = 2.-u[mask]
    r[xmask] = u[xmask]
    xyp = colstack(( rr*r*cos(t),rr*r*sin(t) ))
    dartsxy  = xyp + array( [xx,yy] )

    return dartsxy


  def darts(n):
    """
    get at most n random, uniformly distributed, points in a circle.
    centered at (xx,yy), with radius rr. points are no closer to each other
    than sourceDist.
    """
    
    dartsxy = randomPointsInCircle(n)

    jj = []

    ## remove new nodes that are too close to other 
    ## new nodes
    dists = cdist(dartsxy,dartsxy,'euclidean')
    for j in xrange(n-1):
      if all( dists[j,j+1:] > sourceDist ):
        jj.append(j)

    res = dartsxy[array(jj,dtype=bigint),:]
    lenres = res.shape[0]

    return res,lenres

  
  #@profile  
  def makeNodemap(snum,ltri,lXY,lsXY):
    """
    map and inverse map of relative neighboring vein nodes of all source nodes
    
    - SVdict[source node indices] = list with indices of neighboring vein nodes
    - VSdict[vein node indices] = list with indices of source nodes that have
        this vein as a neighbor

    u_i is relative neighbor of s if for all u_i:
      ||v-s|| < max{ ||u_i-s||, ||u_i-v|| }

      we save time by only checking the vein nodes that belong to the
      surounding simplices.
    """
   
    localtime = time()
    timesum = 0.
    SVdict = {}
    VSdict = defaultdict(list)
    distdict = defaultdict(list)
    
    # simplex -> vertices
    p  = ltri.simplices
    # s -> simplex
    js = ltri.find_simplex(lsXY,bruteforce=True,tol=1e-4) 
    # s -> neighboring simplices including s
    neigh = colstack(( ltri.neighbors[js],js ))
    # s -> potential neighboring points
    vv = ( unique( positive( p[ns,:]-FOUR ) ) \
             for ns in neigh )

    for j,ii in enumerate(vv):
      iin = ii.shape[0]

      distVS = reshape( dist2hd( lXY[ii],lsXY[j:j+1] ), \
                        (iin,) )

      ##  = max { ||u_i-s||, ||u_i-v|| }
      distvv = dist2hd(lXY[ii,:],lXY[ii,:])

      mas = maximum( distvv,distVS )

      ## ||v-s|| < mas
      compare = reshape(distVS,(iin,1)) < mas
      mask = npsum(compare,axis=1) == iin-1
      
      ## element has reached killzone (/is too near a source node)
      if all( distVS[mask]<=killzone ):
        SVdict[j] = ii[mask]

      for d,i in zip(distVS[mask],ii[mask]):
        VSdict[i].append(j)
        distdict[i].append(d)

    VSdict = { i:array(jj) for i,jj in VSdict.iteritems() }
    distdict = { i:array(dd) for i,dd in distdict.iteritems() }

    VSdict2 = {}
    for i,jj in VSdict.iteritems():
      mask = distdict[i] > killzone
      if any(mask):
        VSdict2[i] = jj[mask]

    return VSdict2,SVdict


  #@profile
  def grow_new_veins(vs,lXY,lsXY,lP,o):
    """
    each vein node travels in the average direction of all source nodes
    affecting it.
    """

    for i,jj in vs.iteritems():
      txy = npsum( lXY[i,:]-sXY[jj,:], axis=0 )
      a = arctan2( txy[1],txy[0] )
      xy = array( [cos(a),sin(a)] )*veinNode
      lXY[o,:] = lXY[i,:] - xy 
      lP[o] = i
      o += 1
    return o

  #@profile
  def mask_out_dead(sv,n):

    mask = ones(n,dtype=bool)
    for j,_ in sv.iteritems():
      mask[j] = False
    
    return mask

  #@profile
  def retriangulate_add(tri,xy,o,oo):

    tri.add_points(xy[oo:o,:])

  #@profile
  def retriangulate_new(xyinit,xy,o):
    
    tri = triag( vstack(( xyinit,xy[:o,:]  )), \
               qhull_options='QJ Qc Pp')

    return tri

  ### INITIALIZE

  XY = zeros((vmax,2),dtype=ft)
  P = zeros(vmax,dtype=bigint)-1
  W = zeros(vmax,dtype=float)
  sXY,snum = darts(sinit)

  ctx.set_source_rgb(0.,0.,0.)
  vcirc(sXY[:snum,0], sXY[:snum,1], [leafW]*len(sXY))

  ## (START) VEIN NODES

  ## triangulation needs at least four initial points
  ## in addition we need the initial triangulation 
  ## to contain all source nodes
  ## remember that nodes in tri will be four indices higher than in X,Y

  o = rootNodes
  xyinit = zeros( (FOUR,2) )
  xyinit[:FOUR,:] = array( [[0.,0.],[1.,0.],[1.,1.],[0.,1.]] )


  for i in xrange(rootNodes):
    t = random()*2.*pi
    r = random()*RAD
    xy = C + array( [cos(t),sin(t)] )*r
    XY[i,:] = xy

  ## QJ makes all added points appear in the triangulation
  ## if they are duplicates they are added by adding a random small number to
  ## the node.
  tri = triag( vstack(( xyinit,XY[:o,:]  )), \
               incremental=True,
               qhull_options='QJ Qc')

  ### MAIN LOOP

  itt  = 0
  aggt = 0
  iti = time()

  tri_time = 0
  try:
    while True:

      oo = o

      VSdict,SVdict = makeNodemap(snum,tri,XY,sXY)

      ## grow new vein nodes
      o = grow_new_veins(VSdict, XY, sXY, P, o)
        
      ## mask out dead source nodes
      mask = mask_out_dead(SVdict, snum)


      ## add new points to triangulation
      #retriangulate_add(tri,XY,o,oo)
      tri = retriangulate_new(xyinit,XY,o)

      ## remove dead soure nodes
      sXY  = sXY[mask,:]
      snum = sXY.shape[0]

      if o==oo or snum<5:
        break

      if not itt % 50:
        aggt += time()-iti
        print("""{} | #i: {:6d} | #s: {:9.2f} | """\
              """#vn: {:6d} | #sn: {:6d}"""\
              .format(strftime("%Y-%m-%d %H:%M:%S"),\
                               itt,aggt,o,snum))
        sys.stdout.flush()
        iti = time()

      itt += 1

  except KeyboardInterrupt:
    print
    print 'aborted by keypress'
    print 

  finally:

    ctx.set_source_rgb(FRONT,FRONT,FRONT)

    print('\ndrawing ...\n')

    draw(P,W,o,XY)


    ## save to file
    sur.write_to_png('{:s}.veins.png'.format(OUT))

  return


if __name__ == '__main__' :
    main()

