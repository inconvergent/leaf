#!/usr/bin/python
# -*- coding: utf-8 -*-


import numpy as np
import cairo,Image
from time import time as time
import sys
from scipy.sparse import coo_matrix,csc_matrix 
from itertools import product


def main():
  """
  time to load up the ponies
  """

  ## numpy functions

  cos      = np.cos
  sin      = np.sin
  arctan2  = np.arctan2
  sqrt     = np.sqrt
  random   = np.random.random
  pi       = np.pi
  ft       = np.float64
  bigint   = np.int64
  ones     = np.ones
  zeros    = np.zeros
  array    = np.array
  bool     = np.bool
  tile     = np.tile
  maximum  = np.maximum

  ## GLOBAL-ISH CONSTANTS (SYSTEM RELATED)

  SIZE   = 4000
  BACK   = 1.
  FRONT  = 0.
  OUT    = './img'
  STP    = 1./SIZE
  C      = 0.5
  RAD    = 0.4
  GRAINS = 10

  ## GLOBAL-ISH CONSTANTS (PHYSICAL PROPERTIES)

  sourceDist  = 20.*STP
  killzone    = 5. *STP
  veinNodeRad = 5. *STP
  vmax        = 2  *1e6
  smax        = 2000
  rootWidth   = 10.*STP


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
  vstroke = np.vectorize(stroke)

  def circ(x,y,cr):
    """
    draw circle for each (x,y) with radius cr
    """
    ctx.arc(x,y,cr,0,2.*pi)
    ctx.fill()
    return
  vcirc = np.vectorize(circ)


  def darts(xx,yy,rr,n):
    """
    get at most n random, uniformly distributed, points in a circle.
    centered at (xx,yy), with radius rr.
    """
    t        = 2.*pi*random(n)
    u        = random(n)+random(n)
    r        = zeros(n,dtype=ft)
    mask     = u>1.
    xmask    = np.logical_not(mask)
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

 
  def makeNodemap(oo,snum,distVS,distVV):
    """
    nodemap[i,j] == True if vein node i is relative neightbour of 
    source node j
    u_i is relative neightbour of s if for all u_i:
      ||v-s|| < max{ ||u_i-s||, ||v-u_i|| }
    """
    if oo>1:
      row,col = [],[]

      rowext = row.extend
      colext = col.extend
      for i in xrange(oo):
        repvv = tile(distVV[i,:,None],(1,snum))
        repvs = tile(distVS[i,:],(oo,1))
        ma    = maximum(distVS,repvv)
        jj    = ((repvs<ma).sum(axis=0) == oo-1).nonzero()[0]
        colext(jj)
        rowext([i]*len(jj))

    else:
      col = range(snum)
      row = [0]*snum

    nodemap = coo_matrix( ( [True]*len(col),(row,col) ),\
                shape=(oo,snum),dtype=bool ).tocsr()

    return nodemap

  ### INITIALIZE

  ctx.set_line_width(2./SIZE)

  ## ARRAYS

  X         = zeros(vmax,dtype=ft)
  Y         = zeros(vmax,dtype=ft)
  PARENT    = zeros(vmax,dtype=bigint)
  WIDTH     = zeros(vmax,dtype=float)
  PARENT[0] = 0

  sourceX,sourceY = darts(C,C,RAD,smax)
  snum = sourceX.shape[0]

  ## (START) VEIN NODES

  ## 0 is right, -np.pi/2 is down

  X[0] = C
  Y[0] = C+RAD
  oo   = 1

  ### MAIN LOOP

  itt = 0
  iti = time()
  try:
    while True:
      itt += 1

      ## distance: vein nodes -> source nodes

      distVS = zeros((oo,snum),dtype=ft)
      for i in xrange(oo):
        vsx = (X[i] - sourceX)**2
        vsy = (Y[i] - sourceY)**2
        distVS[i,:] = vsx+vsy
      distVS = sqrt(distVS)
      
      ## distance: vein nodes -> vein nodes
      distVV = zeros((oo,oo),dtype=ft)
      for i in range(oo):
        vvx = (X[:oo,None] - X[i])**2
        vvy = (Y[:oo,None] - Y[i])**2
        distVV[:,i]  = (vvx+vvy)[:,0]
      distVV = sqrt(distVV)

      nodemap = makeNodemap(oo,snum,distVS,distVV)

      ## grow new vein nodes
      cont = False
      for i in xrange(oo):
        mask = nodemap[i,:].nonzero()[1]
        if mask.any():
          cont = True
          tx = (X[i] - sourceX[mask]).sum()
          ty = (Y[i] - sourceY[mask]).sum()
          a  = arctan2(ty,tx)
          X[oo] = X[i] - cos(a)*veinNodeRad
          Y[oo] = Y[i] - sin(a)*veinNodeRad
          PARENT[oo] = i
          oo += 1

      if not cont:
        break

      ## mask out dead source nodes
      #sourcemask = ones(snum,dtype=bool)
      #for i in xrange(distVS.shape[0]):
        #sourcemask[distVS[i,:] < killzone] = False

      sourcemask = ones(snum,dtype=bool)
      for j in xrange(snum):
        vinds = nodemap[:,j].nonzero()[0]
        if (distVS[vinds,j]<killzone).all():
          sourcemask[j] = False

      ## map out dead source nodes
      sourceX = sourceX[sourcemask].copy()
      sourceY = sourceY[sourcemask].copy()
      snum = sourceX.shape[0]

      #timeitt = time()-timeitt
      #print timeitt

      if not itt % 50:
        print itt,oo, time()-iti
        sys.stdout.flush()
        iti = time()

  except KeyboardInterrupt:
    pass
  finally:

    ## show source nodes
    #ctx.set_source_rgb(1,0,0)
    #vcirc(sourceX,sourceY,[sourceDist/2.]*len(sourceX))
    #ctx.set_source_rgb(FRONT,FRONT,FRONT)

    ## simple vein width
    i = oo-1
    while i>1:
      ii = PARENT[i]
      while ii>1:
        WIDTH[ii]+=1.
        ii = PARENT[ii]
      i-=1
    wmax = WIDTH.max()
    WIDTH = sqrt(WIDTH/wmax)*rootWidth
    WIDTH[WIDTH<STP] = STP

    ## show vein nodes
    ctx.set_source_rgb(FRONT,FRONT,FRONT)
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
    ## save to file
    sur.write_to_png('{:s}.png'.format(OUT))

  return


if __name__ == '__main__' : main()

