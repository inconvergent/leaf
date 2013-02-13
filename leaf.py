#!/usr/bin/python
# -*- coding: utf-8 -*-


from numpy import cos, sin, pi, arctan2, sqrt, square, int
from numpy.random import random
import numpy as np
import cairo,Image
from time import time as time
import sys


def main():
  """
  time to load up the ponies
  """

  # GLOBAL-ISH CONSTANTS (SYSTEM RELATED)

  SIZE        = 1000
  BACK        = 1.
  FRONT       = 0.
  OUT         = './img.png'
  STP         = 1./SIZE
  C           = 0.5
  RAD         = 0.4
  MAXITT      = 50

  # GLOBAL-ISH CONSTANTS (PHYSICAL PROPERTIES)

  sourceDist  = 10.*STP
  killzone    = 10.*STP
  veinNodeRad = 5.*STP
  nmax        = 2*1e7
  smax        = 1000


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
    t = pi*2*random(n)
    u = random(n)+random(n)
    r = np.zeros(n,dtype=np.float)
    mask = u>1.
    xmask = np.logical_not(mask)
    r[mask] = 2.-u[mask]
    r[xmask] = u[xmask]
    xp = rr*r*cos(t)
    yp = rr*r*sin(t)
    gridx=xx+xp
    gridy=yy+yp

    o = []
    for i in xrange(n-1):
      dx = gridx[i] - gridx[i+1:]
      dy = gridy[i] - gridy[i+1:]
      dd = sqrt(dx*dx+dy*dy)
      if (dd > sourceDist).all():
        o.append(i)

    o = np.array(o,dtype=np.int)
    return gridx[o],gridy[o]

  # INITIALIZE

  ctx.set_line_width(2./SIZE)

  # ARRAYS

  X      = np.zeros(nmax,dtype=np.float)
  Y      = np.zeros(nmax,dtype=np.float)
  PARENT = np.zeros(nmax,dtype=np.int)

  sourceX,sourceY = darts(C,C,RAD,smax)
  sourcemask = np.zeros(len(sourceX),dtype=np.bool)
  sourcemask[:] = True

  ## SHOW SOURCE NODES
  #ctx.set_source_rgb(1,0,0)
  #vcirc(sourceX,sourceY,[sourceDist/2.]*len(sourceX))
  #ctx.set_source_rgb(FRONT,FRONT,FRONT)

  # (START) VEIN NODES

  ## 0 is right, -PI/2 is down

  X[0] = C
  Y[0] = C
  oo    = 1

  # MAIN LOOP

  itt = 0
  ti = time()
  try:
    while itt<MAXITT:
      itt += 1

      # distances from vein nodes to source nodes
      dd = []
      for i in xrange(oo):
        dx = X[i] - sourceX
        dy = Y[i] - sourceY
        d  = sqrt(dx*dx+dy*dy)
        dd.append(d)
        killmask = d < killzone
        sourcemask[killmask] = False
      
      distances = np.vstack(dd)
      nodemap   = distances.argmin(axis=0)
      
      for i in xrange(oo):
        mask = np.logical_and(nodemap==i,sourcemask)
        if mask.any():
          tx = dx[mask].sum()
          ty = dy[mask].sum()
          a  = arctan2(ty,tx)
          # unit vector of affecting source nodes
          sx = cos(a)
          sy = sin(a)
          
          X[oo] = X[i] - sx*veinNodeRad
          Y[oo] = Y[i] - sy*veinNodeRad
          #print i,oo,o,X[o],Y[o]
          oo += 1
          
      # paint the new vein nodes

      sys.stdout.flush()

    # SHOW SOURCE NODES
    ctx.set_source_rgb(1,0,0)
    vcirc(sourceX[sourcemask],sourceY[sourcemask],\
          [sourceDist/2.]*sourcemask.sum())
    ctx.set_source_rgb(FRONT,FRONT,FRONT)
    
    # SHOW VEIN NODES
    vcirc(X[:oo],Y[:oo],[veinNodeRad/2]*(oo))
    print oo


    print('itt: {:d}  time: {:f}'.format(itt,time()-ti))
  except KeyboardInterrupt:
    pass
  finally:
    sur.write_to_png('{:s}'.format(OUT))

  return


if __name__ == '__main__' : main()

