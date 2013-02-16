#!/usr/bin/python
# -*- coding: utf-8 -*-


import numpy as np
import cairo,Image
from time import time as time
import sys
from scipy.sparse import coo_matrix,csc_matrix 


def main():
  """
  time to load up the ponies
  """

  ## numpy functions

  cos    = np.cos
  sin    = np.sin
  arctan = np.arctan2
  sqrt   = np.sqrt
  random = np.random.random
  pi     = np.pi
  ft     = np.float64
  bigint = np.int64

  ## GLOBAL-ISH CONSTANTS (SYSTEM RELATED)

  SIZE        = 1000
  BACK        = 1.
  FRONT       = 0.
  OUT         = './img.png'
  STP         = 1./SIZE
  C           = 0.5
  RAD         = 0.4
  MAXITT      = 200

  ## GLOBAL-ISH CONSTANTS (PHYSICAL PROPERTIES)

  sourceDist  = 10.*STP
  killzone    = 8. *STP
  veinNodeRad = 5. *STP
  vmax        = 2  *1e6
  smax        = 20


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
    r        = np.zeros(n,dtype=ft)
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

    o = np.array(o,dtype=bigint)
    return gridx[o],gridy[o]

  ### INITIALIZE

  ctx.set_line_width(2./SIZE)

  ## ARRAYS

  X      = np.zeros(vmax,dtype=ft)
  Y      = np.zeros(vmax,dtype=ft)
  PARENT = np.zeros(vmax,dtype=bigint)

  sourceX,sourceY = darts(C,C,RAD,smax)
  snum = sourceX.shape[0]
  sourcemask = np.zeros(snum,dtype=np.bool)
  sourcemask[:] = True

  ## SHOW SOURCE NODES
  #ctx.set_source_rgb(1,0,0)
  #vcirc(sourceX,sourceY,[sourceDist/2.]*len(sourceX))
  #ctx.set_source_rgb(FRONT,FRONT,FRONT)

  ## (START) VEIN NODES

  ## 0 is right, -np.pi/2 is down

  X[0] = C
  Y[0] = C+RAD
  oo   = 1

  ### MAIN LOOP

  itt = 0
  ti  = time()
  iti = time()
  try:
    while True:
      itt += 1
      #if itt > 3:
        #print oo
        #break

      ## distance: vein nodes -> source nodes
      distVS = np.zeros((oo,snum),dtype=ft)
      for i in xrange(oo):
        vsx = (X[i] - sourceX)**2
        vsy = (Y[i] - sourceY)**2
        distVS[i,:] = vsx+vsy
      distVS = sqrt(distVS)
      
      ## distance: vein nodes -> vein nodes
      distVV = np.zeros((oo,oo),dtype=ft)
      for i in range(oo):
        vvx = (X[:oo,None] - X[i])**2
        vvy = (Y[:oo,None] - Y[i])**2
        distVV[:,i]  = (vvx+vvy)[:,0]
      distVV = sqrt(distVV)

      ## mask out dead source nodes
      for i in xrange(oo):
        sourcemask[distVS[i,:] < killzone] = False

      # (for all u_i) ||v-s|| < max{ ||u_i-s||, ||v-u_i|| }
      
      row = []
      col = []
      for i in xrange(oo):
        kk = [k for k in range(oo) if not k==i]
        for j in xrange(snum):
          t  = distVS[i,j]
          ok = True
          for k in kk:
            try:
              ma = np.max(distVS[k,j],distVV[i,k])
            except Exception as e:
              ma = 1.
            if t >= ma:
              ok = False
              break
            
          if ok:
            col.append(j)
            row.append(i)

      nodemap = coo_matrix( ([True]*len(col),(row,col)),\
                  shape=(oo,snum),dtype=np.bool).tocsr()

      ## map: source node -> vein node
      #nodemap = distVS.argmin(axis=0)
     
      for i in xrange(oo):
        #mask = np.logical_and(nodemap==i,sourcemask)
        vmask   = np.logical_and(nodemap[i,:].todense(),sourcemask)
        mask    = np.zeros(snum,dtype=np.bool)
        mask[:] = vmask[:]

        if mask.any():
          tx = (X[i] - sourceX[mask]).sum()
          ty = (Y[i] - sourceY[mask]).sum()
          a  = np.arctan2(ty,tx)
          X[oo] = X[i] - cos(a)*veinNodeRad
          Y[oo] = Y[i] - sin(a)*veinNodeRad
          oo += 1

      if not itt % 20:
        print itt,oo, time()-iti
        sys.stdout.flush()
        iti = time()

      if not sourcemask.any() or itt > MAXITT or oo > vmax:
        break

    print('itt: {:d}  time: {:f}'.format(itt,time()-ti))
  except KeyboardInterrupt:
    pass
  finally:
    ## show source nodes
    #ctx.set_source_rgb(1,0,0)
    #vcirc(sourceX[sourcemask],sourceY[sourcemask],\
          #[sourceDist/2.]*sourcemask.sum())
    
    ## show wein nodes
    ctx.set_source_rgb(FRONT,FRONT,FRONT)
    vcirc(X[:oo],Y[:oo],[veinNodeRad/2]*(oo))

    ## save to file
    sur.write_to_png('{:s}'.format(OUT))

  return


if __name__ == '__main__' : main()

