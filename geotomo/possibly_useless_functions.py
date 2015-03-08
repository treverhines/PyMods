#!/usr/bin/env python
import numpy as np
'''
miscelaneous functions used throughout geotomo
'''

##------------------------------------------------------------------------------
def bias(xmin,xmax,bias_point,bias_factor,N):
  '''
  similar to linspace but skews the distribution towards one point

  arguments:
    xmin: min value in the list to be generated
    xmax: max value in the list to be generated
    bias_point: skew the distribution of entries in the list to be generated
                toward this point
    bias_factor: controls how heavily points are skewed toward the bias point.
                 1.0 will have linear spacing (unskewed), 2.0 will have spacing
                 increase quadratically away from the bias point.
    N: number of entries in the list being generated
  '''                
  xmin_p = (np.sign(xmin - bias_point)*
           np.power(np.abs(xmin - bias_point),1.0/bias_factor) +
           bias_point)
  xmax_p = (np.sign(xmax - bias_point)*
           np.power(np.abs(xmax - bias_point),1.0/bias_factor) +
           bias_point)
  x = np.linspace(xmin_p,xmax_p,N)
  x = (np.sign(x-bias_point)*
      np.power(np.abs(x-bias_point),bias_factor) +
      bias_point)
  return x

##------------------------------------------------------------------------------
def rotation3D(argZ,argY,argX):
  '''
  creates a matrix which rotates a coordinate in 3 dimensional space about the 
  z axis by argz, the y axis by argy, and the x axis by argx, in that order
  '''
  R1 = np.array([[  np.cos(argZ), -np.sin(argZ),           0.0],
                 [  np.sin(argZ),  np.cos(argZ),           0.0],
                 [           0.0,           0.0,           1.0]])

  R2 = np.array([[  np.cos(argY),           0.0,  np.sin(argY)],
                 [           0.0,           1.0,           0.0],
                 [ -np.sin(argY),           0.0,  np.cos(argY)]])

  R3 = np.array([[           1.0,           0.0,           0.0],
                 [           0.0,  np.cos(argX), -np.sin(argX)],
                 [           0.0,  np.sin(argX),  np.cos(argX)]])
  return R1.dot(R2.dot(R3))

##------------------------------------------------------------------------------
def H(x):
  '''
  Heaviside function
  '''
  x = np.array(x)
  return (x >= 0.0).astype(float)

##------------------------------------------------------------------------------
def intH(x):
  '''
  integral of the Heaviside function
  '''
  x = np.array(x)
  return x*(x >= 0.0).astype(float)

##------------------------------------------------------------------------------
def int2H(x):
  '''
  second integral of the Heaviside function
  '''
  x = np.array(x)
  return 0.5*np.power(x,2)*(x >= 0.0).astype(float)

##------------------------------------------------------------------------------
def find_index(A,B):
  '''
  If the entries in A are all found within B then this finds the list of indices 
  C such that B[C] = A.  I use this function to associate x,y coordinates of gps
  stations with the output from pylith.
  '''
  tol = 1.0
  A_no = len(A)
  idx_lst = np.zeros(A_no,int)
  for aidx,a in enumerate(A):
    found = False
    for bidx,b in enumerate(B):
      a_tple = tuple(a)
      b_tple = tuple(b)
      if ((a_tple[0] > (b_tple[0] - tol)) &
          (a_tple[0] < (b_tple[0] + tol)) &
          (a_tple[1] > (b_tple[1] - tol)) &
          (a_tple[1] < (b_tple[1] + tol))):
        idx_lst[aidx] = bidx
        found = True
    if not found:
      raise StandardError('element %s in A not found in B' % aidx)
  return idx_lst

##------------------------------------------------------------------------------
def intersect(vertices1,vertices2):
  '''
  box intersection algorithm... I dont use it anymore
  '''
  v1_xmin = min(vertices1[:,0])
  v1_xmax = max(vertices1[:,0])
  v2_xmin = min(vertices2[:,0])
  v2_xmax = max(vertices2[:,0])
  v1_x = np.array([v1_xmin,v1_xmax])
  v2_x = np.array([v2_xmin,v2_xmax])

  v1_ymin = min(vertices1[:,1])
  v1_ymax = max(vertices1[:,1])
  v2_ymin = min(vertices2[:,1])
  v2_ymax = max(vertices2[:,1])
  v1_y = np.array([v1_ymin,v1_ymax])
  v2_y = np.array([v2_ymin,v2_ymax])

  v1_zmin = min(vertices1[:,2])
  v1_zmax = max(vertices1[:,2])
  v2_zmin = min(vertices2[:,2])
  v2_zmax = max(vertices2[:,2])
  v1_z = np.array([v1_zmin,v1_zmax])
  v2_z = np.array([v2_zmin,v2_zmax])

  x_intersect = False
  y_intersect = False
  z_intersect = False
  if (any((v2_x >= v1_xmin) & (v2_x <= v1_xmax)) |
      any((v1_x >= v2_xmin) & (v1_x <= v2_xmax))):
    x_intersect = True
  if (any((v2_y >= v1_ymin) & (v2_y <= v1_ymax)) |
      any((v1_y >= v2_ymin) & (v1_y <= v2_ymax))):
    y_intersect = True
  if (any((v2_z >= v1_zmin) & (v2_z <= v1_zmax)) |
      any((v1_z >= v2_zmin) & (v1_z <= v2_zmax))):
    z_intersect = True
  if x_intersect & y_intersect & z_intersect:
    return True
  else:
    return False
