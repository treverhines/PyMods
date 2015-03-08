#!/usr/bin/env python
import numpy as np
from misc import rotation3D
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import re
from mpl_toolkits.basemap import Basemap

CFM_DIR = '/cmld/data5/hinest/Data/CFM/CFM5_release_2014/tsurf/CFM5_socal_primary'


def plot_patch(ax,point,strike,dip,length,width,**kwargs):
  argZ = -strike + np.pi/2.0
  argX = dip
  rotation_matrix = rotation3D(argZ,0.0,argX)
  fold_idx = np.array([[0,3],[1,2]])
  coor = np.array([[0.0,  0.0, 0.0],
                   [0.0, -1.0, 0.0],
                   [1.0, -1.0, 0.0],
                   [1.0,  0.0, 0.0]])
  coor[:,0] *= length
  coor[:,1] *= width
  coor = np.array([rotation_matrix.dot(i) for i in coor])
  coor[:,0] += point[0]
  coor[:,1] += point[1]
  coor[:,2] += point[2]
  x = coor[:,0]
  y = coor[:,1]
  z = coor[:,2]
  x = x[fold_idx]
  y = y[fold_idx]
  z = z[fold_idx]
  f = ax.plot_surface(x,y,z,shade=False,**kwargs)
  return f

def fit_plane(points):
  x = points[:,0]
  y = points[:,1]
  z = points[:,2]
  G = np.array([x,y,1.0+0.0*x]).transpose()
  a,b,c = np.linalg.lstsq(G,z)[0]
  #compute strike and dip
  dip_direction = (np.arctan2(b,a) + np.pi)
  strike = (dip_direction + np.pi/2.0)
  dip = np.arctan(np.sqrt(a**2.0 + b**2))
  #compute length and width
  R = rotation3D(strike,0.0,dip)
  Rinv = R.transpose()
  rotated_points = np.array([Rinv.dot(i) for i in points])
  width = abs(min(rotated_points[:,1]) - max(rotated_points[:,1]))
  length = abs(min(rotated_points[:,0]) - max(rotated_points[:,0]))
  #find the top left point
  rotated_tl_point_0 = min(rotated_points[:,0])
  rotated_tl_point_1 = max(rotated_points[:,1])
  rotated_tl_point_2 = np.mean(rotated_points[:,2])
  rotated_tl_point = np.array([rotated_tl_point_0,rotated_tl_point_1,rotated_tl_point_2])
  tl_point = R.dot(rotated_tl_point)
  # convert strike so that it is angle w.r.t. north
  strike = (-strike + np.pi/2.0)%(2.0*np.pi)
  fig = plt.figure()
  ax = Axes3D(fig)
  ax.scatter3D(points[:,0],points[:,1],points[:,2])
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  plot_patch(ax,tl_point,strike,dip,length,width,alpha=0.2)
  plt.show()
  return strike,dip,length,width,tl_point

def fault_geometry(fault_file_name):
  f = open('%s/%s' % (CFM_DIR,fault_file_name),'r')
  fstr = f.read()

  vertex_strings = re.findall('VRTX \d* (.*)',fstr)
  vertex_array = np.array([j.strip().split() for j in vertex_strings],dtype=float)
  geom = fit_plane(vertex_array[:,:3])
  print(geom)



