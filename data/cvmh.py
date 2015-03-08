#!/usr/bin/env python
import subprocess as sp
from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import interp1d
import numpy as np

EXEC_DIR = '/cmld/data5/hinest/Data/CVM-H/bin'
MODEL_DIR = '/cmld/data5/hinest/Data/CVM-H/model'
INPUT_FILE = '/cmld/data5/hinest/Data/CVM-H/file.in'
OUTPUT_FILE = '/cmld/data5/hinest/Data/CVM-H/file.out'

def vel_to_lame(vp,vs,rho):
  mu = rho*np.power(vs,2.0)
  lam = rho*np.power(vp,2.0) - 2.0*mu   
  return mu,lam

def background_model(depth):
  depth = depth*-1.0
  depth_lst = np.array([0.0,
                    5.0,
                    6.0,
                    10.0,
                    15.5,
                    16.5,
                    22.0,
                    31.0,
                    33.0,
                    np.inf])  
  depth_lst *= 1000.0
  vp_lst = np.array([5.0,
                 5.5,
                 6.3,
                 6.3,
                 6.4, 
                 6.7,
                 6.75,
                 6.8,
                 7.8,
                 7.8])
  vp_lst *= 1000.0
  vp_interp = interp1d(depth_lst,vp_lst,kind='linear')
  vp = vp_interp(depth)
  rho = 1865.0 + 0.1579*vp
  nu = 0.25
  vs = vp * np.sqrt((0.5 - nu)/(1.0 - nu))
  return vp,vs,rho


def call_cvmh():
  out = sp.call('%s/vx_lite -s -z dep -m %s < %s > %s' % 
                (EXEC_DIR,MODEL_DIR,INPUT_FILE,OUTPUT_FILE),shell=True)
  return out

def _write_input(lons,lats,depths):
  f = open(INPUT_FILE,'w')
  for values in zip(lons,lats,depths):
    f.write('%s %s %s\n' % values)
  f.close()

def write_grid_input(lon_range,lat_range,depth_range):
  lats,lons,depths = np.meshgrid(lat_range,lon_range,depth_range)
  lats = lats.flatten()
  lons = lons.flatten()
  depths = depths.flatten()  
  _write_input(lons,lats,depths)

def make_interpolator():
  f = open(OUTPUT_FILE,'r')
  f_str = f.read()
  f_lst = f_str.strip().split('\n')
  f_array = np.array([j.strip().split() for j in f_lst])
  lon = np.array(f_array[:,0],dtype=float)
  lat = np.array(f_array[:,1],dtype=float)
  depth = np.array(f_array[:,2],dtype=float)
  rho = np.array(f_array[:,-1],dtype=float)
  vs = np.array(f_array[:,-2],dtype=float)
  vp = np.array(f_array[:,-3],dtype=float)
  #x,y = basemap(lon,lat) 
  points = np.array([lon,lat,-depth]).transpose()
  vp_interp = NearestNDInterpolator(points,vp)
  vs_interp = NearestNDInterpolator(points,vs)
  rho_interp = NearestNDInterpolator(points,rho)
  return (vp_interp,vs_interp,rho_interp)


