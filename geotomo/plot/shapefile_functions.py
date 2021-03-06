#!/usr/bin/env python
'''
This module is used to help plot the quaternary fault traces compiled by the 
usgs
'''
import shapefile
import numpy as np
from matplotlib.collections import LineCollection

def _contained_points(basemap,lon,lat):
  '''
  returns lon and lat points contained within the basemap bounds
  '''
  idx = np.nonzero((basemap.latmin < lat) & 
                   (basemap.lonmin < lon) & 
                   (basemap.latmax > lat) & 
                   (basemap.lonmax > lon))[0]
  return lon[idx],lat[idx]

def _shape_to_line_coordinates(shape,basemap):
  '''
  collects the x and y coordinates for each vertex in a shape and groups together 
  coordinates of connected vertices
  '''
  line_list = []
  points = np.array(shape.points)
  parts = np.array(shape.parts)
  parts = np.concatenate((parts,[-1]))
  for p1,p2 in zip(parts[:-1],parts[1:]):
    lon = points[p1:p2,0]
    lat = points[p1:p2,1]
    if basemap is None:
      xy = lon,lat
    else:
      lon,lat = _contained_points(basemap,lon,lat)
      if len(lon) == 0:
        continue
      xy = basemap(lon,lat)
    line_list += [zip(xy[0],xy[1])]
  return line_list

def shapefile_to_line_collection(fid,basemap=None,**kwargs):
  '''
  returns a LineCollection instance containing all the line segments in the 
  shape file
  
  Parameters
  ----------
    fid: shapefile name
    basemap: basemap instance to georeference the shapes
   
  Returns 
  -------
    collection: LineCollection instance
  '''  
  sf = shapefile.Reader(fid)
  line_list = []
  for s in sf.iterShapes():
    if hasattr(s,'points') and hasattr(s,'parts'):
      line_list += _shape_to_line_coordinates(s,basemap)
  collection = LineCollection(line_list,**kwargs)             
  return collection

def shapefile_to_xy_lists(fid,basemap=None,**kwargs):
  '''
  returns a LineCollection instance containing all the line segments in the 
  shape file
  
  Parameters
  ----------
    fid: shapefile name
    basemap: basemap instance to georeference the shapes
   
  Returns 
  -------
    collection: LineCollection instance
  '''  
  sf = shapefile.Reader(fid)
  line_list = []
  for s in sf.iterShapes():
    if hasattr(s,'points') and hasattr(s,'parts'):
      line_list += _shape_to_line_coordinates(s,basemap)
  return line_list
  
