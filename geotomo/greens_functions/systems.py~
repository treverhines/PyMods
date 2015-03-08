#!/usr/bin/env python
'''
module containing the classes used to simplify defining model geometry
'''
import numpy as np
from numpy import concatenate as cat
from misc import rotation3D
import misc

##-------------------------------------------------------------------------------
class ViscousSystem(dict):
  def __init__(self):
    self.blocks = 0
    self.update({'connectivity':np.zeros((0,0,0),dtype=int),
                 'point':np.zeros((0,3)),
                 'length':np.zeros(0),
                 'width':np.zeros(0),
                 'thickness':np.zeros(0),
                 'strike':np.zeros(0)})

  def add_group(self,point,length,width,thickness,strike,
                length_cuts,width_cuts,thickness_cuts,basemap):
    point_lon = point[0]
    point_lat = point[1]
    point_z = point[2]
    point_x,point_y = basemap(point_lon,point_lat)

    x_grid = cat(([0.0],length_cuts,[length]))
    n_length = len(length_cuts)+1
    y_grid = cat(([0.0],-width_cuts,[-width]))
    n_width = len(width_cuts)+1
    z_grid = np.concatenate(([0.0],-thickness_cuts,[-thickness])) 
    n_thickness = len(thickness_cuts)+1
    new_parameters = n_length*n_width*n_thickness
    self.update({'point':cat((self['point'],
                             np.zeros((new_parameters,3)))),
                 'length':cat((self['length'],
                              np.zeros(new_parameters))),
                 'width':cat((self['width'],
                             np.zeros(new_parameters))),
                 'thickness':cat((self['thickness'],
                                 np.zeros(new_parameters))),
                 'strike':cat((self['strike'],
                              np.zeros(new_parameters)))})

    argZ = -strike + np.pi/2.0
    rotation_matrix = rotation3D(argZ,0.0,0.0)
    x_spacing = np.abs(np.diff(x_grid))
    y_spacing = np.abs(np.diff(y_grid))
    z_spacing = np.abs(np.diff(z_grid))
    connectivity = np.zeros((n_length,n_width,n_thickness),dtype=int)
    pidx = 0
    for xitr,xval in enumerate(x_grid[:-1]):
      for yitr,yval in enumerate(y_grid[:-1]):
        for zitr,zval in enumerate(z_grid[:-1]):
            x,y,z = rotation_matrix.dot([xval,yval,zval])
            x += point_x
            y += point_y
            z += point_z
            lon,lat = basemap(x,y,inverse=True)
            self['point'][self.blocks,:] = np.array([lon,lat,z])
            self['length'][self.blocks] = x_spacing[xitr]
            self['width'][self.blocks] = y_spacing[yitr]
            self['thickness'][self.blocks] = z_spacing[zitr]
            self['strike'][self.blocks] = strike
            connectivity[xitr,yitr,zitr] = self.blocks
            self.blocks += 1

    pad_shape = (n_length+1,n_width,n_thickness)
    connectivity = misc.pad(connectivity,pad_shape,value=-1)
    self['connectivity'] = misc.pad_stack((self['connectivity'],connectivity),value=-1)

##-------------------------------------------------------------------------------
class FaultSystem(dict):
  def __init__(self):
    self.patches = 0
    self.segments = 0
    self.connectivity_tpl = ()
    self.update({'connectivity':np.zeros((0,0),int),
                 'point':np.zeros((0,3),float),
                 'length':np.zeros(0,float),
                 'width':np.zeros(0,float),
                 'segment':np.zeros(0,int),
                 'dip':np.zeros(0,float),
                 'strike':np.zeros(0,float)})

  def add_segment(self,point,length,width,strike,dip,patchsize,basemap):
    point_lon = point[0]
    point_lat = point[1]
    point_z = point[2]
    point_x,point_y = basemap(point_lon,point_lat)
    n_length = np.round(length/patchsize)
    n_width = np.round(width/patchsize)
    new_parameters = n_length*n_width
    self.update({'point':cat((self['point'],
                            np.zeros((new_parameters,3)))),
                'length':cat((self['length'],
                             np.zeros(new_parameters))),
                'width':cat((self['width'],
                            np.zeros(new_parameters))),
                'segment':cat((self['segment'],
                              np.zeros(new_parameters))),
                'dip':cat((self['dip'],
                          np.zeros(new_parameters))),
                'strike':cat((self['strike'],
                             np.zeros(new_parameters)))})

    argZ = -strike + np.pi/2.0
    argX = dip
    rotation_matrix = rotation3D(argZ,0.0,argX)
    x_grid = np.linspace(0.0,1.0,n_length+1)
    y_grid = np.linspace(0.0,-1.0,n_width+1)
    connectivity = np.zeros((n_length,n_width),dtype=int)
    for xitr,xval in enumerate(x_grid[:-1]):
      for yitr,yval in enumerate(y_grid[:-1]):
        x = xval*length
        y = yval*width
        x,y,z = rotation_matrix.dot([x,y,0.0])
        x += point_x
        y += point_y
        z += point_z
        lon,lat = basemap(x,y,inverse=True)        
        self['point'][self.patches,:] = np.array([lon,lat,z])
        self['length'][self.patches] = length/n_length
        self['width'][self.patches] = width/n_width
        self['strike'][self.patches] = strike
        self['dip'][self.patches] = dip
        self['segment'][self.patches] = self.segments
        connectivity[xitr,yitr] = self.patches
        self.patches += 1

    self.segments += 1
    pad_shape = (n_length+1,n_width)
    connectivity = misc.pad(connectivity,pad_shape,value=-1)
    self['connectivity'] = misc.pad_stack((self['connectivity'],connectivity),value=-1)

