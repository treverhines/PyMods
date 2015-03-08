#!/usr/bin/env python                                                                                         
import numpy as np
import matplotlib.axes
import matplotlib.patches
from matplotlib.quiver import Quiver as _Quiver
from matplotlib.collections import EllipseCollection

##------------------------------------------------------------------------------
def _parse_init_args(*args):
  if len(args) == 2:
    u = args[0]
    v = args[1]
    z = None
    x = 0.0*u
    y = 0.0*v
    su = None
    sv = None
  elif len(args) == 3:
    u = args[0]
    v = args[1]
    z = args[2]
    x = 0.0*u
    y = 0.0*v
    su = None
    sv = None
  elif len(args) == 4:
    x = args[0]
    y = args[1]
    u = args[2]
    v = args[3]
    z = None
    su = None
    sv = None
  elif len(args) == 5:
    x = args[0]
    y = args[1]
    u = args[2]
    v = args[3]
    z = args[4]
    su = None
    sv = None
  elif len(args) == 6:
    x = args[0]
    y = args[1]
    u = args[2]
    v = args[3]
    z = None
    su = args[4]
    sv = args[5]
  elif len(args) == 7:
    x = args[0]
    y = args[1]
    u = args[2]
    v = args[3]
    z = args[4]
    su = args[5]
    sv = args[6]

  if z is None:
   return (x,y,u,v),su,sv
  else:
   return (x,y,u,v,z),su,sv

##------------------------------------------------------------------------------
class Quiver(_Quiver):
  def __init__(self,ax,*args,**kwargs):
    quiver_args,su,sv = _parse_init_args(*args)
    scale = kwargs.pop('scale',1.0)
    scale_units = kwargs.pop('scale_units','xy')
    angles = kwargs.pop('angles','xy')
    self.ellipse_kwargs = {}
    self.ellipse_kwargs['edgecolors'] = kwargs.pop('ellipse_edgecolors','k')
    self.ellipse_kwargs['facecolors'] = kwargs.pop('ellipse_facecolors','none')
    self.ellipse_kwargs['linewidths'] = kwargs.pop('ellipse_linewidths',1.0)
    _Quiver.__init__(self,
                     ax,
                     *quiver_args,
                     scale=scale,
                     scale_units=scale_units,
                     angles=angles,
                     **kwargs)
    if (su is not None) & (sv is not None):
      self._update_ellipsoids(su,sv)

  def _update_ellipsoids(self,su,sv):
    self.scale_units = 'xy'
    self.angles = 'xy'
    tips_x = self.X + self.U/self.scale
    tips_y = self.Y + self.V/self.scale
    tips = np.array([tips_x,tips_y]).transpose()
    width = 2.0*su/self.scale
    height = 2.0*sv/self.scale
    angle = 0.0*width # for now...
    if hasattr(self,'ellipsoids'):
      self.ellipsoids.remove()
    self.ellipsoids = EllipseCollection(width,
                                        height,
                                        angle,
                                        units=self.scale_units,
                                        offsets = tips,
                                        transOffset=self.ax.transData,
                                        **self.ellipse_kwargs)
    self.ax.add_collection(self.ellipsoids)

  def set_UVC(self,u,v,C=None,su=None,sv=None):
    if C is None:
      _Quiver.set_UVC(self,u,v)
    else:
      _Quiver.set_UVC(self,u,v,C)
    if (su is not None) & (sv is not None):
      self._update_ellipsoids(su,sv)

