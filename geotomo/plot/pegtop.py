#!/bin/env python
'''
This module was taken from  Ran Novitsky Nof's blog, Ron's Thoughts:
http://rnovitsky.blogspot.com/2010/04/using-hillshade-image-as-intensity.html
'''

from pylab import *
def set_shade(a,cmap=cm.jet,vmin=None,vmax=None,scale=10.0,azdeg=165.0,altdeg=45.0):
  ''' sets shading for data array based on intensity layer 
    or the data's value itself.
  inputs:
    a - a 2-d array or masked array
    intensity - a 2-d array of same size as a (no chack on that)
                      representing the intensity layer. if none is given
                      the data itself is used after getting the hillshade values
                      see hillshade for more details.
    cmap - a colormap (e.g matplotlib.colors.LinearSegmentedColormap
                instance)
    scale,azdeg,altdeg - parameters for hilshade function see there for
                more details
  output:
    rgb - an rgb set of the Pegtop soft light composition of the data and 
             intensity can be used as input for imshow()
  based on ImageMagick's Pegtop_light:
  http://www.imagemagick.org/Usage/compose/#pegtoplight'''
  if vmin is None:
    vmin = a.min()
  if vmax is None:
    vmax = a.max()
  intensity = hillshade(a,scale,azdeg,altdeg)
  # get rgb of normalized data based on cmap
  cnorm = matplotlib.colors.Normalize(vmin,vmax)
  #rgb = cmap(a-a.min())/float(a.max()-a.min()))[:,:,:3]
  rgb = cmap(cnorm(a))[:,:,:3]
  # form an rgb eqvivalent of intensity
  d = intensity.repeat(3).reshape(rgb.shape)
  # simulate illumination based on pegtop algorithm.
  rgb = 2*d*rgb+(rgb**2)*(1-2*d)
  return rgb

def hillshade(data,scale=10.0,azdeg=165.0,altdeg=45.0):
  ''' convert data to hillshade based on matplotlib.colors.LightSource class.
    input: 
         data - a 2-d array of data
         scale - scaling value of the data. higher number = lower gradient 
         azdeg - where the light comes from: 0 south ; 90 east ; 180 north ;
                      270 west
         altdeg - where the light comes from: 0 horison ; 90 zenith
    output: a 2-d array of normalized hilshade 
  '''
  # convert alt, az to radians
  az = azdeg*pi/180.0
  alt = altdeg*pi/180.0
  # gradient in x and y directions
  dx, dy = gradient(data/float(scale))
  slope = 0.5*pi - arctan(hypot(dx, dy))
  aspect = arctan2(dx, dy)
  intensity = sin(alt)*sin(slope) + cos(alt)*cos(slope)*cos(-az - aspect - 0.5*pi)
  intensity = (intensity - intensity.min())/(intensity.max() - intensity.min())
  return intensity
