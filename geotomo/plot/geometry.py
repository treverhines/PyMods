#!/Usr/bin/env python
'''
Module containing plotting functions
'''
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import geotomo.invert_postseismic as invert_postseismic
from misc import rotation3D
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import FancyArrowPatch
from matplotlib.widgets import Slider
from matplotlib.widgets import RadioButtons

#matplotlib.cm.PuBuGn = matplotlib.cm.CMRmap_r.cmap(

##-------------------------------------------------------------------------------
def _fault_patch_map_view(point,strike,dip,length,width,**kwargs):
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
  x = np.concatenate((coor[:,0],[coor[0,0]]))
  y = np.concatenate((coor[:,1],[coor[0,1]]))
  artist = [Line2D(x,y,**kwargs)]
  # look for surface points
  if any(coor[:,2] > -100.0):
    new_kwargs = {}
    new_kwargs.update(kwargs)
    lw = new_kwargs.pop('lw',1.0)
    lw *= 4.0
    surf_idx, = np.nonzero(coor[:,2] > -100.0)
    trace_artist = Line2D(coor[surf_idx,0],coor[surf_idx,1],lw=lw,**new_kwargs) 
    artist += [trace_artist]
  return artist

##-------------------------------------------------------------------------------
def _fault_patch_normal_view(point,strike,dip,length,width,patch_id,**kwargs):
  argZ = -strike + np.pi/2.0
  argX = dip
  rotation_matrix = rotation3D(argZ,0.0,argX)
  inv_rotation_matrix = rotation_matrix.transpose()
  point = inv_rotation_matrix.dot(point)
  xy = point[:2]
  xy[1] -= width
  artist = matplotlib.patches.Rectangle(xy,length,width,label=patch_id,picker=8,**kwargs)
  return artist

##-------------------------------------------------------------------------------
def _fault_patch_arrow(point,strike,dip,length,width,rake,scale,**kwargs):
  argZ = -strike + np.pi/2.0
  argX = dip
  rotation_matrix = rotation3D(argZ,0.0,argX)
  inv_rotation_matrix = np.linalg.inv(rotation_matrix)
  point = inv_rotation_matrix.dot(point)
  xy = point[:2]
  xy[1] -= width
  patch_center_x = xy[0] + length/2.0
  patch_center_y = xy[1] + width/2.0
  slip_x = scale*np.cos(rake)
  slip_y = scale*np.sin(rake)
  artist = FancyArrowPatch((patch_center_x,patch_center_y),
                           (patch_center_x + slip_x,patch_center_y + slip_y),
                           **kwargs)
  return artist

##-------------------------------------------------------------------------------
def _visc_block_map_view(point,strike,length,width,**kwargs):
  argZ = -strike + np.pi/2.0
  rotation_matrix = rotation3D(argZ,0.0,0.0)
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
  x = np.concatenate((coor[:,0],[coor[0,0]]))
  y = np.concatenate((coor[:,1],[coor[0,1]]))
  artist = Line2D(x,y,**kwargs)
  return artist

##-------------------------------------------------------------------------------
def _visc_block_side_view(point,strike,width,thickness,**kwargs):
  argZ = -strike + np.pi/2.0
  rotation_matrix = rotation3D(argZ,0.0,0.0)
  inv_rotation_matrix = rotation_matrix.transpose()
  point_rotated = inv_rotation_matrix.dot(point)
  point_rotated[2] -= thickness
  point_rotated[1] *= -1.0
  artist = Rectangle(point_rotated[1:],width,thickness,**kwargs)
  return artist

##-------------------------------------------------------------------------------
def fault_patches_map_view(context,basemap,**kwargs):
  sources = len(context['point'])
  artist_list = []
  for i in range(sources):
    point_lon = context['point'][i,0]
    point_lat = context['point'][i,1]
    point_z = context['point'][i,2]
    point_x,point_y = basemap(point_lon,point_lat)
    point = np.array([point_x,point_y,point_z])    
    artist_list += _fault_patch_map_view(point,
                                          context['strike'][i],
                                          context['dip'][i],
                                          context['length'][i],
                                          context['width'][i],
                                          **kwargs)
  return artist_list

##-------------------------------------------------------------------------------
def visc_blocks_map_view(context,basemap,**kwargs):
  sources = len(context['point'])
  artist_list = []
  for i in range(sources):
    point_lon = context['point'][i,0]
    point_lat = context['point'][i,1]
    point_z = context['point'][i,2]
    point_x,point_y = basemap(point_lon,point_lat)
    point = np.array([point_x,point_y,point_z])    
    artist_list += [_visc_block_map_view(point,
                                          context['strike'][i],
                                          context['length'][i],
                                          context['width'][i],
                                          **kwargs)]
  return artist_list

##-------------------------------------------------------------------------------
def visc_blocks_side_view(context,basemap,visc_parameters=None,
                          vmin=0.0,vmax=2.0,cmap='YlOrRd',**kwargs):
  cnorm = Normalize(vmin,vmax)
  cmap = ScalarMappable(cnorm,cmap)
  sources = len(context['point'])
  if visc_parameters is None:
    visc_parameters = np.ones(sources)
  artist_list = []
  for i in range(sources):
    point_lon = context['point'][i,0]
    point_lat = context['point'][i,1]
    point_z = context['point'][i,2]
    point_x,point_y = basemap(point_lon,point_lat)
    point = np.array([point_x,point_y,point_z])
    color = cmap.to_rgba(visc_parameters[i])
    artist_list += [_visc_block_side_view(point,
                                          context['strike'][i],
                                          context['width'][i],
                                          context['thickness'][i],
                                          facecolor=color,
                                          **kwargs)]
  return artist_list

##------------------------------------------------------------------------------
def plot_2Dslip(slip_parameters,
                slip_function,
                patch_context,
                fault_segment,
                time,
                basemap,
                vmin=0,vmax=1.0,
                cmap=matplotlib.cm.PuBuGn,
                arrow_length=5000.0,
                ref_time=0.0,
                min_slip=0.01,
                slip_type='total_slip'):

  artist_list = []
  cnorm = matplotlib.colors.Normalize(vmin,vmax)
  slip_cmap = matplotlib.cm.ScalarMappable(cnorm,cmap)
  slip_param_shape = np.shape(slip_parameters)
  slip_function_terms = slip_param_shape[0]
  slip_patch_no = slip_param_shape[1]
  mu = 3.2e10
  Mo = 0.0
  #find fault point

  point_o_defined = False
  for sp in range(slip_patch_no):

    if patch_context['segment'][sp] != fault_segment:
      continue    

    if not point_o_defined:   
      point_lon = patch_context['point'][sp,0]
      point_lat = patch_context['point'][sp,1]
      point_z = patch_context['point'][sp,2]
      point_x,point_y = basemap(point_lon,point_lat)
      point_o = np.array([point_x,point_y,point_z])
      point_o_defined = True

    patch_slip_parameters = slip_parameters[:,sp,0]
    
    if slip_type == 'slip_rate':
      strike_slip = slip_function_derivative(patch_slip_parameters,[time])[0]

    elif slip_type == 'total_slip':
      ref_strike_slip = slip_function(patch_slip_parameters,[ref_time])[0]
      strike_slip = slip_function(patch_slip_parameters,[time])[0]
      strike_slip = strike_slip - ref_strike_slip

    patch_slip_parameters = slip_parameters[:,sp,1]
    if slip_type == 'slip_rate':
      thrust_slip = slip_function_derivative(patch_slip_parmaeters,[time])[0]

    elif slip_type == 'total_slip':
      ref_thrust_slip = slip_function(patch_slip_parameters,[ref_time])[0]
      thrust_slip = slip_function(patch_slip_parameters,[time])[0]
      thrust_slip = thrust_slip - ref_thrust_slip

    slip_mag = np.sqrt(np.power(strike_slip,2.0) + np.power(thrust_slip,2.0))
    color = slip_cmap.to_rgba(slip_mag)
    point_lon = patch_context['point'][sp,0]
    point_lat = patch_context['point'][sp,1]
    point_z = patch_context['point'][sp,2]
    point_x,point_y = basemap(point_lon,point_lat)
    point = np.array([point_x,point_y,point_z]) - point_o
    strike = patch_context['strike'][sp]
    dip = patch_context['dip'][sp]
    length = patch_context['length'][sp]
    width = patch_context['width'][sp]
    artist_list += [_fault_patch_normal_view(point,strike,dip,length,width,str(sp),facecolor=color)]
    Mo += mu*length*width*slip_mag
    rake = np.arctan2(thrust_slip,strike_slip)

    if slip_mag > min_slip:
      artist_list += [_fault_patch_arrow(point,strike,dip,length,width,rake,
                                         5000.0*(slip_mag/vmax),
                                         mutation_scale=20*(slip_mag/vmax),
                                         lw=0.5*slip_mag/vmax,
                                         arrowstyle="simple",
                                         color="black",
                                         zorder=2)]
  Mw = (2.0/3.0*np.log10(Mo) - 6.0)
  print('Mo: %s, Mw: %s' % (Mo,Mw))    

  return artist_list

##-------------------------------------------------------------------------------
def view_model(slip_parameters,
               slip_function,
               visc_parameters,
               patch_context,
               visc_context,
               basemap,
               ref_time = None,
               time_range=[2008.0,2015.0],
               slip_cmax_range=[0.0,10.0],
               visc_cmax_range=[0.0,5.0]):

  if ref_time is None:
    ref_time = 0.0

  current_segment = [0]
  slip_type = ['total_slip']
  segments = np.unique(patch_context['segment'])
  segments = np.array(segments,int)
  segments_str = ['segment %s' % int(i) for i in segments]

  time = np.mean(time_range)
  slip_cmax = np.mean(slip_cmax_range)
  visc_cmax = np.mean(visc_cmax_range)

  fig_slip_ts = plt.figure('SlipTimeSeries',figsize=(10,10))
  fig_slip = plt.figure('Slip',figsize=(10,10))
  fig_visc = plt.figure('Viscosity',figsize=(10,10))

  slip_thrust_ts_ax = fig_slip_ts.add_subplot(211)
  slip_ll_ts_ax = fig_slip_ts.add_subplot(212)

  slip_cmax_ax = fig_slip.add_axes([0.08,0.86,0.76,0.02])
  time_ax = fig_slip.add_axes([0.08,0.84,0.76,0.02])
  slip_ax = fig_slip.add_axes([0.08,0.08,0.76,0.76])
  color_ax = fig_slip.add_axes([0.84,0.08,0.04,0.76])
  radio_ax = fig_slip.add_axes([0.08,0.08,0.2,0.2])
  radio2_ax = fig_slip.add_axes([0.28,0.08,0.2,0.2])

  visc_cmax_ax = fig_visc.add_axes([0.08,0.86,0.76,0.02])
  visc_ax = fig_visc.add_axes([0.08,0.08,0.76,0.76])
  visc_color_ax = fig_visc.add_axes([0.84,0.08,0.04,0.76])

  time_slider = Slider(time_ax,'time',time_range[0],time_range[1],valinit=time)
  slip_cmax_slider = Slider(slip_cmax_ax,'cmax',slip_cmax_range[0],slip_cmax_range[1],valinit=slip_cmax)

  radio = RadioButtons(radio_ax,segments)
  radio2 = RadioButtons(radio2_ax,['total_slip','slip_rate'])

  visc_cmax_slider = Slider(visc_cmax_ax,'cmax',visc_cmax_range[0],visc_cmax_range[1],valinit=visc_cmax)

  # fix this
  visc_ax.set_xlim(200000.0,400000.0)
  visc_ax.set_ylim(-100000.0,0000.0)

  visc_artists = visc_blocks_side_view(visc_context,basemap,visc_parameters=visc_parameters,
                                       vmin=0.0,vmax=visc_cmax)

  for i in visc_artists:
    visc_ax.add_artist(i)

  visc_cnorm = matplotlib.colors.Normalize(0,visc_cmax)
  visc_cmap = matplotlib.cm.ScalarMappable(visc_cnorm,'afmhot_r')
  visc_cmap.set_array(np.arange(0,10))
  fig_visc.colorbar(visc_cmap,cax=visc_color_ax)

  # fix this
  slip_ax.set_xlim(-5000,80000)
  slip_ax.set_ylim(-80000,5000)


  slip_artists = plot_2Dslip(slip_parameters,
                             slip_function,
                             patch_context,
                             current_segment,time,
                             basemap,
                             vmin=0,vmax=slip_cmax,
                             cmap=matplotlib.cm.PuBuGn,
                             arrow_length=5000.0,
                             ref_time=ref_time,
                             min_slip=0.01,
                             slip_type=slip_type[0])
  
  for i in slip_artists:
    slip_ax.add_artist(i)

  color_ax.cla()
  cnorm = matplotlib.colors.Normalize(0,slip_cmax)
  cmap = matplotlib.cm.ScalarMappable(cnorm,'PuBuGn')
  cmap.set_array(np.arange(0,10))
  #cmap = matplotlib.cm.PuBuGn
  #cmap.set_array(np.arange(0,10))
  fig_slip.colorbar(cmap,cax=color_ax)

  def click_function(value):
    slip_ax.cla()
    current_segment[0] = int(value)
    time = time_slider.val
    slip_cmax = slip_cmax_slider.val

    slip_artists = plot_2Dslip(slip_parameters,
                               slip_function,
                               patch_context,
                               current_segment,time,
                               basemap,
                               vmin=0.0,vmax=slip_cmax,
                               cmap=matplotlib.cm.PuBuGn,
                               arrow_length=5000.0,
                               ref_time=ref_time,
                               min_slip=0.01,
                               slip_type=slip_type[0])
    for i in slip_artists:
      slip_ax.add_artist(i)

    #slip_ax.set_xlim(190000,210000)
    #slip_ax.set_ylim(-160000,-140000)
    fig_slip.canvas.draw()

  def click_function2(value):
    slip_ax.cla()
    slip_type[0] = value
    time = time_slider.val
    slip_cmax = slip_cmax_slider.val
    slip_artists = plot_2Dslip(slip_parameters,
                             slip_function,
                             patch_context,
                             current_segment,time,
                             basemap,
                             vmin=0.0,vmax=slip_cmax,
                             cmap=matplotlib.cm.PuBuGn,
                             arrow_length=5000.0,
                             ref_time=ref_time,
                             min_slip=0.01,
                             slip_type=slip_type[0])
    for i in slip_artists:
      slip_ax.add_artist(i)

    fig_slip.canvas.draw()

  def slip_update_function(value):
    slip_ax.cla()
    time = time_slider.val
    slip_cmax = slip_cmax_slider.val
    cnorm = matplotlib.colors.Normalize(0,slip_cmax)
    cmap = matplotlib.cm.ScalarMappable(cnorm,'PuBuGn')
    slip_artists = plot_2Dslip(slip_parameters,
                             slip_function,
                             patch_context,
                             current_segment,time,
                             basemap,
                             vmin=0.0,vmax=slip_cmax,
                             cmap=matplotlib.cm.PuBuGn,
                             arrow_length=5000.0,
                             ref_time=ref_time,
                             min_slip=0.01,
                             slip_type=slip_type[0])
    for i in slip_artists:
      slip_ax.add_artist(i)

    color_ax.cla()
    cnorm = matplotlib.colors.Normalize(0,slip_cmax)
    cmap = matplotlib.cm.ScalarMappable(cnorm,'PuBuGn')
    cmap.set_array(np.arange(0,10))
    fig_slip.colorbar(cmap,cax=color_ax)
    fig_slip.canvas.draw()

  def visc_update_function(value):
    visc_ax.cla()
    visc_color_ax.cla()
    #visc_ax.set_xlim(200000.0,400000.0)
    #visc_ax.set_ylim(-150000.0,0000.0)
    visc_cmax = visc_cmax_slider.val
    visc_artists = visc_blocks_side_view(visc_context,basemap,visc_parameters=visc_parameters,
                                         vmin=0.0,vmax=visc_cmax)
    for i in visc_artists:
      visc_ax.add_artist(i)

    visc_cnorm = matplotlib.colors.Normalize(0,visc_cmax)
    visc_cmap = matplotlib.cm.ScalarMappable(visc_cnorm,'YlOrRd')
    visc_cmap.set_array(np.arange(0,10))
    fig_visc.colorbar(visc_cmap,cax=visc_color_ax)
    fig_visc.canvas.draw()


  def on_pick(event):
    patch_id = event.artist.get_label()
    patch_id = int(patch_id)
    tlist = np.linspace(time_range[0],time_range[1],500)
    ll_slip = slip_function(slip_parameters[:,patch_id,0],tlist)
    thrust_slip = slip_function(slip_parameters[:,patch_id,1],tlist)
    slip_thrust_ts_ax.cla()
    slip_ll_ts_ax.cla()
    slip_thrust_ts_ax.set_xlabel('year',fontsize=16)
    slip_thrust_ts_ax.set_ylabel('thrust slip (meters)',fontsize=16)
    slip_thrust_ts_ax.set_ylim((-5,5))
    slip_ll_ts_ax.set_xlabel('year',fontsize=16)
    slip_ll_ts_ax.set_ylabel('left-lateral slip (meters)',fontsize=16)
    slip_ll_ts_ax.set_ylim((-5,5))
    slip_thrust_ts_ax.plot(tlist,thrust_slip,'k',lw=2)
    slip_ll_ts_ax.plot(tlist,ll_slip,'k',lw=2)
    fig_slip_ts.canvas.draw()

  visc_cmax_slider.on_changed(visc_update_function)
  radio.on_clicked(click_function)
  radio2.on_clicked(click_function2)
  
  time_slider.on_changed(slip_update_function)
  slip_cmax_slider.on_changed(slip_update_function)
  fig_slip.canvas.mpl_connect('pick_event',on_pick)
  plt.show()


