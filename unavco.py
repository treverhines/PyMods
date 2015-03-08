#!/usr/bin/env python
import os
import urllib
import time  as timemod
import numpy as np
import dateutil.parser
import matplotlib
import matplotlib.pyplot as plt
import scipy.sparse
from matplotlib.widgets import Slider
import misc
import pandas
import logging
from geotomo.plot.basemap import Basemap
from geotomo.plot.quiver import Quiver
matplotlib.quiver.Quiver = Quiver # for error ellipses

logger = logging.getLogger(__name__)

TODAY = misc.decyear(*tuple(timemod.gmtime())[:3]) #today in decimal year (eg. 2014.125)

try:
  UNAVCOPATH = os.environ['UNAVCOPATH']
except KeyError:
  print('\ncannot find UNAVOPATH:\n\n'
        'make an empty directory to store data downloaded from\n' 
        'unavco.org and save that directory location in your\n' 
        'environment as "UNAVCOPATH"\n')
  
##------------------------------------------------------------------------------
def _running_weighted_mean(values,sigma,times,dt,cut_times=[],schwarz=False):
  '''
  running mean filter

  This is currently the slowest part of my data processing 
  
  PARAMETERS
  ----------
    values: list of data points
    sigma: data point uncertainties
    times: times at which the data points where acquired
    dt: time window over which data points will be averaged.  This differs from
        the typical running mean because the window is normally defined in terms
        of the number of data points to be averaged
    cut_times: times that the running mean should not average over (e.g. jumps
               in the time series that should not be smoothed)
    schwarz: If true then the output uncertainty represents an upper bound on 
             uncertainty given by Schwarz inequality.  Otherwise, standard error
             propagation is used.  This should be true when data noise is highly
             correlated.

 
  RETURNS
  -------
    out_values: output values
    out_sigma: uncertainty of output values.  This is just the average of the
               uncertainties in the input values used to compute each output value, 
               which is the conservative estimate of uncertainties given by the
               Schwarz inequality
  ''' 
  cut_times = np.sort(cut_times)
  times = np.array(times)
  time_no = len(times)
  out_values = np.zeros(time_no)
  out_sigma = np.zeros(time_no)
  for itr in range(time_no):
    upper_time = times[itr] + dt/2.0
    lower_time = times[itr] - dt/2.0
    if any((upper_time > cut_times) & (times[itr] < cut_times)):
      upper_time = cut_times[(upper_time > cut_times) &
                             (times[itr] < cut_times)][0]
    if any((times[itr] >= cut_times) & (lower_time < cut_times)):
      lower_time = cut_times[(times[itr] >= cut_times) &
                             (lower_time < cut_times)][-1]
    indices = np.nonzero((times >= lower_time) &
                         (times <  upper_time))[0]
    numerator = np.sum(values[indices]/np.power(sigma[indices],2.0))
    denominator = np.sum(1/np.power(sigma[indices],2.0))
    out_values[itr] = numerator/denominator
    if schwarz:
      out_sigma[itr] = np.mean(sigma[indices])
    else:
      out_sigma[itr] = 1.0/np.sqrt(denominator)

  return out_values,out_sigma

##------------------------------------------------------------------------------
def _subsample(times,dt):
  '''
  subsample function
  
  PARAMETERS:
    times: times for each data acquisition
    dt: desired subsample spacing

  RETURNS
    out_indices: indices which can be used to subsample a time series   
  '''
  sort_idx = np.argsort(times)
  times = times[sort_idx]
  diff_times = np.diff(times)
  counter = 0.0
  subsample_idx = [0]
  for itr in range(len(times)-1):
    counter += diff_times[itr]
    if counter >= dt:
      subsample_idx += [itr+1]
      counter = counter%dt
  out_indices = sort_idx[subsample_idx]
  return out_indices

##------------------------------------------------------------------------------
def _make_station(ID,lat,lon,description,time_range,repository):
  '''
  used by __init__ method of StationDB
  '''
  dataframe = pandas.io.parsers.read_csv(
                '%s/%s/%s' % (UNAVCOPATH,repository,ID),
                skiprows=1,
                names=['date','north','east','vert',
                       'northstd','eaststd','vertstd','quality'])

  date = np.array(dataframe['date'])
  date = [dateutil.parser.parse(i) for i in date]
  time = [misc.decyear(i.year,i.month,i.day,i.hour,i.minute) for i in date] 
  time = np.array(time)
  north = np.array(dataframe['north'])
  east = np.array(dataframe['east'])
  vert = np.array(dataframe['vert'])
  northstd = np.array(dataframe['northstd'])
  eaststd = np.array(dataframe['eaststd'])
  vertstd = np.array(dataframe['vertstd'])

  bool_list = (time >= time_range[0]) & (time <= time_range[1])
  keep_idx, = np.nonzero(bool_list)

  time = time[keep_idx]

  north_dict = {}
  east_dict = {}
  vert_dict = {}
  northstd_dict = {}
  eaststd_dict = {}
  vertstd_dict = {}
  if len(time) > 0:
    north_dict['raw'] = north[keep_idx]
    east_dict['raw'] = east[keep_idx]
    vert_dict['raw'] = vert[keep_idx]
    northstd_dict['raw'] = northstd[keep_idx]
    eaststd_dict['raw'] = eaststd[keep_idx]
    vertstd_dict['raw'] = vertstd[keep_idx]
  
  out = {'ID':ID,
         'lat':lat,
         'lon':lon,
         'description':description,
         'time':time,
         'north':north_dict,
         'east':east_dict,
         'vert':vert_dict,
         'northstd':northstd_dict,
         'eaststd':eaststd_dict,
         'vertstd':vertstd_dict}
  return out

##------------------------------------------------------------------------------
def _bounding_box(lat_lst,lon_lst,center,dx,dy,basemap):
  '''
  used by __init__ method of StationDB
  '''
  lat_o = center[0]
  lon_o = center[1]
  x_lst,y_lst = basemap(lon_lst,lat_lst)
  x_o,y_o = basemap(lon_o,lat_o)

  return ((y_lst > (y_o - dy)) &
          (y_lst < (y_o + dy)) &
          (x_lst > (x_o - dx)) &
          (y_lst < (x_o + dx)))

##------------------------------------------------------------------------------
def _bounding_circle(lat_lst,lon_lst,center,radius,basemap):
  '''
  used by __init__ method of StationDB
  '''
  lat_o = center[0]
  lon_o = center[1]
  x_lst,y_lst = basemap(lon_lst,lat_lst)
  x_o,y_o = basemap(lon_o,lat_o)
  x_dif = x_lst - x_o
  y_dif = y_lst - y_o
  distance_square = np.power(x_dif,2.0) + np.power(y_dif,2.0)
  distance = np.sqrt(distance_square)
  return distance < radius

##------------------------------------------------------------------------------
def _create_default_basemap(lat_lst,lon_lst):
  '''
  used by __init__ method of StationDB
  '''
  llcrnrlon = min(lon_lst) - 0.2
  llcrnrlat = min(lat_lst) - 0.2
  urcrnrlon = max(lon_lst) + 0.2
  urcrnrlat = max(lat_lst) + 0.2
  lon_0 = (llcrnrlon + urcrnrlon)/2.0
  lat_0 = (llcrnrlat + urcrnrlat)/2.0
  return Basemap(projection='tmerc',
                 lon_0 = lon_0,
                 lat_0 = lat_0,
                 llcrnrlon = llcrnrlon,
                 llcrnrlat = llcrnrlat,
                 urcrnrlon = urcrnrlon,
                 urcrnrlat = urcrnrlat,
                 resolution = 'h') 

##------------------------------------------------------------------------------
def _detrend_disp(time,disp,sigma,detrend_time,annual,semiannual):
  '''
  used by detrend method of StationDB
  '''
  var = np.power(sigma,2.0)
  cov = np.diag(var)

  design_matrix =  np.array([0.0*time + 1.0,time])
  if annual is True:
    design_matrix = np.concatenate(
                      (design_matrix,
                       [np.sin(2.0*np.pi*time) + np.cos(2.0*np.pi*time)]))

  if semiannual is True:
    design_matrix = np.concatenate(
                      (design_matrix,
                       [np.sin(np.pi*time) + np.cos(np.pi*time)]))

  design_matrix = design_matrix.transpose()

  preseismic_idx, = np.nonzero(time < detrend_time) 

  time_pre = time[preseismic_idx]
  disp_pre = disp[preseismic_idx]
  var_pre = var[preseismic_idx]
  wght_pre = scipy.sparse.diags(1.0/np.sqrt(var_pre),0)

  design_matrix_pre = design_matrix[preseismic_idx,:]
  
  weighted_inv_design = np.linalg.pinv(wght_pre.dot(design_matrix_pre))
  weighted_disp_pre = wght_pre.dot(disp_pre)

  secular_model = weighted_inv_design.dot(weighted_disp_pre)    
  secular_model_cov = weighted_inv_design.dot(weighted_inv_design.transpose())

  disp_secular = design_matrix.dot(secular_model)

  cov_secular = design_matrix.dot(secular_model_cov).dot(design_matrix.transpose())
  var_secular = np.diag(cov_secular)
  std_secular = np.sqrt(var_secular)

  disp_detrended = disp - disp_secular
  var_detrended = var + var_secular
  std_detrended = np.sqrt(var_detrended)

  return disp_detrended,std_detrended,disp_secular,std_secular

##----------------------------------------------------------------------
def _quiver_input(stationdb,time,disp_type,ref_time=None):
  '''
  used in the view method of StationDB
  '''
  x_lst = np.array([],float)
  y_lst = np.array([],float)
  u_lst = np.array([],float)
  v_lst = np.array([],float)
  z_lst = np.array([],float)
  su_lst = np.array([],float)
  sv_lst = np.array([],float)

  for sta in stationdb.itervalues():
    if not sta['north'].has_key(disp_type):
      continue

    x,y = stationdb.basemap(sta['lon'],sta['lat'])
    if (ref_time is not None):
      if ((ref_time > np.min(sta['time'])) &
          (ref_time < np.max(sta['time'])) &
          (time     > np.min(sta['time'])) &
          (time     < np.max(sta['time']))):

        closest_tidx = np.argmin(np.abs(sta['time'] - ref_time))
        uref = sta['east'][disp_type][closest_tidx]
        vref = sta['north'][disp_type][closest_tidx]
        zref = sta['vert'][disp_type][closest_tidx]

        closest_tidx = np.argmin(np.abs(sta['time'] - time))
        su = sta['eaststd'][disp_type][closest_tidx]
        sv = sta['northstd'][disp_type][closest_tidx]
        u = sta['east'][disp_type][closest_tidx] - uref
        v = sta['north'][disp_type][closest_tidx] - vref
        z = sta['vert'][disp_type][closest_tidx] - zref
      
      else:
        u = 0.0
        v = 0.0
        z = 0.0
        su = 0.0
        sv = 0.0

    else:
      if ((time > np.min(sta['time'])) &
          (time < np.max(sta['time']))):
        closest_tidx = np.argmin(np.abs(sta['time'] - time))
        su = sta['eaststd'][disp_type][closest_tidx]
        sv = sta['northstd'][disp_type][closest_tidx]
        u = sta['east'][disp_type][closest_tidx]
        v = sta['north'][disp_type][closest_tidx]
        z = sta['vert'][disp_type][closest_tidx]

      else:
        u = 0.0
        v = 0.0
        z = 0.0
        su = 0.0
        sv = 0.0

    u_lst = np.concatenate((u_lst,[u]))
    v_lst = np.concatenate((v_lst,[v]))
    su_lst = np.concatenate((su_lst,[su]))
    sv_lst = np.concatenate((sv_lst,[sv]))
    z_lst = np.concatenate((z_lst,[z]))
    x_lst = np.concatenate((x_lst,[x]))
    y_lst = np.concatenate((y_lst,[y]))

  return (x_lst,y_lst,u_lst,v_lst,z_lst,su_lst,sv_lst)

##------------------------------------------------------------------------------
def _background_map(basemap,ax,artists):
  '''
  used in the view method of StationDB
  '''
  #ax.patch.set_facecolor([0.0,0.0,1.0,0.2])
  #basemap.drawtopography(ax=ax,vmin=-6000,vmax=4000,alpha=1.0,zorder=0)
  #basemap.drawcoastlines(ax=ax,linewidth=1.5,zorder=1)
  #basemap.drawcountries(ax=ax,linewidth=1.5,zorder=1)
  #basemap.drawstates(ax=ax,linewidth=1,zorder=1)
  #basemap.drawrivers(ax=ax,linewidth=1,zorder=1)
  #basemap.drawmeridians(np.arange(np.floor(basemap.llcrnrlon),np.ceil(basemap.urcrnrlon),1.0),
  #                      labels=[0,0,0,1],dashes=[2,2],
  #                      ax=ax,zorder=1)
  #basemap.drawparallels(np.arange(np.floor(basemap.llcrnrlat),np.ceil(basemap.urcrnrlat),1.0),
  #                      labels=[1,0,0,0],dashes=[2,2],
  #                      ax=ax,zorder=1)
  plt.sca(ax)
  basemap.drawmapscale(units='km',
                     lat=basemap.latmin+(basemap.latmax-basemap.latmin)/10.0,
                     lon=basemap.lonmax-(basemap.lonmax-basemap.lonmin)/5.0,
                     fontsize=16,
                     lon0=(basemap.lonmin+basemap.lonmax)/2.0,
                     lat0=(basemap.latmin+basemap.latmax)/2.0,
                     barstyle='fancy',
                     length=100,zorder=10)
  #basemap.fillcontinents(color=[0.7,0.7,0.7,0.7],zorder=1,ax=ax)
  for a in artists:
    ax.add_artist(a)

##------------------------------------------------------------------------------
def _draw_stations(stationdb,ax,disp_type):
  '''
  used in the view method of StationDB
  '''
  station_point_lst = []
  station_point_label_lst = []
  station_label_lst = []
  for sid,sta in stationdb.iteritems():  
    if not any([sta['north'].has_key(i) for i in disp_type]):
      continue

    (x,y) = stationdb.basemap(sta['lon'],sta['lat'])
    station_point = ax.plot(x,y,'ko',markersize=3,picker=8,zorder=2)
    #ax.text(x,y,sid,fontsize=16)
    station_point_label_lst += [station_point[0].get_label()]
    station_label_lst += [sta['ID']]
    station_point_lst += station_point

  station_label_lst = np.array(station_label_lst)
  station_point_label_lst = np.array(station_point_label_lst)
  #return station_label_lst,station_point_label_lst,station_point_lst
  return station_label_lst,station_point_label_lst

##------------------------------------------------------------------------------
def _draw_scale(basemap,ax,scale_length,quiver_scale,sigma_list,color_list,text_list):
  '''
  used in the view method of StationDB

  scale length is in mm
  '''
  scale_items = len(color_list)

  u_scale = np.array([scale_length])
  sigma_u_scale = np.array(sigma_list)
  sigma_v_scale = np.array(sigma_list)
  v_scale = np.array([0.0])
  z_scale = np.array([0.0])
  x_scale = np.array([basemap.urcrnrx/10.0])
  y_scale = np.array([basemap.urcrnry/10.0])

  x = basemap.urcrnrx/10.0
  y = basemap.urcrnry/15.0
  dy = basemap.urcrnry/20.0

  for i in range(scale_items):
    ax.text(x,y+i*dy,text_list[i],fontsize=16)
    ax.quiver(x_scale,y_scale+i*dy,u_scale,v_scale,sigma_u_scale[i],sigma_v_scale[i],
              scale_units='xy',
              angles='xy',
              width=0.004,
              scale=quiver_scale,
              color=color_list[i])

  ax.text(x,y+scale_items*dy,'%s cm displacement' % np.round(scale_length/10.0,2),
          fontsize=16)

##------------------------------------------------------------------------------
def _plot_tseries(ax,station,direction,disp_type,formats,time_range,ref_time=None):
  '''
  used in the view method of StationDB
  '''
  # 1.0 year ticks
  #logdt = round(np.log2((max(station['time'])-min(station['time']))/5.0))
  dt    = 2.0
  tick_min = round(min(station['time'])/dt)*dt
  tick_max = round(max(station['time'])/dt)*dt + dt
  ticks = np.arange(tick_min,tick_max,dt)

  ax.cla()
  min_disp = np.inf
  max_disp = -np.inf
  for idx,d in enumerate(disp_type):
    if not station[direction].has_key(d):
      continue
    if (idx == 0) & (direction == 'north'):
      ax.set_title('Station %s' % station['ID'],fontsize=16)

    time = station['time']
    if ref_time is not None:
      closest_tidx = np.argmin(np.abs(station['time'] - ref_time))
      ref_disp = station[direction][d][closest_tidx]
      disp = station[direction][d] - ref_disp
    else:
      disp = station[direction][d]
    if any(disp[(time>=time_range[0]) & (time<=time_range[1])] < min_disp):
      min_disp = min(disp[(time>=time_range[0]) & (time<=time_range[1])])
    if any(disp[(time>=time_range[0]) & (time<=time_range[1])] > max_disp):
      max_disp = max(disp[(time>=time_range[0]) & (time<=time_range[1])])
    std = station['%sstd' % direction][d]
    if (d == 'raw') |(d == 'detrended'):
      ax.errorbar(time,disp,std,fmt=formats[idx])
    else:
      ax.plot(time,disp,formats[idx],lw=2)

  ax.set_xticks(ticks)
  diff_time = abs(time_range[0] - time_range[1])
  ax.set_xlim([time_range[0]-diff_time*0.05,time_range[1]+diff_time*0.05])
  diff_disp = abs(min_disp - max_disp) 
  ax.set_ylim([min_disp-diff_disp*.05,max_disp+diff_disp*.05]) 
  ax.ticklabel_format(useOffset=False)
  ax.minorticks_on()
  ax.set_ylabel('%s disp. (mm)' % direction,fontsize=16)

##------------------------------------------------------------------------------
class StationDB(dict):
  '''
  Stores, filters, and plots data downloaded from unavco.org

  behaves like a dictionary where each key is a station ID and the associated
  values are the station metadata or displacements in a particular direction

  EXAMPLE USAGE:

  >> stationdb = StationDB(selection_type='circle',
                           center=[32.258,-115.289]
                           repository='ElMayor')
  >> stationdb['P495']['north']['raw'] % calls uncleaned northing displacements
  >> stationdb.condense(0.1,cut_times=[2010.25]) % filters and subsamples 
  >> stationdb.detrend(2010.25) % detrends data at April 4, 2010
  >> stationdb.view() % displays data

  To do: -modify the hierarchy so that disp_type comes before disp_direction. 
         -fix "add_displacment" so that the new times do not need to match 
          existing times
  '''
  def __init__(self,
               selection_type='all',
               center=None,
               radius=None,
               dx = None,
               dy = None,
               basemap=None,
               time_range=[2000.0,TODAY],
               repository='data'):
    '''
    PARAMETERS
    ----------
      selection_type: Excludes stations within the data repository from being 
                      included in this StationDB instance.  Can be either 'box'
                      (excludes stations outside a bounding box), 'circle'
                      (excludes stations outside a bounding circle), or 'all'
                      (no stations are excluded)
      center: center of either the bounding box or bounding circle 
              (latitude,longitude) (not needed 'all' is specified)
      radius: radius of the bounding circle in meters
      dx: half-width (E-W direction) of the bounding box in meters
      dy: half-height (N-S direction) of the bounding box in meters
      basemap: basemap to be used when view() is called.  A basemap is created
               which includes all stations in the repository is no basemap is 
               provided
      time_range: Displacements measured within this time range will be 
                  downloaded
      repository: Name of directory where data is downloaded (created and 
                  populated with the update_data function
    '''  
      
    dataframe = pandas.io.parsers.read_csv(
                  '%s/%s/metadata.txt' % (UNAVCOPATH,repository),
                  names=['ID','description','lat','lon'])

    station_lat_lst        = np.array(dataframe['lat'])
    station_lon_lst        = np.array(dataframe['lon'])
    station_ID_lst         = np.array(dataframe['ID'])
    station_desc_lst       = np.array(dataframe['description'])

    # exclude some of the stations based on a bounding box of bounding circle
    if basemap is None:
      basemap = _create_default_basemap(station_lat_lst,station_lon_lst)

    if selection_type == 'box':
      bool_lst =  _bounding_box(station_lat_lst,
                                station_lon_lst,
                                center,
                                dx, 
                                dy,
                                basemap)

    if selection_type == 'circle':
      bool_lst = _bounding_circle(station_lat_lst,
                                  station_lon_lst,
                                  center,
                                  radius,
                                  basemap)

    if selection_type == 'all':
      bool_lst = np.ones(len(station_lat_lst))

    station_idx, = np.nonzero(bool_lst)

    station_ID_lst = station_ID_lst[station_idx]
    station_lat_lst = station_lat_lst[station_idx]
    station_lon_lst = station_lon_lst[station_idx]
    station_desc_lst = station_desc_lst[station_idx]

    station_tuple_lst = zip(station_ID_lst,
                            station_lat_lst,
                            station_lon_lst,
                            station_desc_lst)

    for i in station_tuple_lst:
      logger.debug('initiating station %s' % i[0])
      self[i[0]]   = _make_station(i[0],i[1],i[2],i[3],time_range,repository)

    self.basemap = basemap

    return

  def __repr__(self):
    '''
    This class inherited dict.__repr__, which means that all the content of the
    instance will be displayed when called.  I am returning this function to a 
    more appropriate object.__repr__
    '''
    return object.__repr__(self)

  def detrend(self,detrend_time,annual=False,semiannual=False):
    '''
    detrends the raw displacement data and stores the detrended displacements
    as a new dictionary entry for each station
    
    detrended data can be called with:
      >> <stationdb_instance>[<station_id>][<disp_direction>][detrended]  

    PARAMETERS
    ----------
      detrend_time: data before this time will be used to estimate a secular
                    trendline
      annual: whether to fit an annual term to the trendline
      annual: whether to fit an semiannual term to the trendline
    
    
    '''
    for station_ID,sta in self.iteritems():      
      logger.debug('detrending station %s' % station_ID)
      time = sta['time']
      if all(time < detrend_time):
        continue
      if not any(time < detrend_time):
        continue

      n_det,nstd_det,n_sec,nstd_sec = _detrend_disp(
                                        time,
                                        sta['north']['raw'],
                                        sta['northstd']['raw'],
                                        detrend_time,
                                        annual,
                                        semiannual)

      e_det,estd_det,e_sec,estd_sec = _detrend_disp(
                                        time,
                                        sta['east']['raw'],
                                        sta['eaststd']['raw'],
                                        detrend_time,
                                        annual,
                                        semiannual)

      v_det,vstd_det,v_sec,vstd_sec = _detrend_disp(
                                        time,
                                        sta['vert']['raw'],
                                        sta['vertstd']['raw'],
                                        detrend_time,
                                        annual,
                                        semiannual)

      bool_lst        = time >= detrend_time
      keep_idx,       = np.nonzero(bool_lst)

      sta['north']['secular'] = n_sec
      sta['east']['secular'] = e_sec
      sta['vert']['secular'] = v_sec
      sta['northstd']['secular'] = nstd_sec
      sta['eaststd']['secular'] = estd_sec
      sta['vertstd']['secular'] = vstd_sec

      sta['north']['detrended'] = n_det
      sta['east']['detrended'] = e_det
      sta['vert']['detrended'] = v_det
      sta['northstd']['detrended'] = nstd_det
      sta['eaststd']['detrended'] = estd_det
      sta['vertstd']['detrended'] = vstd_det
      sta['postseismic_indices'] = keep_idx
     
  def condense(self,dt,cut_times=[]):
    '''
    filters the raw displacement data with a running mean and then subsamples
    it. This overwrites the 'raw' data and should be used prior to calling the
    'detrend' method

    PARAMETERS
    ----------
      dt: time spacing for each subsample (years) and also the time window for 
          the running mean
      cut_times: list of times (in decimal years) which the running mean should
                 not average over

    '''
    for station_ID,sta in self.iteritems():
      logger.debug('cleaning station %s' % station_ID)
      if len(sta['time']) == 0:
        continue

      north_mean,northstd_mean = _running_weighted_mean(sta['north']['raw'],
                                                        sta['northstd']['raw'],
                                                        sta['time'],
                                                        dt,
                                                        cut_times,schwarz=True)     
      east_mean,eaststd_mean = _running_weighted_mean(sta['east']['raw'],
                                                      sta['eaststd']['raw'],
                                                      sta['time'],
                                                      dt,
                                                      cut_times,schwarz=True)     
      vert_mean,vertstd_mean = _running_weighted_mean(sta['vert']['raw'],
                                                      sta['vertstd']['raw'],
                                                      sta['time'],
                                                      dt,
                                                      cut_times,schwarz=True)     

      subsample_indices = _subsample(sta['time'],dt)

      north_condensed = north_mean[subsample_indices] 
      east_condensed = east_mean[subsample_indices] 
      vert_condensed = vert_mean[subsample_indices] 

      northstd_condensed = northstd_mean[subsample_indices] 
      eaststd_condensed = eaststd_mean[subsample_indices] 
      vertstd_condensed = vertstd_mean[subsample_indices] 
      time_condensed = sta['time'][subsample_indices]
    
      sta['north']['raw'] = north_condensed
      sta['east']['raw'] = east_condensed
      sta['vert']['raw'] = vert_condensed
      sta['northstd']['raw'] = northstd_condensed
      sta['eaststd']['raw'] = eaststd_condensed
      sta['vertstd']['raw'] = vertstd_condensed
      sta['time'] = time_condensed

  def add_displacement(self,disp,context,name,sigma=None):
    '''
    IN PROGRESS
    '''
    if sigma is None:
      sigma = 0.0*disp

    for pidx,p in enumerate(disp):
      sid = context['station_ID'][pidx]
      direction  = context['direction'][pidx]
      time  = context['time'][pidx]
      sta = self[sid]
      if not sta['north'].has_key(name):
        time_no = len(sta['time'])
        sta['north'][name] = np.zeros(time_no)
        sta['northstd'][name] = np.zeros(time_no)
        sta['east'][name]  = np.zeros(time_no)
        sta['eaststd'][name]  = np.zeros(time_no)
        sta['vert'][name]  = np.zeros(time_no)
        sta['vertstd'][name]  = np.zeros(time_no)

      tidx, = np.nonzero(time == sta['time'])
      tidx = tidx[0]     
      if direction == 'n':
        sta['north'][name][tidx] = p
        sta['northstd'][name][tidx] = sigma[pidx]

      if direction == 'e':
        sta['east'][name][tidx]  = p
        sta['eaststd'][name][tidx]  = sigma[pidx]

      if direction == 'v':
        sta['vert'][name][tidx]  = p
        sta['vertstd'][name][tidx]  = sigma[pidx]


  def remove_outliers(self,station_ID,time_range):
    '''
    removes data within a time range for a particular station
    '''
    station = self[station_ID]
    bool_list = (station['time'] < time_range[0]) | (station['time'] > time_range[1])
    keep_idx, = np.nonzero(bool_list)
    if len(keep_idx) == 0:
      self.pop(station_ID)
    else:
      station['time'] = station['time'][keep_idx]
      station['north']['raw'] = station['north']['raw'][keep_idx]
      station['east']['raw'] = station['east']['raw'][keep_idx]
      station['vert']['raw'] = station['vert']['raw'][keep_idx]
      station['vertstd']['raw'] = station['vertstd']['raw'][keep_idx]
      station['eaststd']['raw'] = station['eaststd']['raw'][keep_idx]
      station['northstd']['raw'] = station['northstd']['raw'][keep_idx]

  def data_and_context(self,disp_type='detrended',start_time=2000.0):
    '''
    Unravels each GPS observation of displacement into a vector

    RETURNS
    -------
      data: vector of all detrended displacements
      sigma: uncertainty of all detrended displacements
      context: dictionary with entries that describe the station ID, the time of
               acquisition, and displacement direction for each data point 
               
    '''

    data = []
    sigma = []
    data_context = {'station_ID':[],
                    'time':[],
                    'direction':[]}

    for sid,sta in self.iteritems():
      if not sta['north'].has_key(disp_type):
        continue
      bool_lst = sta['time'] >= start_time
      time_indices, = np.nonzero(bool_lst)

      for idx in time_indices:
        data_context['time'] += [sta['time'][idx]]
        data_context['direction'] += ['n']
        data_context['station_ID'] += [sid]
        data  += [sta['north'][disp_type][idx]]
        sigma += [sta['northstd'][disp_type][idx]]

        data_context['time'] += [sta['time'][idx]]
        data_context['direction'] += ['e']
        data_context['station_ID'] += [sid]
        data  += [sta['east'][disp_type][idx]]
        sigma += [sta['eaststd'][disp_type][idx]]

        data_context['time'] += [sta['time'][idx]]
        data_context['direction'] += ['v']
        data_context['station_ID'] += [sid]
        data  += [sta['vert'][disp_type][idx]]
        sigma += [sta['vertstd'][disp_type][idx]]

    data_context['time'] = np.array(data_context['time'])
    data_context['direction'] = np.array(data_context['direction'])
    data_context['station_ID'] = np.array(data_context['station_ID'])
    data = np.array(data)
    sigma = np.array(sigma)
    return (data,sigma,data_context)

  def view(self,
           disp_types=['raw','secular'],
           ts_formats=None,
           quiver_colors=None,
           ref_time=None,
           time_range=[2000.0,TODAY],
           scale_length=100,
           quiver_scale=0.001,
           artists=[]):
    '''
    Display GPS data in two figures. 1) a map view of displacement vectors, and
    2) a time series plot for selected stations

    PARAMETERS
    ----------
      disp_type: The data types that will be plotted if available.  Must be a
                 vector with elements 'raw','secular','detrended','predicted',
                 or 'residual'.

                 'raw': observered displacements that are not detrended
                 'secular': best fitting secular trend (available after calling
                            the 'detrend' method)
                 'detrended': 'raw' minus 'secular' displacements (available 
                              after calling the 'detrend' method)
                 'predicted': predicted displacements (available after calling 
                              the 'add_predicted' method
                 'residual': 'detrended' minus 'predicted' displacements 
                             (available after calling the 'add_predicted' 
                             method)
      ts_formats: vector of time series line formats for each disp type
                 (e.g. ['ro','b-'])                 
      quiver_colors: vector of quiver arrow colors for each disp type
      ref_time: If provided, zeros each of the displacement types at this time
      time_range: Range of times for the time series plot and map view time
                  slider
      scale_length: real length (in mm) of the map view scale
      quiver_scale: changes lengths of the quiver arrows, 
                    smaller number -> bigger arrow
      artists: additional artists to add to the map view plot
    '''
    if ts_formats is None:
      ts_formats_all = ['b','k','r','m','y','r']
      ts_formats = ts_formats_all[:len(disp_types)]

    if quiver_colors is None:
      quiver_colors_all = ['b','k','r','m','y','r']
      quiver_colors = quiver_colors_all[:len(disp_types)]

    assert len(ts_formats) == len(quiver_colors)
    assert len(ts_formats) == len(disp_types)

    time = time_range[0]

    main_fig = plt.figure('Map View',figsize=(10,11.78))
    sub_fig = plt.figure('Time Series',figsize=(9.0,6.6))

    main_ax = main_fig.add_axes([0.08,0.08,0.76,0.76])
    slider_ax = main_fig.add_axes([0.08,0.88,0.76,0.04])
    color_ax = main_fig.add_axes([0.88,0.08,0.04,0.76])
    sub_ax1 = sub_fig.add_subplot(311)
    sub_ax2 = sub_fig.add_subplot(312)
    sub_ax3 = sub_fig.add_subplot(313)

    time_slider = Slider(slider_ax,'time',time_range[0],time_range[1],valinit=time_range[0])

    _background_map(self.basemap,main_ax,artists)

    station_label_lst,station_point_label_lst = _draw_stations(self,main_ax,disp_types)

    # figure out what uncertainties to use for the quiver scales 
    # a quiver scale will be generated for each shown displacement type
    # if the displacement type has a ts_format with an 'o' or '.' then sigma=1, otherwise sigma=0
    sigma_list = []
    for idx,i in enumerate(disp_types):
      if (i == 'raw') | (i =='detrended'):
        sigma_list += [1.0]
        ts_formats[idx] += '.'
      else:
        sigma_list += [0.0]

    _draw_scale(self.basemap,main_ax,scale_length,quiver_scale,sigma_list,quiver_colors,disp_types)

    Q_lst = []
    for idx,dt in enumerate(disp_types):
      args = _quiver_input(self,time,dt,ref_time=ref_time)                    
      Q_lst += [main_ax.quiver(args[0],args[1],args[2],args[3],args[5],args[6],
                               scale_units='xy',
                               angles='xy',
                               width=0.004,
                               scale=quiver_scale,
                               color=quiver_colors[idx],
                               zorder=3)]

    def _slider_update(time):
      for idx,dt in enumerate(disp_types):  
        args = _quiver_input(self,time,dt,ref_time=ref_time)
        Q_lst[idx].set_UVC(args[2],args[3],su=args[5],sv=args[6])     
      main_fig.canvas.draw() 
      return

    time_slider.on_changed(_slider_update)  

    def _onpick(event):
      #make time series
      artist_idx, = np.nonzero(event.artist.get_label() == station_point_label_lst)
      station_label = station_label_lst[artist_idx[0]]
      station = self[station_label]

      _plot_tseries(sub_ax1,station,'north',disp_types,ts_formats,time_range,ref_time=ref_time)
      _plot_tseries(sub_ax2,station,'east',disp_types,ts_formats,time_range,ref_time=ref_time)
      _plot_tseries(sub_ax3,station,'vert',disp_types,ts_formats,time_range,ref_time=ref_time)

      #adjust main figure
      sub_fig.canvas.draw()
      event.artist.set_markersize(10)
      main_fig.canvas.draw() 
      event.artist.set_markersize(3.0)
      return
  
    main_fig.canvas.mpl_connect('pick_event',_onpick)
    plt.show()


##----------------------------------------------------------------------
def update_data(lat_range,lon_range,repository='data'):
  '''
  Downloads the most recent GPS data from unavco.org
 
  PARAMETERS
  ----------
    lat_range: GPS data will be downloaded for stations within this range of
               latitudes
    lon_range: GPS data will be downloaded for stations within this range of
               longitudes
    repository: The GPS timeseries files and metadata file with be stored in
                this directory.  The full path to the downloaded data will be
                '<UNAVCOPATH>/<repository>'

  HARDCODED PARAMETERS
  --------------------
    reference frame: NAM08 
    start of data acquisition: January 1, 2000
    end of data acquisition: today
  '''  
  if os.path.exists('%s/%s' % (UNAVCOPATH,repository)):
    sp.call(['rm','-r','%s/%s' % (UNAVCOPATH,repository)])
  os.mkdir('%s/%s' % (UNAVCOPATH,repository))
  ref_frame            = 'nam08'
  start_time           = '2000-01-01T00:00:00'
  end_time             = timemod.strftime('%Y-%m-%dT00:00:00')
  string               = ('http://web-services.unavco.org:80/gps/metadata/'
                          'sites/v1?minlatitude=%s&maxlatitude='
                          '%s&minlongitude=%s&maxlongitude=%s&limit=10000' %
                          (lat_range[0],
                           lat_range[1],
                           lon_range[0],
                           lon_range[1]))
  buff                 = urllib.urlopen(string) 
  metadata_file        = open('%s/%s/metadata.txt' % (UNAVCOPATH,repository),'w')
  file_string          = buff.read()
  buff.close()
  file_lst             = file_string.strip().split('\n')
  station_lst          = []
  for i in file_lst[1:]:
    lst                = i.split(',')
    ID                 = lst[0]
    description        = lst[1]
    lat                = float(lst[2])
    lon                = float(lst[3]) 
    string             = ('http://web-services.unavco.org:80/gps/data/position'
                          '/%s/v1?referenceFrame=%s&starttime=%s&endtime='
                          '%s&tsFormat=iso8601' %
                         (ID,ref_frame,start_time,end_time))
    buff               = urllib.urlopen(string)  
    station_exists     = any([j[0] == ID for j in station_lst])
    if station_exists:
      logger.warning('station_exists')
    if ((buff.getcode() != 404) & 
        (not station_exists)):
      logger.info('updating station %s (%s)' % (ID,description))
      file_string      = buff.read()
      out_file         = open('%s/%s/%s' % (UNAVCOPATH,repository,ID),'w')
      out_file.write(file_string)
      out_file.close()      
      station_lst     += [(ID,description,lat,lon)]
      metadata_file.write('%s,%s,%s,%s\n' % (ID,description,lat,lon))
    metadata_file.flush()
    buff.close()   
  metadata_file.close()
  return




