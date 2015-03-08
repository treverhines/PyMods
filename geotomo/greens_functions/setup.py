#!/usr/bin/env python
'''
contains functions which write all the pylith and cubit input/configuration files
'''
import os
import pbs
import numpy as np
from misc import rotation3D
import h5py
from scipy.interpolate import interp1d

##------------------------------------------------------------------------------
def write_config(batchID,modID,fault_idx,visc_idx,
                 mat_file_name,fault_file_name,config_file):
  dimensions = 3
  run_time = 0.03125
  time_step = 0.03125
  surface_freq = 0.03125
  fault_freq = 0.03125
  element = 'TET'

  config_file.write('[pylithapp]\n')
  config_file.write('\n')
  config_file.write('[pylithapp.mesh_generator]\n')
  config_file.write('reader = pylith.meshio.MeshIOCubit\n')
  config_file.write('reorder_mesh=True\n')
  config_file.write('\n')
  config_file.write('[pylithapp.mesh_generator.reader]\n')
  config_file.write('filename = Config/%s/%s_%s/%s/mesh.exo\n' % (batchID,batchID,fault_idx,modID))
  config_file.write('\n')
  config_file.write('[pylithapp.timedependent.normalizer]\n')
  config_file.write('length_scale = 1000.0*m\n')
  config_file.write('shear_modulus = 3.2e+10*Pa\n')
  config_file.write('relaxation_time = 1.0*year\n')
  config_file.write('\n')
  config_file.write('[pylithapp.petsc]\n')
  config_file.write('pc_type = asm\n')
  config_file.write('ksp_type = gmres\n')
  config_file.write('ksp_rtol = 1.0e-14\n')
  config_file.write('ksp_atol = 1.0e-14\n')
  config_file.write('ksp_max_it = 2000\n')
  config_file.write('ksp_gmres_restart = 50\n')
  config_file.write('ksp_converged_reason = true\n')
  config_file.write('\n')
  config_file.write('[pylithapp.timedependent]\n')
  if dimensions == 2:
    config_file.write('bc = [XNeg,XPos,YNeg,YPos,ZNeg,ZZero]\n')
  elif dimensions == 3:
    config_file.write('bc = [XNeg,XPos,YNeg,YPos,ZNeg]\n')
  config_file.write('interfaces = [Fault]\n')
  config_file.write('\n')
  config_file.write('[pylithapp.timedependent.implicit]\n')
  config_file.write('time_step = pylith.problems.TimeStepUniform\n')
  config_file.write('output = [Surface_Grid]\n')
  config_file.write('output.Surface_Grid = pylith.meshio.OutputSolnPoints\n')
  config_file.write('\n')
  config_file.write('[pylithapp.timedependent.implicit.time_step]\n')
  config_file.write('total_time = %s*year\n' % run_time)
  config_file.write('dt = %s*year\n' % time_step)
  config_file.write('\n')
  config_file.write('[pylithapp.timedependent]\n')
  config_file.write('materials = [L1]\n')
  config_file.write('\n')
  config_file.write('[pylithapp.timedependent]\n')
  config_file.write('materials.L1 = pylith.materials.MaxwellIsotropic3D\n')
  config_file.write('\n')
  config_file.write('[pylithapp.timedependent.materials.L1]\n')
  config_file.write('label = L1 Material\n')
  config_file.write('id = 1\n')
  config_file.write('db_properties.label = L1 Material Properties\n')
  config_file.write('db_properties.iohandler.filename = %s\n' % mat_file_name)
  config_file.write('output.cell_data_fields = [maxwell_time]\n')
  config_file.write('output.cell_info_fields = []\n')
  if element == 'QUAD':
    config_file.write('quadrature.cell = pylith.feassemble.FIATLagrange\n')
  elif element == 'TET':
    config_file.write('quadrature.cell = pylith.feassemble.FIATSimplex\n')
  config_file.write('quadrature.cell.dimension = 3\n')
  config_file.write('\n')
  config_file.write('[pylithapp.timedependent.bc.XNeg]\n')
  config_file.write('bc_dof = [0,1,2]\n')
  config_file.write('label = Face_XNeg\n')
  config_file.write('db_initial.label = BC on XNeg\n')
  config_file.write('db_rate.label = BC Rate on XNeg\n')
  config_file.write('db_rate = spatialdata.spatialdb.UniformDB\n')
  config_file.write('db_rate.values = [displacement-rate-x,displacement-rate-y,displacement-rate-z,rate-start-time]\n')
  config_file.write('db_rate.data = [0.0*m/year,0.0*m/year,0.0*m/year,0.0*year]\n')
  config_file.write('\n')
  config_file.write('[pylithapp.timedependent.bc.XPos]\n')
  config_file.write('bc_dof = [0,1,2]\n')
  config_file.write('label = Face_XPos\n')
  config_file.write('db_initial.label = BC on XPos\n')
  config_file.write('db_rate.label = BC Rate on Xpos\n')
  config_file.write('db_rate = spatialdata.spatialdb.UniformDB\n')
  config_file.write('db_rate.values = [displacement-rate-x,displacement-rate-y,displacement-rate-z,rate-start-time]\n')
  config_file.write('db_rate.data = [0.0*m/year,0.0*m/year,0.0*m/year,0.0*year]\n')
  config_file.write('\n')
  if dimensions == 3:
    config_file.write('[pylithapp.timedependent.bc.YNeg]\n')
    config_file.write('bc_dof = [0,1,2]\n')
    config_file.write('label = Face_YNeg\n')
    config_file.write('db_initial.label = BC on YNeg\n')
    config_file.write('db_rate.label = BC Rate on YNeg\n')
    config_file.write('db_rate = spatialdata.spatialdb.UniformDB\n')
    config_file.write('db_rate.values = [displacement-rate-x,displacement-rate-y,displacement-rate-z,rate-start-time]\n')
    config_file.write('db_rate.data = [0.0*m/year,0.0*m/year,0.0*m/year,0.0*year]\n')
    config_file.write('\n')
    config_file.write('[pylithapp.timedependent.bc.YPos]\n')
    config_file.write('bc_dof = [0,1,2]\n')
    config_file.write('label = Face_YPos\n')
    config_file.write('db_initial.label = BC on YPos\n')
    config_file.write('db_rate.label = BC Rate on YPos\n')
    config_file.write('db_rate = spatialdata.spatialdb.UniformDB\n')
    config_file.write('db_rate.values = [displacement-rate-x,displacement-rate-y,displacement-rate-z,rate-start-time]\n')
    config_file.write('db_rate.data = [0.0*m/year,0.0*m/year,0.0*m/year,0.0*year]\n')
    config_file.write('\n')
  elif dimensions == 2:
    config_file.write('[pylithapp.timedependent.bc.ZZero]\n')
    config_file.write('bc_dof = [0,2]\n')
    config_file.write('label = Face_ZZero\n')
    config_file.write('db_initial.label = Dirichlet BC on +Z\n')
    config_file.write('db_rate.label = BC Rate on ZZero\n')
    config_file.write('db_rate = spatialdata.spatialdb.UniformDB\n')
    config_file.write('db_rate.values = [displacement-rate-x,displacement-rate-z,rate-start-time]\n')
    config_file.write('db_rate.data = [0.0*m/year,0.0*m/year,0.0*year]\n')
    config_file.write('\n')
    config_file.write('[pylithapp.timedependent.bc.YNeg]\n')
    config_file.write('bc_dof = [0,2]\n')
    config_file.write('label = Face_YNeg\n')
    config_file.write('db_initial.label = BC on YNeg\n')
    config_file.write('db_rate.label = BC Rate on YNeg\n')
    config_file.write('db_rate = spatialdata.spatialdb.UniformDB\n')
    config_file.write('db_rate.values = [displacement-rate-x,displacement-rate-z,rate-start-time]\n')
    config_file.write('db_rate.data = [0.0*m/year,0.0*m/year,0.0*year]\n')
    config_file.write('\n')
    config_file.write('[pylithapp.timedependent.bc.YPos]\n')
    config_file.write('bc_dof = [0,2]\n')
    config_file.write('label = Face_YPos\n')
    config_file.write('db_initial.label = BC on YPos\n')
    config_file.write('db_rate.label = BC Rate on YPos\n')
    config_file.write('db_rate = spatialdata.spatialdb.UniformDB\n')
    config_file.write('db_rate.values = [displacement-rate-x,displacement-rate-z,rate-start-time]\n')
    config_file.write('db_rate.data = [0.0*m/year,0.0*m/year,0.0*year]\n')
    config_file.write('\n')

  config_file.write('[pylithapp.timedependent.bc.ZNeg]\n')
  config_file.write('bc_dof = [0,1,2]\n')
  config_file.write('label = Face_ZNeg\n')
  config_file.write('db_initial.label = Dirichlet BC on -Z\n')
  config_file.write('db_rate.label = BC Rate on ZNeg\n')
  config_file.write('db_rate = spatialdata.spatialdb.UniformDB\n')
  config_file.write('db_rate.values = [displacement-rate-x,displacement-rate-y,displacement-rate-z,rate-start-time]\n')
  config_file.write('db_rate.data = [0.0*m/year,0.0*m/year,0.0*m/year,0.0*year]\n')
  config_file.write('\n')
  config_file.write('[pylithapp.timedependent.interfaces]\n')
  config_file.write('Fault = pylith.faults.FaultCohesiveKin\n')
  config_file.write('\n')
  config_file.write('[pylithapp.timedependent.interfaces.Fault]\n')
  config_file.write('label = Face_Fault\n')
  config_file.write('edge = Edge_Fault\n')
  if element == 'QUAD':
    config_file.write('quadrature.cell = pylith.feassemble.FIATLagrange\n')
  elif element == 'TET':
    config_file.write('quadrature.cell = pylith.feassemble.FIATSimplex\n')

  config_file.write('quadrature.cell.dimension = 2\n')
  config_file.write('eq_srcs = 0\n')
  config_file.write('eq_srcs.0.origin_time = 0.0*year\n')
  config_file.write('\n')
  config_file.write('[pylithapp.timedependent.interfaces.Fault.eq_srcs.0.slip_function]\n')
  config_file.write('slip.label = Slip 0\n')
  config_file.write('slip.iohandler.filename = %s\n' % fault_file_name)
  config_file.write('slip_time.label = Slip 0 initiation time\n')
  config_file.write('slip_time.iohandler.filename = Config/%s/%s_%s/%s/sliptime.spatialdb\n' % (batchID,batchID,fault_idx,modID))
  config_file.write('\n')

  config_file.write('[pylithapp.problem.formulation.output.Surface_Grid]\n')
  config_file.write('writer = pylith.meshio.DataWriterHDF5\n')
  config_file.write('vertex_data_fields = [displacement,velocity]\n')
  config_file.write('coordsys.space_dim = 3\n')
  config_file.write('coordsys.units = m\n')
  config_file.write('reader.filename = Batch/%s/surface_points.txt\n' % batchID)
  config_file.write('output_freq = time_step\n')
  config_file.write('time_step = %s*year\n' % surface_freq)
  config_file.write('writer.filename = Config/%s/%s_%s/%s/output_Surface.h5\n' %(batchID,batchID,fault_idx,modID))
  config_file.write('\n')
  config_file.write('[pylithapp.problem.interfaces.Fault.output]\n')
  config_file.write('writer = pylith.meshio.DataWriterHDF5\n')
  config_file.write('output_freq = time_step\n')
  config_file.write('time_step = %s*year\n' % fault_freq)
  config_file.write('vertex_info_fields = []\n')
  config_file.write('writer.filename = Config/%s/%s_%s/%s/output_Fault.h5\n' % (batchID,batchID,fault_idx,modID))
  config_file.write('\n')
  config_file.write('[pylithapp.timedependent.materials.L1.output]\n')
  config_file.write('writer = pylith.meshio.DataWriterHDF5\n')
  config_file.write('cell_filter = pylith.meshio.CellFilterAvg\n')
  config_file.write('output_freq = time_step\n')
  config_file.write('time_step = 1000*year\n')
  config_file.write('cell_info_fields = [maxwell_time]\n')
  config_file.write('cell_data_fields = []\n')
  config_file.write('writer.filename = Config/%s/%s_%s/%s/output_L1.h5\n' % (batchID,batchID,fault_idx,modID))

##------------------------------------------------------------------------------
def write_sliptime_spatialdb(file_buf):
  file_buf.write('#SPATIAL.ascii 1\n')
  file_buf.write('SimpleDB {\n')
  file_buf.write('  num-values  = 1\n')
  file_buf.write('  value-names = slip-time\n')
  file_buf.write('  value-units = year\n')
  file_buf.write('  num-locs    = 1\n')
  file_buf.write('  data-dim    = 0\n')
  file_buf.write('  space-dim   = 3\n')
  file_buf.write('  cs-data     = cartesian {\n')
  file_buf.write('    to-meters = 1.0\n')
  file_buf.write('    space-dim = 3\n')
  file_buf.write('  }\n')
  file_buf.write('}\n')
  file_buf.write('0.0 0.0 0.0 0.00\n')

##------------------------------------------------------------------------------
def background_model(depth):
  '''
  This is the SCEC-H background velocity model
  '''
  depth = depth*-1.0
  depth_lst = np.array([-0.1,
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

  # homogeneous model 
  vp = 5963.0 # mu = 3.2e10 lambda = 3.2e10
  vs = 3443.0
  rho = 2700.0
  return vp,vs,rho

##------------------------------------------------------------------------------
def write_mat_spatialdb(visc_context,eta,file_buf,basemap):
  visc_point = visc_context['point']
  visc_point_lon = visc_point[0]
  visc_point_lat = visc_point[1]
  visc_point_z = visc_point[2]
  visc_point_x,visc_point_y = basemap(visc_point_lon,visc_point_lat)
  
  depth_top = visc_point_z
  depth_bot = visc_point_z - visc_context['thickness']

  depth_range = np.linspace(-34000.0,0.0,50)
  depth_range = np.concatenate((depth_range,[depth_bot+10.0],
                                            [depth_bot-10.0],
                                            [depth_top+10.0],
                                            [depth_top-10.0]))

  argZ = -visc_context['strike'] + np.pi/2.0
  rotation_matrix = rotation3D(argZ,0.0,0.0)
  points = np.array([[ 0.5, -0.5,0.0],
                     [ 1.5, -0.5,0.0],
                     [-0.5, -0.5,0.0],
                     [ 0.5,  0.5,0.0],
                     [ 0.5, -1.5,0.0]])

  points[:,0] *= visc_context['length']
  points[:,1] *= visc_context['width']
  points = np.array([rotation_matrix.dot(i) for i in points])
  points[:,0] += visc_point_x
  points[:,1] += visc_point_y
    
  string = ''
  for d in depth_range:
    vp,vs,rho = background_model(d)
    if (d >= depth_bot) & (d < depth_top):
      visc = eta
    else:
      visc = 1.0e30
    string += '%s %s %s %s %s %s %s\n' % (points[0,0],
                                           points[0,1],
                                           d,
                                           vp,vs,rho,
                                           visc)
  for p in points[1:]:    
    for d in depth_range:
      vp,vs,rho = background_model(d)
      string += '%s %s %s %s %s %s 1.0e30\n' % (p[0],
                                                p[1],
                                                d,
                                                vp,vs,rho)
        
  point_no = len(depth_range)*len(points)
  file_buf.write('#SPATIAL.ascii 1\n')
  file_buf.write('SimpleDB {\n')
  file_buf.write('  num-values = 4\n')
  file_buf.write('  value-names =  vp vs density viscosity\n')
  file_buf.write('  value-units =  m/s m/s kg/m**3 Pa*s\n')
  file_buf.write('  num-locs = %s\n' % point_no)
  file_buf.write('  data-dim = 3\n')
  file_buf.write('  space-dim = 3\n')
  file_buf.write('  cs-data = cartesian {\n')
  file_buf.write('    to-meters = 1.0\n')
  file_buf.write('    space-dim = 3\n')
  file_buf.write('  }\n')
  file_buf.write('}\n')
  file_buf.write(string)

##------------------------------------------------------------------------------
def write_mat_spatialdb2(visc_context,eta,file_buf,basemap):
  visc_point = visc_context['point']
  visc_point_lon = visc_point[0]
  visc_point_lat = visc_point[1]
  visc_point_z = visc_point[2]
  visc_point_x,visc_point_y = basemap(visc_point_lon,visc_point_lat)
  
  argZ = -visc_context['strike'] + np.pi/2.0
  rotation_matrix = rotation3D(argZ,0.0,0.0)
  points = np.array([[ 0.5, -0.5, -0.5],
                     [ 1.5, -0.5, -0.5],
                     [-0.5, -0.5, -0.5],
                     [ 0.5,  0.5, -0.5],
                     [ 0.5, -1.5, -0.5],
                     [ 0.5, -0.5,  0.5],
                     [ 0.5, -0.5, -1.5]])
  points[:,0] *= visc_context['length']
  points[:,1] *= visc_context['width']
  points[:,2] *= visc_context['thickness']
  points = np.array([rotation_matrix.dot(i) for i in points])
  points[:,0] += visc_point_x
  points[:,1] += visc_point_y
  points[:,2] += visc_point_z
        
  point_no = 7
  file_buf.write('#SPATIAL.ascii 1\n')
  file_buf.write('SimpleDB {\n')
  file_buf.write('  num-values = 4\n')
  file_buf.write('  value-names =  vp vs density viscosity\n')
  file_buf.write('  value-units =  m/s m/s kg/m**3 Pa*s\n')
  file_buf.write('  num-locs = %s\n' % point_no)
  file_buf.write('  data-dim = 3\n')
  file_buf.write('  space-dim = 3\n')
  file_buf.write('  cs-data = cartesian {\n')
  file_buf.write('    to-meters = 1.0\n')
  file_buf.write('    space-dim = 3\n')
  file_buf.write('  }\n')
  file_buf.write('}\n')
  for itr in range(point_no):
    if itr == 0:
      val = eta
    else:
      val = 1.0e30
    file_buf.write('%s %s %s 5963.0 3443.0 2700.0 %s\n' % (points[itr,0],
                                                           points[itr,1],
                                                           points[itr,2],val))

##------------------------------------------------------------------------------
def write_leftlateral_spatialdb(file_buf):
  slip_dict = {'left-lateral':'1.0 0.0 0.0',
               'right-lateral':'-1.0 0.0 0.0',
               'thrust':'0.0 1.0 0.0',
               'normal':'0.0 -1.0 0.0'}

  slip = slip_dict['left-lateral']
  file_buf.write('#SPATIAL.ascii 1\n')
  file_buf.write('SimpleDB {\n')
  file_buf.write('  num-values = 3\n')
  file_buf.write('  value-names =  left-lateral-slip reverse-slip fault-opening\n')
  file_buf.write('  value-units =  m m m\n')
  file_buf.write('  num-locs = 1\n')
  file_buf.write('  data-dim = 0\n')
  file_buf.write('  space-dim = 3\n')
  file_buf.write('  cs-data = cartesian {\n')
  file_buf.write('    to-meters = 1.0\n')
  file_buf.write('    space-dim = 3\n')
  file_buf.write('  }\n')
  file_buf.write('}\n')
  file_buf.write('0.0 0.0 0.0 %s\n' % slip)

##------------------------------------------------------------------------------
def write_thrust_spatialdb(file_buf):
  slip_dict = {'left-lateral':'1.0 0.0 0.0',
               'right-lateral':'-1.0 0.0 0.0',
               'thrust':'0.0 1.0 0.0',
               'normal':'0.0 -1.0 0.0'}

  slip = slip_dict['thrust']
  file_buf.write('#SPATIAL.ascii 1\n')
  file_buf.write('SimpleDB {\n')
  file_buf.write('  num-values = 3\n')
  file_buf.write('  value-names =  left-lateral-slip reverse-slip fault-opening\n')
  file_buf.write('  value-units =  m m m\n')
  file_buf.write('  num-locs = 1\n')
  file_buf.write('  data-dim = 0\n')
  file_buf.write('  space-dim = 3\n')
  file_buf.write('  cs-data = cartesian {\n')
  file_buf.write('    to-meters = 1.0\n')
  file_buf.write('    space-dim = 3\n')
  file_buf.write('  }\n')
  file_buf.write('}\n')
  file_buf.write('0.0 0.0 0.0 %s\n' % slip)


##------------------------------------------------------------------------------
def write_mesh_journal(fault_context,file_buf,basemap):
  fault_point_lonlat = fault_context['point']
  fault_point_lon = fault_point_lonlat[0]
  fault_point_lat = fault_point_lonlat[1]
  fault_point_z = fault_point_lonlat[2]
  fault_point_x,fault_point_y = basemap(fault_point_lon,fault_point_lat)
  fault_point = np.array([fault_point_x,fault_point_y,fault_point_z])

  fault_length   = fault_context['length']
  fault_width    = fault_context['width']
  fault_strike   = fault_context['strike']
  fault_dip      = fault_context['dip']

  normalizer = fault_width
  normalizer = 10000.0
  fault_tet_size = 0.02*normalizer # 0.02
  domain_tet_size = 1.0*normalizer
  buffer_zone = 0.1*normalizer # 0.1
  domain_x_size = 120.0*normalizer # should be 120  
  domain_y_size = 120.0*normalizer # should be 120     
  domain_depth = -60.0*normalizer # should be -60                
  domain_x0 = fault_point[0]
  domain_y0 = fault_point[1]

  fault_ext_surf_points = np.array([[ -10.0,  10.0, 0.0],
                                    [  10.0,  10.0, 0.0],
                                    [  10.0, -10.0, 0.0],
                                    [ -10.0, -10.0, 0.0]])
  fault_surf_points = np.array([[  0.0,  0.0, 0.0],
                                [  1.0,  0.0, 0.0],
                                [  1.0, -1.0, 0.0],
                                [  0.0, -1.0, 0.0]])
  fault_box_points = np.array([[ 0.0, 0.0, 0.0],
                               [ 0.0, 0.0,-1.0],
                               [ 0.0,-1.0, 0.0],
                               [ 0.0,-1.0,-1.0],
                               [ 1.0, 0.0, 0.0],
                               [ 1.0, 0.0,-1.0],
                               [ 1.0,-1.0, 0.0],
                               [ 1.0,-1.0,-1.0]])
  fault_argZ = -fault_strike + np.pi/2.0
  fault_argX = fault_dip
  fault_rotation = rotation3D(fault_argZ,0.0,fault_argX)

  fault_box_thickness = 2*buffer_zone
  fault_width += fault_tet_size
  fault_length += fault_tet_size

  fault_box_points[:,2] *= fault_box_thickness
  fault_box_points[:,2] += buffer_zone
  fault_box_points[:,1] *= (fault_width +
                            -fault_point[2]/np.sin(fault_dip) +
                            buffer_zone/np.tan(fault_dip))
  fault_box_points[:,1] += -fault_point[2]/np.sin(fault_dip)
  fault_box_points[:,1] += buffer_zone/np.tan(fault_dip)
  fault_box_points[:,1] += 0.5*fault_tet_size
  fault_box_points[:,0] *= fault_length
  fault_box_points[:,0] -= 0.5*fault_tet_size
  fault_box_points = [fault_rotation.dot(i) for i in fault_box_points]
  fault_box_points = np.array(fault_box_points)
  fault_box_points[:,0] += fault_point[0]
  fault_box_points[:,1] += fault_point[1]
  fault_box_points[:,2] += fault_point[2]

  fault_ext_surf_points[:,0] *= fault_length
  fault_ext_surf_points[:,1] *= fault_width
  fault_ext_surf_points = [fault_rotation.dot(i) for i in fault_ext_surf_points]
  fault_ext_surf_points = np.array(fault_ext_surf_points)
  fault_ext_surf_points[:,0] += fault_point[0]
  fault_ext_surf_points[:,1] += fault_point[1]
  fault_ext_surf_points[:,2] += fault_point[2]

  fault_surf_points[:,0] *= fault_length
  fault_surf_points[:,0] -= 0.5*fault_tet_size
  fault_surf_points[:,1] *= fault_width
  fault_surf_points[:,1] += 0.5*fault_tet_size
  fault_surf_points = [fault_rotation.dot(i) for i in fault_surf_points]
  fault_surf_points = np.array(fault_surf_points)
  fault_surf_points[:,0] += fault_point[0]
  fault_surf_points[:,1] += fault_point[1]
  fault_surf_points[:,2] += fault_point[2]

  file_buf.write('set default element tet\n')

  if True:
    for i in fault_box_points:
      file_buf.write('create vertex %.16g %.16g %.16g\n' % (i[0],i[1],i[2]))

    for i in fault_ext_surf_points:
      file_buf.write('create vertex %.16g %.16g %.16g\n' % (i[0],i[1],i[2]))

    if fault_point[2] < (-0.5*fault_tet_size*np.sin(fault_dip)): # if the fault is buried                
      for i in fault_surf_points:
        file_buf.write('create vertex %.16g %.16g %.16g\n' % (i[0],i[1],i[2]))

    file_buf.write('create surface vertex 1 2 4 3\n')
    file_buf.write('create surface vertex 3 4 8 7\n')
    file_buf.write('create surface vertex 7 8 6 5\n')
    file_buf.write('create surface vertex 5 6 2 1\n')
    file_buf.write('create surface vertex 2 4 8 6\n')
    file_buf.write('create surface vertex 1 3 7 5\n')

    file_buf.write('create surface vertex 9 10 11 12\n')

    file_buf.write('create volume surface 1 2 3 4 5 6\n')
    file_buf.write('brick x %.16g y %.16g z %.16g\n' % (domain_x_size,
                                                        domain_y_size,
                                                        -domain_depth))
    file_buf.write('volume 8 move %.16g %.16g %.16g\n' % (domain_x0,
                                                          domain_y0,
                                                          domain_depth/2.0))
    file_buf.write('chop volume 8 with volume 6\n')
    file_buf.write('webcut volume 9 with sheet surface 7\n')
    file_buf.write('delete surface 7\n')
    if fault_point[2] < (-0.5*fault_tet_size*np.sin(fault_dip)): # if the fault is buried               
      file_buf.write('create surface vertex 13 14 15 16\n')
    file_buf.write('imprint all\n')
    file_buf.write('merge all\n')
    file_buf.write('volume 10 size %.16g\n' % domain_tet_size)
    file_buf.write('volume 11 9 size %.16g\n' % fault_tet_size)
    file_buf.write('mesh volume 9 10 11\n')
    file_buf.write('group "Face_YNeg" add node in surface 10\n')
    file_buf.write('group "Face_YPos" add node in surface 12\n')
    file_buf.write('group "Face_XNeg" add node in surface 11\n')
    file_buf.write('group "Face_XPos" add node in surface 13\n')
    file_buf.write('group "Face_ZZero" add node in surface 24 32 29\n')
    file_buf.write('group "Face_ZNeg" add node in surface 9\n')
    if fault_point[2] < (-0.5*fault_tet_size*np.sin(fault_dip)): # if the fault is buried
      file_buf.write('group "Face_Fault" add node in surface 35\n')
      file_buf.write('group "Edge_Fault" add node in curve 77 80 62 78\n')
    else:
      file_buf.write('group "Face_Fault" add node in surface 25\n')
      file_buf.write('group "Edge_Fault" add node in curve 61 62 63\n')
    file_buf.write('nodeset 1 Face_XNeg\n')
    file_buf.write('nodeset 1 name "Face_XNeg"\n')
    file_buf.write('nodeset 2 Face_XPos\n')
    file_buf.write('nodeset 2 name "Face_XPos"\n')
    file_buf.write('nodeset 3 Face_ZNeg\n')
    file_buf.write('nodeset 3 name "Face_ZNeg"\n')
    file_buf.write('nodeset 4 Face_ZZero\n')
    file_buf.write('nodeset 4 name "Face_ZZero"\n')
    file_buf.write('nodeset 5 Face_YNeg\n')
    file_buf.write('nodeset 5 name "Face_YNeg"\n')
    file_buf.write('nodeset 6 Face_YPos\n')
    file_buf.write('nodeset 6 name "Face_YPos"\n')
    file_buf.write('nodeset 7 Face_Fault\n')
    file_buf.write('nodeset 7 name "Face_Fault"\n')
    file_buf.write('nodeset 8 Edge_Fault\n')
    file_buf.write('nodeset 8 name "Edge_Fault"\n')
    file_buf.write('nodeset 5 node in curve in surface 11 13 remove\n')
    file_buf.write('nodeset 6 node in curve in surface 11 13 remove\n')
    if fault_point[2] < (-0.5*fault_tet_size*np.sin(fault_dip)): # if the fault is buried                
      file_buf.write('nodeset 3 node in curve in surface 11 13 10 12 35 remove\n')
      file_buf.write('nodeset 4 node in curve in surface 11 13 10 12 35 remove\n')
    else:
      file_buf.write('nodeset 3 node in curve in surface 11 13 10 12 25 remove\n')
      file_buf.write('nodeset 4 node in curve in surface 11 13 10 12 25 remove\n')
    file_buf.write('block 1 volume all\n')
    file_buf.write('block 1 name "L1"\n')

##------------------------------------------------------------------------------
def setup_batch_run(batchID,fault_context,visc_context,station_ids,basemap):
  f = fault_context
  v = visc_context
  f_parameters = len(f['point'])
  v_parameters = len(v['point'])
  visc_pert = 1.0e18
  slip_pert = 1.0
  os.makedirs('Batch/%s' % batchID)
  model_lst_file = open('Batch/%s/model_list.txt' % batchID,'w')
  for fidx in range(f_parameters):
    for vidx in range(v_parameters):
      modID = '%s_%s_%s' % (batchID,fidx,vidx)
      os.makedirs('Config/%s/%s_%s/%s' % (batchID,batchID,fidx,modID))
      f_context = {'point':f['point'][fidx],
                   'length':f['length'][fidx],
                   'width':f['width'][fidx],
                   'dip':f['dip'][fidx],
                   'strike':f['strike'][fidx]}
      v_context = {'point':v['point'][vidx],
                   'length':v['length'][vidx],
                   'width':v['width'][vidx],
                   'thickness':v['thickness'][vidx],
                   'strike':v['strike'][vidx]}
      slip_ll_file_name = 'Config/%s/%s_%s/%s/slip_leftlateral.spatialdb' % (batchID,batchID,fidx,modID)
      slip_thrust_file_name = 'Config/%s/%s_%s/%s/slip_thrust.spatialdb' % (batchID,batchID,fidx,modID)
      sliptime_file_name = 'Config/%s/%s_%s/%s/sliptime.spatialdb' % (batchID,batchID,fidx,modID)
      visc_file_name = 'Config/%s/%s_%s/%s/material.spatialdb' % (batchID,batchID,fidx,modID)
      mesh_file_name = 'Config/%s/%s_%s/%s/mesh.jou' % (batchID,batchID,fidx,modID)
      config_file_name = 'Config/%s/%s_%s/%s/%s.cfg' % (batchID,batchID,fidx,modID,modID)
      model_lst_file.write('Config/%s/%s_%s/%s\n' % (batchID,batchID,fidx,modID))
      slip_ll_file = open(slip_ll_file_name,'w')
      slip_thrust_file = open(slip_thrust_file_name,'w')
      visc_file = open(visc_file_name,'w')
      mesh_file = open(mesh_file_name,'w')
      config_file = open(config_file_name,'w')
      sliptime_file = open(sliptime_file_name,'w')
      write_leftlateral_spatialdb(slip_ll_file)
      write_thrust_spatialdb(slip_thrust_file)
      write_mat_spatialdb(v_context,visc_pert,visc_file,basemap)
      write_mesh_journal(f_context,mesh_file,basemap)
      write_config(batchID,modID,fidx,vidx,visc_file_name,slip_ll_file_name,config_file)
      write_sliptime_spatialdb(sliptime_file)
      initialize_greens_function_files(batchID,fault_context,visc_context,station_ids)
      slip_ll_file.close()
      slip_thrust_file.close()
      visc_file.close()
      mesh_file.close()
      config_file.close()
      sliptime_file.close()
  model_lst_file.close()

##------------------------------------------------------------------------------
def initialize_greens_function_files(batchID,fault_context,visc_context,station_ids):
  fault_no = len(fault_context['point'])
  visc_no = len(visc_context['point'])
  station_no = len(station_ids)
  e = h5py.File('Batch/%s/elastic_greens_functions.h5' % batchID,'w')
  e.create_dataset('value',(fault_no,2,station_no,3),dtype=float,fillvalue=np.nan)

  for key,val in fault_context.iteritems():
    e['patch_geometry/%s' % key] = val

  e['stations'] = station_ids
  e['disp_direction'] = ['e','n','v']
  e['slip_direction'] = ['left-lateral','thrust']
  e['value_units'] = ['meters']
  e.close()

  v = h5py.File('Batch/%s/viscous_greens_functions.h5' % batchID,'w')
  v.create_dataset('value',(visc_no,fault_no,2,station_no,3),dtype=float,fillvalue=np.nan)
  for key,val in fault_context.iteritems():
    v['patch_geometry/%s' % key] = val

  for key,val in visc_context.iteritems():
    v['block_geometry/%s' % key] = val

  v['stations'] = station_ids
  v['disp_direction'] = ['e','n','v']
  v['slip_direction'] = ['left-lateral','thrust']
  v['value_units'] = ['meters/year']
  v.close()


##------------------------------------------------------------------------------
def write_pbs_files(batchID,cpm,mem_limit,time_limit):
  chain_no = 400
  model_lst_file = open('Batch/%s/model_list.txt' % batchID,'r')
  model_lst_str = model_lst_file.read()
  model_lst = model_lst_str.strip().split('\n')
  PBS_lst = []
  for model_dir in model_lst:
    modID = model_dir.split('/')[-1]
    PBS_file_name  = '%s/pbs.sh' % model_dir
    PBS_lst += [PBS_file_name]
    mpm = int(1.0*mem_limit/cpm)    
    t = int(time_limit)
    PBS_header_kwargs = {
      'N':modID,
      'l':'nodes=1:ppn=%s,pmem=%smb,qos=flux,walltime=%s:00:00'%(cpm,mpm,t),
      'q':'flux',
      'A':'ehetland_flux',
      'r':'n',
      'V':'',
      'M':'hinest@umich.edu',
      'e':'%s/%s.err' % (model_dir,modID),
      'o':'%s/%s.log' % (model_dir,modID)
      }
    PBS_file = pbs.PBSFile(PBS_file_name,'w',**PBS_header_kwargs)
    PBS_file.write('cd $PBS_O_WORKDIR\n')
    PBS_file.write('touch %s/running\n' % model_dir)
    PBS_file.write('./PreProcess.py %s\n' % model_dir)

    PBS_file.write('. pylith_setup.sh\n')
    PBS_file.write('pylith %s/%s.cfg --nodes=%s '
                   '--timedependent.interfaces.Fault.eq_srcs.0.slip_function.'
                   'slip.iohandler.filename=%s/slip_leftlateral.spatialdb\n' % (model_dir,modID,cpm,model_dir))
    PBS_file.write('. ~/.bashrc\n')
    PBS_file.write("./PostProcess.py leftlateral %s\n" % model_dir)

    PBS_file.write('. pylith_setup.sh\n')
    PBS_file.write('pylith %s/%s.cfg --nodes=%s '
                   '--timedependent.interfaces.Fault.eq_srcs.0.slip_function.'
                   'slip.iohandler.filename=%s/slip_thrust.spatialdb\n' % (model_dir,modID,cpm,model_dir))
    PBS_file.write('. ~/.bashrc\n')
    PBS_file.write("./PostProcess.py thrust %s\n" % model_dir)
    PBS_file.write("rm %s/mesh.exo\n" % model_dir)
    PBS_file.write("rm %s/output_*\n" % model_dir)

    PBS_file.close()
  PBS_chain_heads = pbs.chain(PBS_lst,chain_no)
  chain_heads_file = open('Batch/%s/pbs_chain_heads.txt' % batchID,'w')
  for h in PBS_chain_heads:
    chain_heads_file.write('%s\n' % h)
  chain_heads_file.close()

