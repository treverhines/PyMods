#!/usr/bin/env python
'''
functions used for the inverse problem. This includes the system function which 
outputs predicted data given model parameters (e.g. slip and viscosity).  
'''
import numpy as np
import pickle
import misc
import inverse
import slip_functions

##-------------------------------------------------------------------------------
def rotate_slip_direction(M,rotation_list,inverse=False):
  '''
  PARAMETERS
  ----------
    M: array where the length of the second to last dimension is the number of 
       fault patches and the last dimension is 2 (for left lateral and thrust 
       slip on each patch)
    rotation_list:  The basis slip directions on each fault patch will be rotated
                    by these amounts in the counterclockwise direction. 0 means
                    that the basis slip directions will be left-lateral and 
                    thrust; pi/2 means that the basis slip directions will be 
                    thrust and right-lateral; etc.
    inverse: If False, then the resulting matrix is M described in terms of the
             basis slip directions which have been rotated counter-clockwise.  If
             True, then the resulting matrix is M described in terms of basis slip
             directions which have been rotated clockwise.

  RETURNS
  -------
    M_rotated: a rotated matrix with the same dimensions as M
  '''
  Mshape = np.shape(M)
  assert Mshape[-2] == len(rotation_list), ('then length of rake_list must be equal'
                                            ' to the number of fault patches, which'
                                            ' is the second to last dimension of M') 
  M_rotated = np.zeros(Mshape)
  for idx,theta in enumerate(rotation_list):
    if not inverse:
      R = np.array([[ np.cos(theta), np.sin(theta)],
                    [-np.sin(theta), np.cos(theta)]])
    else:
      R = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]])
    M_rotated[...,idx,:] = np.einsum('ij,...j',R,M[...,idx,:])
  
  return M_rotated
  
##-------------------------------------------------------------------------------
def system(slip_parameters,visc_parameters,F,G,slip_func,t):
  '''
  I: number of slip parameters describing slip on each fault patch       
  J: number of fault patches      
  K: number of slip directions (2 unless I decide to include tensile motion) 
  L: number of times where output is specified  
  M: number of viscous regions  

  slip_parameters: IxJxK array
  visc_parameters: M array
  F: LxJxK array
  G: LxMxJxK array
  slip_func: produces an LxJxK array
  '''
  b = slip_func(slip_parameters,t)
  b_int = slip_func(slip_parameters,t,time_integral=True)
  elastic = np.einsum('...jk,...jk',b,F)
  viscous = np.einsum('...jk,m,...jkm',b_int,visc_parameters,G)
  return elastic + viscous  

##-------------------------------------------------------------------------------
def jacobian(slip_parameters,visc_parameters,F,G,slip_func,slip_jac,t):
  '''
  I: number of slip parameters describing slip on each fault patch       
  J: number of fault patches      
  K: number of slip directions (2 unless I decide to include tensile motion) 
  L: number of times where output is specified  
  M: number of viscous regions  

  slip_parameters: IxJxK array
  visc_parameters: M array
  F: LxJxK array
  G: LxMxJxK array
  slip_jac: produces an LxIxJxK array

  outputs: LxIxJxK array

  Evaluates the jacobian of 'system' at the specified slip and viscous
  parameters

  PARAMETERS
  ----------
    slip_parameters: Array of slip parameters, where the first dimension
                     corresponds to the fault patch, the second dimension
                     corresponds to the direction of slip, and the third 
                     dimension corresponds to the term in the slip function
    visc_parameters: Array of viscous parameters, The first and only dimension
                     corresponds to the viscous regions
    F: elastic greens functions matrix
    G: viscous greens functions matrix
    A: slip function matrix
    intA: time integral of the slip function matrix

  RETURNS
  -------
    elastic: jacobian matrix for the elastic parameters. This is a four
             dimensional array, where the first dimension corresponds to 
             each data observation and the last three dimensions correspond
             to the dimensions of slip_parameters
    viscous: jacobian matrix for the viscous parameters. This is a two 
             dimensional array.  The first dimension corresponds to the 
             data observations and the second dimensions corresponds to the
             dimensions of visc_parameters
  '''
  b_jac = slip_jac(slip_parameters,t)
  b_int_jac = slip_jac(slip_parameters,t,time_integral=True)
  b_int = slip_func(slip_parameters,t,time_integral=True)
  # reshape array for easier broadcasting
  b_jac = np.einsum('lijk->iljk',b_jac)
  b_int_jac = np.einsum('lijk->iljk',b_int_jac)
  #G = np.einsum('lmjk->ljkm',G)

  # this can also be done with b_jac*F but I just want to be consistent with my
  # use of einsum
  elastic_term1 = np.einsum('...,...',b_jac,F)
  elastic_term2 = np.einsum('...,m,...m',b_int_jac,visc_parameters,G)    
  elastic_term = elastic_term1 + elastic_term2
  # This is an IxLxJxK matrix and I want a LxIxJxK matrix
  elastic_term = np.einsum('iljk->lijk',elastic_term)

  viscous_term = np.einsum('...jk,...jkm->...m',b_int,G)

  return elastic_term,viscous_term

##-------------------------------------------------------------------------------
def system_wrapper(parameters,F,G,slip_func,t,slip_parameter_shape):
  Fshape = np.shape(F)
  Gshape = np.shape(G)

  slip_no = np.prod(slip_parameter_shape)
  visc_no = Gshape[-1]

  slip_parameters = parameters[:slip_no]
  slip_parameters = np.reshape(slip_parameters,slip_parameter_shape)

  visc_parameters = parameters[-visc_no:]
  
  out = system(slip_parameters,visc_parameters,F,G,slip_func,t)
  
  return out

##-------------------------------------------------------------------------------
def jacobian_wrapper(parameters,F,G,slip_func,slip_jac,t,slip_parameter_shape):
  Gshape = np.shape(G)
  Fshape = np.shape(F)

  slip_no = np.prod(slip_parameter_shape)
  visc_no = Gshape[-1]

  slip_parameters = parameters[:slip_no]
  slip_parameters = np.reshape(slip_parameters,slip_parameter_shape)

  visc_parameters = parameters[-visc_no:]
  
  elastic_jac,visc_jac = jacobian(slip_parameters,visc_parameters,F,G,
                                  slip_func,slip_jac,t)
  
  elastic_jac = np.reshape(elastic_jac,(-1,slip_no))
  visc_jac = np.reshape(visc_jac,(-1,visc_no))
    
  out = np.hstack((elastic_jac,visc_jac))  

  return out

##-------------------------------------------------------------------------------
def regularization_matrix(slip_parameter_shape,
                          visc_no,
                          patch_connectivity,
                          visc_connectivity,
                          slip_order,viscosity_order):
  slip_no = np.prod(slip_parameter_shape)
  param_no = slip_no + visc_no
  reg_matrix = np.zeros((0,param_no),dtype=int)
  slip_indices = np.reshape(range(slip_no),slip_parameter_shape)
  for idx1 in range(slip_parameter_shape[0]):
    for idx2 in range(slip_parameter_shape[2]):
      slip_indices_subset = slip_indices[idx1,:,idx2]
      slip_indices_subset_ext = np.concatenate((slip_indices_subset,[-1]))
      connectivity = slip_indices_subset_ext[patch_connectivity]
      reg_matrix_component = inverse.tikhonov_matrix(connectivity,
                                                     slip_order,
                                                     column_no=param_no)
      reg_matrix = np.vstack((reg_matrix,reg_matrix_component))

  visc_indices = range(slip_no,slip_no+visc_no)  
  visc_indices_ext = np.concatenate((visc_indices,[-1]))
  connectivity = visc_indices_ext[visc_connectivity]
  reg_matrix_component = inverse.tikhonov_matrix(connectivity,
                                                 viscosity_order,
                                                 column_no=param_no)
  reg_matrix = np.vstack((reg_matrix,reg_matrix_component))
  
  return reg_matrix
