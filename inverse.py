#!/usr/bin/env python
import copy
import sys
import os
import numpy as np
import scipy.optimize
import scipy.sparse
import scipy.sparse.linalg
import logging
from misc import funtime
from misc import list_flatten

logger = logging.getLogger(__name__)

class Converger:
  def __init__(self,final,atol=0.01,rtol=0.01,norm=2):
    self.atol = atol
    self.rtol = rtol
    self.norm = norm
    self.final = np.asarray(final)
    self.L2 = np.inf
    return

  def __call__(self,current):
    current = np.asarray(current)
    L2_new = np.linalg.norm(current - self.final,self.norm)    
    if not np.isfinite(L2_new):
      message = 'encountered invalid L2'
      return 3,message

    elif L2_new <= self.atol:
      message = 'converged due to atol:          L2=%s' % L2_new
      return 0,message

    elif abs(L2_new - self.L2) <= self.rtol:
      message = 'converged due to rtol:          L2=%s' % L2_new
      return 0,message

    elif L2_new < self.L2:
      message = 'converging:                     L2=%s' % L2_new
      return 1,message

    elif (L2_new >= self.L2):
      message = 'diverging:                      L2=%s' % L2_new
      return 2,message

  def set(self,current):
    self.current = np.asarray(current)
    self.L2 = np.linalg.norm(current - self.final,self.norm)    
    return

##------------------------------------------------------------------------------
def correlated_noise(var,decay,times):
  N = len(times)
  mean = np.zeros(N)
  t1,t2 = np.meshgrid(times,times)
  cov = var*np.exp(-np.abs(t1 - t2)/decay)
  noise = np.random.multivariate_normal(mean,cov,1)
  return noise[0]

##------------------------------------------------------------------------------
def model_covariance(G,sigma):
  '''
  Returns the model covariance matrix.
  
  Arguments:
    G: system matrix
    sigma: data uncertainty

  '''  
  W = scipy.sparse.diags(1.0/sigma,0)
  Gg = np.linalg.pinv(W.dot(G))
  Cm = Gg.dot(Gg.transpose())
  return Cm

##------------------------------------------------------------------------------
def _remove_zero_rows(M):
  '''
  used in tikhonov_matrix
  '''
  keep_idx, = np.nonzero([np.any(i) for i in M])
  out = M[keep_idx,:]
  return out 

##------------------------------------------------------------------------------
def _linear_to_array_index(val,shape):
  '''
  used in next method of IndexEnumerate
  '''
  N = len(shape)
  indices = np.zeros(N,int)
  for count,dimsize in enumerate(shape[::-1]):
    indices[N-(count+1)] = val%dimsize
    val = val//dimsize
  return indices

##------------------------------------------------------------------------------
class Perturb:
  def __init__(self,v,delta=1):
    self.v = v
    self.delta = delta
    self.itr = 0 
    self.N = len(v)

  def __iter__(self):
    return self

  def next(self):
    if self.itr == self.N:
      raise StopIteration
    else:
      out = copy.deepcopy(self.v)
      out[self.itr] += self.delta
      self.itr += 1
      return out
      

##------------------------------------------------------------------------------
class IndexEnumerate:
  def __init__(self,C):
    '''
    used in tikhonov matrix

    enumerates over the flattened elements of C and their index locations in C
  
    e.g.
  
    >> C = np.array([[1,2],[3,4]])
    >> for idx,val in IndexEnumerate(C):
    ...  print('idx: %s, val: %s' % (idx,val))   
    ...
    idx: [0, 0], val: 1
    idx: [0, 1], val: 2
    idx: [1, 0], val: 3
    idx: [1, 1], val: 4
    '''
    self.C = C
    self.shape = np.shape(C)
    self.size = np.size(C)
    self.itr = 0

  def __iter__(self):
    return self

  def next(self):
    if self.itr == self.size:
      raise StopIteration
    else:
      idx = _linear_to_array_index(self.itr,self.shape)
      self.itr += 1
      return (idx,self.C[tuple(idx)])

##------------------------------------------------------------------------------
def _tikhonov_zeroth_order(C,L):
  '''
  used in tikhonov_matrix
  '''
  for val in C.flat:
    if (val == -1):
      continue
    L[val,val] = 1

  return L

##------------------------------------------------------------------------------
def _tikhonov_first_order(C,L):
  '''
  used in tikhonov_matrix
  '''
  shape = np.shape(C)
  Lrow = 0
  for idx,val in IndexEnumerate(C):
    if val == -1:
      continue
    for idx_pert in Perturb(idx,1):
      if any(idx_pert >= shape):
        continue      

      val_pert = C[tuple(idx_pert)]            
      if val_pert == -1:
        continue

      L[Lrow,val] += -1  
      L[Lrow,val_pert] +=  1
      Lrow += 1

  return L 

##------------------------------------------------------------------------------
def _tikhonov_second_order(C,L):
  '''
  used in tikhonov_matrix
  '''
  shape = np.shape(C)
  for idx,val in IndexEnumerate(C):
    if val == -1:
      continue
    for idx_pert in Perturb(idx,1):
      if any(idx_pert >= shape) | any(idx_pert < 0):
        continue
      val_pert = C[tuple(idx_pert)]      
      if val_pert == -1:
        continue
      L[val,val] += -1  
      L[val,val_pert]  +=  1

    for idx_pert in Perturb(idx,-1):
      if any(idx_pert >= shape) | any(idx_pert < 0):
        continue
      val_pert = C[tuple(idx_pert)]      
      if val_pert == -1:
        continue
      L[val,val] += -1  
      L[val,val_pert]  +=  1
       
  return L

##------------------------------------------------------------------------------
def _tikhonov_second_order_backup(C,L):
  '''
  used in tikhonov_matrix
  '''
  Cshape = np.shape(C)
  for idx,val in IndexEnumerate(C):
      dim = len(idx)
      for d in range(dim):
        # If the second derivative cannot be computed then continue
        if ((idx[d] + 1) >= Cshape[d]) | ((idx[d] - 1) < 0):
          continue  

        idx[d] += 1 
        for_val = C[tuple(idx)]
        idx[d] -= 1 

        idx[d] -= 1 
        back_val = C[tuple(idx)]
        idx[d] += 1 

        if (val == -1) | (for_val == -1) | (back_val == -1):
          continue

        # fill in the regularization matrix
        L[val,val] += -2  
        L[val,for_val]  +=  1
        L[val,back_val] +=  1
       
  return L

## Tikhonov matrix
##------------------------------------------------------------------------------
def tikhonov_matrix(C,n,column_no=None,dtype=None):
  '''
  Parameters
  ----------
    C: connectivity matrix, this can contain '-1' elements which can be used
       to break connections. 
    n: order of tikhonov regularization
    column_no: number of columns in the output matrix
    sparse_type: either 'bsr','coo','csc','csr','dia','dok','lil'

  Returns
  -------
    L: tikhonov regularization matrix saved as a csr sparse matrix

  Example
  -------
    first order regularization for 4 model parameters which are related in 2D 
    space
      >> Connectivity = [[0,1],[2,3]]
      >> L = tikhonov_matrix(Connectivity,1)
      
  '''     
  C = np.array(C)
  # check to make sure all values (except -1) are unique
  idx = C != -1
  params = C[idx] 
  unique_params = set(params)
  assert len(params) == len(unique_params), (
         'all values in C, except for -1, must be unique')

  Cdim = len(np.shape(C))
  max_param = np.max(C) + 1
  if column_no is None:
    column_no = max_param

  assert column_no >= max_param, (
         'column_no must be at least as large as max(C)')

  if n == 0:
    L = np.zeros((column_no,column_no),dtype=dtype)
    L =  _tikhonov_zeroth_order(C,L)

  if n == 1:
    L = np.zeros((Cdim*column_no,column_no),dtype=dtype)
    L = _tikhonov_first_order(C,L)

  if n == 2:
    L = np.zeros((column_no,column_no),dtype=dtype)
    L = _tikhonov_second_order(C,L)

  L = _remove_zero_rows(L)     

  return L


## Compute jacobian matrix w.r.t model parameters                          
##------------------------------------------------------------------------------
@funtime
def jacobian_fd(m_o,
                system,
                system_args=None,
                system_kwargs=None,
                dm=0.01,
                dtype=None):
  '''
  Parameters
  ----------
    system: function where the first argument is a list of model parameters and 
            the output is a data list
    m_o: location in model space where the jacobian will be computed. must be a
         mutable sequence (e.g. np.array or list)
    system_args: additional arguments to system
    system_kargs: additional key word arguments to system
    dm: step size used for the finite difference approximation

  Returns
  -------
    J:  jacobian matrix with dimensions: len(data),len(parameters)
  ''' 
  if system_args is None:
    system_args = []
  if system_kwargs is None:
    system_kwargs = {}

  data_o         = system(m_o,*system_args,**system_kwargs)
  param_no       = len(m_o)
  data_no        = len(data_o)

  Jac            = np.zeros((data_no,param_no),dtype=dtype)
  for p in range(param_no):
    m_o[p]   += dm
    data_pert = system(m_o,*system_args,**system_kwargs)
    m_o[p]   -= dm
    Jac[:,p]  = (data_pert - data_o)/dm

  return Jac

##------------------------------------------------------------------------------
def _objective(system,
               data,
               sigma,
               system_args,
               system_kwargs,
               jacobian,
               jacobian_args,
               jacobian_kwargs,
               reg_matrix,
               lm_matrix,
               data_indices):
  '''
  used for nonlin_lstsq
  '''  
  data = np.asarray(data)
  sigma = np.asarray(sigma)
  def objective_function(model):
    '''
    evaluates the function to be minimized for the given model
    '''
    pred = system(model,*system_args,**system_kwargs)
    res = (pred - data) / sigma
    res = res[data_indices]
    reg = reg_matrix.dot(model)    
    lm = np.zeros(np.shape(lm_matrix)[0])
    return np.hstack((res,reg,lm))

  def objective_jacobian(model):
    '''
    evaluates the jacobian of the objective function at the given model
    '''
    jac = jacobian(model,*jacobian_args,**jacobian_kwargs)
    jac = jac / sigma[:,np.newaxis]
    jac = jac[data_indices,:]
    return np.vstack((jac,reg_matrix,lm_matrix))

  return objective_function,objective_jacobian
##------------------------------------------------------------------------------
def lstsq(G,d,*args,**kwargs):
  '''
  used by nonlin_lstsq
  '''
  out = np.linalg.lstsq(G,d,*args,**kwargs)[0]
  return out

##------------------------------------------------------------------------------
@funtime
def nnls(G,d,*args,**kwargs):
  '''
  used by nonlin_lstsq
  '''
  out = scipy.optimize.nnls(G,d)[0]
  return out

##------------------------------------------------------------------------------
def nonlin_lstsq(system,
                 data,
                 m_o,
                 sigma=None,
                 system_args=None,
                 system_kwargs=None,
                 jacobian=None,
                 jacobian_args=None,
                 jacobian_kwargs=None,
                 solver=lstsq,
                 solver_args=None,
                 solver_kwargs=None,
                 reg_matrix=None,
                 LM_damping=False,
                 LM_param=10.0,
                 LM_factor=2.0,   
                 maxitr=20,
                 rtol=1.0e-2,
                 atol=1.0e-2,
                 data_indices=None,
                 dtype=None):
  '''
  Newtons method for solving a least squares problem

  PARAMETERS
  ----------
  *args 
  -----
    system: function where the first argument is a vector of model parameters 
            and the remaining arguments are system args and system kwargs
    data: vector of data values
    m_o: vector of model parameter initial guesses

  **kwargs 
  -------
    system_args: list of arguments to be passed to system following the model
                 parameters
    system_kwargs: list of key word arguments to be passed to system following 
                   the model parameters
    jacobian: function which computes the jacobian w.r.t the model parameters.
              the first arguments is a vector of parameters and the remaining 
              arguments are jacobian_args and jacobian_kwargs
    jacobian_args: arguments to be passed to the jacobian function 
    jacobian_kwargs: key word arguments to be passed to the jacobian function 
    solver: function which solves "G*m = d" for m, where the first two arguments
            are G and d.  inverse.lstsq, and inverse.nnls are wrappers for 
            np.linalg.lstsq, and scipy.optimize.nnls and can be used here. Using
            nnls ensures that the output model parameters are non-negative
    solver_args: additional arguments for the solver after G and d
    solver_kwargs: additional key word arguments for the solver 
    sigma: data uncertainty vector
    reg_matrix: regularization matrix scaled by the penalty parameter
    LM_damping: flag indicating whether to use the Levenberg Marquart algorithm 
                which damps step sizes in each iteration but ensures convergence
    LM_param: starting value for the Levenberg Marquart parameter 
    LM_factor: the levenberg-Marquart parameter is either multiplied or divided
               by this value depending on whether the algorithm is converging or
               diverging. 
    maxitr: number of steps for the inversion
    rtol: Algorithm stops if relative L2 between successive iterations is below 
          this value  
    atol: Algorithm stops if absolute L2 is below this value 
    data_indices: indices of data that will be used in the inversion. Defaults 
                  to using all data.

  Returns
  -------
    m_new: best fit model parameters
  '''
  param_no = len(m_o)
  data_no = len(data)

  m_o = np.array(m_o,dtype=dtype)
  data = np.array(data,dtype=dtype)

  if sigma is None:
    sigma = np.ones(data_no,dtype=dtype)

  if system_args is None:
    system_args = []

  if system_kwargs is None:
    system_kwargs = {}

  if jacobian is None:
    jacobian = jacobian_fd
    jacobian_args = [system]
    jacobian_kwargs = {'system_args':system_args,
                       'system_kwargs':system_kwargs,
                       'dtype':dtype}

  if jacobian_args is None:
    jacobian_args = []

  if jacobian_kwargs is None:
    jacobian_kwargs = {}

  if solver_args is None:
    solver_args = []

  if solver_kwargs is None:
    solver_kwargs = {}

  if data_indices is None:
    data_indices = range(data_no)

  if reg_matrix is None:
    reg_matrix = np.zeros((0,param_no),dtype=dtype)

  if hasattr(reg_matrix,'todense'):
    reg_matrix = np.array(reg_matrix.todense())

  if LM_damping:
    lm_matrix = LM_param*np.eye(param_no,dtype=dtype)
  else:
    lm_matrix = np.zeros((0,param_no),dtype=dtype)
 
  obj_func,obj_jac = _objective(system,
                                data,
                                sigma,
                                system_args,
                                system_kwargs,
                                jacobian,
                                jacobian_args,
                                jacobian_kwargs,
                                reg_matrix,
                                lm_matrix,
                                data_indices)

  final = np.zeros(data_no+
                   np.shape(reg_matrix)[0] +
                   np.shape(lm_matrix)[0])

  conv = Converger(final,atol=atol,rtol=rtol)
  count = 0
  status = None
  while not ((status == 0) | (status == 3) | (count == maxitr)):
    J = obj_jac(m_o)
    J = np.asarray(J,dtype=dtype)
    d = obj_func(m_o)
    d = np.asarray(d,dtype=dtype)
    m_new = solver(J,-d+J.dot(m_o))
    d_new = obj_func(m_new)
    status,message = conv(d_new)
    logger.debug(message)
    if (status == 1) and LM_damping:
      logger.debug('decreasing LM parameter to %s' % LM_param)
      lm_matrix /= LM_factor
      LM_param /= LM_factor

    while ((status == 2) | (status == 3)) and LM_damping:
      logger.debug('increasing LM parameter to %s' % LM_param)
      lm_matrix *= LM_factor
      LM_param *= LM_factor
      J = obj_jac(m_o)
      J = np.asarray(J,dtype=dtype)
      d = obj_func(m_o)
      d = np.asarray(d,dtype=dtype)
      m_new = solver(J,-d+J.dot(m_o))
      d_new = obj_func(m_new)
      status,message = conv(d_new)
      logger.debug(message)

    m_o = m_new
    conv.set(d_new)
    count += 1
    if count == maxitr:
      logger.debug('converged due to maxitr')

  return m_o

##------------------------------------------------------------------------------
def cross_validate(exclude_groups,*args,**kwargs):
  '''
  cross validation routine.  This function runs nonlin_lstsq to find the optimal
  model parameters while excluding each of the groups of data indices given in 
  exclude groups.  It returns the L2 norm of the predicted data for each of the
  excluded groups minus the observed data

  PARAMETERS
  ----------
    exclude_groups: list of groups of data indices to exclude
    *args: arguments for nonlin_lstsq
    **kwargs: arguments for nonlin_lstsq
  '''                     
  logger.info('starting cross validation iteration')
  system = args[0]
  system_args = kwargs.get('system_args',())
  system_kwargs = kwargs.get('system_kwargs',{})
  data = args[1]
  data_no = len(data)
  sigma = kwargs.get('sigma',np.ones(data_no))
  parameters = args[2]
  group_no = len(exclude_groups)
  param_no = len(parameters)
  residual = np.zeros(data_no)
  for itr,exclude_indices in enumerate(exclude_groups):
    data_indices = [i for i in range(data_no) if not i in exclude_indices]
    pred_params = nonlin_lstsq(*args,
                               data_indices=data_indices,
                               **kwargs)
    pred_data = system(pred_params,*system_args,**system_kwargs)
    residual[exclude_indices] = pred_data[exclude_indices] - data[exclude_indices]
    residual[exclude_indices] /= sigma[exclude_indices] # normalize residuals
    logger.info('finished cross validation for test group %s of '
                '%s' % (itr+1,group_no))
  L2 = np.linalg.norm(residual)
  logger.info('finished cross validation with predicted L2: %s' % L2)
  return L2

##------------------------------------------------------------------------------
def bootstrap(bootstrap_iterations,*args,**kwargs):
  '''
  Bootstraps the uncertainties of the best fit model parameters found by
  nonlin_lstsq
  
  Parameters
  ----------
    bootstrap_iterations: number of bootstrap iterations
    *args: arguments to be given to nonlin_lstsq
    **kwargs: key word arguments to be given to nonlin_lstsq
    bootstrap_log_level: controls the verbosity
  
  Returns
  -------
    parameter_array: array of best fit model parameters for each iteration
  ''' 
  logger.info('starting bootstrap')
  data = args[1]
  parameters = args[2]
  data_no = len(data)
  param_no = len(parameters)
  parameter_array = np.zeros((bootstrap_iterations,param_no))
  for i in range(bootstrap_iterations):
    data_indices = np.random.choice(range(data_no),data_no)
    pred_params = nonlin_lstsq(*args,
                               data_indices=data_indices,
                               **kwargs)
    parameter_array[i,:] = pred_params
    logger.info('finished bootstrap iteration %s of %s' 
                % ((i+1),bootstrap_iterations))
  return parameter_array

##------------------------------------------------------------------------------
def block_bootstrap(bootstrap_iterations,data_groups,*args,**kwargs):
  '''
  Bootstraps the uncertainties of the best fit model parameters found by
  nonlin_lstsq
  
  Parameters
  ----------
    iterations: number of bootstrap iterations
    data_groups: list of data groups where each data group is a list of 
      data indices within that group
    *args: arguments to be given to nonlin_lstsq
    **kwargs: key word arguments to be given to nonlin_lstsq
    bootstrap_log_level: controls the verbosity

  Returns
  -------
    parameter_array: array of best fit model parameters for each iteration
  ''' 
  logger.info('starting bootstrap')
  data = args[1]
  parameters = args[2]
  data_no = len(data)
  param_no = len(parameters)
  group_no = len(data_groups)
  parameter_array = np.zeros((bootstrap_iterations,param_no))
  for i in range(bootstrap_iterations):
    test_groups = np.random.choice(range(group_no),group_no)
    data_indices = list_flatten([data_groups[k] for k in test_groups]) 
    pred_params = nonlin_lstsq(*args,
                               data_indices=data_indices,
                               **kwargs)
    parameter_array[i,:] = pred_params
    logger.info('finished bootstrap iteration %s of %s' 
                 % ((i+1),bootstrap_iterations))
  return parameter_array

