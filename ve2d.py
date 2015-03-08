#!/usr/bin/env python
import numpy as np
from numpy import power
import matplotlib.pyplot as plt
import scipy.integrate
import math

pi = np.pi

##------------------------------------------------------------------------------
def integrate(fun):
  '''
  integral decorator, which I mostly wrote to practice with decorators

  PARAMETERS
  ----------
    fun: function whos only argument is the integration variable, t.

  RETURNS
  -------
    out_fun: the integral of fun which is a function of the upper bound of 
             integration, t. The lower bound of integration, t_o, is an optional
             kwarg for out_fun and defaults to 0.0

  EXAMPLE USAGE
  -------------
    In [25]: @integrate
       ....: def b(t):
       ....:     return t
       ....: 

    In [26]: b(1)
    Out[26]: 0.5

    >> @integrate
       def b(t):
         return t
    >> b(1)
     

  '''
  def out_fun(t,t_o=0.0):
    if hasattr(t,'__iter__'):
      out_val = np.zeros(len(t))
      for itr,t_ in enumerate(t):
        out_val[itr] = scipy.integrate.quad(fun,t_o,t_)[0]
    else:
      out_val = scipy.integrate.quad(fun,t_o,t)[0]
    return out_val     
  return out_fun

##------------------------------------------------------------------------------

def _W2(x,n,D,H):
  atan1 = np.arctan((2*n*H+D)/x)
  atan2 = np.arctan((2*n*H-D)/x)
  return 1.0/np.pi * (atan1 - atan2)


def u2v(x,t,tau1,tau2,b,D,H):
  '''
  return approximation for surface displacements in a 2-D 2 layer earthquake 
  model
  
  PARAMETERS
  ----------
    x: location of output locations in terms of distance from fault trace
    t: time since the earthquake
    tau1: relaxation time of layer 1
    tau2: relaxation time of layer 2
    b: slip function which takes time as its only argument
    D: locking depth (down is positive)
    H: layer 1 thickness
 
  RETURNS
  -------
    out: 2-D array with different locations along the columns and different
         output times along the rows
  '''
  B = integral(b)
  elastic_kernel = 0.5*_W2(x,0,D,H)
  elastic_disp = np.outer(b(t),elastic_kernel)
  viscous_kernel = _W2(x,1,D,H)/(2.0*tau2) - _W2(x,1,D,H)/(2.0*tau1) 
  viscous_disp = np.outer(B(t),viscous_kernel)
  return elastic_disp + viscous_disp
 
##------------------------------------------------------------------------------
def _W3(x,n,m,D,H1,H2):
  atan1 = np.arctan((2*n*H2+2*m*H1+D)/x)
  atan2 = np.arctan((2*n*H2+2*m*H1-D)/x)
  return 1.0/np.pi*(atan1 - atan2)

def u3v(x,t,tau1,tau2,tau3,b,D,H1,H2):
  '''
  return approximation for surface displacements in a 2-D 3 layer earthquake 
  model
  
  PARAMETERS
  ----------
    x: location of output locations in terms of distance from fault trace
    t: time since the earthquake
    tau1: relaxation time of layer 1 (top)
    tau2: relaxation time of layer 2 (middle)
    tau3: relaxation time of layer 3 (bottom)
    b: slip function which takes time as its only argument
    D: locking depth (down is positive)
    H1: layer 1 thickness (top)
    H3: layer 2 thickness (middle)
 
  RETURNS
  -------
    out: 2-D array with different locations along the columns and different
         output times along the rows
  '''
  B = integral(b)
  elastic_kernel = 0.5*_W2(x,0,D,H)
  elastic_disp = np.outer(b(t),elastic_kernel)
  viscous_kernel = ( -_W3(x,0,1,D,H1,H2)/(2*tau1) +
                     (_W3(x,0,1,D,H1,H2)-_W3(x,1,1,D,H1,H2))/(2*tau2) +
                      _W3(x,1,1,D,H1,H2)/(2*tau3))
  viscous_disp = np.outer(B(t),viscous_kernel)
  return elastic_disp + viscous_disp

##------------------------------------------------------------------------------
def elastic_kernel(eps,x):
  den = (power(x,2.0) + power(eps,2.0))
  return x / den

def viscous_kernel(xi,eps,x):
  num1 = 2*xi + eps
  den1 = power(power((2*xi+eps),2) + power(x,2),2)

  num2 = eps - 2*xi
  den2 = power(power((eps-2*xi),2) + power(x,2),2)
  return 2*x*(num1/den1 - num2/den2)

def _elastic_integrand(eps,b,x,t):
  return b(eps,t)*elastic_kernel(eps,x)

def _viscous_integrand(theta,xi,eps,b,tau,x,t):
  return b(eps,theta)/tau(xi)*viscous_kernel(xi,eps,x)

def postseismic_approx(x,t,b,tau,D):
  '''
  return approximation for early postseismic deformation for 2-D earthquake
  model with an arbitrary slip and 1-D viscosity struction.

  PARAMETERS
  ----------
    x: output locations in terms of distance from fault trace
    t: time since the earthquake
    b: slip function which takes depth as the first argument and time as the
       second argument
    tau: viscosity function which takes depth as its only argument
    D: depth above which the lithosphere is elastic (regardless of the tau
       function) and below which there is no slip (regardless of the b 
       function)
  
  RETURNS
  -------
    out: 2-D array with different locations along the columns and different
         output times along the rows
  '''
  disp = np.zeros((len(x),len(t)))
  for xidx,_x in enumerate(x):
    for tidx,_t in enumerate(t):
      elastic_disp = scipy.integrate.nquad(
                       _elastic_integrand,
                       [[0,D]],
                       args=[b,_x,_t])[0]
      viscous_disp = scipy.integrate.nquad(
                       _viscous_integrand,
                       [[0,_t],[D,np.inf],[0,D]],
                       args=[b,tau,_x,_t])[0]
      disp[xidx,tidx] = 1/np.pi*(elastic_disp + viscous_disp)
  return disp  

##------------------------------------------------------------------------------
def savage_2000_fun(x, y, t, H, relaxation_time, T):
  '''                                     
  BROKEN                         
  Arguments:                          
    x:  distance from fault       
    y:  depth                     
    t:  time                      
    H:  locking depth                           
    relaxation_time: relaxation time of lower halfspace 
    T: period                                             
  '''
  seconds_in_a_year = 60 * 60 * 24 * 365.25
  t = seconds_in_a_year * t
  T = seconds_in_a_year * T
  eta = relaxation_time*3.2e10*3.155e7
  mu = 3.2e10
  tau_0 = (mu * T) / (2 * eta)
  max_m = 100
  max_k = 250 / tau_0
  if (y < H):
   sum_S_m = 1 / (2 * pi) * ( -np.arctan((H + y) / x) - np.arctan((H - y) / x) + pi * np.sign(x))
  else:
   sum_S_m = 1 / (2 * pi) * ( -np.arctan((H + y) / x) - np.arctan((y - H) / x) + pi * np.sign(x))
  S_m = np.zeros((len(range(1,max_m)),len(x)))
  C_m = np.zeros(len(range(1,max_m)))
  for m in range(1,max_m):
    if (y < H):
      S_m[m-1, :] = 1 / (2 * pi) * ( np.arctan(((2 * m + 1) * H + y) / x) -\
                                     np.arctan(((2 * m - 1) * H + y) / x) +\
                                     np.arctan(((2 * m + 1) * H - y) / x) -\
                                     np.arctan(((2 * m - 1) * H - y) / x) )
    else:
      S_m[m-1, :] = 1 / (2 * pi) * ( np.arctan(((2 * m + 1) * H + y) / x) -\
                                     np.arctan(((2 * m - 3) * H + y) / x) )
  tau = (mu * t) / (2 * eta)
  for m in range(1,max_m):
    sum_k = 0
    for k in range(int(max_k)):
      sum_k = sum_k + np.exp(-(tau + k * tau_0)) * np.power((tau + k * tau_0),(m - 1))

    C_m[m-1] = tau_0 / scipy.misc.factorial(m - 1) * sum_k +\
               scipy.special.gammainc(m, tau + (max_k + 1) * tau_0) / scipy.misc.factorial(m - 1)

  v = np.zeros(len(x))
  for m in range(1,max_m):
    v = v + (C_m[m-1] - 1) * S_m[m-1, :]

  v = v + sum_S_m
  return v

##  Elastic-Elastic-Elastic Displacement                             
##-------------------------------------------------------------------------------------------                       
def chinnery_and_jovanovich(slip,mu1,h1,mu2,h2,mu3,D,x):
  '''                                                                                                               
  Three layer elastic surface displacements from Chinnery and Jovanovich  
                                                                                                                  
  PARAMETERS
  ----------
    slip: The total left-lateral displacement across the fault
    mu1:  Shear modulus for layer 1 (which contains the fault)   
    h1:   Thickness of layer 1                       
    mu2:  Shear modulus for layer 2 (middle)                    
    h2:   Thickness of layer 2                          
    mu3:  Shear modulus for layer 3 (bottom)                             
    D:    Locking Depth (down is negative)  
    x:    lateral distance from the fault where displacements will be computed

  NOTES
  -----
    The source code uses the numbering scheme from Chinnery and Jovanovich, while this
    docstring uses the reverse numbering scheme. 
  '''
  slip = float(slip)
  mu1  = float(mu1)
  h1   = float(h1)
  mu2  = float(mu2)
  h2   = float(h2)
  mu3  = float(mu3)
  D    = float(D)
  x    = np.double(x)
  U    = x * 0

  l_max = 5
  m_max = 5
  n_max = 5
  a1    = (mu2 - mu3) / (mu2 + mu3)
  a2    = (mu1 - mu2) / (mu1 + mu2)
  a3    = -1
  b1    = 2*mu3 / (mu3 + mu2)
  b2    = 2*mu2 / (mu2 + mu1)
  b3    = 2
  d1    = 2*mu2 / (mu3 + mu2)
  d2    = 2*mu1 / (mu2 + mu1)

  for n in range(n_max):
    W1    =           np.arctan( (  2*n*h1      + D ) / x )
    W2    =      a2 * np.arctan( ( -2*(n+1)*h1  - D ) / x )
    W3    =      a3 * np.arctan( (  2*n*h1      - D ) / x )
    W4    = a2 * a3 * np.arctan( ( -2*(n+1)*h1  + D ) / x )
    U     = U - ( pow(-a2*a3,n) * ( W1 - W2 + W3 - W4 ) )

  for l in range(1,l_max):
    for m in range(m_max):
      for n in range(n_max):
        W1    =           np.arctan( (  2*(l+m)*h2  +  2*(l+n)*h1    + D ) / x )
        W2    =      a2 * np.arctan( ( -2*(l+m)*h2  -  2*(l+n+1)*h1  - D ) / x )
        W3    =      a3 * np.arctan( (  2*(l+m)*h2  +  2*(l+n)*h1    - D ) / x )
        W4    = a2 * a3 * np.arctan( ( -2*(l+m)*h2  -  2*(l+n+1)*h1  + D ) / x )
        Gamma = pow(-a1*a2,m) * pow(-a2*a3,n) * pow(-a1*a3*d2*b2,l) * P(l,m,n)
        U     = U - Gamma*( W1 - W2 + W3 - W4 )

  for l in range( l_max ):
    for m in range( m_max ):
      for n in range( n_max ):
        W1    =      np.arctan( ( -2*(l+m+1)*h2  -  2*(l+n+1)*h1  -  D ) / x )
        W2    = a3 * np.arctan( ( -2*(l+m+1)*h2  -  2*(l+n+1)*h1  +  D ) / x )
        Gamma = a1 * d2 * b2 * pow(-a1*a2,m) * pow(-a2*a3,n) * pow(-a1*a3*d2*b2,l) * Q(l,m,n)
        U     = U + Gamma*(W1 + W2)

  return ( slip / (2*pi) ) * U

##  Coefficient Functions                                   
##-------------------------------------------------------------------------------------------                       
def P(l,m,n):
  N = math.factorial(n+1) * math.factorial(l+m-1)
  D = math.factorial(l) * math.factorial(n) * math.factorial(l-1) * math.factorial(m)
  N = float(N)
  D = float(D)
  return N / D

def Q(l,m,n):
  N = math.factorial(n+l) * math.factorial(l+m)
  D = math.factorial(l) * math.factorial(n) * math.factorial(l) * math.factorial(m)
  N = float(N)
  D = float(D)
  return N / D

