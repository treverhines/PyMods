#!/usr/bin/env python
'''
PyFuncMod
'''

import numpy as np
import math
import scipy.special
import scipy.misc

pi = np.pi

# Calculate Decay Constant
#-------------------------------------------------------------------------------
def Calc_L(T_top,T_bot,T_avg,X_vec):
  '''  
  Description:
    Finds the decay constant for a depth dependent tau
    profile given tau_top,tau_bot, and tau_avg
  Input:
    T_top   - Tau at the top of the layer
    T_bot   - Tau at the bottom of the layer
    T_avg   - Average tau in the layer
    X_vec   - Array of dimensionless depth from 0 to 1
  '''
  X_diff = np.diff(X_vec)
  L_hi  = 50.0
  L_lo  = -50.0
  Tol   = 0.0001
  if (T_top == T_bot):
    return float("NaN")
  while True:    
    tau_high = Calc_Tau_Layer(T_top,T_bot,L_hi,X_vec)
    tau_lo = Calc_Tau_Layer(T_top,T_bot,L_lo,X_vec)
    Mean_hi  = np.sum(X_diff*(tau_high[:-1] + tau_high[1:])/2.0)
    Mean_lo  = np.sum(X_diff*(tau_lo[:-1] + tau_lo[1:])/2.0)
    if ((Mean_hi - T_avg) * ( Mean_lo - T_avg) >= 0):
      print("\nERROR: Average not found within range -50 < L < 50")
      return
    L_mid  = (L_hi + L_lo) / 2
    tau_mid = Calc_Tau_Layer(T_top,T_bot,L_mid,X_vec)
    Mean_mid  = np.sum(X_diff*(tau_mid[:-1] + tau_mid[1:])/2.0)
    if (abs(Mean_mid - T_avg) <= Tol):
      return L_mid
    if ((Mean_hi - T_avg) * ( Mean_mid - T_avg) < 0):
      L_lo  = L_mid
    else:
      L_hi  = L_mid

# Calculate tau for one layer 
#-------------------------------------------------------------------------------
def Calc_Tau_Layer(T_top,T_bot,L,X_vec):
  '''
  Description:
    Outputs depth dependent tau profile
  Input:
    T_Top   - Tau at the top of a layer
    T_bot   - Tau at the bottom of a layer
    L       - Tau decay constant from Calc_L
    X_Vec   - Array of dimensionless depths from 0 to 1
  '''
  Tol      = 1e-16
  T_top    = float(T_top)
  T_bot    = float(T_bot)
  T_bot    = np.power(10,T_bot)
  T_top    = np.power(10,T_top)
  X_vec    = np.array(X_vec)                        
  Zero_vec = X_vec * 0                                                                                             
  if (abs(T_top - T_bot) <= Tol): # handling an instability condition                                            
    Tau_out  = np.log10(T_top) + Zero_vec                                                                     
  elif (abs(L) <= Tol):           # handling an instability condition                                            
    Tau_out  = np.log10(T_top + (T_bot - T_top)*X_vec)                                                   
  else:                                                                                                          
    Tau  = T_top - ((T_top - T_bot) / (1 - pow((T_bot / T_top),L))) * (1 - np.power( (T_bot/T_top) , (L*X_vec) ))
    Tau_out  = np.log10(Tau)                                                                       
  return Tau_out



# Calculate tau for multiple layers 
#-------------------------------------------------------------------------------
def Calc_Tau_Profile(H_Top,H_Bot,T_Top,T_Bot,T_Avg,X_vec):
  '''     
  Description: 
    Generates a tau profile     
  Arguments: 
    H_top   - Top of each layer 
    H_Bot   - Bottom of each layer   
    Tau_Top - Tau at the top of each layer 
    Tau_Bot - Tau at the bottom of each layer
    Tau_Avg - Average tau in each layer 
    X_vec   - Discretization of depth
  '''
  T           = []
  l_idx       = 'NA'
  if (X_vec[0] > H_Top[0]) | (X_vec[-1] < H_Bot[-1]):
    print('X_vec must be between H_Top[0] and H_Bot[1]')
    return
  for x in X_vec:
    for (idx,h) in enumerate(H_Top + [H_Bot[-1]]):
      if x > h:        
        if l_idx != idx - 1:
          l_idx     = idx - 1
          L         = Calc_L(T_Top[l_idx],T_Bot[l_idx],T_Avg[l_idx],np.arange(0,1,pow(2,-8)))
        break
    x_norm    = [(H_Top[l_idx] - x)/(H_Top[l_idx] - H_Bot[l_idx])]
    T        += list(Calc_Tau_Layer(T_Top[l_idx],T_Bot[l_idx],L,x_norm))
  return np.array(T)

