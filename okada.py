#!/usr/bin/env python
import numpy as np
from misc import rotation3D
tol = 1e-10

##------------------------------------------------------------------
def rotate_vectors(x,y,z,R):
  '''
  Rotates [x,y,z] by R
  '''
  points   = len(x)
  vec_list = [R.dot(np.array([x[i],y[i],z[i]])) for i in range(points)]
  vec_list = np.array(vec_list)      
  x_rot    = vec_list[:,0]
  y_rot    = vec_list[:,1]
  z_rot    = vec_list[:,2]
  return np.array([x_rot,y_rot,z_rot])

##------------------------------------------------------------------
def rotate_matrix(xx,yy,zz,xy,xz,yz,R):
  '''
  Rotates [[xx,xy,xz],[xy,yy,yz],[xz,yz,zz]] by R
  '''
  points   = len(xx)
  vec_list = []
  for i in range(points):
    mat         = np.array([[xx[i],xy[i],xz[i]],
                            [xy[i],yy[i],yz[i]],
                            [xz[i],yz[i],zz[i]]])
    mat_rotated = R.dot(mat).dot(R.transpose())
    vec_list   += [mat_rotated.flatten()]

  vec_list  = np.array(vec_list)
  xx_rot    = vec_list[:,0]
  yy_rot    = vec_list[:,4]
  zz_rot    = vec_list[:,8]
  xy_rot    = vec_list[:,1]
  xz_rot    = vec_list[:,2]
  yz_rot    = vec_list[:,5]
  return np.array([xx_rot,yy_rot,zz_rot,xy_rot,xz_rot,yz_rot])

##------------------------------------------------------------------
def okada_disp(x,y,z,U,fault_top_corner,fault_length,fault_strike,
               fault_width,fault_dip,lamb=3.2e10,mu=3.2e10):
  '''
  Description:                                       
    computes displacements at points (x,y,z) for a fault with specified
    geometry and slip components                         

  Arguments:                                        
    x: vector of output location x coordinates            
    y: vector of output location y coordinates            
    z: vector of output location z coordinates 
    fault_top_corner: (x,y,z) coordinates of the fault corner from 
                      which the fault continues in the both the 
                      strike and dip direction.
    fault_length:
    fault_strike: strike using the right hand rule convention
    fault_width:                                    
    fault_dip: This is between 0 and pi/2               
    U: slip vector with three components: left-lateral, thrust, tensile.
    lamb: lambda lame constant
    mu: mu lame constant
  
  output:                                                                      
    tuple where each components is a vector of displacement in the x, y, or z  
    direction                                                       

  example usage:
    >>> x = np.arange(-10,10)
    >>> y = np.arange(-10,10)
    >>> xgrid,ygrid = np.meshgrid(x,y)
    >>> xlst = xgrid.flatten()
    >>> ylst = ygrid.flatten()   
    >>> zlst = 0*ylst
    >>> U = [1.0,0.0,0.0]
    >>> corner  = [1.5,1.5,0.0]
    >>> length = 5 
    >>> width  = 5
    >>> strike = 0.0
    >>> dip = np.pi/2.0
    >>> output = okada_disp(xlst,ylst,zlst,U,corner,length,strike,width,dip)
    >>> plt.quiver(xlst,ylst,output[0],output[1],output[2])
    >>> plt.show()
  '''                         
  if (fault_dip < 0.0) | (fault_dip > np.pi/2.0):
    print('dip must be between 0 and pi/2')
    return
                     
  fault_depth       = fault_top_corner[2]
  fault_depth       = -fault_depth + fault_width*np.sin(fault_dip) # fault depth in Okada92 is depth
  x_trans           = x - fault_top_corner[0]                # at the bottom of the fault, but  
  y_trans           = y - fault_top_corner[1]                # that is inconvenient...
  Zangle            = -fault_strike + np.pi/2.0
  rotation_matrix   = rotation3D(Zangle,0.0,0.0)
  rotation_matrix   = rotation_matrix.transpose() # I want to rotate reference frame, not body
  rotated_points    = rotate_vectors(x_trans,y_trans,z,rotation_matrix)
  x_rot             = rotated_points[0]
  y_rot             = rotated_points[1]
  z_rot             = rotated_points[2]
  y_rot             = y_rot + fault_width*np.cos(fault_dip)
  out               = Okada92(x_rot,y_rot,z_rot,U,fault_length,fault_width,
                              fault_depth,fault_dip,'disp',lamb,mu)  
  unrotated_points  = rotate_vectors(out[0],out[1],out[2],
                                     rotation_matrix.transpose())
  x_out             = unrotated_points[0]
  y_out             = unrotated_points[1]
  z_out             = unrotated_points[2]
  return np.array([x_out,y_out,z_out])

##------------------------------------------------------------------
def okada_strain(x,y,z,U,fault_top_corner,fault_length,
                 fault_strike,fault_width,fault_dip,
                 lamb=3.2e10,mu=3.2e10):
  '''
  NOT VERIFIED
  '''
  if (fault_dip < 0.0) | (fault_dip > np.pi/2.0):
    print('dip must be between 0 and pi/2')
    return
                     
  fault_depth       = fault_top_corner[2]
  fault_depth       = -fault_depth + fault_width*np.sin(fault_dip) # fault depth in Okada92 is depth
  x_trans           = x - fault_top_corner[0]                # at the bottom of the fault, but  
  y_trans           = y - fault_top_corner[1]                # that is inconvenient...
  Zangle            = -fault_strike + np.pi/2.0
  rotation_matrix   = rotation3D(Zangle,0.0,0.0)
  rotation_matrix   = rotation_matrix.transpose() # I want to rotate reference frame, not body
  rotated_points    = rotate_vectors(x_trans,y_trans,z,rotation_matrix)
  x_rot             = rotated_points[0]
  y_rot             = rotated_points[1]
  z_rot             = rotated_points[2]
  y_rot             = y_rot + fault_width*np.cos(fault_dip)
  dudx              = Okada92(x_rot,y_rot,z_rot,U,fault_length,fault_width,
                              fault_depth,fault_dip,'dudx',lamb,mu)  
  dudy              = Okada92(x_rot,y_rot,z_rot,U,fault_length,fault_width,
                              fault_depth,fault_dip,'dudy',lamb,mu)  
  dudz              = Okada92(x_rot,y_rot,z_rot,U,fault_length,fault_width,
                              fault_depth,fault_dip,'dudz',lamb,mu)  

  strain_xx         = dudx[0]
  strain_yy         = dudy[1]
  strain_zz         = dudz[2]
  strain_xy         = 0.5*(dudy[0] + dudx[1])
  strain_xz         = 0.5*(dudz[0] + dudx[2])
  strain_yz         = 0.5*(dudz[1] + dudy[2])
  unrotated_strain  = rotate_matrix(strain_xx,strain_yy,strain_zz,
                                    strain_xy,strain_xz,strain_yz,
                                    rotation_matrix.transpose()) 
  strain_xx_out     = unrotated_strain[0]
  strain_yy_out     = unrotated_strain[1]
  strain_zz_out     = unrotated_strain[2]
  strain_xy_out     = unrotated_strain[3]
  strain_xz_out     = unrotated_strain[4]
  strain_yz_out     = unrotated_strain[5]
  return np.array([strain_xx_out,strain_yy_out,strain_zz_out,
                   strain_xy_out,strain_xz_out,strain_yz_out])

##------------------------------------------------------------------
def traction_force(xx,yy,zz,xy,xz,yz,strike,dip):
  R              = rotation(strike,dip)
  rotated_stress = rotate_matrix(xx,yy,zz,xy,xz,yz,R)
  normal_trac    = rotated_stress[2]
  strike_trac    = rotated_stress[4]
  dip_trac       = rotated_stress[5]
  return np.array([strike_trac,dip_trac,normal_trac])

##------------------------------------------------------------------
def coulumb_stress(xx,yy,zz,xy,xz,yz,strike,dip):
  mu             = 0.4
  R              = rotation(strike,dip)
  rotated_stress = rotate_matrix(xx,yy,zz,xy,xz,yz,R)
  normal_stress  = rotated_stress[2]
  strike_stress  = rotated_stress[4]
  dip_stress     = rotated_stress[5]
  shear_stress   = np.sqrt(np.power(strike_stress,2.0) + 
                           np.power(dip_stress,2.0))
  coulumb_stress = strike_stress - mu*normal_stress
  return coulumb_stress

##------------------------------------------------------------------
def okada_stress(x,y,z,U,fault_top_corner,fault_length,
                 fault_strike,fault_width,fault_dip,
                 lamb=3.2e10,mu=3.2e10):
  '''
  NOT VERIFIED
  '''
  strain_lst = okada_strain(x,y,z,U,fault_top_corner,fault_length,
                            fault_strike,fault_width,fault_dip,
                            lamb=3.2e10,mu=3.2e10)
  C    = np.array([[2*mu + lamb,         lamb,         lamb, 0.0, 0.0, 0.0],
                   [       lamb,  2*mu + lamb,         lamb, 0.0, 0.0, 0.0],
                   [       lamb,         lamb,  2*mu + lamb, 0.0, 0.0, 0.0],
                   [        0.0,          0.0,          0.0,  mu, 0.0, 0.0], 
                   [        0.0,          0.0,          0.0, 0.0, mu,  0.0], 
                   [        0.0,          0.0,          0.0, 0.0, 0.0,  mu]])
  stress = np.array([C.dot(strain) for strain in strain_lst.transpose()])
  stress = stress.transpose()
  print(np.shape(stress))  
  return stress 

##------------------------------------------------------------------  
def Okada92(x,y,z,U,L,W,c,delta,output,lamb=3.2e10,mu=3.2e10):
  '''
  Description:                                                                
    computes displacements at points (x,y,z) for a fault with       
    width W, length L, and depth c.  The fault has one end on the              
    origin and the other end at (x=L,y=0).  Slip on the fault is described 
    by U.                                                            
  Arguments:                                                         
    x: along strike coordinate of output locations (can be a vector)           
    y: perpendicular to strike coordinate of output locations (can be a vector)
    z: depth of output locations. down is negative
    L: length of fault                                                         
    W: width of fault                                                          
    c: depth of the bottom of the fault (a fault which ruptures the surface
       with have d=W, and d<W will give absurd results)                       
    delta: fault dip.  0<delta<pi/2.0 will dip in the -y direction. and        
           pi/2<delta<pi will dip in the +y direction... i think.           
    U: a three components vector with strike-slip, dip-slip, and tensile       
       components of slip                         
    output: either 'disp','dudx','dudy', or 'dudz'                             
                                                                             
  output:                                                                      
    tuple where each components is a vector of either diplacement or 
    displacement derivatives in the x, y, or z direction  
  '''                                            
  sindel      = np.sin(delta)
  cosdel      = np.cos(delta)
  points      = len(x)
  alpha       = (lamb + mu)/(lamb + 2*mu)

  def f(eps,eta,zeta,term,direction):
    X            = np.zeros(points)
    R            = np.zeros(points)
    y_bar        = np.zeros(points)
    c_bar        = np.zeros(points)
    d_bar        = np.zeros(points)
    X11          = np.zeros(points)
    X32          = np.zeros(points)
    X53          = np.zeros(points)
    Y11          = np.zeros(points)
    Y32          = np.zeros(points)
    Y53          = np.zeros(points)
    h            = np.zeros(points)
    theta        = np.zeros(points)
    logReps      = np.zeros(points)
    logReta      = np.zeros(points)
    I1           = np.zeros(points)
    I2           = np.zeros(points)
    I3           = np.zeros(points)
    I4           = np.zeros(points)
    K1           = np.zeros(points)
    K2           = np.zeros(points)
    K3           = np.zeros(points)
    K4           = np.zeros(points)
    D11          = np.zeros(points)
    J1           = np.zeros(points)
    J2           = np.zeros(points)
    J3           = np.zeros(points)
    J4           = np.zeros(points)
    J5           = np.zeros(points)
    J6           = np.zeros(points)
    E            = np.zeros(points)
    F            = np.zeros(points)
    G            = np.zeros(points)
    H            = np.zeros(points)
    P            = np.zeros(points)
    Q            = np.zeros(points)
    Ep           = np.zeros(points)
    Fp           = np.zeros(points)
    Gp           = np.zeros(points)
    Hp           = np.zeros(points)
    Pp           = np.zeros(points)
    Qp           = np.zeros(points)

    d             = c - zeta
    p             = y*cosdel + d*sindel
    q             = y*sindel - d*cosdel
    R             = np.sqrt(np.power(eps,2.0) + 
                            np.power(eta,2.0) + 
                            np.power(q,2.0))

    X             = np.sqrt(np.power(eps,2.0) + 
                            np.power(q,2.0))    
    y_bar         = eta*cosdel + q*sindel
    d_bar         = eta*sindel - q*cosdel
    c_bar         = d_bar + zeta
    h             = q*cosdel - zeta

    idx           = np.abs(q) < tol 
    nidx          = np.abs(q) >= tol 
    theta[nidx]   = np.arctan(eps[nidx]*eta[nidx]/(q[nidx]*R[nidx]))

    idx           = np.abs(R + eps) < tol
    nidx          = np.abs(R + eps) >= tol
    X11[nidx]     = 1.0/(R[nidx]*(R[nidx] + eps[nidx]))
    X32[nidx]     = ((2*R[nidx] + eps[nidx])/(np.power(R[nidx],3.0)*
                     np.power((R[nidx] + eps[nidx]),2.0)))
    X53[nidx]     = ((8*np.power(R[nidx],2.0) + 
                      9*R[nidx]*eps[nidx] + 
                      3*np.power(eps[nidx],2.0))/
                     (np.power(R[nidx],5.0)*
                      (np.power(R[nidx] + eps[nidx],3.0))))
    logReps[idx]  = -np.log(R[idx] - eps[idx]) 
    logReps[nidx] = np.log(R[nidx] + eps[nidx]) 

    idx           = np.abs(R + eta) < tol
    nidx          = np.abs(R + eta) >= tol
    Y11[nidx]     = 1.0/(R[nidx]*(R[nidx] + eta[nidx]))
    Y32[nidx]     = ((2*R[nidx] + eta[nidx])/(np.power(R[nidx],3.0)*
                     np.power((R[nidx] + eta[nidx]),2.0)))
    Y53[nidx]     = ((8*np.power(R[nidx],2.0) + 
                      9*R[nidx]*eta[nidx] + 
                      3*np.power(eta[nidx],2.0))/
                     (np.power(R[nidx],5.0)*
                      (np.power(R[nidx] + eta[nidx],3.0))))
    logReta[idx]  = -np.log(R[idx] - eta[idx])
    logReta[nidx] = np.log(R[nidx] + eta[nidx])
  
    if np.abs(cosdel) >= tol:
      I3          = (y_bar/(cosdel*(R + d_bar)) - 1/np.power(cosdel,2.0)*
                     (logReta - sindel*np.log(R + d_bar)))
      nidx        = np.abs(eps) >= tol
      idx         = np.abs(eps) < tol
      I4[nidx]    = ((sindel*eps[nidx])/(cosdel*(R[nidx] + d_bar[nidx])) + 
                     2.0/np.power(cosdel,2.0)*
                     np.arctan((eta[nidx]*(X[nidx] + q[nidx]*cosdel) + 
                     X[nidx]*(R[nidx] + X[nidx])*sindel)/
                     (eps[nidx]*(R[nidx] + X[nidx])*cosdel)))
      I4[idx]     = (0.5*(eps[idx]*y_bar[idx])/
                     np.power((R[idx] + d_bar[idx]),2.0)) 
    else:
      I3          = (0.5*(eta/(R + d_bar) + (y_bar*q)/
                     np.power((R + d_bar),2.0) - logReta))
      I4          = 0.5*(eps*y_bar)/np.power((R + d_bar),2.0) 


    I2            = np.log(R + d_bar) + I3*sindel
    I1            = -eps/(R + d_bar)*cosdel - I4*sindel
    Y0            = Y11 - np.power(eps,2.0)*Y32
    Z32           = sindel/np.power(R,3.0) - h*Y32
    Z53           = 3*sindel/np.power(R,5.0) - h*Y53
    Z0            = Z32 - np.power(eps,2.0)*Z53
  
    D11           = 1.0/(R*(R + d_bar))
    J2            = eps*y_bar/(R + d_bar)*D11
    J5            = -(d_bar + np.power(y_bar,2.0)/(R + d_bar))*D11

    if np.abs(cosdel) >= tol:
      K1      = eps/cosdel*(D11 - Y11*sindel)
      K3      = 1.0/cosdel*(q*Y11 - y_bar*D11)
      J3      = 1.0/cosdel*(K1 - J2*sindel)
      J6      = 1.0/cosdel*(K3 - J5*sindel)
    else:
      K1       = eps*q/(R + d_bar)*D11
      K3       = (sindel/(R + d_bar)*
                  (np.power(eps,2.0)*D11 - 1))
      J3       = (-eps/np.power((R + d_bar),2.0)*
                  (np.power(q,2.0)*D11 - 0.5))
      J6       = (-y_bar/np.power((R + d_bar),2.0)*
                  (np.power(eps,2.0)*D11 - 0.5))
  
    K4            = eps*Y11*cosdel - K1*sindel
    K2            = 1.0/R + K3*sindel
    J4            = -eps*Y11 - J2*cosdel + J3*sindel
    J1            = J5*cosdel - J6*sindel
    E             = sindel/R - y_bar*q/np.power(R,3.0)
    F             = d_bar/np.power(R,3.0) + np.power(eps,2.0)*Y32*sindel
    G             = 2.0*X11*sindel - y_bar*q*X32
    H             = d_bar*q*X32 + eps*q*Y32*sindel
    P             = cosdel/np.power(R,3.0) + q*Y32*sindel
    Q             = 3*c_bar*d_bar/np.power(R,5.0) - (zeta*Y32 + Z32 + Z0)*sindel
    Ep            = cosdel/R + d_bar*q/np.power(R,3.0)
    Fp            = y_bar/np.power(R,3.0) + np.power(eps,2.0)*Y32*cosdel
    Gp            = 2.0*X11*cosdel + d_bar*q*X32
    Hp            = y_bar*q*X32 + eps*q*Y32*cosdel
    Pp            = sindel/np.power(R,3.0) - q*Y32*cosdel
    Qp            = (3*c_bar*y_bar/np.power(R,5.0) + 
                     q*Y32 - (zeta*Y32 + Z32 + Z0)*cosdel)

    if output == 'disp':
      if direction == 'strike':
        if term == 'A':
          f1   = theta/2.0 + alpha/2.0*eps*q*Y11  
          f2   = alpha/2.0*q/R
          f3   = (1-alpha)/2.0*logReta - alpha/2.0*np.power(q,2.0)*Y11
        if term == 'B':
          f1   = -eps*q*Y11 - theta - (1- alpha)/alpha*I1*sindel
          f2   = -q/R + (1-alpha)/alpha*y_bar/(R+d_bar)*sindel
          f3   = np.power(q,2.0)*Y11 - (1 - alpha)/alpha*I2*sindel
        if term == 'C':
          f1   = (1-alpha)*eps*Y11*cosdel - alpha*eps*q*Z32
          f2   = ((1-alpha)*(cosdel/R + 2*q*Y11*sindel) - 
                  alpha*c_bar*q/np.power(R,3.0))
          f3   = ((1-alpha)*q*Y11*cosdel - 
                  alpha*(c_bar*eta/np.power(R,3.0) - 
                  zeta*Y11 + np.power(eps,2.0)*Z32))
      if direction == 'dip':
        if term == 'A':
          f1  = alpha/2.0*q/R
          f2  = theta/2.0 + alpha/2.0*eta*q*X11
          f3  = (1-alpha)/2.0*logReps - alpha/2.0*np.power(q,2.0)*X11
        if term == 'B':
          f1  = -q/R + (1 - alpha)/alpha*I3*sindel*cosdel
          f2  = (-eta*q*X11 - theta - (1-alpha)/alpha*
                 eps/(R + d_bar)*sindel*cosdel)
          f3  = np.power(q,2.0)*X11 + (1-alpha)/alpha*I4*sindel*cosdel
        if term == 'C':
          f1  = ((1-alpha)*cosdel/R - q*Y11*sindel - 
                 alpha*c_bar*q/np.power(R,3.0))
          f2  = (1 - alpha)*y_bar*X11 - alpha*c_bar*eta*q*X32
          f3  = (-d_bar*X11 - eps*Y11*sindel - 
                 alpha*c_bar*(X11 - np.power(q,2.0)*X32))
      if direction == 'tensile':
        if term == 'A':
          f1 = -(1 - alpha)/2.0*logReta - alpha/2.0*np.power(q,2.0)*Y11
          f2 = -(1 - alpha)/2.0*logReps - alpha/2.0*np.power(q,2.0)*X11
          f3 = theta/2.0 - alpha/2.0*q*(eta*X11 + eps*Y11)
        if term == 'B':
          f1 = np.power(q,2.0)*Y11 - (1 - alpha)/alpha*I3*np.power(sindel,2.0) 
          f2 = (np.power(q,2.0)*X11 + 
                (1 - alpha)/alpha*eps / (R + d_bar)*np.power(sindel,2.0))
          f3 = (q*(eta*X11 + eps*Y11) - theta - 
                (1 - alpha)/alpha*I4*np.power(sindel,2.0))
        if term == 'C':
          f1 = (-(1 - alpha)*(sindel/R + q*Y11*cosdel) - 
                alpha*(zeta*Y11 - np.power(q,2.0)*Z32))
          f2 = ((1 - alpha)*2.0*eps*Y11 *sindel + d_bar*X11 - 
                alpha*c_bar*(X11 - np.power(q,2.0)*X32))
          f3 = ((1 - alpha)*(y_bar*X11 + eps*Y11*cosdel) + 
                alpha*q*(c_bar*eta*X32 + eps*Z32))

    if output == 'dudx':
      if direction == 'strike':
        if term == 'A':
          f1 = -(1-alpha)/2.0*q*Y11 - alpha/2.0*np.power(eps,2.0)*q*Y32
          f2 = -alpha/2.0*eps*q/np.power(R,3.0)
          f3 = (1 - alpha)/2.0*eps*Y11 + alpha/2.0*eps*np.power(q,2.0)*Y32
        if term == 'B':
          f1 =  np.power(eps,2.0)*q*Y32 -(1 - alpha)/alpha*J1*sindel
          f2 =  eps*q/np.power(R,3.0) -(1 - alpha)/alpha*J2*sindel
          f3 = -eps*np.power(q,2.0)*Y32 -(1 - alpha)/alpha*J3*sindel
        if term == 'C':
          f1 = (1 - alpha)*Y0*cosdel - alpha*q*Z0
          f2 = (-(1 - alpha) *eps*(cosdel/np.power(R,3.0) + 
                2.0*q*Y32*sindel) + alpha* 3.0*c_bar*eps*q / np.power(R,5.0))
          f3 = (-(1 - alpha)*eps*q*Y32*cosdel + 
                alpha*eps*(3.0*c_bar*eta/np.power(R,5.0) - zeta*Y32 - Z32 - Z0))
      if direction == 'dip':
        if term == 'A':
          f1 = -alpha/2.0*eps*q/np.power(R,3.0)
          f2 = -q/2.0*Y11 - alpha/2.0*eta*q/np.power(R,3.0)
          f3 = (1 - alpha)/2.0*1.0/R + alpha/2.0*np.power(q,2.0)/np.power(R,3.0)
        if term == 'B':
          f1 = eps*q/np.power(R,3.0) + (1 - alpha)/alpha*J4*sindel*cosdel
          f2 = (eta*q/np.power(R,3.0) + q*Y11 + 
                (1 - alpha)/alpha*J5*sindel*cosdel)
          f3 = (-np.power(q,2.0)/np.power(R,3.0) + 
                (1- alpha)/alpha*J6*sindel*cosdel)
        if term == 'C':
          f1 = (-(1 - alpha)* eps/np.power(R,3.0)*cosdel + 
                eps*q*Y32*sindel + alpha*3*c_bar*eps*q / np.power(R,5.0))
          f2 = (-(1 - alpha)* y_bar/np.power(R,3.0) + 
                alpha*3*c_bar*eta*q/np.power(R,5.0))
          f3 = (d_bar/np.power(R,3.0) - Y0*sindel + 
                alpha*c_bar/np.power(R,3.0)*
                (1 - 3*np.power(q,2.0)/np.power(R,2.0)))
      if direction == 'tensile':
        if term == 'A':
          f1 = -(1 - alpha)/2.0*eps*Y11 + alpha/2.0*eps*np.power(q,2.0)*Y32
          f2 = (-(1 - alpha)/2.0*1.0/R + 
                alpha/2.0*np.power(q,2.0)/np.power(R,3.0))
          f3 = -(1 - alpha)/2.0*q*Y11 - alpha/2.0*np.power(q,3.0)*Y32
        if term == 'B':
          f1 = (-eps*np.power(q,2.0)*Y32 - 
                (1 - alpha)/alpha*J4*np.power(sindel,2.0))
          f2 = (-np.power(q,2.0)/np.power(R,3.0) - 
                (1 - alpha)/alpha*J5*np.power(sindel,2.0))
          f3 = np.power(q,3.0)*Y32 - (1 - alpha)/alpha*J6*np.power(sindel,2.0)
        if term == 'C':
          f1 = ((1 - alpha)*eps/np.power(R,3.0)*sindel + eps*q*Y32*cosdel + 
                alpha*eps*(3*c_bar*eta/np.power(R,5.0) - 2.0*Z32 - Z0))
          f2 = ((1 - alpha)*2.0*Y0*sindel - d_bar/np.power(R,3.0) + 
                alpha*c_bar/np.power(R,3.0)*(1 - 3.0*
                np.power(q,2.0)/np.power(R,2.0)))
          f3 = (-(1- alpha)*(y_bar/np.power(R,3.0) - Y0*cosdel) - 
                alpha*(3*c_bar*eta*q/np.power(R,5.0) - q *Z0))
  
    if output == 'dudy':
      if direction == 'strike':
        if term == 'A':
          f1 = (1-alpha)/2.0*eps*Y11*sindel + d_bar/2.0*X11 + alpha/2.0*eps *F
          f2 = alpha/2.0*E
          f3 = (1 - alpha)/2.0*(cosdel/R + q*Y11*sindel) - alpha/2.0*q*F
        if term == 'B':
          f1 = -eps*F - d_bar*X11 + (1- alpha)/alpha*(eps*Y11 + J4)*sindel
          f2 = -E + (1- alpha)/alpha*(1.0/R + J5)*sindel
          f3 = q*F - (1 - alpha)/alpha*(q*Y11 - J6)*sindel
        if term == 'C':
          f1 = -(1.0 - alpha)*eps*P*cosdel - alpha*eps*Q
          f2 = (2*(1.0 - alpha)*(d_bar/np.power(R,3.0) - Y0*sindel)*sindel - 
                y_bar/np.power(R,3.0)*cosdel - 
                alpha*((c_bar + d_bar)/np.power(R,3.0)*sindel - 
                eta/np.power(R,3.0) - 3.0*c_bar*y_bar*q/np.power(R,5.0)))
          f3 = (-(1-alpha)*q/np.power(R,3.0) + 
                (y_bar/np.power(R,3.0) - Y0*cosdel)*sindel + 
                alpha*((c_bar + d_bar)/np.power(R,3.0)*cosdel + 
                3.0*c_bar*d_bar*q/np.power(R,5.0) - (Y0*cosdel + q*Z0)*sindel))
      if direction == 'dip':
        if term == 'A':
          f1 = alpha/2.0*E
          f2 = (1 -alpha)/2.0*d_bar*X11 + eps/2.0*Y11*sindel + alpha/2.0*eta*G
          f3 = (1 -alpha)/2.0*y_bar*X11- alpha/2.0*q*G
        if term == 'B':
          f1 = -E       + (1- alpha)/alpha*J1*sindel*cosdel
          f2 = -eta*G   - eps*Y11*sindel + (1- alpha)/alpha*J2*sindel*cosdel
          f3 = q*G     + (1- alpha)/alpha*J3*sindel*cosdel
        if term == 'C':
          f1 = (-(1 - alpha)*eta/np.power(R,3.0) + Y0*np.power(sindel,2.0) - 
                alpha*((c_bar+d_bar)/np.power(R,3.0)*sindel - 
                3*c_bar*y_bar*q/np.power(R,5.0)))
          f2 = ((1 - alpha)*(X11 - np.power(y_bar,2.0)*X32) - 
                alpha*c_bar*((d_bar + 2.0*q*cosdel)*X32 - y_bar*eta*q*X53))
          f3 = (eps*P*sindel + y_bar*d_bar*X32 + 
                alpha*c_bar*((y_bar + 2*q*sindel)*X32 - 
                y_bar*np.power(q,2.0)*X53))
      if direction == 'tensile':
        if term == 'A':
          f1 = -(1 - alpha)/2.0*(cosdel/R + q*Y11*sindel) - alpha/2.0*q*F
          f2 = -(1 - alpha)/2.0*y_bar*X11 - alpha/2.0*q*G
          f3 = (1 - alpha)/2.0*(d_bar*X11 + eps*Y11*sindel) + alpha/2.0*q*H
        if term == 'B':
          f1 = q*F - (1- alpha)/alpha*J1*np.power(sindel,2.0)
          f2 = q*G - (1- alpha)/alpha*J2*np.power(sindel,2.0)
          f3 = -q*H - (1- alpha)/alpha*J3*np.power(sindel,2.0)
        if term == 'C':
          f1 = ((1 - alpha)*(q/np.power(R,3.0) + Y0*sindel*cosdel) + 
                alpha*(zeta/np.power(R,3.0)*cosdel + 
                3.0*c_bar*d_bar*q/np.power(R,5.0) - q*Z0*sindel))
          f2 = (-(1-alpha)*2.0*eps*P*sindel - y_bar*d_bar*X32 + 
                alpha*c_bar*((y_bar + 2.0*q*sindel)*X32 - 
                y_bar*np.power(q,2.0)*X53))
          f3 = (-(1 - alpha)*(eps*P*cosdel - X11 + np.power(y_bar,2.0)*X32) + 
                alpha*c_bar*((d_bar + 2.0*q*cosdel)*X32 - y_bar*eta*q*X53) + 
                alpha*eps*Q)

    if output == 'dudz':
      if direction == 'strike':
        if term == 'A':
          f1 = ((1 - alpha)/2.0*eps *Y11*cosdel + y_bar/2.0*X11 + 
                alpha/2.0*eps*Fp)
          f2 = alpha/2.0*Ep
          f3 = -(1 - alpha)/2.0*(sindel/R - q*Y11*cosdel) - alpha/2.0*q*Fp
        if term == 'B':
          f1 = -eps*Fp - y_bar*X11 + (1 - alpha)/alpha*K1*sindel
          f2 = -Ep + (1 - alpha)/alpha*y_bar*D11*sindel
          f3 = q*Fp + (1 - alpha)/alpha*K2*sindel
        if term == 'C':
          f1 = (1 - alpha)*eps*Pp*cosdel - alpha*eps*Qp
          f2 = (2*(1-alpha)*(y_bar/np.power(R,3.0) - Y0*cosdel)*sindel + 
                d_bar/np.power(R,3.0)*cosdel - 
                alpha*((c_bar + d_bar)/np.power(R,3.0)*cosdel + 
                3*c_bar*d_bar*q/np.power(R,5.0)))
          f3 = ((y_bar/np.power(R,3.0) - Y0*cosdel)*cosdel - 
                alpha*((c_bar + d_bar)/np.power(R,3.0)*sindel - 
                3*c_bar*y_bar*q/np.power(R,5.0)- Y0*np.power(sindel,2.0) + 
                q*Z0*cosdel))
        if term == 'D':
          f1   = (1-alpha)*eps*Y11*cosdel - alpha*eps*q*Z32
          f2   = ((1-alpha)*(cosdel/R + 2*q*Y11*sindel) - 
                  alpha*c_bar*q/np.power(R,3.0))
          f3   = ((1-alpha)*q*Y11*cosdel - 
                  alpha*(c_bar*eta/np.power(R,3.0) - 
                  zeta*Y11 + np.power(eps,2.0)*Z32))

      if direction == 'dip':
        if term == 'A':
          f1 = alpha/2.0*Ep
          f2 = (1 - alpha)/2.0*y_bar*X11 + eps/2.0*Y11*cosdel + alpha/2.0*eta*Gp
          f3 = -(1 - alpha)/2.0*d_bar*X11 - alpha/2.0*q*Gp
        if term == 'B':
          f1 = -Ep - (1 - alpha)/alpha*K3*sindel*cosdel
          f2 = (-eta*Gp - eps*Y11*cosdel - 
                (1 - alpha)/alpha*eps*D11*sindel*cosdel)
          f3 = q*Gp - (1 - alpha)/alpha*K4*sindel*cosdel
        if term == 'C':
          f1 = (-q/np.power(R,3.0) + Y0*sindel*cosdel - 
                alpha*((c_bar + d_bar)/np.power(R,3.0)*cosdel + 
                3*c_bar*d_bar*q/np.power(R,5.0)))
          f2 = ((1 - alpha)*y_bar*d_bar*X32 - 
                alpha*c_bar*((y_bar - 2*q*sindel)*X32 + d_bar*eta*q*X53))
          f3 = (-eps*Pp*sindel + X11 - np.power(d_bar,2.0)*X32 - 
                alpha*c_bar*((d_bar - 2*q*cosdel)*X32 - 
                d_bar*np.power(q,2.0)*X53))
        if term == 'D':
          f1  = ((1-alpha)*cosdel/R - q*Y11*sindel - 
                 alpha*c_bar*q/np.power(R,3.0))
          f2  = (1 - alpha)*y_bar*X11 - alpha*c_bar*eta*q*X32
          f3  = (-d_bar*X11 - eps*Y11*sindel - 
                 alpha*c_bar*(X11 - np.power(q,2.0)*X32))
      if direction == 'tensile':
        if term == 'A':
          f1 = (1 - alpha)/2.0*(sindel/R - q*Y11*cosdel) - alpha/2.0*q*Fp
          f2 = (1 - alpha)/2.0*d_bar*X11 - alpha/2.0*q*Gp
          f3 = (1 - alpha)/2.0*(y_bar*X11 + eps*Y11*cosdel) + alpha/2.0*q*Hp
        if term == 'B':
          f1 = q*Fp + (1 - alpha)/alpha*K3*np.power(sindel,2.0)
          f2 = q*Gp + (1 - alpha)/alpha*eps*D11*np.power(sindel,2.0)
          f3 = -q*Hp + (1 - alpha)/alpha*K4*np.power(sindel,2.0)
        if term == 'C':
          f1 = (-eta/np.power(R,3.0) + Y0*np.power(cosdel,2.0) - 
                alpha*(zeta/np.power(R,3.0)*sindel- 
                3*c_bar*y_bar*q/np.power(R,5.0) - 
                Y0*np.power(sindel,2.0) + q*Z0*cosdel))
          f2 = ((1 - alpha)*2*eps*Pp*sindel - X11 + np.power(d_bar,2.0)*X32 - 
                alpha*c_bar*((d_bar - 2*q*cosdel)*X32 - 
                d_bar*np.power(q,2.0)*X53))
          f3 = ((1 - alpha)*(eps*Pp*cosdel + y_bar*d_bar*X32) + 
                alpha*c_bar*((y_bar - 2*q*sindel)*X32 + d_bar*eta*q*X53) + 
                alpha*eps*Qp)
        if term == 'D':
          f1 = (-(1 - alpha)*(sindel/R + q*Y11*cosdel) - 
                alpha*(zeta*Y11 - np.power(q,2.0)*Z32))
          f2 = ((1 - alpha)*2.0*eps*Y11 *sindel + d_bar*X11 - 
                alpha*c_bar*(X11 - np.power(q,2.0)*X32))
          f3 = ((1 - alpha)*(y_bar*X11 + eps*Y11*cosdel) + 
                alpha*q*(c_bar*eta*X32 + eps*Z32))
    return (f1,f2,f3)

  ux = np.zeros(points)
  uy = np.zeros(points)
  uz = np.zeros(points)
  for itr,direction in enumerate(['strike','dip','tensile']):
    p          = y*cosdel + (c - z)*sindel
    p_         = y*cosdel + (c + z)*sindel # it is unclear if p_ is needed
                                           # based on the text. this could be   
    fI = f(x,p,z,'A',direction)            # the cause of an error
    fII = f(x,p - W,z,'A',direction)
    fIII = f(x-L,p,z,'A',direction)
    fIV = f(x-L,p - W,z,'A',direction)
    uA1 = fI[0] - fII[0] - fIII[0] + fIV[0]  
    uA2 = fI[1] - fII[1] - fIII[1] + fIV[1]  
    uA3 = fI[2] - fII[2] - fIII[2] + fIV[2]  
  
    fI = f(x,p_,-z,'A',direction)
    fII = f(x,p_ - W,-z,'A',direction)
    fIII = f(x-L,p_,-z,'A',direction)
    fIV = f(x-L,p_ - W,-z,'A',direction)
    uhatA1 = fI[0] - fII[0] - fIII[0] + fIV[0]  
    uhatA2 = fI[1] - fII[1] - fIII[1] + fIV[1]  
    uhatA3 = fI[2] - fII[2] - fIII[2] + fIV[2]  
  
    fI = f(x,p,z,'B',direction)
    fII = f(x,p - W,z,'B',direction)
    fIII = f(x-L,p,z,'B',direction)
    fIV = f(x-L,p - W,z,'B',direction)
    uB1 = fI[0] - fII[0] - fIII[0] + fIV[0]  
    uB2 = fI[1] - fII[1] - fIII[1] + fIV[1]  
    uB3 = fI[2] - fII[2] - fIII[2] + fIV[2]  
  
    fI = f(x,p,z,'C',direction)
    fII = f(x,p - W,z,'C',direction)
    fIII = f(x-L,p,z,'C',direction)
    fIV = f(x-L,p - W,z,'C',direction)
    uC1 = fI[0] - fII[0] - fIII[0] + fIV[0]  
    uC2 = fI[1] - fII[1] - fIII[1] + fIV[1]  
    uC3 = fI[2] - fII[2] - fIII[2] + fIV[2]  
  
    if output == 'dudz':
      fI = f(x,p,z,'D',direction)
      fII = f(x,p - W,z,'D',direction)
      fIII = f(x-L,p,z,'D',direction)
      fIV = f(x-L,p - W,z,'D',direction)
      uD1 = fI[0] - fII[0] - fIII[0] + fIV[0]  
      uD2 = fI[1] - fII[1] - fIII[1] + fIV[1]  
      uD3 = fI[2] - fII[2] - fIII[2] + fIV[2]  
      ux += U[itr]/(2*np.pi)*(uA1 + uhatA1 + uB1 + uD1 + z*uC1)
      uy += (U[itr]/(2*np.pi)*((uA2 + uhatA2 + uB2 + uD2 + z*uC2)*cosdel - 
             (uA3 + uhatA3 + uB3 + uD3 + z*uC3)*sindel))
      uz += (U[itr]/(2*np.pi)*((uA2 + uhatA2 + uB2 - uD2 - z*uC2)*sindel + 
             (uA3 + uhatA3 + uB3 - uD3 - z*uC3)*cosdel))
    else:
      ux += U[itr]/(2*np.pi)*(uA1 - uhatA1 + uB1 + z*uC1)
      uy += (U[itr]/(2*np.pi)*((uA2 - uhatA2 + uB2 + z*uC2)*cosdel - 
             (uA3 - uhatA3 + uB3 + z*uC3)*sindel))
      uz += (U[itr]/(2*np.pi)*((uA2 - uhatA2 + uB2 - z*uC2)*sindel + 
             (uA3 - uhatA3 + uB3 - z*uC3)*cosdel))
  return np.array([ux,uy,uz]) 

##------------------------------------------------------------------
def Okada85(x,y,L,W,d,delta,U):                                                
  '''                                                                           
  Description:                                                                  
    computes displacements at points (x,y) for a fault with                
    width W, length L, and depth d.  The fault has one end on the               
    origin and the other end at (x=L,y=0).  Slip on the fault is                
    described by U.                                                             
  Arguments:                                                                    
    x: along strike coordinate of output locations (can be a vector)            
    y: perpendicular to strike coordinate of output locations (can be a vector) 
    L: length of fault                                                          
    W: width of fault                                                           
    d: depth of the bottom of the fault (a fault which ruptures the surface     
       with have d=W, and d<W will give absurd results)                         
    delta: fault dip.  0<delta<pi/2.0 will dip in the -y direction. and         
           pi/2<delta<pi will dip in the +y direction... i think.               
    U: a three components vector with strike-slip, dip-slip, and tensile        
       components of slip                                                       
                                                                                
  output:                                                                       
    tuple where each components is a vector of displacement in the x, y, or z  
    direction                                                                   
  '''                                                                           
  mu = 3.2e10                                                                   
  lamb = 3.2e10                                                                 
  x = np.array(x)                                                               
  y = np.array(y)                                                               
  cosdel = np.cos(delta)                                                        
  sindel = np.sin(delta)                                                        
  p = y*cosdel + d*sindel                                                       
  q = y*sindel - d*cosdel                                                       
  r = np.sqrt(np.power(y,2.0) +                                                 
              np.power(x,2.0) +                                                 
              np.power(d,2.0))
  def f(eps,eta):
    y_til = eta*cosdel + q*sindel
    d_til = eta*sindel - q*cosdel
    R = np.sqrt(np.power(eps,2.0) +
                np.power(eta,2.0) +
                np.power(q,2.0))
    X = np.sqrt(np.power(eps,2.0) +
                np.power(q,2.0))

    if cosdel < tol:
      I1 = (-mu/(2*(lamb + mu))*
            eps*q/np.power((R + d_til),2.0))
      I3 = (mu/(2*(lamb + mu))*
            (eta/(R + d_til) +
             y_til*q/np.power(R + d_til,2.0) -
             np.log(R + eta)))
      I2 = (mu/(lamb + mu)*
            -np.log(R + eta) - I3)
      I4 = (-mu/(lamb + mu)*
            q/(R + d_til))
      I5 = (-mu/(lamb + mu)*
            eps*sindel/(R + d_til))
    else:
      I5 = (mu/(lamb + mu)*
            (2.0/cosdel)*
            np.arctan((eta*(X + q*cosdel) + X*(R + X)*sindel)/
                      (eps*(R + X)*cosdel)))
      I4 = (mu/(lamb + mu)*
            1.0/cosdel*
            (np.log(R + d_til) - sindel*np.log(R + eta)))
      I3 = (mu/(lamb + mu)*
            (y_til/(cosdel*(R+d_til)) - np.log(R + eta)) +
            sindel/cosdel*I4)
      I2 = (mu/(lamb + mu)*
            -np.log(R + eta) - I3)
      I1 = (mu/(lamb + mu)*
            (-eps/(cosdel*(R + d_til))) -
            sindel/cosdel*I5)

    u1 = (-U[0]/(2*np.pi)*
          (eps*q/(R*(R + eta)) +
           np.arctan(eps*eta/(q*R)) +
           I1*sindel))
    u2 = (-U[0]/(2*np.pi)*
          (y_til*q/(R*(R + eta)) +
           q*cosdel/(R + eta) +
           I2*sindel))
    u3 = (-U[0]/(2*np.pi)*
          (d_til*q/(R*(R + eta)) +
           q*sindel/(R + eta) +
           I4*sindel))

    u1 += (-U[1]/(2*np.pi)*
           (q/R - I3*sindel*cosdel))
    u2 += (-U[1]/(2*np.pi)*
           (y_til*q/(R*(R + eps)) +
            cosdel*np.arctan(eps*eta/(q*R)) -
            I1*sindel*cosdel))
    u3 += (-U[1]/(2*np.pi)*
           (d_til*q/(R*(R + eps)) +
            sindel*np.arctan(eps*eta/(q*R)) -
            I5*sindel*cosdel))

    u1 += (U[2]/(2*np.pi)*
           (np.power(q,2.0)/(R*(R + eta)) -
            I3 * np.power(sindel,2.0)))
    u2 += (U[2]/(2*np.pi)*
           (-d_til*q/(R*(R + eps)) -
            sindel*(eps*q/(R*(R + eta)) - np.arctan(eps*eta/(q*R))) -
            I1*np.power(sindel,2.0)))
    u3 += (U[2]/(2*np.pi)*
           (y_til*q/(R*(R + eps)) +
            cosdel*(eps*q/(R*(R + eta)) - np.arctan(eps*eta/(q*R))) -
            I5*np.power(sindel,2.0)))
    return (u1,u2,u3)

  disp1 = f(x,p)[0] - f(x,p - W)[0] - f(x - L,p)[0] + f(x-L,p-W)[0]
  disp2 = f(x,p)[1] - f(x,p - W)[1] - f(x - L,p)[1] + f(x-L,p-W)[1]
  disp3 = f(x,p)[2] - f(x,p - W)[2] - f(x - L,p)[2] + f(x-L,p-W)[2]
  return (disp1,disp2,disp3)



