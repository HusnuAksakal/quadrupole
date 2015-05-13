#!/usr/bin/python
"""
val[0] = x current position; val[1] = current velocity
val[1] = y current position; val[1] = current velocity
"""
from __future__ import division
import collections
import numpy as np
from pylab import *
from numpy import *
from scipy import *
from scipy.stats import maxwell
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import normal
from matplotlib.pylab import *  # for random
import matplotlib.cm as cm

import pylab, matplotlib,sys, os, scipy, numpy
import matplotlib.pyplot as plt

#from matplotlib.mlab import griddata
#from visual.graph import *
#from scipy.special.orthogonal import hermite
#from scipy.special.orthogonal import laguerre

from PhysicsConstants import *

massK  = 39.0026    ;massNa = 22.990      # sodium    # potassium
massRb = 85.4678    ;massLi = 6.941        # lithium      #rubidium 
massF  = 18.998404  ;massCs = 132.90546    # cesium     # Fluorine   charge -1
massCl  = 35.4527   ;massBr = 79.904      # Bromine charge -1  # Chlorine  charge -1
massMg =  24.305    ;massCa = 40.078     #calcium charge +2  #Magnesium charge +2
massSr = 87.62      ;massIn = 114.818 # indium +3 #strontium charge +2

mass =  massK*one_amu            # [kg] 
T = 298.15                       # Kelvin
Np = 1                           # parcacik sayisi
sigmax = 685.e-6                 # ion beam size 
sigmay = 685.e-6                 # ion beam size 
sigmaz = 675.e-6                 # ion beam size 
ener = (2.52e-3)*eV              # energy in J

v = np.sqrt(2.*ener/mass)          # initial velocity
print 'v=', v 
#trapparam=dict(Udc=10,Vac=500,f=1.0e6)
re  = 2.95*mm                      # electrode radius   
r0  = re/1.146                     # Inscribed radius (for best trap)
z0  = 12.*mm
gL  = z0/3.
lring= 3.*mm
Lq  = 2.* (z0+gL+lring)            # Quadrupole length #127*mm  
r_p = 2.*re+r0 +1.*mm
r_pmax =r_p + lring
f   = 0.85e6                           # Mhz
omega = 2.*np.pi*f 
d_2 = pow (r0,2)+2.*pow (z0,2)
tend = 2*15./f 
ksimax = tend #(omega/2.)*tend
tstart = 0;
dt     = 1.5/f #4*0.125/f  

nx = 2*32 ;  ny = 2*32;  nz = 2*32

elocx=r0+re ; elocy=r0+re   # electrode location
nx = 2*32 ;  ny = 2*32;  nz = 2*32
xmin=-r_pmax+r_pmax; xmax=r_pmax+r_pmax;
ymin=-r_pmax+r_pmax; ymax=r_pmax+r_pmax;
zmin=-Lq/2.+Lq/2.;  zmax=Lq/2.+Lq/2.;

dx = (xmax-xmin)/nx; dy = (ymax-ymin)/ny ; dz = (zmax-zmin)/nz

x = np.linspace(xmin, xmax, num=nx,endpoint=True) 
y = np.linspace(ymin, ymax, num=ny,endpoint=True) 
z = np.linspace(zmin, zmax, num=nz,endpoint=True) 

#rho = np.zeros((nx, ny, nz))
#rho[(nx-1)//2,(ny-1)//2,(nz-1)//2] = 0 #q*rand(Nz-1)/(dx*dy*dz) # ornek ->3//2 = 1 


def trapparam(m):
 az, qz = -0.0, 0.392 #0.0197,-0.735 #0.0645,-0.56571 #0.022,-0.60 #0.020,-0.572  
 Udc    = -az*m*d_2*pow(omega, 2)/(16.*echarge)
 Vac    = -qz*m*d_2*pow(omega, 2)/(8.*echarge)
 ay = ax = 8.*echarge*Udc/(m*d_2*omega*omega)   
 qy = qx = 4.*echarge*Vac/(m*d_2*omega*omega)   
 return ax,ay,az,qx,qy,qz,Udc,Vac

ax,ay,az,qx,qy,qz,Udc,Vac = trapparam(mass)
print 'ax, qx,Udc, Vac =',ax, qx,Udc,Vac

epsilon = 0.45e-5 ## Absolute Error tolerance
imax = 300    ## Maximum number of iterations allowed

def SolveLaplace(t,nx, ny, nz,dx,dy,dz, epsilon, imax):
  T  = np.zeros((len(x), len(y), len(z) ))
  mask  = np.zeros((len(x), len(y), len(z) ))
  print 'len-T=',len(T)
  global  Vac,Udc 
 
  ## Set boundary conditions for the problem
  
  for k in range ( nz ):
    for j in range ( ny  ):   
      for i in range ( nx  ):
        # left pole   
        #print 'z[k]=',z[k],k
        r1 = sqrt((x[i] +elocx-r_pmax )**2 + (y[j] -r_pmax )**2);
        if r1<=(re)  and abs(z[k]) <= Lq/2.+Lq/2. :
          T[i,j,k] = -Vac*np.cos(omega*t)

        # right  pole
        r2 = sqrt((x[i] -elocx-r_pmax)**2 + (y[j]-r_pmax)**2);
        if r2<=re and abs(z[k]) <= Lq:
          T[i,j,k] =- Vac*np.cos(omega*t);
    #    elif r2 > (re) : T[i,j,k] =0
  
       #  top pole
        r3 = sqrt((x[i] -r_pmax)**2 + (y[j] +elocy-r_pmax)**2);
        if r3 <= re and abs(z[k]) <=Lq/2.+Lq/2.: 
          T[i,j,k]= Vac*np.cos(omega*t);
#          print '3-i,j,k=',i,j,k
#          time.sleep(1)
   #     elif r3 > (re) : T[i,j,k] =0
  
        # bottom pole
        r4 = sqrt((x[i] -r_pmax)**2 + (y[j] -elocy-r_pmax)**2);
        if r4<=re and abs(z[k]) <=Lq/2.+Lq/2. : 
           T[i,j,k]= Vac*np.cos(omega*t);
        
        r=sqrt(x[i]**2+y[j]**2);
        if r<(r0+2*re) and abs(z[k]) <=Lq/2.+Lq/2. : 
            mask[i,j,k]=1;
        elif r>(r0+2*re): mask[i,j,k]=0;
        
        # first ring
        r5 = sqrt((y[j]-r_pmax )**2+(x[i]-r_pmax )**2);
        if z[k] >= (z0-gL-lring/2.) and z[k] <= (z0-gL+lring/2.) and \
            r5 >= r_p and r5 <=r_pmax:
           T[i,j,k]=Udc;

        # second ring
        if z[k]>= (z0+gL-lring/2.) and z[k]<= (z0+gL+lring/2.)  and\
             r5>=r_p and r5 <=r_pmax:
            T[i,j,k]=Udc;
        
        T[i,j,0]=0
        T[0,j,k]=0
        T[i,0,k]=0 
        T[i,j,nz-1]=0
        T[nx-1,j,k]=0
        T[i,ny-1,k]=0
          
  Ex = np.zeros((len(x),len(y),len(z) ));
  Ey = np.zeros((len(x),len(y),len(z) ));
  Ez = np.zeros((len(x),len(y),len(z) ));
 
  ## Store previous grid values to check against error tolerance
  TN = T + np.zeros((len(x), len(y), len(z)))
  #TN=np.zeros((len(x), len(x), len(x)))
  err =TN - T# np.zeros((len(x), len(y), len(z))) #
  ## Constants
  it = 1                   # Iteration counter
  # white and black pixels: white have i+j+k even; black have i+j+k odd
  white = [(i, j, k)   for i in range(1,nx-1) \
                       for j in range(1,ny-1) \
                       for k in range(1,nz-1) if (i+j+k)%2 == 0]
  black = [(i, j, k)   for i in range(1,nx-1) \
                       for j in range(1,ny-1) \
                       for k in range(1,nz-1) if (i+j+k)%2 == 1]
  ## Iterative procedure
  dx2=dx**2; 
  dy2=dy**2; 
  dz2=dz**2; 
  ome= 2.0/(1.0+np.sin(np.pi*dz/Lq))
  while it <= imax:
    #for i in range (1,nx-1): 
    # for j in range(1,ny-1): 
    #  for k in range(1,nz-1):
       
    for (i, j, k) in white+black: # loop over white pixels then black pixels
       #if mask[i,j,k]==1:
        TN[i,j,k]=( dy2*dz2*(T[i-1,j,k] + T[i+1,j,k]) + \
                   dx2*dz2*(T[i,j-1,k] + T[i,j+1,k]) + \
                   dx2*dy2*(T[i,j,k-1] + T[i,j,k+1]))/(2*(dy2*dz2+ dx2*dz2 +dx2*dy2)) 
                   #+ (dx2+dy2+dz2)*rho[i,j,k]              
 #      T[i,j,k]+= ome*(TN[i,j,k] - T[i,j,k])
        err[i,j,k] += np.abs(TN[i,j,k] - T[i,j,k])
    err /= (nx*ny*nz)
    T = TN + np.zeros((nx, ny,nz))
    it =it+1
    errmax = np.max(np.max(err))
    if errmax < epsilon:
       print("Convergence after ", it, " iterations.")
       for n in range (1,nx-1):
        for m in range (1,ny-1):
         for k in range (1,nz-1):
          Ex[n,m,k] =-( (T[n+1,m,k]-T[n-1,m,k]) )/(2*(x[n+1]-x[n]));
          Ey[n,m,k] =-( (T[n,m+1,k]-T[n,m-1,k]) )/(2*(y[m+1]-y[m]));
          Ez[n,m,k] =-( (T[n,m,k+1]-T[n,m,k-1]) )/(2*(z[k+1]-z[k]));
       return Ex,Ey,Ez, T
  print("No convergence after ", it, " iterations.")
  return False

def PlotSolution(Ex,Ey,Ez,x,y,z,nx,ny,nz,dx,dy,dz,T):
#   Set up x and y vectors for meshgrid
#   X, Y,Z = np.meshgrid(x,y,z)
   X, Z= np.meshgrid(x,z)
   X, Y= np.meshgrid(x,y)
   
   plot(z,T[round(nx/2)-1,round(ny/2)-1,:],label='z');
   plot(x,T[:,round(ny/2)-1,round(nz/2)-1],label='x');
   plot(y,T[round(nx/2)-1,:,round(nz/2)-1],label='y');
   legend() 
   xlabel("z,x,y");ylabel("T");  show()
   
   plt.contour(x, y, T[:,:,round(nz/2)-1], 64, rstride=1, cstride=1, cmap=cm.cool)
   plt.colorbar()
   plt.quiver(x,y,Ex[:,:,round(nz/2)-1], Ey[:,:,round(nz/2)-1],angles='xy')
   plt.xlabel("x");    plt.ylabel("y")
   plt.show()
 
   plt.contour(z, x, T[:,round(ny/2)-1,:], 32, rstride=1, cstride=1, cmap=cm.cool)
   plt.colorbar()
   plt.quiver(z, x, Ex[:,round(ny/2)-1,:], Ez[:,round(ny/2)-1,:],angles='xy')
   plt.xlabel("z")
   plt.ylabel("x") ;   plt.show()

   fig = plt.figure()
   ax = fig.gca(projection='3d')  
   surf = ax.plot_surface(X,Z,T[:,round(ny/2)-1,:],rstride=1,cstride=1,cmap=cm.cool) # antialiased=False
   plt.xlabel("X")
   plt.ylabel("Y")
   plt.show()
      
   figa = plt.figure()
   ax1 = figa.gca(projection='3d')
   ax1.plot_surface(X,Y,T[:,:,round(nz/2)-1]);
   ax1.set_xlabel('X');ax1.set_ylabel('Y');
   ax1.set_title('Fields in the middle of the quadrupole')
   plt.show()
   
   plot(x,Ex[    :        ,round(ny/2)-1,round(nz/2)-1],label='Ex' );
   plot(y,Ey[round(nx/2)-1,    :         ,round(nz/2)-1],label='Ey' );
   plot(z,Ez[round(nx/2)-2, round(ny/2)-2,    :      ],label='Ez');
   legend()
   show()
   
   plt.contour(x, y, Ex[:,:,round(ny/2)-1], 32, rstride=1, cstride=1, cmap=cm.cool)
   plt.colorbar()
   plt.xlabel("x")
   plt.ylabel("y Elecric field countor Ex") ;   plt.show()

   plt.contour(x, y, Ey[:,:,round(nz/2)-1], 32, rstride=1, cstride=1, cmap=cm.cool)
   plt.colorbar()
   plt.xlabel("x")
   plt.ylabel("y Elecric field countor Ey") ;   plt.show()

   plt.contour(x, z, Ez[:,round(ny/2)-1,:], 32, rstride=1, cstride=1, cmap=cm.cool)
   plt.colorbar()
   plt.xlabel("x")
   plt.ylabel("z Elecric field countor Ez") ;   plt.show()
       
   plt.quiver(X,Y, Ex[-1,-1,round(nz/2)-1],Ey[:,:,round(nz/2)-1]); #,angles='xy'
   plt.xlabel("x")
   plt.ylabel("y")
   plt.show()

   plt.quiver(Z,X, Ez[-1,round(ny/2)-1,-1],Ex[-1,round(ny/2)-1,-1]); #,angles='xy'
   plt.xlabel("z")
   plt.ylabel("x")
   plt.show()
   
   plt.quiver(Z,Y, Ez[round(nx/2)-1,:,:],Ey[round(nx/2)-1,:,:]); #,angles='xy'
   plt.xlabel("z")
   plt.ylabel("y")
   plt.show()
   
   fig2 = plt.figure()
   plt.contourf(x, y, T[:,:,round(nz/2)-1], 32, rstride=1, cstride=1, cmap=cm.cool)
   plt.colorbar()
   plt.xlabel("x")
   plt.ylabel("y")
   plt.show()

   fig3 = pylab.figure()
   #ax = fig.gca(projection='3d', azim=-60, elev=20)
   ax=Axes3D(fig3)
   surf = ax.plot_surface(X, Y, T[:,:,round(nz/2)-1], rstride=1, cstride=1, linewidth=0.25,cmap=pylab.cm.jet ) # , norm=matplotlib.colors.LogNorm()
#   ax.contour(X, Y, T.T, levels, zdir='z', offset=-1.0, norm=matplotlib.colors.LogNorm())
#   ax.contourf(X, Y, T[:,0,:], 1, zdir='x', offset=-3)
#   ax.contourf(X, Y, T.T, 1, zdir='y', offset=L)
#   ax.set_zlim(-3.0, 3.0)
   ax.set_xlabel('x')
   ax.set_ylabel('y')
   ax.set_zlabel('u'); # fig.colorbar(surf)
   pylab.show()
   

## Size of plate and mesh in 3 dimension
# f= n* (m/(2.*np.pi*kB*T))**1.5 * exp(-m(u**2+v**2+w**2)/2*K*T)
# where n number density and u,v,w are velocities in 3 dimension
def idistrb(j,m):
  const = (m/(2.*np.pi*kB*T))**0.5 
  #fvx = [const*np.exp(-m*xp*xp /(2.*kB*T))] # 4.*np.pi*xp*xp  
  #fvy = [const*np.exp(-m*yp*yp /(2.*kB*T))] #*4.*np.pi*yp*yp   
  #fvz = [const*np.exp(-m*zp*zp /(2.*kB*T))]  
  val=[[],[],[],[],[]]
  x0 = np.random.normal(loc= 600.0e-6,scale=sigmax, size=j)
  y0 = np.random.normal(loc=-600.0e-6,scale=sigmay, size=j)
  z0 = np.random.normal(loc=-800.0e-6,scale=sigmaz, size=j) #np.zeros(j) #
  xp0, yp0, zp0 =maxwell.rvs(loc=0,scale=v, size=j), \
                 maxwell.rvs(loc=0,scale=v, size=j), \
                 maxwell.rvs(loc=0,scale=v, size=j)  
                 #v*random(j),v*random(j),v*random(j)
#  xp0,yp0,zp0 =np.random.uniform(0.,v,j),\
               # np.random.uniform(0.,v,j),\
               #np.random.uniform(0.,v,j)
#  xp, yp, zp = maxwell.rvs(v, size=j),  \
#                maxwell.rvs(v, size=j),  \ 
#                maxwell.rvs(v, size=j)  
#                #fmb(m, T, xp0,yp0,zp0) loc=v,scale=0, 
#  xpm, ypm, zpm = max(xp),  max(yp), max(zp)
  #i=0
  #while i<=j-1: 
  # if xpm<xp[i] and ypm<yp[i] and zpm < zp[i]  : 
  xxp = [x0, xp0]        # initial position & velocity
  yyp = [y0, yp0]        # initial position & velocity  
  zzp = [z0, zp0]        
  val =np.array([[xxp], [yyp], [zzp], [j],[m]])
  #  i+=1
  # else:
    #xp0, yp0, zp0 =np.random.uniform(0.,v,j), \
                    # np.random.uniform(0.,v,j), np.random.uniform(0.,v,j)
    #xp0,yp0,zp0 =v*random(j),v*random(j),v*random(j)
    #i+=0  
  xxp, yyp, zzp,j,m= val[0,0], val[1,0], val[2,0], val[3,0], val[4,0]
#  xxp, yyp, zzp =[array([0.0e-0]), array([2.6e-3])],\
                  #[array([0.0]), array([2.6e-3])], \ 
                  #[array([0.0e-1]), array([2.6e-3])]
  x, y, z   = [xxp[0]], [yyp[0]], [zzp[0]]
  xp, yp, zp  = [xxp[1]], [yyp[1]], [zzp[1]]
  return xxp,yyp,zzp,x,y,z,xp,yp,zp

def comin():
  cminxxp=[sum(xxp[0][i] for i in range(1,Np)),sum(xxp[1][i] for i in range(1,Np))] 
  cminyyp=[sum(yyp[0][i] for i in range(1,Np)),sum(yyp[1][i] for i in range(1,Np))] 
  cminzzp=[sum(zzp[0][i] for i in range(1,Np)),sum(zzp[1][i] for i in range(1,Np))] 
  masscm = mass*(Np-1)
  charge= echarge*(Np-1)
  valcm =np.array([[cminxxp], [cminyyp], [cminzzp]])
  return valcm , charge, masscm
#valcm , chargecm, masscm =comin()
#cminxxp,cminyyp,cminzzp = valcm[0,0],valcm[1,0],valcm[2,0]
#cmxi ,cmyi, cmzi = cminxxp[0],cminyyp[0],cminzzp[0]
#cmvxi ,cmvyi, cmvzi = cminxxp[1],cminyyp[1],cminzzp[1]

def lasercw(xpp):
 x0,y0,z0 = xxp[0], yyp[0], zzp[0]
 xpp, ypp, zpp = xxp[1], yyp[1], zzp[1]
 power        = 300.e-3   # Watt
 lambda_laser = 279.6e-9  # for mg 3s1/2-3p3/2
 W1           = 70.e-6    #micron np.sqrt(lambda_laser*zr/np.pi)
 k1           = 2.*pi/lambda_laser
 w1           = k1*clight              # angular frequency 
 gamma = 42.0e6            # FWHM Mg 3S1/2 -> 3P3/2
 w0 = (clight/279.6e-9)*2*np.pi # mg transition  
# x0 = np.random.normal(loc=100e-6,scale=W1, size=j)
# y0 = np.random.normal(loc=100e-6,scale=W1, size=j)
 IntensL = (2.*power/np.pi*W1*W1)*exp(-2.*(y0/W1)**2 -2.*(z0/W1)**2 )
 I0sat = 15.73             # ( = c k**3 gamma)  watt/cm^2
 delta = abs(w0-w1)
 S = (IntensL/I0sat) * (gamma/2.)**2 /((delta-xpp/clight)**2 + (gamma/2.)**2) 
 R = 2.*np.pi*gamma* S/(1.+2.*S)
 Fx = hbar*k1*R
 """
 gamma = (1.0/7.7e-9) # FWHM of 4s_1/2->4P_1/2 Ca ion
 lambda_f1= 396.847e-9 # 4s1/2-> 4p1/2 for Ca ion
 Fx = hbar*(s*(gamma/2.)/(1.+s + 4.*(delta-kz*zzp[1])**2/gamma**2 ) )*\
       (ell*yl/(xl**2 + yl**2))
 Fy = -hbar*(s*(gamma/2.)/(1.+s + 4.*(delta-kz*zzp[1])**2/gamma**2 ) )*\
      (ell*xl/(xl**2 + yl**2))
 Fz = -hbar*(s*(gamma/2.)/(1.+s + 4.*(delta-kz*zzp[1])**2/gamma**2 ) )*kz 
 """
# def incomplete(): 
 N= n0* (gama/2.)**2 / ((w0-w1*(1.-xxp[1]/c))**2 + (gama/2)**2) 
 Nph = N0*dt* Nfactor(w1*(1.-xxp[1]/c)) # number of photon 
 cth = 2.*random(Nph)-1    # get cos (theta)
 sth = np.sqrt(1.0-cth**2) # get sine (theta)
 phi = 2* np.pi*random(Nph) 
 sthcph = sth*cos(phi)
 sthsph = sth*sin(phi)
 dvz=-xxp[1]*cth ;  dvy=-xxp[1]*sthsph ;  dvx=-xxp[1]*sthsph
 return Fx

def impact():
  b=(k*q1*q2/(m*v0**2))*np.cot(theta/2.)
  return b

from scipy.interpolate import Rbf

def interpol(t,xx,yy,zz):
  # X, Y,Z = np.meshgrid(x,y,z)
  # X, Z= np.meshgrid(x,z)
  # X, Y= np.meshgrid(x,y)
   Ex,Ey,Ez,T= SolveLaplace(t,nx,ny,nz,dx,dy,dz, epsilon,imax)
   # print 'len-Ex,len-x,len-y,lenz=',len(Ex),len(x),len(y),len(z)
   """
   for n in range (nx-1):
     for m in range (ny-1):
      for k in range (nz-1):
       Ex1 = Rbf(x[n], y[m], z[k], Ex[n,m,k],epsilon=(x[n+1]-x[n]))
       Ey1 = Rbf(x[n], y[m], z[k], Ey[n,m,k],epsilon=(y[n+1]-y[n]))
       Ez1 = Rbf(x[n], y[m], z[k], Ez[n,m,k],epsilon=(z[n+1]-z[n])) #,epsilon=2
       #print 'ksi,Ex1,,Ey1,,Ez1=',ksi,Ex1,Ey1,Ez1
       Ex2= Ex1(xx,yy,zz)
       Ey2= Ey1(xx,yy,zz)
       Ez2= Ez1(xx,yy,zz)
       return Ex2,Ey2,Ez2   
   """
   """ 
   for n in range (nx-1):
    for m in range (ny-1):
     for k in range (nz-1):
     # if xxp[0]>x[n] and xxp[0]<=x[n+1] :
     #   if yyp[0]>y[m] and yyp[0]<=y[m+1] :
     #     if zzp[0]>z[k] and zzp[0]<=z[k+1] :
            Ex2=Ex[n,m,k]+(Ex[n+1,m,k]-Ex[n,m,k])*((xx-x[n])/(x[n+1]-x[n]))  
            Ey2=Ey[n,m,k]+(Ey[n,m+1,k]-Ey[n,m,k])*((yy-y[m])/(y[m+1]-y[m]))
            Ez2=Ez[n,m,k]+(Ez[n,m,k+1]-Ez[n,m,k])*((zz-z[k])/(z[k+1]-z[k]))
            return Ex2,Ey2,Ez2
   #plt.quiver(xxp[0],yyp[0],Ex1[:,:,0], Ey1); plt.show()
   #plt.quiver(xxp[0],zzp[0],Ex1, Ez1); plt.show()
   #print 'Ex,Ey2,Ez=',Ex2,Ey2,Ez2
   """
   """
   for n in range (nx-1):
    for m in range (ny-1):
     for k in range (nz-1):
   
      xd = (xx-x[n])/(x[n+1]-x[n])
      yd = (yy-y[n])/(y[n+1]-y[n])
      zd = (zz-z[n])/(z[n+1]-z[n])
   
      Ex = Ex[x[n], y[m], z[k] ]*(1-xd)*(1-yd)*(1-zd)  +\
         Ex[x[n+1], y[m], z[k]]*(1-xd)*(1-yd)* zd   +\
         Ex[x[n], y[m+1], z[k]]*(1-xd)* yd * (1-zd) +\
         Ex[x[n], y[m], z[k+1]]*(1-xd)* yd * zd     +\
         Ex[x[n+1], y[m+1], z[k]]*yd*(1-xd)* (1-zd) +\
         Ex[x[n+1], y[m], z[k+1]]*yd *(1-xd)* zd      +\
         Ex[x[n], y[m+1], z[k+1]]*  yd * xd *(1-zd)   +\
         Ex[x[n+1], y[m+1], z[k+1]]*yd *xd *zd            

      Ey = Ey[x[n], y[m], z[k] ]*(1-xd)*(1-yd)*(1-zd)  +\
         Ey[x[n+1], y[m], z[k]]*(1-xd)*(1-yd)* zd   +\
         Ey[x[n], y[m+1], z[k]]*(1-xd)* yd * (1-zd) +\
         Ey[x[n], y[m], z[k+1]]*(1-xd)* yd * zd     +\
         Ey[x[n+1], y[m+1], z[k]]*yd*(1-xd)* (1-zd) +\
         Ey[x[n+1], y[m], z[k+1]]*yd *(1-xd)* zd      +\
         Ey[x[n], y[m+1], z[k+1]]*  yd * xd *(1-zd)   +\
         Ey[x[n+1], y[m+1], z[k+1]]*yd *xd *zd            
      Ez = Ez[x[n], y[m], z[k] ]*(1-xd)*(1-yd)*(1-zd)  +\
         Ez[x[n+1], y[m], z[k]]*(1-xd)*(1-yd)* zd   +\
         Ez[x[n], y[m+1], z[k]]*(1-xd)* yd * (1-zd) +\
         Ez[x[n], y[m], z[k+1]]*(1-xd)* yd * zd     +\
         Ez[x[n+1], y[m+1], z[k]]*yd*(1-xd)* (1-zd) +\
         Ez[x[n+1], y[m], z[k+1]]*yd *(1-xd)* zd      +\
         Ez[x[n], y[m+1], z[k+1]]*  yd * xd *(1-zd)   +\
         Ez[x[n+1], y[m+1], z[k+1]]*yd *xd *zd            
      return Ex, Ey, Ez
    
   """
   #print 'Ex=',Ex,Ey,Ez
   for n in range (1,nx-1):
    for m in range (1,ny-1):
     for k in range (1,nz-1):
          i,j,l=abs(int( round( np.floor(xx / x[n]) ))),\
            abs(int( round( np.floor(yy / y[m]) ))),\
            abs(int( round( np.floor(zz / z[k]) )))
      #print 'a,b,c=',a,b,c
     # if (a,b,c) <=1:
          Ex2=((xx-x[i])/(x[i+1]-x[i]))*Ex[i+1,j,l]+((xx-x[i+1])/(x[i]-x[i+1]))*Ex[i,j,l]
          Ey2=((yy-y[j])/(y[j+1]-y[j]))*Ey[i,j+1,l]+((yy-y[j+1])/(y[j]-y[j+1]))*Ey[i,j,l]
          Ez2=((zz-z[l])/(z[l+1]-z[l]))*Ez[i,j,l+1]+((zz-z[l+1])/(z[l]-z[l+1]))*Ez[i,j,j]
          print 'Ex2=',Ex2,Ey2,Ez2
          return Ex2,Ey2,Ez2   
def rk4_two(ksi, h,m,f3,f4,f5,f6,xxp,yyp,zzp,Coulomb,ax,ay,az,qx,qy,qz):
   global d_2   
   Ex2,Ey2,Ez2= interpol(ksi,xxp[0],yyp[0],zzp[0])
   
   def accn(ksi, xxp,yyp,zzp,m,f3,f4,f5,f6):

       if Coulomb==1: 
         for i in range( Np):
          const=4.*ke * echarge/(omega*omega)
          fcx, fcy, fcz = 0.,0.,0.
          for j in range( Np):
           if j == i : continue
           ri=np.sqrt((xxp[0][i]-xxp[0][j])**2+(yyp[0][i]-yyp[0][j])**2+(zzp[0][i]-zzp[0][j])**2+ 1.0e-10)
           fcx +=const*(echarge/m)*(xxp[0][i]-xxp[0][j]) / ri**3 
           fcy +=const*(echarge/m)*(yyp[0][i]-yyp[0][j]) / ri**3
           fcz +=const*(echarge/m)*(zzp[0][i]-zzp[0][j]) / ri**3  	        
       else:   fcx=0; fcy=0; fcz=0         
       if f3 !=0:
         hexa = 3.*f3*zzp[0]/np.sqrt(d_2)
         hexaz=f3*(3.*(xxp[0]**2 +yyp[0]**2)- 6.*zzp[0]**2)/(2.*np.sqrt(d_2))
       else : hexa =0; hexaz=0
       if f4 !=0:
         octo =f4*(12.*(xxp[0]**2+yyp[0]**2)- 48.*zzp[0]**2)/(8.*d_2)
         octoz=f4*(-48.*(xxp[0]**2+yyp[0]**2)*zzp[0]+32.*zzp[0]**3)/(8.*d_2)
       else : octo=0; octoz=0
       if f5 !=0: 
         deca =f5*(60.*(xxp[0]**2+yyp[0]**2)*zzp[0] -80.*zzp[0]**3)/(8.*d_2*np.sqrt(d_2))
         decaz=f5*(15.*(xxp[0]**2+yyp[0]**2)**2 -120.*(xxp[0]**2+yyp[0]**2)*zzp[0]**2 +40.*zzp[0]**4)/(8.*d_2*np.sqrt(d_2))
       else : deca =0; decaz =0  
       if f6 !=0: 
         dodeca =f6*(30.*(xxp[0]**2+yyp[0]**2)**2 - 360.*(xxp[0]**2+yyp[0]**2)*zzp[0]**2 +240.*zzp[0]**4)/(16.*d_2*d_2)
         dodecaz=f6*(-180.*(xxp[0]**2+yyp[0]**2)**2 *zzp[0] +480.*(xxp[0]**2+yyp[0]**2)*zzp[0]**3-96.*zzp[0]**5)/(16.*d_2*d_2)
       else : dodeca =0; dodecaz =0
       beta=1.e-4
         
       accx =(echarge/m)*Ex2 #-(ax-2.*qx*np.cos(2.*ksi))*(1.+ hexa+ octo + deca+dodeca )*xxp[0] + fcx -beta*xxp[1]
	#-Fx*(4./(omega*omega*m)) #+2*beta*xxp[1] -(1./m)*xxp[0]#  
       accy = (echarge/m)*Ey2 #-(ay-2.*qy*np.cos(2.*ksi))*(1.+ hexa+ octo+deca+dodeca )*yyp[0] + fcy  -beta*yyp[1]
	 #+2*beta*yyp[1]-(1./m)*yyp[0] # 
       accz = (echarge/m)*Ez2 #-(az-2.*qz*np.cos(2.*ksi))*(zzp[0]+ hexaz +octoz + decaz+dodecaz) + fcz -beta*zzp[1]
       return accx,accy,accz 

   def vel(ksi, xxp,yyp,zzp,m):
        return  xxp[1], yyp[1],zzp[1]
     
   """ 
   k1 = [[0],[0]]	# k1=[[velx,vely,velz],[accx,accy,accz]]
   k2 = [[0],[0]]
   k3 = [[0],[0]]  
   k4 = [[0],[0]]
   
   k1[0] = vel(ksi,xxp,yyp,zzp,m)   # for x, y,z
   k1[1] = accn(ksi,xxp,yyp,zzp,m,f3,f4,f5,f6)  # for vx, vy,vz
 
   k2[0] = vel(ksi+h/2.,  xxp +k1[0][0]*h/2., yyp +k1[0][1]*h/2., zzp +k1[0][2]*h/2.,m)
   k2[1] = accn(ksi+h/2., xxp +k1[1][0]*h/2., yyp +k1[1][1]*h/2., zzp +k1[1][2]*h/2.,m,f3,f4,f5,f6)
   
   k3[0] =vel(ksi+h/2.,  xxp+k2[0][0]*h/2., yyp+k2[0][1]*h/2.,  zzp+k2[0][2]*h/2.,m)
   k3[1] =accn(ksi+h/2., xxp+k2[1][0]*h/2., yyp+k2[1][1]*h/2., zzp+k2[1][2]*h/2. ,m,f3,f4,f5,f6)
   
   k4[0] =vel(ksi+h,  xxp+k3[0][0]*h,  yyp+k3[0][1]*h,  zzp+k3[0][2]*h ,m)
   k4[1] =accn(ksi+h, xxp+k3[1][0]*h, yyp+k3[1][1]*h,  zzp +k3[1][2]*h ,m,f3,f4,f5,f6)
   
   for i in range (2):
     xxp[i] = xxp[i] +(k1[i][0] + 2.0*k2[i][0] + 2.0*k3[i][0] + k4[i][0])*h/6.0
     yyp[i] = yyp[i] +(k1[i][1] + 2.0*k2[i][1] + 2.0*k3[i][1] + k4[i][1])*h/6.0
     zzp[i] = zzp[i] +(k1[i][2] + 2.0*k2[i][2] + 2.0*k3[i][2] + k4[i][2])*h/6.0
   """
   k1 = [[0],[0]]            # k1=[[vel],[acc]]
   k2 = [[0],[0]]
   k3 = [[0],[0]]            #[0.,0.],[0.,0.],[0,0]
   k4 = [[0],[0]]
   temp1= [[0],[0]];
   temp2= [[0],[0]];
   temp3 =[[0],[0]]

   k1[0] = vel(ksi,xxp,yyp,zzp,m)   # for x, y,z
   k1[1] = accn(ksi,xxp,yyp,zzp,m,f3,f4,f5,f6)  # for vx, vy,vz
   for i in range(2):
      temp1[i]=xxp[i]+k1[i][0]*h/2. 
      temp2[i]=yyp[i]+k1[i][1]*h/2. 
      temp3[i]=zzp[i]+k1[i][2]*h/2.
   k2[0] = vel(ksi+h/2., temp1,temp2,temp3,m)
   k2[1] = accn(ksi+h/2., temp1,temp2,temp3,m,f3,f4,f5,f6)
   for i in range(2):
      temp1[i]=xxp[i]+k2[i][0]*h/2. 
      temp2[i]=yyp[i]+k2[i][1]*h/2. 
      temp3[i]=zzp[i]+k2[i][2]*h/2.
   k3[0] =vel(ksi+h/2., temp1, temp2, temp3,m)
   k3[1] =accn(ksi+h/2., temp1, temp2, temp3,m,f3,f4,f5,f6)
   for i in range(2):
      temp1[i]=xxp[i]+k3[i][0]*h 
      temp2[i]=yyp[i]+k3[i][1]*h 
      temp3[i]=zzp[i]+k3[i][2]*h
   k4[0] =vel(ksi+h, temp1,temp2,temp3,m)
   k4[1] =accn(ksi+h, temp1, temp2, temp3,m,f3,f4,f5,f6)
   for i in range (2):
     xxp[i] = xxp[i] +(k1[i][0] + 2.0*k2[i][0] + 2.0*k3[i][0] + k4[i][0])*h/6.0
     yyp[i] = yyp[i] +(k1[i][1] + 2.0*k2[i][1] + 2.0*k3[i][1] + k4[i][1])*h/6.0
     zzp[i] = zzp[i] +(k1[i][2] + 2.0*k2[i][2] + 2.0*k3[i][2] + k4[i][2])*h/6.0
   
   return xxp , yyp , zzp # x and xprime values in a 'xxp', y and yprime in 'yyp'
"""
void create_random_gas(){
     vmb = sqrt(-2.0*kb*T/m_gas*log(1-sqrt(collrand()))); 
     //for velocity
     theta_gas = collrand()*2.0*pi;
     phi_gas = acos(collrand()*2.0-1.0);
     gas_v[0] = vmb*cos(theta_gas)*sin(phi_gas);
     gas_v[1] = vmb*sin(theta_gas)*sin(phi_gas);
     gas_v[2] = vmb*cos(phi_gas);
   //cout<<"normal: "<<gas_v[0]<<"  "<<gas_v[1]<<"  "<<gas_v[2]<<endl;   
     // for location
     theta_gas = collrand()*2.0*pi;
     phi_gas = acos(collrand()*2.0-1.0);
}    """
#def __init__(self, voltdata=None,timedata=None, **traits):
#        # express the coordinates in polar form
#        x = r[:, 0]
#        y = r[:, 1]
#        z = r[:, 2]
#        rho = np.sqrt(x**2 + y**2)
#        theta = np.arctan(x/y)

class Multipole:
 def __init__(self):
   self.xxp, self.yyp, self.zzp, self.x, self.y, self.z, self.xp, self.yp, self.zp =idistrb(Np,mass)
   #self.Ex1,self.Ey1,self.Ez1,self.T=SolveLaplace(ksi,nx,ny,nz,dx,dy,dz,epsilon, imax)
   """
   rrp[0,0]  = x0 ;      #initial x position
   rrp[0,1]  = vx0;      #initial x velocity
   rrp[1,0]  = y0 ;    #initial y position
   rrp[1,1]  = vy0;      #initial y velocity
   rrp[2,0]  = z0 ;    #initial z position
   rrp[2,1]  = vz0;      #initial z velocity
   """
   print 'xxp,yyp,zzp=',self.xxp, self.yyp,self.zzp
   self.ax,self.ay,self.az,self.qx,self.qy,\
   self.qz,self.Udc,self.Vac= trapparam(mass)

   plt.hist(self.x) ; plt.hist(self.y) ; plt.hist(self.z) ; show()
   #plt.hist(self.xp) ; plt.hist(self.yp); plt.hist(self.zp) ; show()

   #fig = plt.figure()
   #ag = fig.add_subplot(111, projection='3d')
   #ag.scatter(self.x, self.y, self.z, c='r', marker='o'); show()

   #fig = plt.figure()
   #agcm = fig.add_subplot(111, projection='3d')
   #agcm.scatter(cmxi, cmyi, cmzi, c='r', marker='o')
   #agcm.scatter(x[0], y[0], z[0], c='r', marker='o'); show()

 def quad(self,f3,f4,f5,f6,Coulomb):
   ksi = 0.0         	# Stating time
   h =  dt #0.01      	# Runge-Kutta step size,time (ksi) increment
   tm = [0.0]         	
   global mass,ksimax
   xxp=self.xxp[:];yyp=self.yyp[:]; zzp=self.zzp[:]
   xp=self.xp[:];yp=self.yp[:]; zp=self.zp[:]
   x=self.x[:];y=self.y[:]; z=self.z[:]
   ax,ay,az,qx,qy,qz=self.ax,self.ay,self.az,self.qx,self.qy,self.qz
   print 'f1,f2,f3=',f3,f4,f5,f6
   while ksi < ksimax:
     rk4_two(ksi,h,mass, f3,f4,f5,f6,  xxp,  yyp, zzp,Coulomb,ax,ay,az,qx,qy,qz)
     ksi = ksi + h
     tm.append(ksi)
     xp.append(xxp[1])
     x.append(xxp[0])
     yp.append(yyp[1])
     y.append(yyp[0])
     zp.append(zzp[1])
     z.append(zzp[0])
     print 'simulation' ,(ksi/tend)*100 , '% is completed'
   return xp, x, yp,y,zp,z,tm
 def hexa(self,f3,f4,f5,f6,Coulomb): 
   ksi = 0.0         	# Stating time
   h = 0.01      	# Runge-Kutta step size, time (ksi) increment  
   tm = [0.0]         	# Lists to store time, position & velocity
   xxp=self.xxp[:];yyp=self.yyp[:]; zzp=self.zzp[:]
   xp=self.xp[:];yp=self.yp[:]; zp=self.zp[:]
   x=self.x[:];y=self.y[:]; z=self.z[:]
   ax,ay,az,qx,qy,qz=self.ax,self.ay,self.az,self.qx,self.qy,self.qz
   global mass,ksimax
   print 'f3,f4,f5,f6=',f3,f4,f5,f6  
   #print 'xxp,yyp,zzp=',self.xxp, self.yyp,self.zzp
   while ksi < ksimax:
     rk4_two(ksi,h,mass, f3,f4,f5,f6, xxp, yyp, zzp,Coulomb,ax,ay,az,qx,qy,qz)  # Do one step RK integration
     ksi = ksi + h
     tm.append(ksi)
     xp.append(xxp[1])
     x.append(xxp[0])
     yp.append(yyp[1])
     y.append(yyp[0])
     zp.append(zzp[1])
     z.append(zzp[0])
   return xp, x, yp,y,zp,z,tm
 
 def octa(self,f3,f4,f5,f6,Coulomb):
   ksi = 0.0         	# Stating time
   h = 0.01             # Runge-Kutta step size, time (ksi) increment  
   tm = [0.0]         	# Lists to store time, position & velocity
   xxp=self.xxp[:];yyp=self.yyp[:]; zzp=self.zzp[:]
   xp=self.xp[:];yp=self.yp[:]; zp=self.zp[:]
   x=self.x[:];y=self.y[:]; z=self.z[:]
   global mass,ksimax
   #print 'f3,f4,f5,f6=',f3,f4,f5,f6
   ax,ay,az,qx,qy,qz=self.ax,self.ay,self.az,self.qx,self.qy,self.qz
   while ksi < ksimax:
     rk4_two(ksi,h,mass, f3,f4,f5,f6, xxp, yyp, zzp,Coulomb,ax,ay,az,qx,qy,qz)  #
     ksi = ksi + h
     tm.append(ksi)
     xp.append(xxp[1])
     x.append(xxp[0])
     yp.append(yyp[1])
     y.append(yyp[0])
     zp.append(zzp[1])
     z.append(zzp[0])
   return xp, x, yp,y,zp,z,tm # @staticmethod # @classmethod

 def deca(self,f3,f4,f5,f6,Coulomb):
   ksi = 0.0         	# Stating time
   h = 0.01      	# R-Kutta step size, time (ksi) increment  
   tm = [0.0]         	
   xxp=self.xxp[:];yyp=self.yyp[:]; zzp=self.zzp[:]
   xp=self.xp[:];yp=self.yp[:]; zp=self.zp[:]
   x=self.x[:];y=self.y[:]; z=self.z[:]
   ax,ay,az,qx,qy,qz=self.ax,self.ay,self.az,self.qx,self.qy,self.qz
   global mass,ksimax
   print 'f3,f4,f5,f6=',f3,f4,f5,f6
   while ksi < ksimax:
     rk4_two(ksi,h,mass, f3,f4,f5,f6, xxp, yyp, zzp,Coulomb,ax,ay,az,qx,qy,qz) 
     ksi = ksi + h
     tm.append(ksi)
     xp.append(xxp[1])
     x.append(xxp[0])
     yp.append(yyp[1])
     y.append(yyp[0])
     zp.append(zzp[1])
     z.append(zzp[0])
   return xp, x, yp,y,zp,z,tm
 def dodeca(self,f3,f4,f5,f6,Coulomb):
   ksi = 0.0         	# Stating time
   h = 0.01      	# R-Kutta step size, time (ksi) increment  
   tm = [0.0]         	
   xxp=self.xxp[:];yyp=self.yyp[:]; zzp=self.zzp[:]
   xp=self.xp[:];yp=self.yp[:]; zp=self.zp[:]
   x=self.x[:];y=self.y[:]; z=self.z[:]
   ax,ay,az,qx,qy,qz=self.ax,self.ay,self.az,self.qx,self.qy,self.qz
   global mass,ksimax
   print 'f3,f4,f5,f6=',f3,f4,f5,f6
   while ksi < ksimax:
     rk4_two(ksi,h,mass, f3,f4,f5,f6, xxp, yyp, zzp,Coulomb,ax,ay,az,qx,qy,qz) 
     ksi = ksi + h
     tm.append(ksi)
     xp.append(xxp[1])
     x.append(xxp[0])
     yp.append(yyp[1])
     y.append(yyp[0])
     zp.append(zzp[1])
     z.append(zzp[0])
   return xp, x, yp,y,zp,z,tm

if __name__ == "__main__":

 Ex,Ey,Ez,T = SolveLaplace(0*dt,nx, ny,nz, dx,dy,dz, epsilon, imax)
 PlotSolution(Ex,Ey,Ez,x,y,z, nx, ny,nz, dx,dy,dz, T)

 #print 'len-Ex,len-T =',len(Ex),len(T)
 qp=Multipole()
 quxp, qux, quyp,quy,quzp,quz,tm =qp.quad(0.,0.,0.,0.,1)
 print 'len-quxp=',len(quxp),len(qux),len(tm)
  
 plot (tm, qux)
 xlabel("tm")
 ylabel("x");show()

 plot (tm, quy); 
 xlabel("tm")
 ylabel("y");show()

 plot (tm, quz);  xlabel("tm")
 ylabel("x");show()

 plot (quxp, qux);  xlabel("xp")
 ylabel("x");show()

 plot (quyp, quy); xlabel("yp")
 ylabel("y");show()

 plot (quzp, quz);  xlabel("zp")
 ylabel("zp");show()

 fig = plt.figure()
 ay = fig.add_subplot(111, projection='3d')
 ay.plot(qux,quy,quz,label='q' )
 ay.set_xlabel('$x$')
 ay.set_ylabel('$y$')
 ax.set_zlabel('$z $')
 plt.legend(loc=1,borderaxespad=0) 
 plt.savefig('10_sp_m3dxyz.png')
 plt.show()


# quxpwc, quxwc, quypwc,quywc,quzpwc,quzwc,tm =qp.quad(0.,0.,0.,0.,0)#without coulomb
# hxp, hx, hyp,hy,hzp,hz,htm  =qp.hexa(0.1,0.,0.,0.,1)
# hxp1, hx1, hyp1,hy1,hzp1,hz1,htm1 =qp.hexa(0.2,0.,0.,0.,1)
# hxp2, hx2, hyp2,hy2,hzp2,hz2,htm2 =qp.hexa(-0.2,0.,0.,0.,1)
# oxp, ox, oyp,oy,ozp,oz,otm =qp.octa(0.1,0.1,0.0,0.,1)
# oxp1, ox1, oyp1,oy1,ozp1,oz1,otm1 =qp.octa(0.2,0.2,0.0,0.,1)
# oxp2, ox2, oyp2,oy2,ozp2,oz2,otm2 =qp.octa(-0.2,-0.2,0.0,0.,1)
# dxp, dx, dyp,dy,dzp,dz,dtm =qp.deca(0.1,0.1,0.1,0.,1)
# dxp1, dx1, dyp1,dy1,dzp1,dz1,dtm1 =qp.deca(-0.1,-0.1,-0.1,0.,1)
# dodxp, dodx, dodyp,dody,dodzp,dodz,dodtm =qp.dodeca(0.1,0.1,0.1,0.1,1)
# dodxp2, dodx2, dodyp2,dody2,dodzp2,dodz2,dodtm2 =qp.dodeca(-0.1,-0.1,-0.1,-0.1,1)
# print 'len=',len(qux),len(hx),len(ox),len(dx),len(dodx),len(tm) 
 """  
#[x[i][0] butun zamanlarda 0. partticle
# x[0] 0. zamandaki all particles
 """
def particles():
 c1 = np.array([qux[i][0] for i in range(int(len(tm))) ]) # all time x of 0. particle
 c2 = np.array([quy[i][0] for i in range(int(len(tm))) ])
 c3 = np.array([quz[i][0] for i in range(int(len(tm))) ])
 c4 = np.array([quxp[i][0] for i in range(int(len(tm))) ])
 c5 = np.array([quyp[i][0] for i in range(int(len(tm))) ])
 c6 = np.array([quzp[i][0] for i in range(int(len(tm))) ])
 """
 c1wc = np.array([quxwc[i][0] for i in range(int(len(tm))) ]) # all time x of 0. particle
 c2wc = np.array([quywc[i][0] for i in range(int(len(tm))) ])
 c3wc = np.array([quzwc[i][0] for i in range(int(len(tm))) ])#without coulomb
 c4wc = np.array([quxpwc[i][0] for i in range(int(len(tm))) ])
 c5wc = np.array([quypwc[i][0] for i in range(int(len(tm))) ])
 c6wc = np.array([quzpwc[i][0] for i in range(int(len(tm))) ])

 ch1 = np.array([hx[i][0] for i in range(int(len(tm))) ]) # all time x of 0. particle
 ch2 = np.array([hy[i][0] for i in range(int(len(tm))) ])
 ch3 = np.array([hz[i][0] for i in range(int(len(tm))) ])
 ch4 = np.array([hxp[i][0] for i in range(int(len(tm))) ])
 ch5 = np.array([hyp[i][0] for i in range(int(len(tm))) ])
 ch6 = np.array([hzp[i][0] for i in range(int(len(tm))) ])

 ch11 = np.array([hx1[i][0] for i in range(int(len(tm))) ]) # all time x of 0. particle
 ch21 = np.array([hy1[i][0] for i in range(int(len(tm))) ])
 ch31 = np.array([hz1[i][0] for i in range(int(len(tm))) ])
 ch41 = np.array([hxp1[i][0] for i in range(int(len(tm))) ])
 ch51 = np.array([hyp1[i][0] for i in range(int(len(tm))) ])
 ch61 = np.array([hzp1[i][0] for i in range(int(len(tm))) ])

 ch12 = np.array([hx2[i][0] for i in range(int(len(tm))) ]) # all time x of 0. particle
 ch22 = np.array([hy2[i][0] for i in range(int(len(tm))) ])
 ch32 = np.array([hz2[i][0] for i in range(int(len(tm))) ])
 ch42 = np.array([hxp2[i][0] for i in range(int(len(tm))) ])
 ch52 = np.array([hyp2[i][0] for i in range(int(len(tm))) ])
 ch62 = np.array([hzp2[i][0] for i in range(int(len(tm))) ])

 co1 = np.array([ox[i][0] for i in range(int(len(tm))) ]) # all time x of 0. particle
 co2 = np.array([oy[i][0] for i in range(int(len(tm))) ])
 co3 = np.array([oz[i][0] for i in range(int(len(tm))) ])
 co4 = np.array([oxp[i][0] for i in range(int(len(tm))) ])
 co5 = np.array([oyp[i][0] for i in range(int(len(tm))) ])
 co6 = np.array([ozp[i][0] for i in range(int(len(tm))) ])

 co11 = np.array([ox1[i][0] for i in range(int(len(tm))) ]) # all time x of 0. particle
 co21 = np.array([oy1[i][0] for i in range(int(len(tm))) ])
 co31 = np.array([oz1[i][0] for i in range(int(len(tm))) ])
 co41 = np.array([oxp1[i][0] for i in range(int(len(tm))) ])
 co51 = np.array([oyp1[i][0] for i in range(int(len(tm))) ])
 co61 = np.array([ozp1[i][0] for i in range(int(len(tm))) ])
 
 co12 = np.array([ox2[i][0] for i in range(int(len(tm))) ]) # all time x of 0. particle
 co22 = np.array([oy2[i][0] for i in range(int(len(tm))) ])
 co32 = np.array([oz2[i][0] for i in range(int(len(tm))) ])
 co42 = np.array([oxp2[i][0] for i in range(int(len(tm))) ])
 co52 = np.array([oyp2[i][0] for i in range(int(len(tm))) ])
 co62 = np.array([ozp2[i][0] for i in range(int(len(tm))) ])


 cd1 = np.array([dx[i][0] for i in range(int(len(tm))) ]) # all time x of 0. particle
 cd2 = np.array([dy[i][0] for i in range(int(len(tm))) ])
 cd3 = np.array([dz[i][0] for i in range(int(len(tm))) ])
 cd4 = np.array([dxp[i][0] for i in range(int(len(tm))) ])
 cd5 = np.array([dyp[i][0] for i in range(int(len(tm))) ])
 cd6 = np.array([dzp[i][0] for i in range(int(len(tm))) ])

 cd11 = np.array([dx1[i][0] for i in range(int(len(tm))) ]) # all time x of 0. particle
 cd21 = np.array([dy1[i][0] for i in range(int(len(tm))) ])
 cd31 = np.array([dz1[i][0] for i in range(int(len(tm))) ])
 cd41 = np.array([dxp1[i][0] for i in range(int(len(tm))) ])
 cd51 = np.array([dyp1[i][0] for i in range(int(len(tm))) ])
 cd61 = np.array([dzp1[i][0] for i in range(int(len(tm))) ])

 cdod1 = np.array([dodx[i][0] for i in range(int(len(tm))) ]) # all time x of 0. particle
 cdod2 = np.array([dody[i][0] for i in range(int(len(tm))) ])
 cdod3 = np.array([dodz[i][0] for i in range(int(len(tm))) ])
 cdod4 = np.array([dodxp[i][0] for i in range(int(len(tm))) ])
 cdod5 = np.array([dodyp[i][0] for i in range(int(len(tm))) ])
 cdod6 = np.array([dodzp[i][0] for i in range(int(len(tm))) ])

 cdod12 = np.array([dodx2[i][0] for i in range(int(len(tm))) ]) # all time x of 0. particle
 cdod22 = np.array([dody2[i][0] for i in range(int(len(tm))) ])
 cdod32 = np.array([dodz2[i][0] for i in range(int(len(tm))) ])
 cdod42 = np.array([dodxp2[i][0] for i in range(int(len(tm))) ])
 cdod52 = np.array([dodyp2[i][0] for i in range(int(len(tm))) ])
 cdod62 = np.array([dodzp2[i][0] for i in range(int(len(tm))) ])

 return c1,c2,c3,c4,c5,c6, \
        c1wc,c2wc,c3wc,c4wc,c5wc,c6wc, \
        ch1,ch2,ch3,ch4,ch5,ch6, \
        ch11,ch21,ch31,ch41,ch51,ch61,\
        ch12,ch22,ch32,ch42,ch52,ch62,\
        co1,co2,co3,co4,co5,co6, \
        co11,co21,co31,co41,co51,co61, \
        co12,co22,co32,co42,co52,co62, \
        cd1,cd2,cd3,cd4,cd5,cd6, \
        cd11,cd21,cd31,cd41,cd51,cd61, \
        cdod1,cdod2,cdod3,cdod4,cdod5,cdod6,\
        cdod12,cdod22,cdod32,cdod42,cdod52,cdod62
 """
""" 
c1,c2,c3,c4,c5,c6,\
c1wc,c2wc,c3wc,c4wc,c5wc,c6wc, \
ch1,ch2,ch3,ch4,ch5,ch6, \
ch11,ch21,ch31,ch41,ch51,ch61, \
ch12,ch22,ch32,ch42,ch52,ch62,\
co1,co2,co3,co4,co5,co6, \
co11,co21,co31,co41,co51,co61, \
co12,co22,co32,co42,co52,co62, \
cd1,cd2,cd3,cd4,cd5,cd6, \
cd11,cd21,cd31,cd41,cd51,cd61, \
cdod1,cdod2,cdod3,cdod4,cdod5,cdod6,\
cdod12,cdod22,cdod32,cdod42,cdod52,cdod62 =particles()
"""
#plt.scatter(oxrt,oyrt)
#plt.xlabel('ox_rt')
#plt.ylabel('oy_rt')
#plt.show()

def rms():
    int1 = [sum(c4[i]*c4[i]) for i in range(int(len(tm)))]
    vxrms = sqrt(int1/Np) 
    int2 = [sum(c5[i]*c5[i]) for i in range(int(len(tm)))]
    vyrms =sqrt(int2/Np) 
    int3 = [sum(quzp[i][0]*quzp[i][0]) for i in range(int(len(tm)))]
    vzrms =sqrt(int3/Np)
    return xrms, yrms, zrms
#xrms, yrms, zrms=rms()

KEq =0.5*mass* (c4**2 + c5**2 +  c6**2)
KEh = 0.5*mass*(ch4**2 + ch5**2 + ch6**2)
KEh1= 0.5*mass*(ch41**2 + ch51**2 + ch61**2)
KEh2= 0.5*mass*(ch42**2 + ch52 + ch62)
KEo = 0.5*mass*(co4**2 + co5**2 +co6**2)
KEd = 0.5*mass*(cd4**2 + cd5**2 + cd6**2)
KEdd= 0.5*mass*(cdod4**2+ cdod5**2 + cdod6**2)
tempq = KEq /(3*kB/2.)
temph = KEh /(3*kB/2.)  
tempo = KEo /(3*kB/2.)
tempd = KEd /(3*kB/2.)
tempdd = KEdd /(3*kB/2.)  

#plot(tm, tempq,label='$T_{q}$')
#plot(tm, temph,label='$T_{h}$')
#plot(tm, tempo,label='$T_{o}$')
#plot(tm, tempd,label='$T_{d}$')
#plot(tm, tempdd,label='$T_{dd}$')
#xlabel(r'$\xi$')
#ylabel(r'Temperature')
#legend()
#savefig('1_tp_temp.png')
#show()

#plot (tm,np.fft.fft(c3),label='q')
#plot (tm,np.fft.fft(ch3),label='h')
#plot (tm,np.fft.fft(co3),label='o')
#plot (tm,np.fft.fft(cd3),label='d')
#xlabel(r'$\xi$')
#ylabel(r' fft-c ')
#legend()
#show()     

#plot (x,Fx)
#xlabel('x ')
#ylabel('Fx')
#show()

xmin = min(c1)


if Np==100:
 plot(c1,c2,label='q')
 plot(ch1,ch2,label='h')
 plot(co1,co2,label='o')
 plot(cd1,cd2,label='d')
 plot(cdod1,cdod2,label='dd')
 xlabel('$x$ ')
 ylabel('$y$')
 legend(loc=0)
 savefig('1_sp_xy_qhoddd.png')
 show()

 plot(c1,c3,label='q')
 plot(ch1,ch3,label='h')
 plot(co1,co3,label='o')
 plot(cd1,cd3,label='d')
 plot(cdod1,cdod3,label='dd')
 xlabel('$x$ ')
 ylabel('$z$')
 legend(loc=2)
 savefig('2_sp_xz_qhoddd.png')
 show()


 plot(tm, KEq, label='q')
 plot(tm, KEdd, label='dd')
 xlabel(r'$\xi$')
 ylabel(r' Kinetic energy')
 legend(loc=1,borderaxespad=0)
 savefig('3_sp_KE.png')
 show()
 
 plot (tm,ch3,label='$f_{3}=0.1$')
 plot (tm,ch31,label='$f_{3}=0.2$')
 plot (tm,ch32,label='$f_{3}=-0.2$')
 xlabel(r'$\xi$ ')
 ylabel(' z ')
 legend(loc=4,borderaxespad=0)
 savefig('4_spz_f3.png')
 show() 

 plot (tm,co3,label='$f_{3},f_{4}=0.1$')
 plot (tm,co31,label='$f_{3},f_{4}=0.2$')
 plot (tm,co32,label='$f_{3},f_{4}=-0.2$')
 xlabel(r'$\xi$ ')
 ylabel(' z ')
 legend(loc=4,borderaxespad=0) # sag alt ? 
 savefig('4_spz_f4.png')
 show() 

 plot (tm,cd3,label='$f_{3},f_{4},f_{5}=0.1$')
 plot (tm,cd31,label='$f_{3},f_{4},f_{5}=-0.1$')
 xlabel(r'$\xi$ ')
 ylabel(' z ')
 legend(loc=4,borderaxespad=0) # sag alt ? 
 savefig('4_spz_f4.png')
 show() 

 plot(tm,cdod3,label='$f_{3},f_{4},f_{5},f_{6}=0.1$')
 plot(tm,cdod32,label='$f_{3},f_{4},f_{5},f_{6}=-0.1$')
 xlabel(r'$\xi$ ')
 ylabel(' z ')
 legend(loc=4,borderaxespad=0)
 savefig('4_spz_f3456.png')
 show() 

 plot (tm,c1,label='q')
 plot (tm,ch1,label='h')
 plot (tm,co1,label='o')
 plot (tm,cd1,label='d')
 plot (tm,cdod1, label='dd')
 xlabel(r'$\xi$')
 ylabel(' $x$ ')
 legend(loc=0,borderaxespad=0) # sag alt
 savefig('5_sp_mx.png')
 show()   
 
 plot (tm,c3,label='q')
 plot (tm,ch3,label='h')
 plot (tm,co3,label='o')
 plot (tm,cd3,label='d')
 plot (tm,cdod3, label='dd')
 xlabel(r'$\xi$ ')
 ylabel(' $z$ ')
 legend(loc=0,borderaxespad=0) # sag alt
 savefig('6_sp_mz.png')
 show()     
    
 
 plot(c1,c4, label='q') ; 
 plot(ch1,ch4, label='h') ; 
 plot(co1,co4, label='o') ; 
 plot(cd1,cd4, label='d') ;
 plot (cdod1, cdod4, label='dd') 
 xlabel('$x $ ')
 ylabel('$\.x $')
 legend(loc=1,borderaxespad=0) #sag ust
 savefig('7_sp_mpsx.png')
 show()
 
 plot(c3,c6, label='q') ; 
 plot(ch3,ch6, label='h') ; 
 plot(co3,co6, label='o') ; 
 plot(cd3,cd6, label='d') ; 
 plot(cdod3,cdod6, label='dd') ; 
 xlabel('$z$')
 ylabel('$\.z$')
 legend(loc=1,borderaxespad=0) # sag ust
 savefig('8_sp_mpsz.png')
 show()

 zmin = min(c3); length_of_array = len(c1)
 zmin_array = [zmin] * length_of_array
 fig = plt.figure()
 ax = fig.add_subplot(111, projection='3d')
 ax.plot(c1,c2,c3,label='q' )
 ax.plot(ch1,ch2,ch3,label='h' )
 ax.plot(co1,co2,co3,label='o' )
 ax.plot(cd1,cd2,cd3,label='d' )
 ax.plot(cdod1,cdod2,cdod3,label='dd')
 ax.plot(c1, c2, zmin_array, zdir='z',linewidth=2)
 ax.plot(co1, co2, zmin_array, zdir='z',linewidth=2)
 ax.plot(ch1, ch2, zmin_array, zdir='z',linewidth=2)
 ax.plot(cd1, cd2, zmin_array, zdir='z',linewidth=2)
 ax.plot(cdod1, cdod2, zmin_array, zdir='z',linewidth=2)
 ax.set_xlabel('$x$')
 ax.set_ylabel('$y$')
 ax.set_zlabel('$z $')
 plt.legend(loc=1,borderaxespad=0) 
 plt.savefig('9_sp_m3dxyz.png')
 plt.show()

 fig = plt.figure()
 ay = fig.add_subplot(111, projection='3d')
 ay.plot(c1,c2,c3,label='q' )
 ay.plot(ch1,ch2,ch3,label='h' )
 ay.plot(co1,co2,co3,label='o' )
 ay.plot(cd1,cd2,cd3,label='d' )
 ay.plot(cdod1,cdod2,cdod3,label='dd')
 ax.set_xlabel('$x$')
 ax.set_ylabel('$y$')
 ax.set_zlabel('$z $')
 plt.legend(loc=1,borderaxespad=0) 
 plt.savefig('10_sp_m3dxyz.png')
 plt.show()


def com():
 for j in range(1,Np): # butun zamanlarda 0 haric diger parcaciklar
   d1=np.array( [sum(qux[i][j]) for i in range(int(len(tm))) ])
   d2=np.array( [sum(quy[i][j]) for i in range(int(len(tm))) ])
   d3=np.array( [sum(quz[i][j]) for i in range(int(len(tm))) ])
   d4=np.array( [sum(quxp[i][j]) for i in range(int(len(tm))) ])
   d5=np.array( [sum(quyp[i][j]) for i in range(int(len(tm))) ])
   d6=np.array( [sum(quzp[i][j]) for i in range(int(len(tm))) ])

   dh1=np.array( [sum(hx[i][j]) for i in range(int(len(tm))) ])
   dh2=np.array( [sum(hy[i][j]) for i in range(int(len(tm))) ])
   dh3=np.array( [sum(hz[i][j]) for i in range(int(len(tm))) ])
   dh4=np.array( [sum(hxp[i][j]) for i in range(int(len(tm))) ])
   dh5=np.array( [sum(hyp[i][j]) for i in range(int(len(tm))) ])
   dh6=np.array( [sum(hzp[i][j]) for i in range(int(len(tm))) ])

   do1=np.array( [sum(ox[i][j]) for i in range(int(len(tm))) ])
   do2=np.array( [sum(oy[i][j]) for i in range(int(len(tm))) ])
   do3=np.array( [sum(oz[i][j]) for i in range(int(len(tm))) ])
   do4=np.array( [sum(oxp[i][j]) for i in range(int(len(tm))) ])
   do5=np.array( [sum(oyp[i][j]) for i in range(int(len(tm))) ])
   do6=np.array( [sum(ozp[i][j]) for i in range(int(len(tm))) ])

   dd1=np.array( [sum(dx[i][j]) for i in range(int(len(tm))) ])
   dd2=np.array( [sum(dy[i][j]) for i in range(int(len(tm))) ])
   dd3=np.array( [sum(dz[i][j]) for i in range(int(len(tm))) ])
   dd4=np.array( [sum(dxp[i][j]) for i in range(int(len(tm))) ])
   dd5=np.array( [sum(dyp[i][j]) for i in range(int(len(tm))) ])
   dd6=np.array( [sum(dzp[i][j]) for i in range(int(len(tm))) ])

   ddod1=np.array( [sum(dx[i][j]) for i in range(int(len(tm))) ])
   ddod2=np.array( [sum(dy[i][j]) for i in range(int(len(tm))) ])
   ddod3=np.array( [sum(dz[i][j]) for i in range(int(len(tm))) ])
   ddod4=np.array( [sum(dxp[i][j]) for i in range(int(len(tm))) ])
   ddod5=np.array( [sum(dyp[i][j]) for i in range(int(len(tm))) ])
   ddod6=np.array( [sum(dzp[i][j]) for i in range(int(len(tm))) ])

 b1q=c1-d1 # difference between cm and test particle
 b2q=c2-d2
 b3q=c3-d3 # x y ve z de 1 particle vs CM distance

 b1h=c1-dh1 # difference between cm and test particle
 b2h=c2-dh2
 b3h=c3-dh3 # x y ve z de 1 particle vs CM distance

 b1o=c1-do1 # difference between cm and test particle
 b2o=c2-do2
 b3o=c3-do3 # x y ve z de 1 particle vs CM distance

 b1d=c1-dd1 # difference between cm and test particle
 b2d=c2-dd2
 b3d=c3-dd3 # x y ve z de 1 particle vs CM distance

 b1dd=c1-ddod1 # difference between cm and test particle
 b2dd=c2-ddod2
 b3dd=c3-ddod3 # x y ve z de 1 particle vs CM distance

 deltarq=np.sqrt(b1q**2 + b2q**2 + b3q**2)
 deltarh=np.sqrt(b1h**2 + b2h**2 + b3h**2)
 deltaro=np.sqrt(b1o**2 + b2o**2 + b3o**2)
 deltard=np.sqrt(b1d**2 + b2d**2 + b3d**2)
 deltardd=np.sqrt(b1dd**2 + b2dd**2 + b3dd**2)

 return d1,d2,d3,d4,d5,d6,b1q,b2q,b3q,b1h,b2h,b3h,\
 b1o,b2o,b3o,b1d,b2d,b3d,b1dd,b2dd,b3dd, \
 deltarq,deltarh,deltaro,deltard,deltardd,\
 dh1,dh2,dh3,dh4,dh5,dh6, \
 do1,do2,do3,do4,do5,do6, \
 dd1,dd2,dd3,dd4,dd5,dd6, \
 ddod1,ddod2,ddod3,ddod4,ddod5,ddod6

#do1rt,do2rt,do3rt,do4rt,do5rt,do6rt, \  
if Np>1:
 d1,d2,d3,d4,d5,d6,b1q,b2q,b3q,b1h,b2h,b3h,\
 b1o,b2o,b3o,b1d,b2d,b3d,b1dd,b2dd,b3dd, \
 deltarq,deltarh,deltaro,deltard,deltardd,\
 dh1,dh2,dh3,dh4,dh5,dh6, \
 do1,do2,do3,do4,do5,do6, \
 dd1,dd2,dd3,dd4,dd5,dd6, \
 ddod1,ddod2,ddod3,ddod4,ddod5,ddod6 =com()

 KEqcom =0.5*mass* (d4**2 + d5**2 +  d6**2)
 KEhcom = 0.5*mass*(dh4**2 + dh5**2 + dh6**2)
 KEocom = 0.5*mass*(do4**2 + do5**2 +do6**2)
 KEdcom = 0.5*mass*(dd4**2 + dd5**2 + dd6**2)
 KEddcom= 0.5*mass*(ddod4**2+ ddod5**2 + ddod6**2)
 tempqcom = KEqcom /(3*kB/2.)  
 temphcom = KEhcom /(3*kB/2.)  
 tempocom = KEocom /(3*kB/2.)
 tempdcom = KEdcom /(3*kB/2.)
 tempddcom = KEddcom /(3*kB/2.)  
 plot(tm, tempqcom,label='$T_{q}$')
 plot(tm, temphcom,label='$T_{h}$')
 plot(tm, tempocom,label='$T_{o}$')
 plot(tm, tempdcom,label='$T_{d}$')
 plot(tm, tempddcom,label='$T_{dd}$')
 xlabel(r'$\xi$')
 ylabel(r'Temperature')
 legend()
 savefig('10_com_temp.png')
 show()

 plot(tm,c1wc,label='w/C')
 plot(tm,c1,label='C')
 xlabel(r'$\xi$')
 ylabel('$x$')
 legend()
 savefig('10_tp_q_wwc.png')
 show()
 
 
 plt.scatter(do1,do2)
 plt.xlabel('x_o ')
 plt.ylabel('y_o')
 plt.show()
 
# plt.scatter(tm,c1/ch1)
# plt.scatter(tm,c1/co1)
# plt.scatter(tm,c1/cd1)
# plt.scatter(tm,c1/cdod1)
# plt.xlabel('tm ')
# plt.ylabel('c1/co1')
# plt.show()

 #plt.scatter(qux,quy)
 #plt.scatter(hx,hy)
 #plt.scatter(ox,oy)
 #plt.scatter(dx,dy)
 #plt.xlabel('x'); plt.ylabel('y')
 #plt.show()
 #plt.scatter( np.fft.fft(c1).real ,np.fft.fft(c3).real  ,c='b')
 #plt.xlabel('c1fftx'); plt.ylabel('c1ffty')
 #plt.show()

# fig1 = plt.figure()
# ag1 = fig1.add_subplot(111, projection='3d')
# ag1.scatter(np.fft.fft(qux).real, np.fft.fft(quy).real, np.fft.fft(quz).real, c='r', marker='o'); 
# ag1.set_xlabel('x [m]')
# ag1.set_ylabel('y [m]')
# ag1.set_zlabel('z [m]')
# plt.savefig('mp_fft_m3dxyz.png')
# show()


# fig2 = plt.figure()
# ag2 = fig2.add_subplot(111, projection='3d')
# ag2.scatter(np.fft.fft(c1).real, np.fft.fft(c2).real, np.fft.fft(c3).real, c='b', marker='o'); 
# ag2.set_xlabel('$x $')
# ag2.set_ylabel('$y $')
# ag2.set_zlabel('$z $')
# plt.savefig('b_fft_q_3dxyz.png')
# show()

#plt.scatter([fftx[10][j] for j in range(Np)],[ffty[10][j] for j in range (Np)],c='b')
#plt.xlabel('fftx[10]'); plt.ylabel('ffty[10]')
#plt.show()

#plt.scatter([fftx[10][j] for j in range(Np)],[fftz[10][j] for j in range (Np)],c='r')
#plt.xlabel('fftx[10]'); plt.ylabel('fftz[10]')
#plt.show()

# plt.scatter( np.fft.fft(qux ).real ,np.fft.fft(quz ).real  ,c='r')
# plt.xlabel('fftx');
# plt.ylabel('fftz')
# plt.show()

 plt.hist2d(hx[314],hy[314],(50,50),cmap=cm.jet)
 plt.xlabel('x');
 plt.ylabel('y')
 plt.colorbar()
 plt.savefig('11_b_hist2d.png')
 plt.show()


# plt.scatter(hx[314],hy[314],hz[314])
# plt.xlabel('x');
# plt.ylabel('y')
# plt.zlabel('z')
# plt.savefig('12_bhexa_3d_scat.png')
# plt.show()
    
# plt.hist2d(qux,quy,bins=40)
# plt.xlabel('qux'); plt.ylabel('quy')
# plt.colorbar()
# plt.show()

# plt.hist2d(hx,hy,bins=40)
# plt.xlabel('hx'); plt.ylabel('hy')
# plt.colorbar()
# plt.show()

 meffx = mass*(Np-1) + mass*(c4/sqrt(d4**2 + d5**2 + d6**2))
 meffy = mass*(Np-1) + mass*(c5/sqrt(d4**2 + d5**2 + d6**2))
 meffz = mass*(Np-1) + mass*(c6/sqrt(d4**2 + d5**2 + d6**2))

 meffxh = mass*(Np-1) + mass*(ch4/sqrt(dh4**2 + dh5**2 + dh6**2))
 meffyh = mass*(Np-1) + mass*(ch5/sqrt(dh4**2 + dh5**2 + dh6**2))
 meffzh = mass*(Np-1) + mass*(ch6/sqrt(dh4**2 + dh5**2 + d6**2))

 meffxo = mass*(Np-1) + mass*(co4/sqrt(do4**2 + do5**2 + do6**2))
 meffyo = mass*(Np-1) + mass*(co5/sqrt(do4**2 + do5**2 + do6**2))
 meffzo = mass*(Np-1) + mass*(co6/sqrt(do4**2 + do5**2 + do6**2))

 meffxd = mass*(Np-1) + mass*(cd4/sqrt(dd4**2 + dd5**2 + dd6**2))
 meffyd = mass*(Np-1) + mass*(cd5/sqrt(dd4**2 + dd5**2 + dd6**2))
 meffzd = mass*(Np-1) + mass*(cd6/sqrt(dd4**2 + dd5**2 + dd6**2))

 meffxdd = mass*(Np-1) + mass*(cdod4/sqrt(ddod4**2 + ddod5**2 + ddod6**2))
 meffydd = mass*(Np-1) + mass*(cd5/sqrt(ddod4**2 + ddod5**2 + ddod6**2))
 meffzdd = mass*(Np-1) + mass*(cd6/sqrt(ddod4**2 + ddod5**2 + ddod6**2))

 plot(tm,deltarq, label='$\Delta r _{q}$')
 plot(tm,deltarh, label='$\Delta r _{h}$')
 plot(tm,deltaro, label='$\Delta r _{o}$')
 plot(tm,deltard, label='$\Delta r _{d}$')
 plot(tm,deltardd,label='$\Delta r _{dd}$') 
 xlabel(r'$\xi$')
 ylabel(r'$\Delta r $')
 legend()
 savefig('12_deltar_ksi.png')
 show()

 
 plot(tm, b1q, label='b1')
 plot(tm, b2q, label='b2')
 plot(tm, b3q, label='b3')
 xlabel(r'$\xi$')
 ylabel(r'distance between CoM and test particle in quad')
 legend(frameon=True)
 savefig('13_b_q_o.png')
 show() 

# plot(tm, b1h, label='b1')
# plot(tm, b2h, label='b2')
# plot(tm, b3h, label='b3')
# xlabel(r'$\xi$')
# ylabel(r'distance between CoM and test particle in hexapole')
# legend(frameon=True)
# savefig('b_meff_o.png')
# show() 

# plot(tm, b1o, label='b1')
# plot(tm, b2o, label='b2')
# plot(tm, b3o, label='b3')
# xlabel(r'$\xi$')
# ylabel(r'distance between CoM and test particle octopole [m]')
# legend(frameon=True)
# savefig('b_meff_o.png')
# show() 

# plot(tm, b1d, label='b1')
# plot(tm, b2d, label='b2')
# plot(tm, b3d, label='b3')
# xlabel(r'$\xi$')
# ylabel(r'distance between CoM and test particle octopole [m]')
# legend(frameon=True)
# savefig('b_meff_d.png')
# show() 
 
 plot(tm,meffx, label='$m_{eff,x}^{q}$')
 plot(tm,meffxh, label='$m_{eff,x}^{h}$')
 plot(tm,meffxo, label='$m_{eff,x}^{o}$')
 plot(tm,meffxd, label='$m_{eff,x}^{d}$')
 plot(tm,meffxdd, label='$m_{eff,x}^{dd}$')
 xlabel(r'$\xi$')
 ylabel(r'$m_{eff,x}$')
 legend(frameon=True,loc=2, borderaxespad=0)
 savefig('14_b_meff_x.png')
 show()   

 plot(tm,meffy, label='$m_{eff,y}^{q}$')
 plot(tm,meffyh, label='$m_{eff,y}^{h}$')
 plot(tm,meffyo, label='$m_{eff,y}^{o}$')
 plot(tm,meffyd, label='$m_{eff,y}^{d}$')
 plot(tm,meffydd, label='$m_{eff,y}^{dd}$')
 xlabel(r'$\xi$')
 ylabel(r'm_{eff,y}')
 legend(loc=2, borderaxespad=0)
 savefig('15_b_meff_y.png')
 show()  

 plot(tm,meffz, label='$m_{eff,z}^{q}$')
 plot(tm,meffzh, label='$m_{eff,z}^{h}$')
 plot(tm,meffzo, label='$m_{eff,z}^{o}$')
 plot(tm,meffzd, label='$m_{eff,z}^{d}$')
 plot(tm,meffzdd, label='$m_{eff,z}^{dd}$')
 xlabel(r'$\xi$')
 ylabel(r' $ m_{eff,z} $ ')
 legend(frameon=True,loc=3, borderaxespad=0)
 savefig('16_b_meff_z.png') #bbox_to_anchor=(0, 0), 
 show()  

 plot(tm,d1, label= r'$x_{CoM}^{q}$',linewidth=1) ;
 plot(tm,dh1, label=r'$x_{CoM}^{h}$',linewidth=1) ;
 plot(tm,do1, label=r'$x_{CoM}^{o}$',linewidth=1) ;
 plot(tm,dd1, label=r'$x_{CoM}^{d}$',linewidth=1) ;
 plot(tm,ddod1, label=r'$x_{CoM}^{dd}$',linewidth=1) ;
# plot(tm,c1, label=r'$x_{q,tp}$',linewidth=1) ;  
 xlabel(r'$\xi$ ')
 ylabel('$x_{CoM}$')
 legend(loc=3, borderaxespad=0)
 savefig('17_b_com_xksi.png')
 show()
 
 plot(tm,d3, label=r'$z_{CoM}^{q}$',linewidth=1) ;
 plot(tm,dh3, label=r'$z_{CoM}^{h}$',linewidth=1) ;
 plot(tm,do3, label=r'$z_{CoM}^{o}$',linewidth=1) ;
 plot(tm,dd3, label=r'$z_{CoM}^{d}$',linewidth=1) ;
 plot(tm,ddod3, label=r'$z_{CoM}^{dd}$',linewidth=1) ;
# plot(tm,c3, label=r'$z_{q,tp}$',linewidth=1) ;  
 xlabel(r'$\xi$')  
 ylabel('$\.z_{CoM}$')
 legend(loc=3, borderaxespad=0)
 savefig('18_b_com_zksi.png')
 show()

 plot(d1,d4, label='$q$')
 plot(dh1,dh4, label='$h$')
 plot(do1,do4, label='$o$')
 plot(dd1,dd4, label='$d$')
 plot(ddod1,ddod4, label='$dd$')
# plot(c1,c4, label='tp')
 xlabel('$ x $')
 ylabel('$ \.x_{CoM}$')
 legend(loc=0)
 savefig('19_b_com_ps_xxp.png')
 show() 

 plot( d3,d6, label= '$q$')
 plot(dh3,dh6, label='$h$')
 plot(do3,do6, label='$o$')
 plot(dd3,dd6, label='$d$')
 plot(ddod3,ddod6, label='$dd$')
# plot(c3,c6, label='tp')
 xlabel('$ z_{CoM}$ ')
 ylabel('$\.z_{CoM}$')
 legend(loc=0)
 savefig('20_b_com_ps_x.png')
 show() 

 plot(d1,d2,label='q')
 plot(dh1,dh2,label='h')
 plot(do1,do2,label='o')
 plot(dd1,dd2,label='d')
 plot(ddod1,ddod2,label='dd')
 xlabel('$x$ ')
 ylabel('$y$')
 legend(loc=0)
 savefig('21_com_xy_qhoddd.png')
 show()

 fig = plt.figure()
 ay = fig.add_subplot(111, projection='3d')
 ay.plot(d1,d2,d3,label='q' )
 ay.plot(dh1,dh2,dh3,label='h' )
 ay.plot(do1,do2,do3,label='o' )
 ay.plot(dd1,dd2,dd3,label='d' )
 ay.plot(ddod1,ddod2,ddod3,label='dd')
 ay.set_xlabel('$ x $')
 ay.set_ylabel('$ y $')
 ay.set_zlabel('$ z $')
 plt.legend(loc=1,borderaxespad=0) 
 plt.savefig('21_b_m3dxyz.png')
 plt.show()
 
