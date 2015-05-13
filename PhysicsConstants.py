import scipy.constants as sc
import numpy as np
#Physical Constants
"""
sc.R      molar gas constant
sc.alpha  fine - structure constant
sc.N_A    Avogadro constant
sc.k      Boltzmann constant
sc.sigma  Stefan-Boltzmann constant
sc.Wien   Wien displacement law constant
sc.Rydberg Rydberg constant
sc.m_e     electron mass
sc.m_p     proton mass
sc.m_n     neutron mass
sc.c       speed of light in vacuum
sc.mu_0    the magnetic constant
sc.epsilon_0  the electric constant (vacuum permittivity),
sc.h        the Planck constant
sc.hbar     the Planck constant divided by 2*pi
sc.G        Newtonian constant of gravitation
sc.g        standard acceleration of gravity
sc.e        elementary charge
"""

#define global variables
q_e_2_m_e = -1.758820150e11 #electron charge-to-mass ratio
q_e_2_m_p = 9.57883392e7    #proton charge-to-mass ratio
m_e = 9.10938215e-31
m_e_2_m_p = 5.4461702177e-4
m_p_2_m_e = 1836.15267247
one_amu = 1.660538782e-27 # atomic mass unit
m_p = 1.672621637e-27
m_p_amu = 1.00727646677
m_4He = 6.64465620e-27
q_e = 1.602176487e-19
c_light = 299792458
mu_0 = 4.*np.pi*1.e-7
N_A = 6.02214179e23
k_B = 1.3806504e-23       # J/K boltzmann constant
E_conv = 931494.028233      #multiply E by this to obtain energy keV
			    #Note: I calculate E with m in amu and c_light=1
			    #Thus, E_conv = c_light**2*one_amu/q_e

amu = 1.6605402E-27          #  mass ion in kg
ke  = 8.987551787E9          #  1/(4*pi*eps0) [N*(m/C)^2]
echarge = 1.60217656535e-19  # [Coulomb]
mm = 1.0e-3
clight = 2.99792458e8                # speed of light
emass = 9.10938291e-31         #kg
hplanck  = 6.62606957e-34      # J s
hbar  = hplanck/(2.0*np.pi)         # J s 
e0 = 8.8541e-12   #C^2/(Nm^2) Permittivity of vacuum
kB = 1.3806503e-23           # J/K
eV = 1.602176565e-19             # [J]
