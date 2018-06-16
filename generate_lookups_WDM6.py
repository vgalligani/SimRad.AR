# Python script to _________________________
# V. Galligani 
# CIMA, UBA-CONICET, Argentina

# Dependencies: 
# conda install -c conda-forge wrf-python

import matplotlib
matplotlib.use('Agg') 
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.special as ssp
import sys
from netCDF4 import num2date, date2num, Dataset
import pyart
import numpy as np
from scipy import constants

from pytmatrix.fortran_tm import pytmatrix
from pytmatrix.tmatrix import Scatterer
from pytmatrix import psd, orientation, radar, tmatrix_aux, refractive
from pytmatrix.refractive import ice_refractive, mg_refractive

from psd_extra import NEWhybridPSD

import pickle

#------------------------------------------------------------------------------
# The following are function definitions for WDM6 (mp=16)
#------------------------------------------------------------------------------

# Function for drop radius dependent ovalization
def drop_ar(D_eq):
    if D_eq < 0.7:
        return 1.0;
    elif D_eq < 1.5:
        return 1.173 - 0.5165*D_eq + 0.4698*D_eq**2 - 0.1317*D_eq**3 - 8.5e-3*D_eq**4
    else:
        return 1.065 - 6.25e-2*D_eq - 3.99e-3*D_eq**2 + 7.66e-4*D_eq**3 - 4.095e-5*D_eq**4
        
# Function to calculate psd values from wrf output
# RAIN
def get_psd_param_rain(q,qn):
    # WDM 6 Double Moment for rain - set up to fit psd.UnnormalizedGammaPSD in scatterer class
    rho_r = 1000.;
    lamdarmax = 8E4;
    #rho = 1.2;
    Lambda = (np.pi*rho_r*ssp.gamma(5.)*qn / (6.*ssp.gamma(2.)*q))**(1./3.);
    if Lambda  > lamdarmax:
        Lambda = lamdarmax; 
    N0 = Lambda**2. * qn;
    return [N0/1000000., Lambda/1000.]
# SNOW
def get_psd_param_snow(q):
    # WDM 6 Double Moment for snow 
    rho_s = 300.;
    n0s = 2e6;
    #rho = 0.6;
    lamdasmax = 1E5; 
    Lambda = (np.pi*rho_s*n0s / (q))**(1./4.);
    if Lambda  > lamdasmax:
        Lambda = lamdasmax;
    N0 = n0s 
    return [N0/1000., Lambda/1000.]
# GRAU
def get_psd_param_grau(q):
    # WDM 6 Double Moment for grau 
    rho_g = 400.;
    n0g = 4e6;
    #rho = 0.8;
    lamdagmax = 6e4; 
    Lambda = (np.pi*rho_g*n0g / (q))**(1./4.);
    if Lambda  > lamdagmax:
        Lambda = lamdagmax;
    N0 = n0g
    return [N0/1000., Lambda/1000.]
# CLOUD
def get_psd_param_cloud(q,qn):
    rho_c = 1000.;
    #rho = 0.75;
    Lambda = (np.pi*rho_c*qn / (3.*q))**(1./3.);
    lamdacmax = 1e10; 
    if Lambda  > lamdacmax:
        Lambda = lamdacmax;  
    N0 = (Lambda**3.) * qn * 3.;
    return [N0/1e9, Lambda/1000.] 

# Function to calculate radar variables from psd values
# set unnormalized Gamma PSD - for rain 
# the PSD form is N(D) = N0 D**mu exp(-Lambda*D)
# and return radar variables: 
def get_radar_variables_unnormalizedGamma(N0=None,Lambda=None,mu=None,D_max=None):
    scatterer.psd = psd.UnnormalizedGammaPSD(N0=N0, Lambda=Lambda, mu=mu, D_max=D_max)
    return [10.*np.log10(radar.refl(scatterer)), 10.*np.log10(radar.Zdr(scatterer))]
    #return [10.*np.log10(radar.refl(scatterer)), 10.*np.log10(radar.Zdr(scatterer)), radar.Kdp(scatterer), radar.Ai(scatterer)]

# set Exponential particle size distribution (PSD) - for snow, graupel
# the PSD form is N(D) = N0 exp(-Lambda*D)
# and return radar variables: 
def get_radar_variables_Exponential(N0=None,Lambda=None,D_max=None):
    scatterer.psd = psd.ExponentialPSD(N0=N0, Lambda=Lambda, D_max=D_max)
    return [10.*np.log10(radar.refl(scatterer)), 10.*np.log10(radar.Zdr(scatterer))]

# Set the PSD for cloud 
# The PSD form is N(D)=N0 * D**2 exp(-(Lambda*D)**3)   where N0 is 3*N*lambda^3 
# and return radar variables:   
def get_radar_variables_cloudPSD(N0=None,Lambda=None,D_max=None): 
    scatterer.psd = NEWhybridPSD(N0=N0, Lambda=Lambda, D_max=D_max)
    #return [10.*np.log10(radar.refl(scatterer)), 10.*np.log10(radar.Zdr(scatterer)), radar.Kdp(scatterer), radar.Ai(scatterer)]
    return [10.*np.log10(radar.refl(scatterer)), 10.*np.log10(radar.Zdr(scatterer))]

# This function creates the logarithmic scaled vectors with 'dim' entries 
# and range '[q_..._min, q_..._max]' 
def create_logarithmic_scaled_vectors(vec_min=None,vec_max=None, dim=None):
    vec = np.logspace(np.log10(vec_min), np.log10(vec_max), num=dim, endpoint=True, base=10.0);
    return [vec]

# INPUT VARIABLES: Min/Max of WRF-Output-Variables, dimension
const = dict([('q_rain_min', 1e-8),      # input 1
    ('q_rain_max', 0.013),     # input 2
    ('qn_rain_min', 100.),      # input 3
    ('qn_rain_max', 2.6e6),     # input 4
    ('q_snow_min',  1e-8),     # input 5
    ('q_snow_max',  0.003),     # input 6
    ('q_grau_min',  1e-8),      # input 5
    ('q_grau_max',  0.008),     # input 6
    ('q_clou_min',  1e-10),
    ('q_clou_max',  0.013), 
    ('qn_clou_min', 100.),
    ('qn_clou_max', 2.0e8), 
    ('dim', 200)])        
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Define some general scatterer parameters 
wavelengths = constants.c/np.array([3e9, 5.6e9, 10e9]) * 1e3  # in mm

names       = ['Reflectivity','Diff. Refl.', 'Spec. Phase', 'Spec. Atten.']
ylab        = ['dBZ','dB','deg/km','deg/km']

theta_radar = (0.5, 0.9, 1.3, 1.9, 2.3, 3, 3.5, 5, 6.9, 9.1, 11.8, 15.1)


# Initialize a scatterer object for RAIN 
ref_indices_rain = [complex(8.983, 0.989), complex(8.590, 1.670), complex(7.718, 2.473)]


# (1) CREATE INPUT LOOKUPS
# logarithmic scaled vectors with 'dim' entries and range '[q_..._min, q_..._max]'
# Created a function to do this: 
# q_rain_vec = np.logspace(np.log10(q_rain_min), np.log10(q_rain_max), num=dim, endpoint=True, base=10.0);
# qn_rain_vec = np.logspace(np.log10(qn_rain_min), np.log10(qn_rain_max), num=dim, endpoint=True, base=10.0);

[q_rain_vec]  = create_logarithmic_scaled_vectors(const['q_rain_min'], const['q_rain_max'], const['dim'])
[q_snow_vec]  = create_logarithmic_scaled_vectors(const['q_snow_min'], const['q_snow_max'], const['dim'])
[q_grau_vec]  = create_logarithmic_scaled_vectors(const['q_grau_min'], const['q_grau_max'], const['dim'])
[q_clou_vec]  = create_logarithmic_scaled_vectors(const['q_clou_min'], const['q_clou_max'], const['dim'])
[qn_rain_vec] = create_logarithmic_scaled_vectors(const['qn_rain_min'], const['qn_rain_max'], const['dim']) 
[qn_clou_vec]  = create_logarithmic_scaled_vectors(const['qn_clou_min'], const['qn_clou_max'], const['dim']) 

dim = const['dim']


#initialize Lambda, N0
N0_rain = np.zeros([dim,dim])
Lambda_rain = np.zeros([dim,dim])
N0_snow = np.zeros([dim])
Lambda_snow = np.zeros([dim])
N0_grau = np.zeros([dim])
Lambda_grau = np.zeros([dim])
N0_clou = np.zeros([dim,dim])
Lambda_clou = np.zeros([dim,dim])

for i in range(dim):
    for j in range(dim):
        [N0_rain[i,j], Lambda_rain[i,j]] = get_psd_param_rain(q_rain_vec[i],qn_rain_vec[j]);
        [N0_clou[i,j], Lambda_clou[i,j]] = get_psd_param_cloud(q_clou_vec[i],qn_clou_vec[j]);
    [N0_snow[i], Lambda_snow[i]] = get_psd_param_snow(q_snow_vec[i])
    [N0_grau[i], Lambda_grau[i]] = get_psd_param_grau(q_grau_vec[i]) 
    
# (2) SCATTERER OBJECT FOR RAIN AND CREATE LOOKUP TABLE 
# This is an object-oriented interface to the Fortran 77 
# T-matrix code. To use it, create an instance of the class, set the properties 
# of the scatterer, and then run one of the commands to retrieve the amplitude 
# and/or phase matrices. So, initialize a scatterer object for RAIN 
# INFO ON GEOMETRY: geom_tuple = (thet0, thet, phi0, phi, alpha, beta) where, 
# The Euler angle alpha of the scatterer orientation (alpha) 
# The Euler angle beta of the scatterer orientation (beta)
# The zenith angle of the incident beam (thet0)
# The zenith angle of the scattered beam (thet)
# The azimuth angle of the incident beam (phi0)
# The azimuth angle of the scattered beam (phi)
# e.g. geom_horiz_back is (90.0, 90.0, 0.0, 180.0, 0.0, 0.0) 

ref_indices_rain = [complex(8.983, 0.989), complex(8.590, 1.670), complex(7.718, 2.473)]
scatterer        = Scatterer()
theta_radar      = (0.5, 0.9, 1.3, 1.9, 2.3, 3, 3.5, 5, 6.9, 9.1, 11.8, 15.1) # The angles for the radar are:
scatterer.alpha  = 0.0 
scatterer.beta   = 0.0 
scatterer.phi0   = 0.0 
scatterer.thet   = 90.0-theta_radar[0]
scatterer.thet0  = 90.0-theta_radar[0]
scatterer.phi    = 180.0 
geom_back        = scatterer.get_geometry() 
scatterer.phi    = 0.0 
geom_forw        = scatterer.get_geometry() 

# so assuming perfect backscattering (no gaussian function) 
# geom_tuple = (theta_radar, theta_radar, 0.0, 180.0, 0.0, 0.0)
# Set geometry to backscattering! 
# CAREFUL: for Kdp need forward scattering. <-------------------------------------------------
scatterer.set_geometry(geom_back) 

# set up orientation averaging, Gaussian PDF with mean=0 and std=7 deg
scatterer.or_pdf = orientation.gaussian_pdf(7.0)      # orientation PDF
scatterer.orient = orientation.orient_averaged_fixed  # averaging method

# set up PSD integration
scatterer.psd_integrator                 = psd.PSDIntegrator()
scatterer.psd_integrator.D_max           = 5.  # maximum diameter considered
scatterer.psd_integrator.geometries      = (geom_forw, geom_back)
#scatterer.psd_integrator.geometries     = (tmatrix_aux.geom_horiz_back, tmatrix_aux.geom_horiz_forw)    # ????????? 
scatterer.psd_integrator.axis_ratio_func = lambda D: 1.0/drop_ar(2.*D)       # This only for rain maybe (?)

scatterer.wavelength = wavelengths[1]
scatterer.m          = ref_indices_rain[1]

# initialize lookup table
scatterer.psd_integrator.init_scatter_table(scatterer)  
 
Zh_RAIN  = np.zeros([dim,dim])
Zdr_RAIN = np.zeros([dim,dim])

# for rain: mu = 1 as: Nr(D) = Lambda**2 qn D exp(-Lambda*D) -> mu = 1, N0 = lambda**2 * qn
for i in range(dim):
    for j in range(dim):
        [Zh_RAIN[i,j], Zdr_RAIN[i,j]] = get_radar_variables_unnormalizedGamma(N0_rain[i,j],Lambda_rain[i,j],mu=1.,D_max=5.);

# (3) SCATTERER OBJECT FOR SNOW AND CREATE LOOKUP TABLE 
del scatterer 

# initialize a scatterer object for SNOW 
# [m] The first element is taken as the matrix and the second as the inclusion. 
# If len(m)>2, the media are mixed recursively so that the last element is used 
# as the inclusion and the second to last as the matrix, then this mixture is used 
# as the last element on the next iteration, and so on. The effective complex refractive 
# index is then returned.
# Use MG air-in-ice (e.g., Eriksson 2015) 
m_air = [1,1,1];

# inclusion 
m_ice = ([complex(1.7831, 0.0001), complex(1.7831, 0.0001), complex(1.7831, 0.0002)]) # For -10K (matrix)

#  mix: Volume fractions of the media, len(mix)==len(m)
f_inclusion = (0.3-1)/(0.9167-1)/100;
f_matrix    = 1-f_inclusion; 
mix = [f_matrix, f_inclusion]

ref_indices_snow1 = mg_refractive([m_air[0],m_ice[0]], mix)   # HACER MAS ELEGANTE? NO PUDE HACERLO DIRECTAMENTE ... ? 
ref_indices_snow2 = mg_refractive([m_air[1],m_ice[1]], mix)
ref_indices_snow3 = mg_refractive([m_air[2],m_ice[2]], mix)

ref_indices_snow = [ref_indices_snow1,ref_indices_snow2,ref_indices_snow3]

scatterer = Scatterer()

scatterer.alpha = 0.0 
scatterer.beta  = 0.0 
scatterer.phi0  = 0.0 
scatterer.thet  = 90.0-theta_radar[0]
scatterer.thet0 = 90.0-theta_radar[0]
scatterer.phi   = 180.0 
geom_forw       = scatterer.get_geometry() 
scatterer.phi   = 0.0 
geom_back       = scatterer.get_geometry() 

# set up orientation averaging, Gaussian PDF with mean=0 and std=7 deg
scatterer.or_pdf = orientation.gaussian_pdf(7.0)      # orientation PDF
scatterer.orient = orientation.orient_averaged_fixed  # averaging method

# set up PSD integration
scatterer.psd_integrator                 = psd.PSDIntegrator()
scatterer.psd_integrator.D_max           = 8.  # maximum diameter considered
scatterer.psd_integrator.geometries      = (geom_back, geom_forw)
#scatterer.psd_integrator.axis_ratio_func = lambda D: 1.0/drop_ar(2*D)       # Delete for snow by now

scatterer.wavelength = wavelengths[1]
scatterer.m          = ref_indices_snow[1]

# initialize lookup table
scatterer.psd_integrator.init_scatter_table(scatterer)  

Zh_SNOW  = np.zeros([dim,])
Zdr_SNOW = np.zeros([dim,])

for i in range(dim):
	[Zh_SNOW[i], Zdr_SNOW[i]] = get_radar_variables_Exponential(N0_snow[i],Lambda_snow[i],D_max=8.);

       
# SAVE VARIABLES
# sio.savemat('Zh_s_wdm6', mdict={'Zh_s': Zh})
# sio.savemat('Zdr_s_wdm6', mdict={'Zdr_s': Zdr})

# (4) SCATTERER OBJECT FOR GRAUPEL AND CREATE LOOKUP TABLE 
del scatterer 

# initialize a scatterer object for GRAUPEL 
# [m] The first element is taken as the matrix and the second as the inclusion. 
# If len(m)>2, the media are mixed recursively so that the last element is used 
# as the inclusion and the second to last as the matrix, then this mixture is used 
# as the last element on the next iteration, and so on. The effective complex refractive 
# index is then returned.
# Use MG air-in-ice (e.g., Eriksson 2015) 
m_air = [1,1,1];
# inclusion 
m_ice = ([complex(1.7831, 0.0001), complex(1.7831, 0.0001), complex(1.7831, 0.0002)]) # For -10K (matrix)
#  mix: Volume fractions of the media, len(mix)==len(m)
f_inclusion = (0.4-1)/(0.9167-1)/100;
f_matrix    = 1-f_inclusion; 
mix = [f_matrix, f_inclusion]

ref_indices_grau1 = mg_refractive([m_air[0],m_ice[0]], mix)   # HACER MAS ELEGANTE? NO PUDE HACERLO DIRECTAMENTE ... ? 
ref_indices_grau2 = mg_refractive([m_air[1],m_ice[1]], mix)
ref_indices_grau3 = mg_refractive([m_air[2],m_ice[2]], mix)

ref_indices_grau = [ref_indices_grau1,ref_indices_grau2,ref_indices_grau3]

# t-matrix in pytmatrix 
scatterer       = Scatterer()
scatterer.alpha = 0.0 
scatterer.beta  = 0.0 
scatterer.phi0  = 0.0 
scatterer.thet  = 90.0-theta_radar[0]
scatterer.thet0 = 90.0-theta_radar[0]
scatterer.phi   = 180.0 
geom_forw       = scatterer.get_geometry() 
scatterer.phi   = 0.0 
geom_back       = scatterer.get_geometry() 

# set up orientation averaging, Gaussian PDF with mean=0 and std=7 deg
scatterer.or_pdf = orientation.gaussian_pdf(7.0)      # orientation PDF
scatterer.orient = orientation.orient_averaged_fixed  # averaging method

# set up PSD integration
scatterer.psd_integrator                 = psd.PSDIntegrator()
scatterer.psd_integrator.D_max           = 8.  # maximum diameter considered
scatterer.psd_integrator.geometries      = (geom_back, geom_forw)
#scatterer.psd_integrator.axis_ratio_func = lambda D: 1.0/drop_ar(2*D)       # Delete for snow by now

scatterer.wavelength = wavelengths[1]
scatterer.m          = ref_indices_grau[1]

# initialize lookup table
scatterer.psd_integrator.init_scatter_table(scatterer)  

Zh_GRAU  = np.zeros([dim,])
Zdr_GRAU = np.zeros([dim,])

for i in range(dim):
	[Zh_GRAU[i], Zdr_GRAU[i]] = get_radar_variables_Exponential(N0_grau[i],Lambda_grau[i],D_max=8.);
          
# SAVE VARIABLES
# sio.savemat('Zh_g_wdm6', mdict={'Zh_g': Zh})
# sio.savemat('Zdr_g_wdm6', mdict={'Zdr_g': Zdr}) 
  
# (5) SCATTERER OBJECT FOR CLOUD AND CREATE LOOKUP TABLE 
del scatterer 

# initialize a scatterer object for CLOUD 
ref_indices_rain = [complex(8.983, 0.989), complex(8.590, 1.670), complex(7.718, 2.473)]

# t-matrix in pytmatrix
scatterer = Scatterer()

scatterer.alpha = 0.0 
scatterer.beta  = 0.0 
scatterer.phi0  = 0.0 
scatterer.thet  = 90.0-theta_radar[0]
scatterer.thet0 = 90.0-theta_radar[0]
scatterer.phi   = 180.0 
geom_forw       = scatterer.get_geometry() 
scatterer.phi   = 0.0 
geom_back       = scatterer.get_geometry() 

# set up orientation averaging, Gaussian PDF with mean=0 and std=7 deg
#scatterer.or_pdf = orientation.gaussian_pdf(7.0)      # orientation PDF
scatterer.orient = orientation.orient_averaged_fixed  # averaging method

# set up PSD integration
scatterer.psd_integrator                 = psd.PSDIntegrator()
scatterer.psd_integrator.D_max           = 0.05  # maximum diameter considered
scatterer.psd_integrator.geometries      = (geom_back, geom_forw)
#scatterer.psd_integrator.axis_ratio_func = lambda D: 1.0/drop_ar(2*D)       # This only for rain maybe (?)

# Set geometry to backscattering! 
# CAREFUL: for Kdp need forward scattering <-------------------------------------- 
scatterer.set_geometry(geom_back) 

scatterer.wavelength = wavelengths[1]
scatterer.m          = ref_indices_rain[1]

# initialize lookup table
scatterer.psd_integrator.init_scatter_table(scatterer)     
  
Zh_CLOUD = np.zeros([dim,dim])
Zdr_CLOUD = np.zeros([dim,dim])

for i in range(dim):
	for j in range(dim):
		[Zh_CLOUD[i,j], Zdr_CLOUD[i,j]] = get_radar_variables_cloudPSD(3.*N0_clou[i,j]*Lambda_clou[i,j],Lambda_clou[i,j],D_max=0.05);
 
      
# SAVE VARIABLES
# sio.savemat('Zh_c_wdm6', mdict={'Zh_c': Zh})
# sio.savemat('Zdr_c_wdm6', mdict={'Zdr_c': Zdr})  
  

# SAVE LOOKUP TABLE TO THIS FILE 
f = open('WDM6_LOOKUPTABLE.pckl', 'wb')
pickle.dump([Zh_RAIN, Zdr_RAIN, Zh_SNOW, Zdr_SNOW, Zh_GRAU, Zdr_GRAU, Zh_CLOUD, Zdr_CLOUD], f)
f.close()   