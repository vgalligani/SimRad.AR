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
from functions import create_logarithmic_scaled_vectors 
from functions import return_constants  

import pickle


const=return_constants()

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
    rho_r = 1000.;
    lamdarmax = 5.E4;
    #rho = 1.2;
    Lambda = (np.pi*rho_r*qn / q)**(1./3.); 
    if Lambda  > lamdarmax:
        Lambda = lamdarmax; 
    N0 = Lambda * qn;
    return [N0/1000., Lambda/1000.]
# SNOW
def get_psd_param_snow(q,qn):
    rho_s = 100.;
    lamdasmax = 1E5; 
    Lambda = (np.pi*rho_s*qn / (q))**(1./3.);
    if Lambda  > lamdasmax:
        Lambda = lamdasmax;
    N0 = Lambda * qn; 
    return [N0/1000., Lambda/1000.]
# GRAU
def get_psd_param_grau(q,qn):
    rho_g = 900.;
    lamdagmax = 6e4; 
    Lambda = (np.pi*rho_g*qn / (q))**(1./3.);
    if Lambda  > lamdagmax:
        Lambda = lamdagmax;
    N0 = Lambda * qn; 
    return [N0/1000., Lambda/1000.]
# CLOUD
#def get_psd_param_cloud(q,qn):
#    rho_c = 1000.;
#    #rho = 0.75;
#    Lambda = (np.pi*rho_c*qn / (3.*q))**(1./3.);
#    lamdacmax = 1e10; 
#    if Lambda  > lamdacmax:
#        Lambda = lamdacmax;  
#    N0 = (Lambda**3.) * qn * 3.;
#    return [N0/1e9, Lambda/1000.] 

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
#def get_radar_variables_cloudPSD(N0=None,Lambda=None,D_max=None): 
#   scatterer.psd = NEWhybridPSD(N0=N0, Lambda=Lambda, D_max=D_max)
#    #return [10.*np.log10(radar.refl(scatterer)), 10.*np.log10(radar.Zdr(scatterer)), radar.Kdp(scatterer), radar.Ai(scatterer)]
#    return [10.*np.log10(radar.refl(scatterer)), 10.*np.log10(radar.Zdr(scatterer))]

# This function creates the logarithmic scaled vectors with 'dim' entries 
# and range '[q_..._min, q_..._max]' 


# INPUT VARIABLES: Min/Max of WRF-Output-Variables, dimension
#const = dict([('q_rain_min', 1e-12),      # input 1
#    ('q_rain_max', 0.013),               # input 2
#    ('qn_rain_min', 1E-8),               # input 3
#    ('qn_rain_max', 2.6e6),              # input 4
#    ('q_snow_min',  1e-8),      # input 5
#    ('q_snow_max',  0.003),     # input 6
#    ('qn_snow_min', 1E-2),      # input 3
#    ('qn_snow_max', 2.6e6),     # input 4   
#    ('q_grau_min',  1e-8),      # input 5
#    ('q_grau_max',  0.008),     # input 6
#    ('qn_grau_min', 1E-2),      # input 
#    ('qn_grau_max', 2.6e6),     # input 4   
#    ('q_clou_min',  1e-10),
#    ('q_clou_max',  0.013),
#    ('dim', 200)])        
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Define some general scatterer parameters 
wavelengths = constants.c/np.array([3e9, 5.6e9, 10e9]) * 1e3  # in mm

names       = ['Reflectivity','Diff. Refl.', 'Spec. Phase', 'Spec. Atten.']
ylab        = ['dBZ','dB','deg/km','deg/km']

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
[qn_snow_vec]  = create_logarithmic_scaled_vectors(const['qn_snow_min'], const['qn_snow_max'], const['dim']) 
[qn_grau_vec]  = create_logarithmic_scaled_vectors(const['qn_grau_min'], const['qn_grau_max'], const['dim']) 

dim = const['dim']


#initialize Lambda, N0
N0_rain = np.zeros([dim,dim])
Lambda_rain = np.zeros([dim,dim])
N0_snow = np.zeros([dim,dim])
Lambda_snow = np.zeros([dim,dim])
N0_grau = np.zeros([dim,dim])
Lambda_grau = np.zeros([dim,dim])
N0_clou = np.zeros([dim,dim])
Lambda_clou = np.zeros([dim,dim])

for i in range(dim):
    for j in range(dim):
        [N0_rain[i,j], Lambda_rain[i,j]] = get_psd_param_rain(q_rain_vec[i],qn_rain_vec[j]);
        [N0_snow[i,j], Lambda_snow[i,j]] = get_psd_param_snow(q_snow_vec[i],qn_snow_vec[j]);
        [N0_grau[i,j], Lambda_grau[i,j]] = get_psd_param_grau(q_grau_vec[i],qn_grau_vec[j]); 
    #[N0_clou[i,j], Lambda_clou[i,j]] = get_psd_param_cloud(q_clou_vec[i],qn_clou_vec[j]);
    
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

elev_radar      = (0.5, 0.9, 1.3, 1.9, 2.3, 3, 3.5, 5, 6.9, 9.1, 11.8, 15.1) # The angles for the radar are:

Zh_RAIN  = np.zeros([dim,dim])
Zdr_RAIN = np.zeros([dim,dim])


# Geometry of forward and backward directions: 
# horiz_back: [90,90,0,180,0,0]
# vert_back: [0,180,0,0,0,0]

    
#for iz in range(len(elev_radar)): 

scatterer        = Scatterer()

scatterer.psd_integrator   = psd.PSDIntegrator() 

# This is from Thurai et al., J. Atmos. Ocean Tech., 24, 2007
scatterer.psd_integrator.axis_ratio_func = lambda D: 1.0/drop_ar(D)

scatterer.wavelength = wavelengths[1]
scatterer.m          = ref_indices_rain[1]    

scatterer.alpha  = 0.0 
scatterer.beta   = 0.0 
scatterer.phi0   = 0.0 
scatterer.thet   = 90.0 - elev_radar[0]
scatterer.thet0  = 90.0 - elev_radar[0]
scatterer.phi    = 0.0 
geom_back       = scatterer.get_geometry() 
scatterer.phi   = 180.0 
geom_forw       = scatterer.get_geometry()     


# set up orientation averaging, Gaussian PDF with mean=0 and std=7 deg
scatterer.or_pdf = orientation.gaussian_pdf(7.0)      # orientation PDF according to Bringi and Chandrasekar (2001)
scatterer.orient = orientation.orient_averaged_fixed  # averaging method

# set up PSD integration   
scatterer.psd_integrator.D_max            = 10.  # maximum diameter considered [mm]
scatterer.psd_integrator.geometries      = (tmatrix_aux.geom_horiz_back, tmatrix_aux.geom_horiz_forw)
scatterer.psd_integrator.geometries      = (geom_forw, geom_back)
   

scatterer.psd_integrator.init_scatter_table(scatterer)  

# for rain: mu = 0 for morrison
for i in range(dim):
    for j in range(dim):
        [Zh_RAIN[i,j], Zdr_RAIN[i,j]] = get_radar_variables_unnormalizedGamma(N0_rain[i,j],Lambda_rain[i,j],mu=0.,D_max=10.);




# (3) SCATTERER OBJECT FOR SNOW AND CREATE LOOKUP TABLE 

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
#  mix: Volume fractions of the media, len(mix)==len(m)
snow_density = 0.1; 
f_inclusion  = (snow_density-1)/(0.9167-1)/100;
f_matrix     = 1-f_inclusion; 
mix          = [f_matrix, f_inclusion]

ref_indices_snow1 = mg_refractive([m_air[0],m_ice[0]], mix)   
ref_indices_snow2 = mg_refractive([m_air[1],m_ice[1]], mix)
ref_indices_snow3 = mg_refractive([m_air[2],m_ice[2]], mix)

ref_indices_snow = [ref_indices_snow1,ref_indices_snow2,ref_indices_snow3]

Zh_SNOW  = np.zeros([dim,dim])
Zdr_SNOW = np.zeros([dim,dim])

# "In terms of aspect-ratio, Straka et al. (2000) report values ranging between 
# 0.6 and 0.8 for dry aggregates and between 0.6 and 0.9 for graupels while 
# Garrett et al. (2015) reports a median aspect-ratio of 0.6 for aggregates and
# a strong mode in graupel aspect-ratios around 0.9. In terms of orientation distributions,
# both Ryzhkov et al. (2011) and Augros et al. (2016) consider a Gaussian distribution 
#with zero mean and a standard deviation of 40 for aggregates and graupels in their simulation

#for iz in range(len(elev_radar)): 

scatterer        = Scatterer()

scatterer.psd_integrator   = psd.PSDIntegrator()   

scatterer.wavelength = wavelengths[1]
scatterer.m          = ref_indices_snow[1]    
scatterer.axis_ratio = 1/0.6; 

scatterer.alpha  = 0.0 
scatterer.beta   = 0.0 
scatterer.phi0   = 0.0 
scatterer.thet   = 90.0 - elev_radar[0]
scatterer.thet0  = 90.0 - elev_radar[0]
scatterer.phi    = 0.0 
geom_back       = scatterer.get_geometry() 
scatterer.phi   = 180.0 
geom_forw       = scatterer.get_geometry()     

print('ok here')

# set up orientation averaging, Gaussian PDF with mean=0 and std=7 deg
scatterer.or_pdf = orientation.gaussian_pdf(40.0)      # orientation PDF according to Bringi and Chandrasekar (2001)
scatterer.orient = orientation.orient_averaged_fixed   # averaging method

# set up PSD integration   
scatterer.psd_integrator.D_max            = 50.  # maximum diameter considered [mm]
scatterer.psd_integrator.geometries      = (geom_forw, geom_back)

scatterer.psd_integrator.init_scatter_table(scatterer)  


# for rain: mu = 0 for morrison
for i in range(dim):
    for j in range(dim):
        [Zh_SNOW[i,j], Zdr_SNOW[i,j]] = get_radar_variables_unnormalizedGamma(N0_snow[i,j],Lambda_snow[i,j],mu=0.,D_max=8.);



# (4) SCATTERER OBJECT FOR GRAUPEL AND CREATE LOOKUP TABLE 

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

grau_density = 0.4; 
f_inclusion  = (grau_density-1)/(0.9167-1)/100;
f_matrix    = 1-f_inclusion; 
mix = [f_matrix, f_inclusion]

ref_indices_grau1 = mg_refractive([m_air[0],m_ice[0]], mix)   # HACER MAS ELEGANTE? NO PUDE HACERLO DIRECTAMENTE ... ? 
ref_indices_grau2 = mg_refractive([m_air[1],m_ice[1]], mix)
ref_indices_grau3 = mg_refractive([m_air[2],m_ice[2]], mix)

ref_indices_grau = [ref_indices_grau1,ref_indices_grau2,ref_indices_grau3]

Zh_GRAU  = np.zeros([dim,dim])
Zdr_GRAU = np.zeros([dim,dim])


#for iz in range(len(elev_radar)): 

scatterer        = Scatterer()

scatterer.psd_integrator   = psd.PSDIntegrator()   

scatterer.wavelength = wavelengths[1]
scatterer.m          = ref_indices_grau[1]    
scatterer.axis_ratio = 1.0/0.6; 

scatterer.alpha  = 0.0 
scatterer.beta   = 0.0 
scatterer.phi0   = 0.0 
scatterer.thet   = 90.0 - elev_radar[0]
scatterer.thet0  = 90.0 - elev_radar[0]
scatterer.phi    = 0.0 
geom_back       = scatterer.get_geometry() 
scatterer.phi   = 180.0 
geom_forw       = scatterer.get_geometry()     

# set up orientation averaging, Gaussian PDF with mean=0 and std=7 deg
scatterer.or_pdf = orientation.gaussian_pdf(40.0)      # orientation PDF according to Bringi and Chandrasekar (2001)
scatterer.orient = orientation.orient_averaged_fixed  # averaging method

# set up PSD integration   
scatterer.psd_integrator.D_max            = 8.  # maximum diameter considered [mm]
scatterer.psd_integrator.geometries      = (geom_forw, geom_back)

scatterer.psd_integrator.init_scatter_table(scatterer)  

# for rain: mu = 0 for morrison
for i in range(dim):
    for j in range(dim):
        [Zh_GRAU[i,j], Zdr_GRAU[i,j]] = get_radar_variables_unnormalizedGamma(N0_grau[i,j],Lambda_grau[i,j],mu=0.,D_max=8.);

#del scatterer 


#==============================================================================
  

# SAVE LOOKUP TABLE TO THIS FILE 
f = open('MORRISON_LOOKUPTABLE.pckl', 'wb')
pickle.dump([Zh_RAIN, Zdr_RAIN, Zh_SNOW, Zdr_SNOW, Zh_GRAU, Zdr_GRAU], f)
f.close()   

