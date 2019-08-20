#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 11:04:48 2018

@author: vgalligani
"""
# Python script to _________________________
# V. Galligani 
# CIMA, UBA-CONICET, Argentina

# Dependencies: 
# conda install -c conda-forge wrf-python

import getpass
import matplotlib
matplotlib.use('Agg') 
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import scipy
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
from pytmatrix import refractive, tmatrix_aux

ref   = refractive.ice_refractive('IOP_2008_ASCIItable.dat')
const = return_constants()

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
    #if Lambda  > lamdasmax:
    #    Lambda = lamdasmax;
    N0 = Lambda * qn; 
    return [N0/1000., Lambda/1000.]
# GRAU
def get_psd_param_grau(q,qn):
    rho_g = 400.;
    lamdagmax = 6e4; 
    Lambda = (np.pi*rho_g*qn / (q))**(1./3.);
    #if Lambda  > lamdagmax:
    #    Lambda = lamdagmax;
    N0 = Lambda * qn; 
    return [N0/1000., Lambda/1000.]
# ICE
def get_psd_param_ice(q,qn):
    rho_i = 500.;
    Lambda = (np.pi*rho_i*qn / (q))**(1./3.);
    lamdaimax = 1./1E-6; 
    lamdaimin = 1./(2.*(125.E-6)+100E-6)
    N0 = Lambda * qn; 
    #if Lambda < lamdaimin :
    #    Lambda = lamdaimax
    #    N0 = (Lambda**4)*qn/(np.pi*rho_i)
    if Lambda  > lamdaimax:
        Lambda = lamdaimax;
        #N0 = (Lambda**4)*qn/(np.pi*rho_i)
    N0 = Lambda * qn; 

    return [N0/1000., Lambda/1000.]
## CLOUD
def get_psd_param_cloud(q):
    # For cloud droplets it is a function of the droplet number mixing ratio 
    # Nc, based on observations from Martin et al. (1994). This is the 
    # parameter "PGAM" in the code. The implementation of the Martin et al. (1994) 
    # observations is described in the paper Morrison and Grabowski (2007, JAS).
    # N is "NDCNST", and the default setting is 250 cm-3.   
    Nc  = (250*1E6)  #cm-3 TO m-3
    bb  = 1./3.
    rho = 997
                                                                                      
    mu = (0.0005714*Nc) + 0.2714
    mu = 1./(mu*mu) - 1.
    if mu<2:
        mu=2
    elif mu>10:
        mu=10
    Lambda = (np.pi*rho*Nc)/(6*q)*scipy.special.gamma(mu+4.)/scipy.special.gamma(mu+1.)
    Lambda=Lambda**bb
    N0 = (Nc * Lambda**(mu+1.))/(scipy.special.gamma(mu+1.))
    return [N0/1000., Lambda/1000.,mu]
    

# set Exponential particle size distribution (PSD) 
# the PSD form is N(D) = N0 exp(-Lambda*D)
# and return radar variables: 
def get_radar_variables_unnormalizedGamma(N0=None,Lambda=None,mu=None,D_max=None):
    scatterer.psd = psd.UnnormalizedGammaPSD(N0=N0, Lambda=Lambda, mu=mu, D_max=D_max)
    return [radar.refl(scatterer), radar.Zdr(scatterer), radar.ldr(scatterer)] 
            
def get_radar_variables_unnormalizedGamma_fwScattering(N0=None,Lambda=None,mu=None,D_max=None):
    scatterer.psd = psd.UnnormalizedGammaPSD(N0=N0, Lambda=Lambda, mu=mu, D_max=D_max)
    return [radar.Ai(scatterer, h_pol=True), radar.Ai(scatterer, h_pol=False), radar.Kdp(scatterer)] 
                             
    #return [10.*np.log10(radar.refl(scatterer)), 10.*np.log10(radar.Zdr(scatterer)), 
    # radar.Kdp(scatterer), radar.Ai(scatterer)]

# set Exponential particle size distribution (PSD) - for snow, graupel
# the PSD form is N(D) = N0 exp(-Lambda*D)
# and return radar variables: 
def get_radar_variables_Exponential(N0=None,Lambda=None,D_max=None):
    scatterer.psd = psd.ExponentialPSD(N0=N0, Lambda=Lambda, D_max=D_max)
    return [radar.refl(scatterer), radar.Zdr(scatterer), radar.ldr(scatterer)] 

def get_radar_variables_Exponential_fwScattering(N0=None,Lambda=None,D_max=None):
    scatterer.psd = psd.ExponentialPSD(N0=N0, Lambda=Lambda, D_max=D_max)
    return [radar.Ai(scatterer, h_pol=True), radar.Ai(scatterer, h_pol=False), 
            radar.Kdp(scatterer)] 
    
    
def setup_scatterer_snow(elev_radar): 
 
     scatterer                  = Scatterer()
     scatterer.psd_integrator   = psd.PSDIntegrator()   
     scatterer.axis_ratio       = 1./0.6;     
     scatterer.alpha  = 0.0 
     scatterer.beta   = 0.0 
     scatterer.phi0   = 0.0 
     scatterer.thet   = 90.0 - elev_radar[0]
     scatterer.thet0  = 90.0 - elev_radar[0]
     scatterer.phi    = 0.0 
     geom_forw        = scatterer.get_geometry() 
     scatterer.phi    = 180.0 
     geom_back        = scatterer.get_geometry()     
     # set up orientation averaging, Gaussian PDF with mean=0 and std=7 deg
     scatterer.or_pdf = orientation.gaussian_pdf(1)         # orientation PDF according to Bringi and Chandrasekar (2001)
     scatterer.orient = orientation.orient_averaged_fixed   # averaging method
      
     return [scatterer, geom_forw, geom_back]  


def setup_scatterer_rain(elev_radar):
    
    scatterer        = Scatterer()
    scatterer.psd_integrator   = psd.PSDIntegrator() 
    scatterer.psd_integrator.axis_ratio_func = lambda D: 1.0/drop_ar(D)   
    scatterer.alpha  = 0.0 
    scatterer.beta   = 0.0 
    scatterer.phi0   = 0.0 
    scatterer.thet   = 90.0 - elev_radar[0]
    scatterer.thet0  = 90.0 - elev_radar[0]
    scatterer.phi    = 0.0 
    geom_forw        = scatterer.get_geometry() 
    scatterer.phi    = 180.0 
    geom_back        = scatterer.get_geometry()     
    scatterer.or_pdf = orientation.gaussian_pdf(7.0)      # orientation PDF according to Bringi and Chandrasekar (2001)
    scatterer.orient = orientation.orient_averaged_fixed  # averaging method    

    return [scatterer, geom_forw, geom_back]   

def setup_scatterer_grau(elev_radar):
    scatterer                  = Scatterer()
    scatterer.psd_integrator   = psd.PSDIntegrator()    
    scatterer.axis_ratio       = 1.       # 1./0.8 (original); 
    scatterer.alpha  = 0.0 
    scatterer.beta   = 0.0 
    scatterer.phi0   = 0.0 
    scatterer.thet   = 90.0 - elev_radar[0]
    scatterer.thet0  = 90.0 - elev_radar[0]
    scatterer.phi    = 0.0 
    geom_forw        = scatterer.get_geometry() 
    scatterer.phi    = 180.0 
    geom_back        = scatterer.get_geometry()    
    
    # set up orientation averaging, Gaussian PDF with mean=0 and std=7 deg
    scatterer.or_pdf = orientation.gaussian_pdf(1)      # orientation PDF according to Bringi and Chandrasekar (2001)
    scatterer.orient = orientation.orient_averaged_fixed  # averaging method

    return [scatterer, geom_forw, geom_back]   
    

# airmatrix and ice incluions 
airmatrix = 1

# Define some general scatterer parameters 
wavelengths = constants.c/np.array([3e9, 5.5e9, 10e9]) * 1e3  # in mm

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
[q_ice_vec]   = create_logarithmic_scaled_vectors(const['q_ice_min'], const['q_ice_max'], const['dim'])

[qn_rain_vec]  = create_logarithmic_scaled_vectors(const['qn_rain_min'], const['qn_rain_max'], const['dim']) 
[qn_snow_vec]  = create_logarithmic_scaled_vectors(const['qn_snow_min'], const['qn_snow_max'], const['dim']) 
[qn_grau_vec]  = create_logarithmic_scaled_vectors(const['qn_grau_min'], const['qn_grau_max'], const['dim']) 
[qn_ice_vec]   = create_logarithmic_scaled_vectors(const['qn_ice_min'], const['qn_ice_max'], const['dim']) 

dim = const['dim']

#initialize Lambda, N0
N0_rain     = np.zeros([dim,dim])
Lambda_rain = np.zeros([dim,dim])
N0_snow     = np.zeros([dim,dim])
Lambda_snow = np.zeros([dim,dim])
N0_grau     = np.zeros([dim,dim])
Lambda_grau = np.zeros([dim,dim])
N0_clou     = np.zeros([dim])
Lambda_clou = np.zeros([dim])
mu_clou     = np.zeros([dim])
N0_ice      = np.zeros([dim,dim])
Lambda_ice  = np.zeros([dim,dim])

for i in range(dim):
    for j in range(dim):
        [N0_rain[i,j], Lambda_rain[i,j]] = get_psd_param_rain(q_rain_vec[i],qn_rain_vec[j]);
        [N0_snow[i,j], Lambda_snow[i,j]] = get_psd_param_snow(q_snow_vec[i],qn_snow_vec[j]);
        [N0_grau[i,j], Lambda_grau[i,j]] = get_psd_param_grau(q_grau_vec[i],qn_grau_vec[j]); 
        [N0_ice[i,j],  Lambda_ice[i,j]]  = get_psd_param_ice(q_ice_vec[i],qn_ice_vec[j]); 
    [N0_clou[i], Lambda_clou[i], mu_clou[i]] = get_psd_param_cloud(q_clou_vec[i]);


#-------------------------------------------------------------  
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

Zh_RAIN  = np.zeros([3, dim, dim])
Zdr_RAIN = np.zeros([3, dim, dim])
LDR_RAIN = np.zeros([3, dim, dim])
Aih_RAIN = np.zeros([3, dim, dim])
Aiv_RAIN = np.zeros([3, dim, dim])
KDP_RAIN = np.zeros([3, dim, dim])


for ifreq in range(3):
    [scatterer, geom_forw, geom_back]    = setup_scatterer_rain(elev_radar)
    scatterer.wavelength                 = wavelengths[ifreq]
    scatterer.m                          = ref_indices_rain[ifreq]    
    scatterer.psd_integrator.D_max       = 10.  # maximum diameter considered [mm] 10
    scatterer.psd_integrator.geometries  = (geom_forw, geom_back)
    scatterer.psd_integrator.init_scatter_table(scatterer)   
     
    for i in range(dim):
        for j in range(dim):
            [Zh_RAIN[ifreq,i,j], Zdr_RAIN[ifreq,i,j], LDR_RAIN[ifreq,i,j]] = get_radar_variables_unnormalizedGamma(N0_rain[i,j],Lambda_rain[i,j],mu=0.,D_max=10.);

    scatterer.set_geometry(geom_forw)   #geom_horiz_forw = (90.0, 90.0, 0.0, 0.0, 0.0, 0.0) #horiz. forward scatter 
    for i in range(dim):
        for j in range(dim):
            [Aih_RAIN[ifreq,i,j], Aiv_RAIN[ifreq,i,j], KDP_RAIN[ifreq,i,j]] = get_radar_variables_unnormalizedGamma_fwScattering(N0_rain[i,j],Lambda_rain[i,j],mu=0.,D_max=10.);
        
    del scatterer 

#-------------------------------------------------------------
# (3) SCATTERER OBJECT FOR SNOW AND CREATE LOOKUP TABLE 
m_air = [1,1,1];
# inclusion wavelengths = constants.c/np.array([3e9, 5.6e9, 10e9]) * 1e3  # in mm
m_ice = ([ref(tmatrix_aux.wl_S, 0.5), ref(tmatrix_aux.wl_C, 0.5), ref(tmatrix_aux.wl_X, 0.5)]) # For -10K (matrix)
m_ice_pure = ([ref(tmatrix_aux.wl_S, 0.9), ref(tmatrix_aux.wl_C, 0.9), ref(tmatrix_aux.wl_X, 0.9)]) # For -10K (matrix)

snow_density = 0.1; 
if airmatrix == 1:

    f_inclusion  = (snow_density-1)/(0.9167-1)/100;
    f_matrix     = 1-f_inclusion; 
    mix          = [f_matrix, f_inclusion]

    #  If len(m)==2, the first element is taken as the matrix and the second as 
    #  the inclusion. So here: [m_air, m_ice_pure] means air is the matrix 
    #  and ice the inclusion
    ref_indices_snow1 = mg_refractive([m_air[0],m_ice_pure[0]], mix)   
    ref_indices_snow2 = mg_refractive([m_air[1],m_ice_pure[1]], mix)
    ref_indices_snow3 = mg_refractive([m_air[2],m_ice_pure[2]], mix)

else:

    f_inclusion  = (0.9167-1)/(snow_density-1)/100;
    f_matrix    = 1-f_inclusion; 
    mix = [f_matrix, f_inclusion]
    #  If len(m)==2, the first element is taken as the matrix and the second as 
    #  the inclusion. So here: [m_air, m_ice_pure] means air is the matrix 
    #  and ice the inclusion
    ref_indices_snow1 = mg_refractive([m_ice_pure[0],m_air[0]], mix)   
    ref_indices_snow2 = mg_refractive([m_ice_pure[1],m_air[1]], mix)
    ref_indices_snow3 = mg_refractive([m_ice_pure[2],m_air[2]], mix)

ref_indices_snow = [ref_indices_snow1,ref_indices_snow2,ref_indices_snow3]

Zh_SNOW  = np.zeros([3,dim,dim])
Zdr_SNOW = np.zeros([3,dim,dim])
Aih_SNOW = np.zeros([3,dim,dim])
Aiv_SNOW = np.zeros([3,dim,dim])
KDP_SNOW = np.zeros([3,dim,dim])
LDR_SNOW = np.zeros([3,dim,dim])


for ifreq in range(3):
    [scatterer, geom_forw, geom_back]    = setup_scatterer_snow(elev_radar)
    scatterer.wavelength                 = wavelengths[ifreq]
    scatterer.m                          = ref_indices_snow[ifreq]    
    scatterer.psd_integrator.D_max       = 50.  # maximum diameter considered [mm] 50
    scatterer.psd_integrator.geometries  = (geom_forw, geom_back)
    scatterer.psd_integrator.init_scatter_table(scatterer)   
    for i in range(dim):
        for j in range(dim):
            [Zh_SNOW[ifreq,i,j], Zdr_SNOW[ifreq,i,j], LDR_SNOW[ifreq,i,j]] = get_radar_variables_unnormalizedGamma(N0_snow[i,j],Lambda_snow[i,j],mu=0.,D_max=50.);

    scatterer.set_geometry(geom_forw)   #geom_horiz_forw = (90.0, 90.0, 0.0, 0.0, 0.0, 0.0) #horiz. forward scatter 
    for i in range(dim):
        for j in range(dim):
            [Aih_SNOW[ifreq,i,j], Aiv_SNOW[ifreq,i,j], KDP_SNOW[ifreq,i,j]] = get_radar_variables_unnormalizedGamma_fwScattering(N0_snow[i,j],Lambda_snow[i,j],mu=0.,D_max=50.);
        
    del scatterer


# ------------------------------------------------------------
# (4) SCATTERER OBJECT FOR GRAUPEL AND CREATE LOOKUP TABLE 
m_air = [1,1,1];
# inclusion 
#  mix: Volume fractions of the media, len(mix)==len(m)
grau_density = 0.4; 

if airmatrix == 1:

    f_inclusion  = (grau_density-1)/(0.9167-1)/100;
    f_matrix    = 1-f_inclusion; 
    mix = [f_matrix, f_inclusion]
    ref_indices_grau1 = mg_refractive([m_air[0],m_ice_pure[0]], mix)   # HACER MAS ELEGANTE? NO PUDE HACERLO DIRECTAMENTE ... ? 
    ref_indices_grau2 = mg_refractive([m_air[1],m_ice_pure[1]], mix)
    ref_indices_grau3 = mg_refractive([m_air[2],m_ice_pure[2]], mix)

else: 
    f_inclusion  = (0.9167-1)/(grau_density-1)/100;
    f_matrix    = 1-f_inclusion; 
    mix = [f_matrix, f_inclusion]
    ref_indices_grau1 = mg_refractive([m_ice_pure[0], m_air[0]], mix)   # HACER MAS ELEGANTE? NO PUDE HACERLO DIRECTAMENTE ... ? 
    ref_indices_grau2 = mg_refractive([m_ice_pure[1], m_air[1]], mix)
    ref_indices_grau3 = mg_refractive([m_ice_pure[2], m_air[2]], mix)

ref_indices_grau = [ref_indices_grau1, ref_indices_grau2, ref_indices_grau3]

Zh_GRAU  = np.zeros([3,dim,dim])
Zdr_GRAU = np.zeros([3,dim,dim])
Aih_GRAU = np.zeros([3,dim,dim])
Aiv_GRAU = np.zeros([3,dim,dim])
KDP_GRAU = np.zeros([3,dim,dim])
LDR_GRAU = np.zeros([3,dim,dim])




for ifreq in range(3):
    [scatterer, geom_forw, geom_back]    = setup_scatterer_grau(elev_radar)
    scatterer.wavelength                 = wavelengths[ifreq]
    scatterer.m                          = ref_indices_snow[ifreq]    
    scatterer.psd_integrator.D_max       = 50.  # maximum diameter considered [mm] 50
    scatterer.psd_integrator.geometries  = (geom_forw, geom_back)
    scatterer.psd_integrator.init_scatter_table(scatterer)   
    for i in range(dim):
        for j in range(dim):
            [Zh_GRAU[ifreq,i,j], Zdr_GRAU[ifreq,i,j], LDR_GRAU[ifreq,i,j]] = get_radar_variables_unnormalizedGamma(N0_grau[i,j],Lambda_grau[i,j],mu=0.,D_max=50.);

    scatterer.set_geometry(geom_forw)   #geom_horiz_forw = (90.0, 90.0, 0.0, 0.0, 0.0, 0.0) #horiz. forward scatter 
    for i in range(dim):
        for j in range(dim):
            [Aih_GRAU[ifreq,i,j], Aiv_GRAU[ifreq,i,j], KDP_GRAU[ifreq,i,j]] = get_radar_variables_unnormalizedGamma_fwScattering(N0_grau[i,j],Lambda_grau[i,j],mu=0.,D_max=50.);
            
    del scatterer




#Zh_ICE  = np.zeros([dim,dim])
#Zdr_ICE = np.zeros([dim,dim])
#Aih_ICE = np.zeros([dim,dim])
#Aiv_ICE = np.zeros([dim,dim])
#KDP_ICE = np.zeros([dim,dim])
#LDR_ICE = np.zeros([dim,dim])
#
#scatterer        = Scatterer()
#
#scatterer.psd_integrator   = psd.PSDIntegrator() 
#scatterer.wavelength       = wavelengths[1]
#scatterer.m                = m_ice[1]    
#scatterer.axis_ratio = 1./0.6 #1.0/0.6; 
#
#
#scatterer.alpha  = 0.0 
#scatterer.beta   = 0.0 
#scatterer.phi0   = 0.0 
#scatterer.thet   = 90.0 - elev_radar[0]
#scatterer.thet0  = 90.0 - elev_radar[0]
#scatterer.phi    = 0.0 
#geom_forw       = scatterer.get_geometry() 
#scatterer.phi   = 180.0 
#geom_back     = scatterer.get_geometry()    
#
## set up orientation averaging, Gaussian PDF with mean=0 and std=7 deg
#scatterer.or_pdf = orientation.gaussian_pdf(10.0)      # orientation PDF according to Bringi and Chandrasekar (2001)
#scatterer.orient = orientation.orient_averaged_fixed  # averaging method
#
#scatterer.psd_integrator.D_max            = 1.5 #0.8 # maximum diameter considered [mm]
#scatterer.psd_integrator.geometries      = (geom_forw, geom_back)
#   
##scatterer.psd_integrator.num_points = 3000 
##tema D resulto. pero probar con esto ....
##Die ca 10 micrones? 
#
#
#scatterer.psd_integrator.init_scatter_table(scatterer)  
## for rain: mu = 0 for morrison
#for i in range(dim):
#    for j in range(dim):
#        [Zh_ICE[i,j], Zdr_ICE[i,j], LDR_ICE[i,j]] = get_radar_variables_unnormalizedGamma(N0_ice[i,j],Lambda_ice[i,j],mu=0.,D_max=1.5);
#
#scatterer.set_geometry(geom_forw)   #geom_horiz_forw = (90.0, 90.0, 0.0, 0.0, 0.0, 0.0) #horiz. forward scatter 
#for i in range(dim):
#    for j in range(dim):
#        [Aih_ICE[i,j], Aiv_ICE[i,j], KDP_ICE[i,j]] = get_radar_variables_unnormalizedGamma_fwScattering(N0_ice[i,j],Lambda_ice[i,j],mu=0.,D_max=1.5);
#
#del scatterer
#
#Zh_CLOUD  = np.zeros([dim])
#Zdr_CLOUD = np.zeros([dim])
#LDR_CLOUD = np.zeros([dim])
#Aih_CLOUD = np.zeros([dim])
#Aiv_CLOUD = np.zeros([dim])
#KDP_CLOUD = np.zeros([dim])
#
#scatterer                  = Scatterer()
#scatterer.psd_integrator   = psd.PSDIntegrator() 
#scatterer.wavelength       = wavelengths[1]
#scatterer.m                = ref_indices_rain[1]    
#
#scatterer.alpha  = 0.0 
#scatterer.beta   = 0.0 
#scatterer.phi0   = 0.0 
#scatterer.thet   = 90.0 - elev_radar[0]
#scatterer.thet0  = 90.0 - elev_radar[0]
#scatterer.phi    = 0.0 
#geom_forw       = scatterer.get_geometry() 
#scatterer.phi   = 180.0 
#geom_back     = scatterer.get_geometry()    
#
## set up orientation averaging, Gaussian PDF with mean=0 and std=7 deg
## scatterer.or_pdf = orientation.gaussian_pdf(7.0)      # orientation PDF according to Bringi and Chandrasekar (2001)
## scatterer.orient = orientation.orient_averaged_fixed  # averaging method
#
## set up PSD integration   
#scatterer.psd_integrator.D_max            = 0.001  # maximum diameter considered [mm] 10
#scatterer.psd_integrator.geometries      = (tmatrix_aux.geom_horiz_back, tmatrix_aux.geom_horiz_forw)
#scatterer.psd_integrator.geometries      = (geom_forw, geom_back)
#   
#scatterer.psd_integrator.init_scatter_table(scatterer)  
#
## for rain: mu = 0 for morrison
#for i in range(dim):
#    [Zh_CLOUD[i], Zdr_CLOUD[i], LDR_CLOUD[i]] = get_radar_variables_unnormalizedGamma(N0_clou[i],Lambda_clou[i],mu=mu_clou[i],D_max=0.001);
#
#scatterer.set_geometry(geom_forw)   #geom_horiz_forw = (90.0, 90.0, 0.0, 0.0, 0.0, 0.0) #horiz. forward scatter 
#for i in range(dim):
#    [Aih_CLOUD[i], Aiv_CLOUD[i], KDP_CLOUD[i]] = get_radar_variables_unnormalizedGamma_fwScattering(N0_clou[i],Lambda_clou[i],mu=mu_clou[i],D_max=0.001);
#



#-------------------------------------------------------------  
if airmatrix == 1:

    # SAVE LOOKUP TABLE TO THIS FILE 
    if getpass.getuser() == 'vgalligani':
        f = open('/home/victoria.galligani/Work/Studies/WRF_radar_simulator/UNIX_MORR_LOOKUPTABLE_airmatrix_graupelAR1_ALLBANDS.pckl', 'wb')
    elif getpass.getuser() == 'victoria.galligani':
        f = open('/Users/victoria.galligani/Work/Studies/WRF_radar_simulator/MAC_MORR_LOOKUPTABLE_airmatrix_graupelAR1_ALLBANDS.pckl', 'wb')
        
    pickle.dump([Zh_RAIN, Zdr_RAIN, Zh_SNOW, Zdr_SNOW, Zh_GRAU, Zdr_GRAU,               
                 LDR_RAIN, Aih_RAIN, Aiv_RAIN, KDP_RAIN, LDR_SNOW, Aih_SNOW, Aiv_SNOW, KDP_SNOW, 
                 LDR_GRAU, Aih_GRAU, Aiv_GRAU, KDP_GRAU], f) 
    f.close()   
