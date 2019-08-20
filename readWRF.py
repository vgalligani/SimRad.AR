# Read WRF output files 
# readWRFvariables is the main function. It outputs the relevant wrfout variables
# from the wrf ncfile and the microphysics parameterization of interest
# IN:   ncfile: the wrfout file 
#       mp: the WRF microphysics parameterization e.g., WDM6 is mp=16      
# OUT:  WRF variables of interest Qx and Nx
#       z_level:
#       lat:
#       lon:
#       u:
#       v:
#       w:
#       qr:
#       qs:
#       qc:
#       qg:
#       qi:
#       qnr:
#       qnc:
# 
# V. Galligani 
# CIMA, UBA-CONICET
#
# Dependencies: 
# conda install -c conda-forge wrf-python

import wrf
from wrf import getvar
from netCDF4 import Dataset
import sys 
import numpy as np
from scipy import integrate

def airdensity(p,t): 
    """ Function to calculate the air density by trivial
    application of the ideal gas law
    """
    Re  = 287.04    
    rho = p / ( Re * t )       
    return rho


def mixr2massconc(mixr,  pres, temp):
    """ Function to calculate the mass cocentration
    from mass mixing ratio assuming a mixture of an 
    atmospheric species and dry air. Output in kg/m3 
    """
    Re  = 287.04    
    rho = airdensity( pres, temp )          
    massconc = rho * mixr               
    return massconc

def readWRFvariables(strfile, mp):
    ncfile = Dataset(strfile,'r')
    pressure = wrf.g_pressure.get_pressure(ncfile)
    temp     = wrf.g_temp.get_tk(ncfile)
    itime    = 0
    u        =  wrf.g_wind.get_u_destag(ncfile)  
    v        =  wrf.g_wind.get_v_destag(ncfile)  
    w        =  wrf.g_wind.get_w_destag(ncfile)  
    lat      =  wrf.getvar( ncfile,"lat") 
    lon      =  wrf.getvar( ncfile,"lon")
    geopo_p  = wrf.g_geoht.get_height(ncfile) # geopotential height as Mean Sea Level (MSL)
    Re       = 6.3781e6
    z_level  = Re*geopo_p/(Re-geopo_p)
    
    if (mp == 16):
        print('WDM6: mp=16')
        
        #qnc = mixr2massconc( getvar(ncfile, "QNCLOUD") , pressure, temp ) 
        qnc = mixr2massconc( np.squeeze(ncfile.variables["QNCLOUD"][itime,:,:,:] ), pressure, temp ) 
        qnr = mixr2massconc( np.squeeze(ncfile.variables["QNRAIN" ][itime,:,:,:] ), pressure, temp )       

        qr = mixr2massconc( np.squeeze(ncfile.variables["QRAIN"][itime,:,:,:]  ), pressure, temp )        
        qs = mixr2massconc( np.squeeze(ncfile.variables["QSNOW"][itime,:,:,:]  ), pressure, temp )        
        qi = mixr2massconc( np.squeeze(ncfile.variables["QICE"][itime,:,:,:]   ), pressure, temp )        
        qc = mixr2massconc( np.squeeze(ncfile.variables["QCLOUD"][itime,:,:,:] ), pressure, temp )       
        qg = mixr2massconc( np.squeeze(ncfile.variables["QGRAUP"][itime,:,:,:] ), pressure, temp )     

    elif (mp == 6): 
        print('WSM6: mp=6')
 
        qr = mixr2massconc( np.squeeze(ncfile.variables["QRAIN"][itime,:,:,:]   ), pressure, temp )        
        qs = mixr2massconc( np.squeeze(ncfile.variables["QSNOW"][itime,:,:,:]   ), pressure, temp )        
        qi = mixr2massconc( np.squeeze(ncfile.variables["QICE"][itime,:,:,:]    ), pressure, temp )        
        qc = mixr2massconc( np.squeeze(ncfile.variables["QCLOUD"][itime,:,:,:]  ), pressure, temp )       
        qg = mixr2massconc( np.squeeze(ncfile.variables["QGRAUP"][itime,:,:,:]  ), pressure, temp )  

        thrs       = 1E-8; 
        qr_clipped = qr.data.clip(min=thrs); qr_clipped[qr_clipped==thrs]=np.nan
        qs_clipped = qs.data.clip(min=thrs); qs_clipped[qs_clipped==thrs]=np.nan
        qi_clipped = qi.data.clip(min=thrs); qi_clipped[qi_clipped==thrs]=np.nan
        qg_clipped = qg.data.clip(min=thrs); qg_clipped[qg_clipped==thrs]=np.nan
        qc_clipped = qc.data.clip(min=thrs); qc_clipped[qc_clipped==thrs]=np.nan
        qr_int     = integrate.trapz(qr.data, z_level.data, axis=0)
        qs_int     = integrate.trapz(qs.data, z_level.data, axis=0)
        qg_int     = integrate.trapz(qg.data, z_level.data, axis=0)
        qi_int     = integrate.trapz(qi.data, z_level.data, axis=0)
        qc_int     = integrate.trapz(qc.data, z_level.data, axis=0)        
        qtotal_int = qr_int+qs_int+qg_int+qi_int+qc_int
        
    elif (mp == 2):
        print('Lin: mp=2')
            
        qr = mixr2massconc( np.squeeze(ncfile.variables["QRAIN"][0,:,:,:]   ), pressure, temp )        
        qs = mixr2massconc( np.squeeze(ncfile.variables["QSNOW"][0,:,:,:]   ), pressure, temp )        
        qc = mixr2massconc( np.squeeze(ncfile.variables["QCLOUD"][0,:,:,:]  ), pressure, temp )       
        qh = mixr2massconc( np.squeeze(ncfile.variables["QHAIL"][0,:,:,:]   ), pressure, temp )  
        
    elif (mp == 10):
        print('MORR: mp=10')
        
        #qnc = mixr2massconc( getvar(ncfile, "QNCLOUD") , pressure, temp ) 
        qnr =  np.squeeze(ncfile.variables["QNRAIN" ][0,:,:,:] )      
        qns =  np.squeeze(ncfile.variables["QNSNOW"][0,:,:,:] )
        qng =  np.squeeze(ncfile.variables["QNGRAUPEL"][0,:,:,:] )
        qni =  np.squeeze(ncfile.variables["QNICE"][0,:,:,:] )
        
        qr =  np.squeeze(ncfile.variables["QRAIN"][0,:,:,:]  )      
        qs =  np.squeeze(ncfile.variables["QSNOW"][0,:,:,:]  )    
        qi =  np.squeeze(ncfile.variables["QICE"][0,:,:,:]   ) 
        qc =  np.squeeze(ncfile.variables["QCLOUD"][0,:,:,:] )   
        qg =  np.squeeze(ncfile.variables["QGRAUP"][0,:,:,:] )   


        qi[np.where(qi<1E-8)] = 0.
        qni[np.where(qi<1E-8)] = 0.            #0.8E-9    
        qr[np.where(qr<1E-8)] = 0.
        qnr[np.where(qr<1E-8)] = 0.            #0.8E-9    
        qs[np.where(qs<1E-8)] = 0.
        qns[np.where(qs<1E-8)] = 0.            #0.8E-9    
        qg[np.where(qg<1E-8)] = 0.
        qng[np.where(qg<1E-8)] = 0.            #0.8E-9            
        
        #qnc = mixr2massconc( getvar(ncfile, "QNCLOUD") , pressure, temp ) 
        qnr = mixr2massconc( qnr , pressure, temp )       
        qns = mixr2massconc( qns , pressure, temp ) 
        qng = mixr2massconc( qng , pressure, temp ) 
        qni = mixr2massconc( qni , pressure, temp ) 

        qr = mixr2massconc( qr , pressure, temp )        
        qs = mixr2massconc( qs , pressure, temp )        
        qi = mixr2massconc( qi , pressure, temp )        
        qc = mixr2massconc( qc , pressure, temp )       
        qg = mixr2massconc( qg , pressure, temp )    
        
    else: 
        print('Selected microphysics parameterization not included')
        sys.exit()
         
    if (mp == 6): 
        return z_level, lat, lon, u, v, w, qr_clipped, qs_clipped, qc_clipped, qg_clipped, qi_clipped, qtotal_int
    elif (mp == 10): 
        return z_level, lat, lon, u, v, w, qr, qs, qc, qg, qi, qnr, qns, qng, qni
#==============================================================================
            
        















