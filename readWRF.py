# Python script to read WRF output files 
# readWRFvariables is the main function. It outputs the relevant wrfout variables
# from ncfile and the microphysics parameterization of interenst
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
# CIMA, UBA-CONICET, Argentina

# Dependencies: 
# conda install -c conda-forge wrf-python

###### PARA BORRAR USAR DE REFE POR AHORA
# lats, lons = wrf.latlon_coords(temp)
     
#------------------------------------------------------------------------------
#ncfile = Dataset("/home/victoria.galligani/Work/Studies/TEPEMAI_01132011/WDM6/wrfout_d01_2011-01-13_22:00:00")
#mp     = 16
#------------------------------------------------------------------------------
import wrf
from wrf import getvar
from netCDF4 import Dataset
import sys 
import numpy as np

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

    if (mp == 16):
        print('WDM6: mp=16')
        
        #qnc = mixr2massconc( getvar(ncfile, "QNCLOUD") , pressure, temp ) 
        qnc = mixr2massconc( np.squeeze(ncfile.variables["QNCLOUD"][0,:,:,:] ), pressure, temp ) 
        qnr = mixr2massconc( np.squeeze(ncfile.variables["QNRAIN" ][0,:,:,:] ), pressure, temp )       

        qr = mixr2massconc( np.squeeze(ncfile.variables["QRAIN"][0,:,:,:]  ), pressure, temp )        
        qs = mixr2massconc( np.squeeze(ncfile.variables["QSNOW"][0,:,:,:]  ), pressure, temp )        
        qi = mixr2massconc( np.squeeze(ncfile.variables["QICE"][0,:,:,:]   ), pressure, temp )        
        qc = mixr2massconc( np.squeeze(ncfile.variables["QCLOUD"][0,:,:,:] ), pressure, temp )       
        qg = mixr2massconc( np.squeeze(ncfile.variables["QGRAUP"][0,:,:,:] ), pressure, temp )     

    elif (mp == 6): 
        print('WSM6: mp=6')
 
        qr = mixr2massconc( np.squeeze(ncfile.variables["QRAIN"][0,:,:,:]   ), pressure, temp )        
        qs = mixr2massconc( np.squeeze(ncfile.variables["QSNOW"][0,:,:,:]   ), pressure, temp )        
        qi = mixr2massconc( np.squeeze(ncfile.variables["QICE"][0,:,:,:]    ), pressure, temp )        
        qc = mixr2massconc( np.squeeze(ncfile.variables["QCLOUD"][0,:,:,:]  ), pressure, temp )       
        qg = mixr2massconc( np.squeeze(ncfile.variables["QGRAUP"][0,:,:,:]  ), pressure, temp )  
        
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
        
        qr =  np.squeeze(ncfile.variables["QRAIN"][0,:,:,:]  )      
        qs =  np.squeeze(ncfile.variables["QSNOW"][0,:,:,:]  )    
        qi =  np.squeeze(ncfile.variables["QICE"][0,:,:,:]   ) 
        qc =  np.squeeze(ncfile.variables["QCLOUD"][0,:,:,:] )   
        qg =  np.squeeze(ncfile.variables["QGRAUP"][0,:,:,:] )   
               
        #qnc = mixr2massconc( getvar(ncfile, "QNCLOUD") , pressure, temp ) 
        qnr = mixr2massconc( qnr , pressure, temp )       
        qns = mixr2massconc( qns , pressure, temp ) 
        qng = mixr2massconc( qng , pressure, temp ) 

        qr = mixr2massconc( qr , pressure, temp )        
        qs = mixr2massconc( qs , pressure, temp )        
        qi = mixr2massconc( qi , pressure, temp )        
        qc = mixr2massconc( qc , pressure, temp )       
        qg = mixr2massconc( qg , pressure, temp )    
        
    
        # Add a max and min tresholds to have zero valued pixels 
        qr.data[np.where(np.logical_or(qr<1E-8,  qnr<1E-8))] = 0.
        qnr.data[np.where(np.logical_or(qr<1E-8, qnr<1E-8))] = 0.

        qs.data[np.where(np.logical_or(qs<1E-8,  qns<1E-8))] = 0.
        qns.data[np.where(np.logical_or(qs<1E-8, qns<1E-8))] = 0.
                 
        qg.data[np.where(np.logical_or(qg<1E-8,  qng<1E-8))] = 0.
        qng.data[np.where(np.logical_or(qg<1E-8, qng<1E-8))] = 0.                 
        #qr.data[np.where(qr<0.1E-7)] = 0.
 
        #qnr.data[np.where(qnr<1E-2)] = 0.
         
    else: 
        print('Selected microphysics parameterization not included')
        sys.exit()


    u   =  wrf.g_wind.get_u_destag(ncfile) #wrf.getvar( ncfile,"U")#ncfile.variables["U"]
    v   =  wrf.g_wind.get_v_destag(ncfile) #wrf.getvar( ncfile,"V")#ncfile.variables["V"] 
    w   =  wrf.g_wind.get_w_destag(ncfile) #wrf.getvar( ncfile,"W")#ncfile.variables["W" ]
    lat =  wrf.getvar( ncfile,"lat")#ncfile.variables["XLONG"]
    lon =  wrf.getvar( ncfile,"lon")#ncfile.variables["XLAT"]

    #    check_u_Darray = xr.DataArray(np.squeeze(u[0,:,:,:]) coords=[, lat, lon], dims=['time', 'lat','lon'])



    geopo_p = wrf.g_geoht.get_height(ncfile) # geopotential height as Mean Sea Level (MSL)
    Re      = 6.3781e6
    z_level = Re*geopo_p/(Re-geopo_p)
    
    
#==============================================================================
#if (mp == 16):  
#    qnr.data[np.where(np.isnan(qr.data)==1)] = 0. 
#   qnc.data[np.where(np.isnan(qc.data)==1)] = 0.    
#    return z_level, lat, lon, u, v, w, qr, qs, qc, qg, qi, qnr, qnc
# elif (mp == 10):
#     qns.data[np.where(np.isnan(qs.data)==1)] = 0.   

    return z_level, lat, lon, u, v, w, qr, qs, qc, qg, qi, qnr, qns, qng
#==============================================================================
            
        







