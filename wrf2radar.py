#-----------------------------------------------------------------------------
# Python script
# V. Galligani <victoria.galligani@cima.fcen.uba.ar>
# CIMA, UBA-CONICET
#-----------------------------------------------------------------------------

import pickle
import readWRF
import numpy as np
import scipy.interpolate as intp
import regionmask
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from pyart.core import transforms
import time
import scipy.interpolate as intp
import functions as fun 
import getpass
import sys

#-----------------------------------------------------------------------------
# Definition of internal functions
def sum_nan_arrays(a,b):
    ma = np.isnan(a)
    mb = np.isnan(b)
    return np.where(ma&mb, np.nan, np.where(ma,0,a) + np.where(mb,0,b))

def sum_total(a, b, c):
        a_b       = sum_nan_arrays( a, b)
        a_b_c_d   = sum_nan_arrays( a_b, c)
        total_out = a_b_c_d 
        del a_b, a_b_c_d
        return total_out

def find_nearest(array,THEvalue):
    idx    = int( (np.abs( (array.ravel()) - THEvalue)).argmin() )
    output =  array[idx]
    if np.isnan(output):
        idx = np.nan
    return idx, output

def find_indx_regular(indices, lut_q, lut_qn, qwrf, qnwrf, Zh, Zdr):     
    # Here hard coded the number of elevation angles in my lookup table!     
    qdelta = np.abs(np.log10(lut_q[100]) - np.log10(lut_q[99]))
    qndelta = np.abs(np.log10(lut_qn[100]) - np.log10(lut_qn[99]))     
    
    Zh_out  = np.zeros([np.shape(qwrf)[0],np.shape(qwrf)[1],np.shape(qwrf)[2]]); Zh_out[:]=np.nan
    Zdr_out = np.zeros([np.shape(qwrf)[0],np.shape(qwrf)[1],np.shape(qwrf)[2]]); Zdr_out[:]=np.nan
    
    for j in np.arange(np.shape(indices)[1]):
        indq  =  int( np.abs( np.log10(qwrf[indices[0][j],indices[1][j],indices[2][j]])  - np.log10(lut_q[0]))/ qdelta )
        indn =   int( np.abs( np.log10(qnwrf[indices[0][j],indices[1][j],indices[2][j]]) - np.log10(lut_qn[0])) / qndelta )
        #for jj in np.arange(12): 
        Zh_out[indices[0][j],indices[1][j],indices[2][j]]   = Zh[indq,indn];
        Zdr_out[indices[0][j],indices[1][j],indices[2][j]]  = Zdr[indq,indn];

    return Zh_out, Zdr_out


#-----------------------------------------------------------------------------

def main_wrf2radar(radar_site, mp, max_range, mode, ncfile, ftable):
    
    const      = fun.return_constants()
    start_time = time.time()      
    airmatrix  = 1      
                    
    #------------------------------------------------------------------------------
    # Radar de  config: 
    # Initial radar elevation angles: 
    theta_radar      = (0.5, 0.9, 1.3, 1.9, 2.3, 3, 3.5, 5, 6.9, 9.1, 11.8, 15.1)                                       
    radar_antenna_bw = 1.0;  
    #-----------------------------------------------------------------------------
    
    if (radar_site == 1):                       # AR7 (Parana)  
        radar_lon     = -60.539722 
        radar_lat     = -31.858334
    elif (radar_site == 0):                     # Anguil
        radar_lon     = -64.0103
        radar_lat     = -36.5257 
    
    elif (radar_site == 2):                     # RMA5 (Bernardo de Irigoyen)
        radar_lon     = -26.278056
        radar_lat     = -53.670833
    
    elif (radar_site == 3):                     # RMA1 (CORDOBA)
        radar_lon     = -31.441389
        radar_lat     = -64.191944
    
    elif (radar_site == 4):                     # RMA10 (ESPORA)
        radar_lon     = -38,734444
        radar_lat     = -62,163611
    
    elif (radar_site == 5):                     # RMA2 (EZEIZA)
        radar_lon     = -34.787778
        radar_lat     = -58.536667
    
    elif (radar_site == 6):                     # RMA3 (LAS LOMITAS)
        radar_lon     = -24.730556
        radar_lat     = -60.551667 
    
    elif (radar_site == 7):                     # RMA6 (MAR DEL PLATA)
        radar_lon     = -37.913056
        radar_lat     = -57.527778
    
    elif (radar_site == 8):                     # RMA8 (MERCEDES)
        radar_lon     = -29.195833
        radar_lat     = -58.044722
    
    elif (radar_site == 9):                     # RMA7 (NEUQUEN)
        radar_lon     = -38.876667
        radar_lat     = -68.145278
    
    elif (radar_site == 10):                     # AR5 (PERGAMINO)
        radar_lon     = -33.946111
        radar_lat     = -60.5625     
        
    elif (radar_site == 11):                     # RMA4 (RESISTENCIA)
        radar_lon     = -27.451944
        radar_lat     = -59.051111   
        
    elif (radar_site == 12):                     # RMA9 (RIO GRANDE)
        radar_lon     = -53.785
        radar_lat     = -67.76    
        
    elif (radar_site == 13):                     # RMA11 (TERMAS RIO HONDO)
        radar_lon     = np.nan; 
        radar_lat     = np.nan; 
        
    elif (radar_site == 14):                     # RMA0 (BARILOCHE)
        radar_lon     = -41.139722
        radar_lat     = -71.149722  
    
    radar_window = [[radar_lon-5, radar_lat-5], [radar_lon-5, radar_lat+5], [radar_lon+5, radar_lat+5], [radar_lon+5, radar_lat-5]]
    radar_mask   = regionmask.Regions_cls('RADARES', [0], ['radar_site'], ['site'], [radar_window])
         
    #-----------------------------------------------------------------------------
    # READ WRFOUT and LOOKUP TABLES 
                 
    if (mp == 16):

        print('update !')
#        # Load lookup tables
#        f = open('WDM6_LOOKUPTABLE.pckl', 'rb')
#        Zh_RAIN, Zdr_RAIN, Zh_SNOW, Zdr_SNOW, Zh_GRAU, Zdr_GRAU, Zh_CLOUD, Zdr_CLOUD = pickle.load(f)
#        f.close() 
#        
#        from generate_lookups_WDM6 import const, create_logarithmic_scaled_vectors 
#        
#        [q_rain_vec]  = fun.create_logarithmic_scaled_vectors(const['q_rain_min'], const['q_rain_max'], const['dim'])
#        [q_snow_vec]  = fun.create_logarithmic_scaled_vectors(const['q_snow_min'], const['q_snow_max'], const['dim'])
#        [q_grau_vec]  = fun.create_logarithmic_scaled_vectors(const['q_grau_min'], const['q_grau_max'], const['dim'])
#        [q_clou_vec]  = fun.create_logarithmic_scaled_vectors(const['q_clou_min'], const['q_clou_max'], const['dim'])
#        [qn_rain_vec] = fun.create_logarithmic_scaled_vectors(const['qn_rain_min'], const['qn_rain_max'], const['dim']) 
#        [qn_clou_vec] = fun.create_logarithmic_scaled_vectors(const['qn_clou_min'], const['qn_clou_max'], const['dim']) 
#        q_rain_grid, qn_rain_grid = np.meshgrid(q_rain_vec, qn_rain_vec) 
#        q_clou_grid, qn_clou_grid = np.meshgrid(q_clou_vec, qn_clou_vec) 
#    
#        [z_level, lat, lon, u, v, w, qr, qs, qc, qg, qi, qnr, qnc] = readWRF.readWRFvariables(ncfile, mp)
#    
#        ncfile2 = "/home/victoria.galligani/Work/Studies/TEPEMAI_01132011/WDM6/wrfout_d01_2011-01-13_22:00:00"
#        [z_level2, lat2, lon2, u2, v2, w2, qr2, qs2, qc2, qg2, qi2, qnr2, qnc2] = readWRF.readWRFvariables(ncfile2, 16)
#    
#        themask       = radar_mask.mask(lon,lat)                          # Mask has nans and zeros 
#        dstacked      = np.rollaxis(np.stack([themask] * qs.shape[0]),0)  # Mask has nans and zeros 
#    
#        qnr_restrict = np.ma.masked_array(qnr, dstacked)
#        qnc_restrict = np.ma.masked_array(qnc, dstacked)
        
    elif (mp == 10):
        
        apply_mask = 1; 

        f = open(ncfile, 'rb')
            
        [Zh_RAIN, Zdr_RAIN, Zh_SNOW, Zdr_SNOW, Zh_GRAU, Zdr_GRAU,               
                 LDR_RAIN, Aih_RAIN, Aiv_RAIN, KDP_RAIN, LDR_SNOW, Aih_SNOW, Aiv_SNOW, KDP_SNOW, 
                 LDR_GRAU, Aih_GRAU, Aiv_GRAU, KDP_GRAU] = pickle.load(ftable)
        f.close()   
        
        [z_level, lat, lon, u, v, w, qr, qs, qc, qg, qi, qnr, qns, qng, qni] = readWRF.readWRFvariables(ncfile, mp)
        
        [q_rain_vec]  = fun.create_logarithmic_scaled_vectors(const['q_rain_min'],  const['q_rain_max'],  const['dim'])
        [q_snow_vec]  = fun.create_logarithmic_scaled_vectors(const['q_snow_min'],  const['q_snow_max'],  const['dim'])
        [q_grau_vec]  = fun.create_logarithmic_scaled_vectors(const['q_grau_min'],  const['q_grau_max'],  const['dim'])
        [q_clou_vec]  = fun.create_logarithmic_scaled_vectors(const['q_clou_min'],  const['q_clou_max'],  const['dim'])
        [q_ice_vec]   = fun.create_logarithmic_scaled_vectors(const['q_ice_min'],   const['q_ice_max'],   const['dim'])
        [qn_rain_vec] = fun.create_logarithmic_scaled_vectors(const['qn_rain_min'], const['qn_rain_max'], const['dim']) 
        [qn_snow_vec] = fun.create_logarithmic_scaled_vectors(const['qn_snow_min'], const['qn_snow_max'], const['dim']) 
        [qn_grau_vec] = fun.create_logarithmic_scaled_vectors(const['qn_grau_min'], const['qn_grau_max'], const['dim']) 
        [qn_ice_vec]  = fun.create_logarithmic_scaled_vectors(const['qn_ice_min'],  const['qn_ice_max'],  const['dim']) 
    
        q_rain_grid, qn_rain_grid = np.meshgrid(q_rain_vec, qn_rain_vec) 
        q_snow_grid, qn_snow_grid = np.meshgrid(q_snow_vec, qn_snow_vec) 
        q_grau_grid, qn_grau_grid = np.meshgrid(q_grau_vec, qn_grau_vec) 
        q_ice_grid , qn_ice_grid  = np.meshgrid(q_ice_vec, qn_ice_vec) 
        
        themask       = radar_mask.mask(lon,lat)                        # Mask has nans and zeros 
        dstacked      = np.rollaxis(np.stack([themask]*qs.shape[0]),0)  # Mask has nans and zeros 
        ddstacked     = np.rollaxis(np.stack([dstacked]*12),0)          # Mask has nans and zeros 
        
        qnr_restrict = np.ma.masked_array(qnr, dstacked)
        qns_restrict = np.ma.masked_array(qns, dstacked)
        qng_restrict = np.ma.masked_array(qng, dstacked)    
        qni_restrict = np.ma.masked_array(qni, dstacked)    
    
    elif (mp == 6):
        
        apply_mask = 0; 

        ffile = open(ftable, 'rb')
        [Zh_RAIN, Zdr_RAIN, Zh_SNOW, Zdr_SNOW, Zh_GRAU, Zdr_GRAU, 
             LDR_RAIN, Aih_RAIN, Aiv_RAIN, KDP_RAIN, LDR_SNOW, Aih_SNOW, Aiv_SNOW, KDP_SNOW,
             LDR_GRAU, Aih_GRAU, Aiv_GRAU, KDP_GRAU] = pickle.load(ffile, encoding="latin1" ) 
        ffile.close()   

        f = open(ncfile, 'rb')
        [z_level, lat, lon, u, v, w, qr, qs, qc, qg, qi, qtotal] = readWRF.readWRFvariables(ncfile, mp)
        f.close()   

        [q_rain_vec]  = fun.create_logarithmic_scaled_vectors(const['q_rain_min'], const['q_rain_max'], const['dim'])
        [q_snow_vec]  = fun.create_logarithmic_scaled_vectors(const['q_snow_min'], const['q_snow_max'], const['dim'])
        [q_grau_vec]  = fun.create_logarithmic_scaled_vectors(const['q_grau_min'], const['q_grau_max'], const['dim'])
        [q_clou_vec]  = fun.create_logarithmic_scaled_vectors(const['q_clou_min'], const['q_clou_max'], const['dim'])
        [q_ice_vec]   = fun.create_logarithmic_scaled_vectors(const['q_ice_min'], const['q_ice_max'], const['dim'])
        
        themask       = radar_mask.mask(lon,lat)                       # Mask has nans and zeros 
        dstacked      = np.rollaxis(np.stack([themask]*qs.shape[0]),0)  # Mask has nans and zeros 
        ddstacked     = np.rollaxis(np.stack([dstacked]*12),0)  # Mask has nans and zeros 
    
    qs_restrict  = np.ma.masked_array(qs,  dstacked)    # Has zeros and data.
    qr_restrict  = np.ma.masked_array(qr,  dstacked)
    qg_restrict  = np.ma.masked_array(qg,  dstacked)
    qi_restrict  = np.ma.masked_array(qi,  dstacked)
    qc_restrict  = np.ma.masked_array(qc,  dstacked)
    
    u_restrict   = np.ma.masked_array(u,   dstacked)
    v_restrict   = np.ma.masked_array(v,   dstacked)
    w_restrict   = np.ma.masked_array(w,   dstacked)
    z_restrict   = np.ma.masked_array(z_level,   dstacked)
    lon_restrict = np.ma.masked_array(lon, themask)
    lat_restrict = np.ma.masked_array(lat, themask)
    
    # get nonzero-entries
    indices_r = np.ma.nonzero(qr_restrict)
    indices_c = np.ma.nonzero(qc_restrict)
    indices_s = np.ma.nonzero(qs_restrict)
    indices_g = np.ma.nonzero(qg_restrict)
    indices_i = np.ma.nonzero(qi_restrict)
         
    #------------------------------------------------------------------------------                                  
    # EXTRACT FROM LOOKUP TABLE VALUES OF INTEREST
                        
    if (mp == 16):
        display('get from wrf2radar_old')
    
    elif (mp == 10):
        display('get from wrf2radar_old')
        
    elif (mp == 6):
        
        # RAIN      
        Zh_r_out = np.zeros((3, qr.shape[0], qr.shape[1], qr.shape[2]));     Zh_r_out[:]  = np.nan
        Zdr_r_out = np.zeros((3, qr.shape[0], qr.shape[1], qr.shape[2]));    Zdr_r_out[:]  = np.nan
        LDR_r_out = np.zeros((3, qr.shape[0], qr.shape[1], qr.shape[2]));    LDR_r_out[:]  = np.nan
        KDP_r_out = np.zeros((3, qr.shape[0], qr.shape[1], qr.shape[2]));   KDP_r_out[:]  = np.nan
        Aih_r_out = np.zeros((3, qr.shape[0], qr.shape[1], qr.shape[2]));    Aih_r_out[:]  = np.nan
        Aiv_r_out = np.zeros((3, qr.shape[0], qr.shape[1], qr.shape[2]));     Aiv_r_out[:]  = np.nan
        # SNOW      
        Zh_s_out = np.zeros((3, qr.shape[0], qr.shape[1], qr.shape[2]));     Zh_s_out[:]  = np.nan
        Zdr_s_out = np.zeros((3, qr.shape[0], qr.shape[1], qr.shape[2]));    Zdr_s_out[:]  = np.nan
        LDR_s_out = np.zeros((3, qr.shape[0], qr.shape[1], qr.shape[2]));    LDR_s_out[:]  = np.nan
        KDP_s_out = np.zeros((3, qr.shape[0], qr.shape[1], qr.shape[2]));    KDP_s_out[:]  = np.nan
        Aih_s_out = np.zeros((3, qr.shape[0], qr.shape[1], qr.shape[2]));    Aih_s_out[:]  = np.nan
        Aiv_s_out = np.zeros((3, qr.shape[0], qr.shape[1], qr.shape[2]));    Aiv_s_out[:]  = np.nan
        # GRAU      
        Zh_g_out = np.zeros((3, qr.shape[0], qr.shape[1], qr.shape[2]));     Zh_g_out[:]  = np.nan
        Zdr_g_out = np.zeros((3, qr.shape[0], qr.shape[1], qr.shape[2]));    Zdr_g_out[:]  = np.nan
        LDR_g_out = np.zeros((3, qr.shape[0], qr.shape[1], qr.shape[2]));    LDR_g_out[:]  = np.nan
        KDP_g_out = np.zeros((3, qr.shape[0], qr.shape[1], qr.shape[2]));    KDP_g_out[:]  = np.nan
        Aih_g_out = np.zeros((3, qr.shape[0], qr.shape[1], qr.shape[2]));    Aih_g_out[:]  = np.nan
        Aiv_g_out = np.zeros((3, qr.shape[0], qr.shape[1], qr.shape[2]));    Aiv_g_out[:]  = np.nan 
        
        for ifreq in range(3):
            Zh_r_out[ifreq,:]  = intp.griddata(  q_rain_vec, Zh_RAIN[ifreq,:],   qr, method='linear') 
            Zdr_r_out[ifreq,:] = intp.griddata(  q_rain_vec, Zdr_RAIN[ifreq,:],  qr, method='linear') 
            LDR_r_out[ifreq,:] = intp.griddata(  q_rain_vec, LDR_RAIN[ifreq,:],  qr, method='linear') 
            KDP_r_out[ifreq,:] = intp.griddata(  q_rain_vec, KDP_RAIN[ifreq,:],  qr, method='linear') 
            Aih_r_out[ifreq,:] = intp.griddata(  q_rain_vec, Aih_RAIN[ifreq,:],  qr, method='linear') 
            Aiv_r_out[ifreq,:] = intp.griddata(  q_rain_vec, Aiv_RAIN[ifreq,:],  qr, method='linear') 
        
            # SNOW  
            Zh_s_out[ifreq,:]  = intp.griddata(  q_snow_vec, Zh_SNOW[ifreq,:],   qs, method='linear') 
            Zdr_s_out[ifreq,:] = intp.griddata(  q_snow_vec, Zdr_SNOW[ifreq,:],  qs, method='linear') 
            LDR_s_out[ifreq,:] = intp.griddata(  q_snow_vec, LDR_SNOW[ifreq,:],  qs, method='linear') 
            KDP_s_out[ifreq,:] = intp.griddata(  q_snow_vec, KDP_SNOW[ifreq,:],  qs, method='linear') 
            Aih_s_out[ifreq,:] = intp.griddata(  q_snow_vec, Aih_SNOW[ifreq,:],  qs, method='linear') 
            Aiv_s_out[ifreq,:] = intp.griddata(  q_snow_vec, Aiv_SNOW[ifreq,:],  qs, method='linear') 
           
            # GRAU 
            Zh_g_out[ifreq,:]  = intp.griddata(  q_grau_vec, Zh_GRAU[ifreq,:],   qg, method='linear') 
            Zdr_g_out[ifreq,:] = intp.griddata(  q_grau_vec, Zdr_GRAU[ifreq,:],  qg, method='linear') 
            LDR_g_out[ifreq,:] = intp.griddata(  q_grau_vec, LDR_GRAU[ifreq,:],  qg, method='linear') 
            KDP_g_out[ifreq,:] = intp.griddata(  q_grau_vec, KDP_GRAU[ifreq,:],  qg, method='linear') 
            Aih_g_out[ifreq,:] = intp.griddata(  q_grau_vec, Aih_GRAU[ifreq,:],  qg, method='linear') 
            Aiv_g_out[ifreq,:] = intp.griddata(  q_grau_vec, Aiv_GRAU[ifreq,:],  qg, method='linear') 
        
        # TOTAL
        Zh_t_out  = sum_total(Zh_r_out, Zh_s_out, Zh_g_out) 
        Zdr_t_out = sum_total(Zdr_r_out, Zdr_s_out, Zdr_g_out) 
        LDR_t_out = sum_total(LDR_r_out, LDR_s_out, LDR_g_out) 
        KDP_t_out = sum_total(KDP_r_out, KDP_s_out, KDP_g_out) 
        Aih_t_out = sum_total(Aih_r_out, Aih_s_out, Aih_g_out) 
        Aiv_t_out = sum_total(Aiv_r_out, Aiv_s_out, Aiv_g_out)            
    
    # APPLY MASK? (dummy in case domain is similar in size to radar range)
    if apply_mask == 0: 
        
        LDR_t_out_restrict  = LDR_t_out     
        KDP_t_out_restrict = KDP_t_out   
        Zh_t_out_restrict  = Zh_t_out 
        Aih_t_out_restrict = Aih_t_out 
        Aiv_t_out_restrict = Aiv_t_out 
    
        Zdr_t_out_restrict  = Zdr_t_out
        Zdr_rain_out_restrict = Zdr_r_out
        Zdr_snow_out_restrict = Zdr_s_out
        Zdr_grau_out_restrict = Zdr_g_out
    
        Zh_r_out_restrict  = Zh_r_out
        Zdr_r_out_restrict = Zdr_r_out
        LDR_r_out_restrict = LDR_r_out
        Aih_r_out_restrict = Aih_r_out
        Aiv_r_out_restrict = Aiv_r_out
        KDP_r_out_restrict = KDP_r_out
    
        Zh_s_out_restrict  = Zh_s_out
        Zdr_s_out_restrict = Zdr_s_out
        LDR_s_out_restrict = LDR_s_out
        Aih_s_out_restrict = Aih_s_out
        Aiv_s_out_restrict = Aiv_s_out
        KDP_s_out_restrict = KDP_s_out
    
        Zh_g_out_restrict  = Zh_g_out
        Zdr_g_out_restrict = Zdr_g_out
        LDR_g_out_restrict = LDR_g_out
        Aih_g_out_restrict = Aih_g_out
        Aiv_g_out_restrict = Aiv_g_out
        KDP_g_out_restrict = KDP_g_out
        
    elif apply_mask == 1: 

        # APPLY MASK! 
        LDR_t_out_restrict  = np.ma.masked_array(LDR_t_out, stacked_column_mass)      
        KDP_t_out_restrict = np.ma.masked_array(KDP_t_out, stacked_column_mass)      
        Zh_t_out_restrict  = np.ma.masked_array(Zh_t_out, stacked_column_mass)    
        Aih_t_out_restrict = np.ma.masked_array(Aih_t_out, stacked_column_mass)      
        Aiv_t_out_restrict = np.ma.masked_array(Aiv_t_out, stacked_column_mass)      
    
        Zdr_t_out_restrict  = np.ma.masked_array(Zdr_t_out,  stacked_column_mass)  
        Zdr_rain_out_restrict = np.ma.masked_array(Zdr_r_out, stacked_column_mass)  
        Zdr_snow_out_restrict = np.ma.masked_array(Zdr_s_out, stacked_column_mass)  
        Zdr_grau_out_restrict = np.ma.masked_array(Zdr_g_out, stacked_column_mass)  
    
        Zh_r_out_restrict  = np.ma.masked_array(Zh_r_out,  stacked_column_mass)
        Zdr_r_out_restrict = np.ma.masked_array(Zdr_r_out, stacked_column_mass)
        LDR_r_out_restrict = np.ma.masked_array(LDR_r_out, stacked_column_mass)
        Aih_r_out_restrict = np.ma.masked_array(Aih_r_out, stacked_column_mass)
        Aiv_r_out_restrict = np.ma.masked_array(Aiv_r_out, stacked_column_mass)
        KDP_r_out_restrict = np.ma.masked_array(KDP_r_out, stacked_column_mass)
    
        Zh_s_out_restrict  = np.ma.masked_array(Zh_s_out,  stacked_column_mass)
        Zdr_s_out_restrict = np.ma.masked_array(Zdr_s_out, stacked_column_mass)
        LDR_s_out_restrict = np.ma.masked_array(LDR_s_out, stacked_column_mass)
        Aih_s_out_restrict = np.ma.masked_array(Aih_s_out, stacked_column_mass)
        Aiv_s_out_restrict = np.ma.masked_array(Aiv_s_out, stacked_column_mass)
        KDP_s_out_restrict = np.ma.masked_array(KDP_s_out, stacked_column_mass)
    
        Zh_g_out_restrict  = np.ma.masked_array(Zh_g_out,  stacked_column_mass)
        Zdr_g_out_restrict = np.ma.masked_array(Zdr_g_out, stacked_column_mass)
        LDR_g_out_restrict = np.ma.masked_array(LDR_g_out, stacked_column_mass)
        Aih_g_out_restrict = np.ma.masked_array(Aih_g_out, stacked_column_mass)
        Aiv_g_out_restrict = np.ma.masked_array(Aiv_g_out, stacked_column_mass)
        KDP_g_out_restrict = np.ma.masked_array(KDP_g_out, stacked_column_mass)
        
    #------------------------------------------------------------------------------
    #   Transformation of WRF coordinates to cartesian coordinates for radar site
    #------------------------------------------------------------------------------
    
    dim1 = z_level.shape[0]
    dim2 = lon.shape[0]
    dim3 = lon.shape[1]
    
    x = np.zeros((dim2,dim3)); #x[:] =  np.NAN
    y = np.zeros((dim2,dim3)); #y[:] =  np.NAN
    phi = np.zeros((dim2,dim3));
    
    indices_lon = np.ma.nonzero(lon_restrict)
    
    for j in np.arange(np.shape(indices_lon)[1]):
        x[indices_lon[0][j],indices_lon[1][j]], y[indices_lon[0][j],
          indices_lon[1][j]] = transforms.geographic_to_cartesian_aeqd( 
          (lon_restrict[indices_lon[0][j],indices_lon[1][j]]), 
             (lat_restrict[indices_lon[0][j],indices_lon[1][j]]), 
             radar_lon, radar_lat, R=6370997)
    
    x_restrict   = np.ma.masked_array(x, themask)
    y_restrict   = np.ma.masked_array(y, themask)
    
    # Calculate range and elevation assuming 4/3 effective radius for each model 
    # cartesian coordinate gridpoint relatively to the position of radar origin 
    # (determined with ix,iy,iz)
    # Use a maximum range
    R = 6371.0 * 1000.0 * 4.0 / 3.0         # effective radius of earth in meters.
    
    r              = np.empty((dim1,dim2,dim3));      # r[:] = np.NAN  
    z              = np.empty((dim1,dim2,dim3)); 
    theta_e        = np.empty((dim1,dim2,dim3));      # Applying the 4/3 Reff approx.
    theta_geometry = np.empty((dim1,dim2,dim3));      # crsim calculates this for each pixel 
    
    # get nonzero-entries
    indices_x = np.ma.nonzero(x_restrict)
    indices_y = np.ma.nonzero(y_restrict)
    
    for j in np.arange(np.shape(indices_lon)[1]):
        for iz in range(z_restrict.shape[0]):        
            r[iz,indices_lon[0][j],indices_lon[1][j]]  = np.sqrt(
                    x_restrict[indices_lon[0][j],indices_lon[1][j]]**2 +
                     y_restrict[indices_lon[0][j],indices_lon[1][j]]**2 + 
                      z_restrict[iz,indices_lon[0][j],indices_lon[1][j]]**2)
    
    masked_r = np.ma.masked_where( (np.ma.masked_array(r, dstacked)) >= max_range, 
                                  (np.ma.masked_array(r, dstacked)))   
    
    # Now apply 4/3 approximation. This model assumes that the effective Earth radius
    # is 1/3 larger tan the real one, so Reff=4/3Re. This then allows to calculate 
    # analytically the height h and surface distance s relative to the radar site at a height h=0. 
    # Elevantion angle theta from H=sqrt(r2+Reff2+2rReffsintheta)-Reff
    # Note: theta in radians, transform to degrees
    theta_e    = (np.arcsin((z_level**2 - masked_r**2 + 2*z_level*R) / (2*masked_r*R)))* (180/np.pi)            
    
    # And this has to be different to the geometric theta 
    theta_geom      = np.arccos(z_level/masked_r); theta_geom = theta_geom * (180/np.pi)
    theta_geom_rads = np.arccos(z_restrict/masked_r);
    
    # Azimuth angle! 
    azimuth_angle = np.rad2deg( np.arctan2(x_restrict, y_restrict) )
    azimuth_angle[np.where(azimuth_angle<0)] = azimuth_angle[np.where(azimuth_angle<0)] + 360
    
    # Observation from radar only at theta_radar, so for each (x,y) find the index in z that 
    # corresponds to the zenith angle of interest.
    # If abs(array-value) > 0.01, return NaN
    # Add gaussian approach from Xue et al. (2006, An OSSE Framework
    # Based on the Ensemble Square Root Kalman Filter for Evaluating the Impact of 
    # Data from Radar Networks on Thunderstorm Analysis and Forecasting) 
    zindeces        = np.zeros((len(theta_radar),dim2,dim3));    zindeces[:]        = np.nan
    value           = np.zeros((len(theta_radar),dim2,dim3));    value[:]           = np.nan
    zindeces_max_bw = np.zeros((len(theta_radar),dim2,dim3));    zindeces_max_bw[:] = np.nan
    zindeces_min_bw = np.zeros((len(theta_radar),dim2,dim3));    zindeces_min_bw[:] = np.nan
    value_max_bw    = np.zeros((len(theta_radar),dim2,dim3));    value_max_bw[:]    = np.nan
    value_min_bw    = np.zeros((len(theta_radar),dim2,dim3));    value_min_bw[:]    = np.nan
    gridlevels      = np.zeros((len(theta_radar),dim2,dim3));    gridlevels[:]      = np.nan
    
    vr_doppler_weighted      = np.zeros((len(theta_radar),dim2,dim3));   vr_doppler_weighted[:]     = np.nan                          
    vradial_weighted_intep   = np.zeros((len(theta_radar),dim2,dim3));   vradial_weighted_intep[:]  = np.nan       
                                     
    Zh_rain_weighted         = np.zeros((len(theta_radar),dim2,dim3));   Zh_rain_weighted[:]        = np.nan
    Zh_rain_weighted_intep   = np.zeros((len(theta_radar),dim2,dim3));   Zh_rain_weighted_intep[:]  = np.nan
                                     
    Zh_snow_weighted         = np.zeros((len(theta_radar),dim2,dim3));   Zh_snow_weighted[:]        = np.nan
    Zh_snow_weighted_intep   = np.zeros((len(theta_radar),dim2,dim3));   Zh_snow_weighted_intep[:]  = np.nan      
                                     
    Zh_grau_weighted         = np.zeros((len(theta_radar),dim2,dim3));   Zh_grau_weighted[:]        = np.nan
    Zh_grau_weighted_intep   = np.zeros((len(theta_radar),dim2,dim3));   Zh_grau_weighted_intep[:]  = np.nan                                          
                                     
    Zh_ice_weighted          = np.zeros((len(theta_radar),dim2,dim3));   Zh_ice_weighted[:]         = np.nan
    Zh_ice_weighted_intep    = np.zeros((len(theta_radar),dim2,dim3));   Zh_ice_weighted_intep[:]   = np.nan       
                                     
    Zh_r_theta_radar         = np.zeros((len(theta_radar),dim2,dim3));   Zh_r_theta_radar[:]        = np.nan
    
    Zh_total_weighted_intep  = np.zeros((len(theta_radar),dim2,dim3));   Zh_total_weighted_intep[:] = np.nan      
    
    ZDR_total_weighted_intep = np.zeros((len(theta_radar),dim2,dim3));   ZDR_total_weighted_intep[:] = np.nan                           
    ZDR_rain_weighted_intep  = np.zeros((len(theta_radar),dim2,dim3));   ZDR_rain_weighted_intep[:]  = np.nan                           
    ZDR_snow_weighted_intep  = np.zeros((len(theta_radar),dim2,dim3));   ZDR_snow_weighted_intep[:]  = np.nan                           
    ZDR_grau_weighted_intep  = np.zeros((len(theta_radar),dim2,dim3));   ZDR_grau_weighted_intep[:]  = np.nan                           
    
    KDP_total_weighted_intep  = np.zeros((len(theta_radar),dim2,dim3));  KDP_total_weighted_intep[:] = np.nan                           
    
    elapsed_time = time.time() - start_time
    print('Before entering the radar geometry transformation modules:')
    print(elapsed_time/60)   # 1.49025276899
    

    if mode == 1: 
        start_time = time.time()   
        interplevels = 10 ;         # OROGINAL 20
        # TEST WITH IX=2649 WHICH IS  [-63.444153, -38.562912]                         
        # indices_lon HELPS WHEN DOMAIN OF WRF SIMULATION IS MUCH MUCH LARGER THAN DOMAIN OF RADAR
        indices_lon = np.ma.nonzero(lon_restrict)
        for ix in np.arange(np.shape(indices_lon)[1]):
            col_theta_e                       = theta_e[:,indices_lon[0][ix],indices_lon[1][ix]].stack()
            col_theta_geom                    = theta_geom[:,indices_lon[0][ix],indices_lon[1][ix]].stack()
            for it in range(len(theta_radar)):
        
                # 4/3 APPROX          
                zindeces[it, indices_lon[0][ix], indices_lon[1][ix]], value[it, indices_lon[0][ix], indices_lon[1][ix]] = find_nearest(col_theta_e.values,theta_radar[it])
        
                # NOW TAKE INTO ACCOUNT BANDWIDTH 
                zindeces_max_bw[it,indices_lon[0][ix], indices_lon[1][ix]], value_max_bw[it,indices_lon[0][ix], indices_lon[1][ix]] = find_nearest( col_theta_e.values,theta_radar[it]+(radar_antenna_bw/2))
                zindeces_min_bw[it,indices_lon[0][ix], indices_lon[1][ix]], value_min_bw[it,indices_lon[0][ix], indices_lon[1][ix]] = find_nearest( col_theta_e.values,theta_radar[it]-(radar_antenna_bw/2))
        
                # HOW MANY GRID LEVELS ARE FOUND IN THIS 1 DEGREE BEAMWIDTH 
                gridlevels[it,indices_lon[0][ix],indices_lon[1][ix]] = zindeces_max_bw[it,indices_lon[0][ix], indices_lon[1][ix]] - zindeces_min_bw[it,indices_lon[0][ix], indices_lon[1][ix]]
                            
                # Get the right zindex Zh_rain 
                if (np.isnan(zindeces[it,indices_lon[0][ix],indices_lon[1][ix]])!=1):
                    
                    # Calculate Zh_r_theta_radar to compare with Gaussing weighted Zh_r
                    Zh_r_theta_radar[it,indices_lon[0][ix],indices_lon[1][ix]] =  Zh_r_out_restrict[int(zindeces[it,indices_lon[0][ix],indices_lon[1][ix]]),indices_lon[0][ix],indices_lon[1][ix]]
        
                    # This is the number of gridpoints between bwtop and bwbottom
                    gridpoints = gridlevels[it,indices_lon[0][ix],indices_lon[1][ix]]     
        
                    # These are the empty array that needs to be interpolated             
                    Zh_riii           = np.empty(int(gridpoints));      Zh_riii[:] = np.nan    
                    Zh_siii           = np.empty(int(gridpoints));      Zh_siii[:] = np.nan    
                    Zh_giii           = np.empty(int(gridpoints));      Zh_giii[:] = np.nan    
                    Zh_iiii           = np.empty(int(gridpoints));      Zh_iiii[:] = np.nan     
                    Zh_tiii           = np.empty(int(gridpoints));      Zh_tiii[:] = np.nan   
                    elev_iii          = np.empty(int(gridpoints));      elev_iii[:] = np.nan  
                    Gain_g            = np.empty(int(gridpoints));    
                    layerdepth        = np.empty(int(gridpoints));    
                    zgrid_iii         = np.empty(int(gridpoints));
                    vradial_iii       = np.empty(int(gridpoints));
                    Zdr_tiii          = np.empty(int(gridpoints));      Zdr_tiii[:] = np.nan   
                    Zdr_riii          = np.empty(int(gridpoints));      Zdr_riii[:] = np.nan   
                    Zdr_siii          = np.empty(int(gridpoints));      Zdr_siii[:] = np.nan               
                    Zdr_giii          = np.empty(int(gridpoints));      Zdr_giii[:] = np.nan   
                    KDP_tiii          = np.empty(int(gridpoints));      KDP_tiii[:] = np.nan                                     
                    azimuth_angle_iii = np.empty(int(gridpoints));  
                    
                    
                                    
                    if (gridpoints != 0): 
                        # ------------------------------------------------------------------ 
                        # OPTION 1: BUILD GAUSSIAN AROUND ZINDEX_MIN : ZINDEX_MAX DISREGARDING 
                        #              THE VERTICAL RESOLUTION/DIFFERENCE BETWEEN THETA_E AND BW/2. 
                        # 
                        # OPTION 2: INTERPOLATE within this layer! DEFINE A HIGHER RESOLUTION 
                        #              GRID BETWEEN zindex_min and zindex_max and interpolate 
                        #              radar variables (Zh, ZDR, KDP, etc) and z_height 
                        # ------------------------------------------------------------------ 
                        # OPTION 1:
                        n=0
                        for iii in range(int(zindeces_min_bw[it,indices_lon[0][ix],indices_lon[1][ix]]), int(zindeces_max_bw[it,indices_lon[0][ix],indices_lon[1][ix]])):
                            
                            Zh_riii[n]           = Zh_r_out_restrict[int(iii),indices_lon[0][ix],indices_lon[1][ix]]
                            Zh_siii[n]           = Zh_s_out_restrict[int(iii),indices_lon[0][ix],indices_lon[1][ix]]
                            Zh_giii[n]           = Zh_g_out_restrict[int(iii),indices_lon[0][ix],indices_lon[1][ix]]
                            Zh_tiii[n]           = Zh_t_out_restrict[int(iii),indices_lon[0][ix],indices_lon[1][ix]]
        
                            Zdr_tiii[n]          = Zdr_t_out_restrict[int(iii),indices_lon[0][ix],indices_lon[1][ix]]
                            Zdr_riii[n]          = Zdr_rain_out_restrict[int(iii),indices_lon[0][ix],indices_lon[1][ix]]
                            Zdr_siii[n]          = Zdr_snow_out_restrict[int(iii),indices_lon[0][ix],indices_lon[1][ix]]
                            Zdr_giii[n]          = Zdr_grau_out_restrict[int(iii),indices_lon[0][ix],indices_lon[1][ix]] 
        
                            KDP_tiii[n]          = KDP_t_out_restrict[int(iii),indices_lon[0][ix],indices_lon[1][ix]]
        
                            elev_iii[n]          = theta_e[int(iii),indices_lon[0][ix],indices_lon[1][ix]]
                            elev_center          = value[it,indices_lon[0][ix],indices_lon[1][ix]]
                            azimuth_angle_iii[n] = azimuth_angle[indices_lon[0][ix],indices_lon[1][ix]]
                
                            vr1            = u_restrict[int(iii),indices_lon[0][ix],indices_lon[1][ix]]*np.cos(np.deg2rad(elev_iii[n]))*np.sin( np.deg2rad(azimuth_angle_iii[n]) )
                            vr2            = v_restrict[int(iii),indices_lon[0][ix],indices_lon[1][ix]]*np.cos(np.deg2rad(elev_iii[n]))*np.cos( np.deg2rad(azimuth_angle_iii[n]) )
                            vr3            = w_restrict[int(iii),indices_lon[0][ix],indices_lon[1][ix]]*np.sin(np.deg2rad(elev_iii[n]))
                            vradial_iii[n] = vr1 + vr2 + vr3
        
                            n += 1
        
                        # ------------------------------------------------------------------ 
                        # OPTION 2:
                        theta_interp_start = theta_e[int(zindeces_min_bw[it,indices_lon[0][ix],indices_lon[1][ix]]),indices_lon[0][ix],indices_lon[1][ix]]
                        theta_interp_end   = theta_e[int(zindeces_max_bw[it,indices_lon[0][ix],indices_lon[1][ix]]),indices_lon[0][ix],indices_lon[1][ix]]
                        theta_interp_vals  = np.linspace(theta_interp_start.values, theta_interp_end.values, interplevels)
        
                        Zr_interp_indx  = np.interp( theta_interp_vals, elev_iii, Zh_riii)
                        Zs_interp_indx  = np.interp( theta_interp_vals, elev_iii, Zh_siii)
                        Zg_interp_indx  = np.interp( theta_interp_vals, elev_iii, Zh_giii)
                        Zt_interp_indx  = np.interp( theta_interp_vals, elev_iii, Zh_tiii)
        
                        Zdr_interp_indx = np.interp( theta_interp_vals, elev_iii, Zdr_tiii)   
                        Zdr_rain_interp_indx = np.interp( theta_interp_vals, elev_iii, Zdr_riii)   
                        Zdr_snow_interp_indx = np.interp( theta_interp_vals, elev_iii, Zdr_siii)   
                        Zdr_grau_interp_indx = np.interp( theta_interp_vals, elev_iii, Zdr_giii)   
        
                        KDP_interp_indx = np.interp( theta_interp_vals, elev_iii, KDP_tiii)   
                        
                        vradial_interp_indx  = np.interp( theta_interp_vals, elev_iii, vradial_iii)
        
                        zgrid_interp_start = z_restrict[int(zindeces_min_bw[it,indices_lon[0][ix],indices_lon[1][ix]]),indices_lon[0][ix],indices_lon[1][ix]]
                        zgrid_interp_end   = z_restrict[int(zindeces_max_bw[it,indices_lon[0][ix],indices_lon[1][ix]]),indices_lon[0][ix],indices_lon[1][ix]]                 
                        zgrid_interp_vals  = np.linspace(zgrid_interp_start, zgrid_interp_end, interplevels)
                        layerdepth         = zgrid_interp_vals[1]-zgrid_interp_vals[0]             
                        
                        # make nans those theta_interp_elev_vals that fall outisde the theta_radar +- bw/2
                        theta_interp_vals[np.where(np.logical_or( theta_interp_vals[:] < (theta_radar[it]-(radar_antenna_bw/2)), theta_interp_vals[:] > (theta_radar[it]+(radar_antenna_bw/2))))] = np.nan
                        Zr_interp_indx[np.where(np.isnan(theta_interp_vals) == 1)] = np.nan 
                        Zs_interp_indx[np.where(np.isnan(theta_interp_vals) == 1)] = np.nan  
                        Zg_interp_indx[np.where(np.isnan(theta_interp_vals) == 1)] = np.nan 
                        Zt_interp_indx[np.where(np.isnan(theta_interp_vals) == 1)] = np.nan 
                        
                        Zdr_interp_indx[np.where(np.isnan(theta_interp_vals) == 1)] = np.nan 
                        Zdr_rain_interp_indx[np.where(np.isnan(theta_interp_vals) == 1)] = np.nan 
                        Zdr_snow_interp_indx[np.where(np.isnan(theta_interp_vals) == 1)] = np.nan 
                        Zdr_grau_interp_indx[np.where(np.isnan(theta_interp_vals) == 1)] = np.nan 
        
                        KDP_interp_indx[np.where(np.isnan(theta_interp_vals) == 1)]      = np.nan 
                        
                        vradial_interp_indx[np.where(np.isnan(theta_interp_vals) == 1)]  = np.nan 
                                       
                        # c) apply gaussian equation here
                        Gain_g     = np.empty(interplevels);    
                        for iii in range(interplevels):                   
                            top_sq           = ((theta_interp_vals[iii]-elev_center)/radar_antenna_bw)**2
                            Gain_g[iii]      = np.exp(-4*np.log(4)*top_sq)
        
                        Zh_rain_weighted_intep[it,indices_lon[0][ix],indices_lon[1][ix]]   = np.nansum(Gain_g*Zr_interp_indx*layerdepth)/np.nansum(Gain_g*layerdepth)  
                        Zh_snow_weighted_intep[it,indices_lon[0][ix],indices_lon[1][ix]]   = np.nansum(Gain_g*Zs_interp_indx*layerdepth)/np.nansum(Gain_g*layerdepth)  
                        Zh_grau_weighted_intep[it,indices_lon[0][ix],indices_lon[1][ix]]   = np.nansum(Gain_g*Zg_interp_indx*layerdepth)/np.nansum(Gain_g*layerdepth)  
                        Zh_total_weighted_intep[it,indices_lon[0][ix],indices_lon[1][ix]]  = np.nansum(Gain_g*Zt_interp_indx*layerdepth)/np.nansum(Gain_g*layerdepth)  
        
                        ZDR_total_weighted_intep[it,indices_lon[0][ix],indices_lon[1][ix]] = np.nansum(Gain_g*Zdr_interp_indx*layerdepth)/np.nansum(Gain_g*layerdepth)  
                        ZDR_rain_weighted_intep[it,indices_lon[0][ix],indices_lon[1][ix]]  = np.nansum(Gain_g*Zdr_rain_interp_indx*layerdepth)/np.nansum(Gain_g*layerdepth)  
                        ZDR_snow_weighted_intep[it,indices_lon[0][ix],indices_lon[1][ix]]  = np.nansum(Gain_g*Zdr_snow_interp_indx*layerdepth)/np.nansum(Gain_g*layerdepth)  
                        ZDR_grau_weighted_intep[it,indices_lon[0][ix],indices_lon[1][ix]]  = np.nansum(Gain_g*Zdr_grau_interp_indx*layerdepth)/np.nansum(Gain_g*layerdepth)  
        
                        KDP_total_weighted_intep[it,indices_lon[0][ix],indices_lon[1][ix]] = np.nansum(Gain_g*KDP_interp_indx*layerdepth)/np.nansum(Gain_g*layerdepth)  
            
                        vradial_weighted_intep[it,indices_lon[0][ix],indices_lon[1][ix]]   = np.nansum(Gain_g*vradial_interp_indx*layerdepth)/np.nansum(Gain_g*layerdepth)  
        
                        del Gain_g, Zh_riii, elev_iii, Zr_interp_indx, Zs_interp_indx, Zg_interp_indx, 
                        vradial_interp_indx, zgrid_interp_start, zgrid_interp_end, zgrid_interp_vals, Zdr_interp_indx
                    
                    # If its the central ray! 
                    else:
                        if (np.isnan(Zh_t_out_restrict[int(zindeces[it,indices_lon[0][ix],indices_lon[1][ix]]),indices_lon[0][ix],indices_lon[1][ix]])!=1):   
                            Zh_rain_weighted_intep[it,indices_lon[0][ix],indices_lon[1][ix]]   = Zh_r_out_restrict[int(zindeces[it,indices_lon[0][ix],indices_lon[1][ix]]),indices_lon[0][ix],indices_lon[1][ix]]
                            Zh_snow_weighted_intep[it,indices_lon[0][ix],indices_lon[1][ix]]   = Zh_s_out_restrict[int(zindeces[it,indices_lon[0][ix],indices_lon[1][ix]]),indices_lon[0][ix],indices_lon[1][ix]]  
                            Zh_grau_weighted_intep[it,indices_lon[0][ix],indices_lon[1][ix]]   = Zh_g_out_restrict[int(zindeces[it,indices_lon[0][ix],indices_lon[1][ix]]),indices_lon[0][ix],indices_lon[1][ix]]
                            Zh_total_weighted_intep[it,indices_lon[0][ix],indices_lon[1][ix]]  = Zh_t_out_restrict[int(zindeces[it,indices_lon[0][ix],indices_lon[1][ix]]),indices_lon[0][ix],indices_lon[1][ix]]
                            
                            ZDR_total_weighted_intep[it,indices_lon[0][ix],indices_lon[1][ix]] = Zdr_t_out_restrict[int(zindeces[it,indices_lon[0][ix],indices_lon[1][ix]]),indices_lon[0][ix],indices_lon[1][ix]]
                            ZDR_rain_weighted_intep[it,indices_lon[0][ix],indices_lon[1][ix]]  = Zdr_rain_out_restrict[int(zindeces[it,indices_lon[0][ix],indices_lon[1][ix]]),indices_lon[0][ix],indices_lon[1][ix]]
                            ZDR_snow_weighted_intep[it,indices_lon[0][ix],indices_lon[1][ix]]  = Zdr_snow_out_restrict[int(zindeces[it,indices_lon[0][ix],indices_lon[1][ix]]),indices_lon[0][ix],indices_lon[1][ix]]
                            ZDR_grau_weighted_intep[it,indices_lon[0][ix],indices_lon[1][ix]]  = Zdr_grau_out_restrict[int(zindeces[it,indices_lon[0][ix],indices_lon[1][ix]]),indices_lon[0][ix],indices_lon[1][ix]]
                            
                            KDP_total_weighted_intep[it,indices_lon[0][ix],indices_lon[1][ix]] = KDP_t_out_restrict[int(zindeces[it,indices_lon[0][ix],indices_lon[1][ix]]),indices_lon[0][ix],indices_lon[1][ix]]
                            
                            azimuth_angle_central = azimuth_angle[indices_lon[0][ix],indices_lon[1][ix]]
                            vr1            = u_restrict[int(zindeces[it,indices_lon[0][ix],indices_lon[1][ix]]),indices_lon[0][ix],indices_lon[1][ix]]*np.cos(np.deg2rad(elev_center))*np.sin( np.deg2rad(azimuth_angle_central) )
                            vr2            = v_restrict[int(zindeces[it,indices_lon[0][ix],indices_lon[1][ix]]),indices_lon[0][ix],indices_lon[1][ix]]*np.cos(np.deg2rad(elev_center))*np.cos( np.deg2rad(azimuth_angle_central) )
                            vr3            = w_restrict[int(zindeces[it,indices_lon[0][ix],indices_lon[1][ix]]),indices_lon[0][ix],indices_lon[1][ix]]*np.sin(np.deg2rad(elev_center))
                            
                            vradial_weighted_intep[it,indices_lon[0][ix],indices_lon[1][ix]]   = vr1 + vr2 + vr3
                    
                    
    elif mode == 0:

        elev_center_save    = np.zeros((len(theta_radar),dim2,dim3));   elev_center_save[:]        = np.nan
        azimuth_angles_save = np.zeros((len(theta_radar),dim2,dim3));   azimuth_angles_save[:]        = np.nan
  
        Zh_r_central    = np.zeros((len(theta_radar),dim2,dim3));   Zh_r_central[:]        = np.nan
        Zh_s_central    = np.zeros((len(theta_radar),dim2,dim3));   Zh_s_central[:]        = np.nan
        Zh_g_central    = np.zeros((len(theta_radar),dim2,dim3));   Zh_g_central[:]        = np.nan
        Zh_t_central    = np.zeros((len(theta_radar),dim2,dim3));   Zh_t_central[:]        = np.nan

        Zdr_t_central    = np.zeros((len(theta_radar),dim2,dim3));   Zdr_t_central[:]        = np.nan
        Zdr_r_central    = np.zeros((len(theta_radar),dim2,dim3));   Zdr_r_central[:]        = np.nan
        Zdr_s_central    = np.zeros((len(theta_radar),dim2,dim3));   Zdr_s_central[:]        = np.nan
        Zdr_g_central    = np.zeros((len(theta_radar),dim2,dim3));   Zdr_g_central[:]        = np.nan

        KDP_t_central    = np.zeros((len(theta_radar),dim2,dim3));   KDP_t_central[:]        = np.nan
        vradial_central  = np.zeros((len(theta_radar),dim2,dim3));   KDP_t_central[:]        = np.nan
        
        indices_lon = np.ma.nonzero(lon_restrict)            
        for ix in np.arange(np.shape(indices_lon)[1]):
            col_theta_e                       = theta_e[:,indices_lon[0][ix],indices_lon[1][ix]].stack()
            col_theta_geom                    = theta_geom[:,indices_lon[0][ix],indices_lon[1][ix]].stack()
        for it in range(len(theta_radar)):
    
            # 4/3 APPROX          
            zindeces[it, indices_lon[0][ix], indices_lon[1][ix]], value[it, indices_lon[0][ix], indices_lon[1][ix]] = find_nearest(col_theta_e.values,theta_radar[it])
    
            # Get the right zindex Zh_rain 
            if (np.isnan(zindeces[it,indices_lon[0][ix],indices_lon[1][ix]])!=1):
                
                # Calculate Zh_r_theta_radar to compare with Gaussing weighted Zh_r
                Zh_r_central[it,indices_lon[0][ix],indices_lon[1][ix]] =  Zh_r_out_restrict[int(zindeces[it,indices_lon[0][ix],indices_lon[1][ix]]),indices_lon[0][ix],indices_lon[1][ix]]
                Zh_s_central[it,indices_lon[0][ix],indices_lon[1][ix]] =  Zh_s_out_restrict[int(zindeces[it,indices_lon[0][ix],indices_lon[1][ix]]),indices_lon[0][ix],indices_lon[1][ix]]
                Zh_g_central[it,indices_lon[0][ix],indices_lon[1][ix]] =  Zh_g_out_restrict[int(zindeces[it,indices_lon[0][ix],indices_lon[1][ix]]),indices_lon[0][ix],indices_lon[1][ix]]
                Zh_t_central[it,indices_lon[0][ix],indices_lon[1][ix]] =  Zh_t_out_restrict[int(zindeces[it,indices_lon[0][ix],indices_lon[1][ix]]),indices_lon[0][ix],indices_lon[1][ix]]
    
    
                Zdr_t_central[it,indices_lon[0][ix],indices_lon[1][ix]] = Zdr_t_out_restrict[int(zindeces[it,indices_lon[0][ix],indices_lon[1][ix]]),indices_lon[0][ix],indices_lon[1][ix]]
                Zdr_r_central[it,indices_lon[0][ix],indices_lon[1][ix]] = Zdr_rain_out_restrict[int(zindeces[it,indices_lon[0][ix],indices_lon[1][ix]]),indices_lon[0][ix],indices_lon[1][ix]]
                Zdr_s_central[it,indices_lon[0][ix],indices_lon[1][ix]] = Zdr_snow_out_restrict[int(zindeces[it,indices_lon[0][ix],indices_lon[1][ix]]),indices_lon[0][ix],indices_lon[1][ix]]
                Zdr_g_central[it,indices_lon[0][ix],indices_lon[1][ix]] = Zdr_grau_out_restrict[int(zindeces[it,indices_lon[0][ix],indices_lon[1][ix]]),indices_lon[0][ix],indices_lon[1][ix]]
    
                KDP_t_central[it,indices_lon[0][ix],indices_lon[1][ix]] = KDP_t_out_restrict[int(zindeces[it,indices_lon[0][ix],indices_lon[1][ix]]),indices_lon[0][ix],indices_lon[1][ix]]
    
    
                elev_center          = value[it,indices_lon[0][ix],indices_lon[1][ix]]
                azimuth_angle_iii    = azimuth_angle[indices_lon[0][ix],indices_lon[1][ix]]
            
                vr1            = u_restrict[int(zindeces[it,indices_lon[0][ix],indices_lon[1][ix]]),indices_lon[0][ix],indices_lon[1][ix]]*np.cos(np.deg2rad(elev_center))*np.sin( np.deg2rad(azimuth_angle_iii) )
                vr2            = v_restrict[int(zindeces[it,indices_lon[0][ix],indices_lon[1][ix]]),indices_lon[0][ix],indices_lon[1][ix]]*np.cos(np.deg2rad(elev_center))*np.cos( np.deg2rad(azimuth_angle_iii) )
                vr3            = w_restrict[int(zindeces[it,indices_lon[0][ix],indices_lon[1][ix]]),indices_lon[0][ix],indices_lon[1][ix]]*np.sin(np.deg2rad(elev_center))
                vradial_central[it,indices_lon[0][ix],indices_lon[1][ix]] = vr1 + vr2 + vr3
                
                elev_center_save[it,indices_lon[0][ix],indices_lon[1][ix]]    = elev_center; 
                azimuth_angles_save[it,indices_lon[0][ix],indices_lon[1][ix]] = azimuth_angle_iii; 




    
    #------------------------------------------------------------------------------
    z_FOR_theta_radar      =  np.zeros((len(theta_radar),dim2,dim3));       z_FOR_theta_radar[:]      = np.nan
    z_FOR_theta_radar_GEOM =  np.zeros((len(theta_radar),dim2,dim3));       z_FOR_theta_radar_GEOM[:] = np.nan
    
    for ix in np.arange(np.shape(indices_lon)[1]):
         for it in range(len(theta_radar)):
             z_FOR_theta_radar[it,indices_lon[0][ix], indices_lon[1][ix]] = np.nan
             z_FOR_theta_radar_GEOM[it,indices_lon[0][ix], indices_lon[1][ix]] = np.nan
             if (zindeces[it,indices_lon[0][ix], indices_lon[1][ix]] != 0):
                 if (np.isnan(value[it,indices_lon[0][ix], indices_lon[1][ix]] )!= 1):
                     integer_index=int(zindeces[it,indices_lon[0][ix], indices_lon[1][ix]])
                     z_FOR_theta_radar[it,indices_lon[0][ix], indices_lon[1][ix]]      = z_restrict[integer_index,indices_lon[0][ix], indices_lon[1][ix]]
                     #z_FOR_theta_radar_GEOM[it,indices_lon[0][ix], indices_lon[1][ix]] = z_restrict[geom_integer_index,indices_lon[0][ix], indices_lon[1][ix]]
    
    
    elapsed_time = time.time() - start_time
    print('ONLY BANDWIDTH etc LOOP:')
    print(elapsed_time/60)
    
    if mp == 10:
        if mode == 1:
            return [Zh_rain_weighted_intep, Zh_snow_weighted_intep, Zh_grau_weighted_intep, Zh_ice_weighted_intep, Zh_total_weighted_intep, 
        vradial_weighted_intep, theta_e, r, masked_r, z_FOR_theta_radar, lon_restrict, lat_restrict, Zh_r_out_restrict, 
        Zh_s_out_restrict, Zh_g_out_restrict, Zh_i_out_restrict, z_restrict, Zh_t_out_restrict, 
        Zdr_r_out_restrict,Zdr_s_out_restrict,Zdr_g_out_restrict,Zdr_i_out_restrict,Zdr_t_out_restrict, 
        LDR_r_out_restrict, LDR_s_out_restrict, LDR_g_out_restrict, LDR_i_out_restrict, LDR_t_out_restrict,
        KDP_r_out_restrict, KDP_s_out_restrict, KDP_g_out_restrict, KDP_i_out_restrict, KDP_t_out_restrict,
        Aiv_r_out_restrict, Aiv_s_out_restrict, Aiv_g_out_restrict, Aiv_i_out_restrict, Aiv_t_out_restrict,
        Aih_r_out_restrict, Aih_s_out_restrict, Aih_g_out_restrict, Aih_i_out_restrict, Aih_t_out_restrict, 
        u_restrict, v_restrict, w_restrict, ZDR_total_weighted_intep, 
        ZDR_rain_weighted_intep, ZDR_snow_weighted_intep, ZDR_grau_weighted_intep, ZDR_grau_weighted_intep,
        KDP_total_weighted_intep]
            
#            pickle.dump([Zh_rain_weighted_intep, Zh_snow_weighted_intep, Zh_grau_weighted_intep, Zh_ice_weighted_intep, Zh_total_weighted_intep, 
#                 vradial_weighted_intep, theta_e, r, masked_r, z_FOR_theta_radar, lon_restrict, lat_restrict, Zh_r_out_restrict, 
#                 Zh_s_out_restrict, Zh_g_out_restrict, Zh_i_out_restrict, z_restrict, Zh_t_out_restrict, 
#                 Zdr_r_out_restrict,Zdr_s_out_restrict,Zdr_g_out_restrict,Zdr_i_out_restrict,Zdr_t_out_restrict, 
#                 LDR_r_out_restrict, LDR_s_out_restrict, LDR_g_out_restrict, LDR_i_out_restrict, LDR_t_out_restrict,
#                 KDP_r_out_restrict, KDP_s_out_restrict, KDP_g_out_restrict, KDP_i_out_restrict, KDP_t_out_restrict,
#                 Aiv_r_out_restrict, Aiv_s_out_restrict, Aiv_g_out_restrict, Aiv_i_out_restrict, Aiv_t_out_restrict,
#                 Aih_r_out_restrict, Aih_s_out_restrict, Aih_g_out_restrict, Aih_i_out_restrict, Aih_t_out_restrict, 
#                 u_restrict, v_restrict, w_restrict, ZDR_total_weighted_intep, 
#                 ZDR_rain_weighted_intep, ZDR_snow_weighted_intep, ZDR_grau_weighted_intep, ZDR_grau_weighted_intep,
#                 KDP_total_weighted_intep], fout) 
        
        elif (mode == 0):
            print('add')
            
    
    elif mp == 6:
        if mode == 1:
            return [Zh_rain_weighted_intep, Zh_snow_weighted_intep, Zh_grau_weighted_intep, Zh_total_weighted_intep, 
                 vradial_weighted_intep, theta_e, r, masked_r, z_FOR_theta_radar, lon_restrict, lat_restrict, Zh_r_out_restrict, 
                 Zh_s_out_restrict, Zh_g_out_restrict, z_restrict, Zh_t_out_restrict, 
                 Zdr_r_out_restrict,Zdr_s_out_restrict,Zdr_g_out_restrict,Zdr_t_out_restrict, 
                 LDR_r_out_restrict, LDR_s_out_restrict, LDR_g_out_restrict, LDR_t_out_restrict,
                 KDP_r_out_restrict, KDP_s_out_restrict, KDP_g_out_restrict, KDP_t_out_restrict,
                 Aiv_r_out_restrict, Aiv_s_out_restrict, Aiv_g_out_restrict, Aiv_t_out_restrict,
                 Aih_r_out_restrict, Aih_s_out_restrict, Aih_g_out_restrict, Aih_t_out_restrict, 
                 u_restrict, v_restrict, w_restrict, ZDR_total_weighted_intep, 
                 ZDR_rain_weighted_intep, ZDR_snow_weighted_intep, ZDR_grau_weighted_intep, ZDR_grau_weighted_intep,
                 KDP_total_weighted_intep]
            
#            pickle.dump([Zh_rain_weighted_intep, Zh_snow_weighted_intep, Zh_grau_weighted_intep, Zh_total_weighted_intep, 
#                 vradial_weighted_intep, theta_e, r, masked_r, z_FOR_theta_radar, lon_restrict, lat_restrict, Zh_r_out_restrict, 
#                 Zh_s_out_restrict, Zh_g_out_restrict, z_restrict, Zh_t_out_restrict, 
#                 Zdr_r_out_restrict,Zdr_s_out_restrict,Zdr_g_out_restrict,Zdr_t_out_restrict, 
#                 LDR_r_out_restrict, LDR_s_out_restrict, LDR_g_out_restrict, LDR_t_out_restrict,
#                 KDP_r_out_restrict, KDP_s_out_restrict, KDP_g_out_restrict, KDP_t_out_restrict,
#                 Aiv_r_out_restrict, Aiv_s_out_restrict, Aiv_g_out_restrict, Aiv_t_out_restrict,
#                 Aih_r_out_restrict, Aih_s_out_restrict, Aih_g_out_restrict, Aih_t_out_restrict, 
#                 u_restrict, v_restrict, w_restrict, ZDR_total_weighted_intep, 
#                 ZDR_rain_weighted_intep, ZDR_snow_weighted_intep, ZDR_grau_weighted_intep, ZDR_grau_weighted_intep,
#                 KDP_total_weighted_intep], fout)    
        elif mode == 0:
            return [Zh_r_central, Zh_s_central,Zh_g_central,Zh_t_central, 
                     Zdr_t_central, Zdr_r_central, Zdr_s_central, Zdr_g_central, 
                     KDP_t_central, elev_center_save, azimuth_angles_save, vradial_central, 
                     theta_e, r, masked_r, z_FOR_theta_radar, lon_restrict, lat_restrict, Zh_r_out_restrict, 
                     Zh_s_out_restrict, Zh_g_out_restrict, z_restrict, Zh_t_out_restrict, 
                     Zdr_r_out_restrict,Zdr_s_out_restrict,Zdr_g_out_restrict,Zdr_t_out_restrict, 
                     LDR_r_out_restrict, LDR_s_out_restrict, LDR_g_out_restrict, LDR_t_out_restrict,
                     KDP_r_out_restrict, KDP_s_out_restrict, KDP_g_out_restrict, KDP_t_out_restrict,
                     Aiv_r_out_restrict, Aiv_s_out_restrict, Aiv_g_out_restrict, Aiv_t_out_restrict,
                     Aih_r_out_restrict, Aih_s_out_restrict, Aih_g_out_restrict, Aih_t_out_restrict, 
                     u_restrict, v_restrict, w_restrict]
            
            
#            pickle.dump([Zh_r_central, Zh_s_central,Zh_g_central,Zh_t_central, 
#                     Zdr_t_central, Zdr_r_central, Zdr_s_central, Zdr_g_central, 
#                     KDP_t_central, elev_center_save, azimuth_angles_save, vradial_central, 
#                     theta_e, r, masked_r, z_FOR_theta_radar, lon_restrict, lat_restrict, Zh_r_out_restrict, 
#                     Zh_s_out_restrict, Zh_g_out_restrict, z_restrict, Zh_t_out_restrict, 
#                     Zdr_r_out_restrict,Zdr_s_out_restrict,Zdr_g_out_restrict,Zdr_t_out_restrict, 
#                     LDR_r_out_restrict, LDR_s_out_restrict, LDR_g_out_restrict, LDR_t_out_restrict,
#                     KDP_r_out_restrict, KDP_s_out_restrict, KDP_g_out_restrict, KDP_t_out_restrict,
#                     Aiv_r_out_restrict, Aiv_s_out_restrict, Aiv_g_out_restrict, Aiv_t_out_restrict,
#                     Aih_r_out_restrict, Aih_s_out_restrict, Aih_g_out_restrict, Aih_t_out_restrict, 
#                     u_restrict, v_restrict, w_restrict], f)    

    
