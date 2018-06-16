# Python script to _________________________
# V. Galligani 
# CIMA, UBA-CONICET, Argentina
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

def find_nearest(array,THEvalue):
    idx    = int( (np.abs( (array.ravel()) - THEvalue)).argmin() )
    output =  array[idx]
    if np.isnan(output):
        idx = np.nan
    return idx, output


#indices=indices_r; lut_q=q_rain_vec; lut_qn=qn_rain_vec; qwrf=qr_restrict; qnwrf=qnr_restrict; Zh=Zh_RAIN; Zdr=Zdr_RAIN;

def find_indx_regular(indices, lut_q, lut_qn, qwrf, qnwrf, Zh, Zdr): 
    
    # Here hard coded the number of elevation angles in my lookup table! 
    
    qdelta = np.abs(np.log10(lut_q[100]) - np.log10(lut_q[99]))
    qndelta = np.abs(np.log10(lut_qn[100]) - np.log10(lut_qn[99]))     
    
    Zh_out  = np.zeros([np.shape(qwrf)[0],np.shape(qwrf)[1],np.shape(qwrf)[2]]); 
    Zdr_out = np.zeros([np.shape(qwrf)[0],np.shape(qwrf)[1],np.shape(qwrf)[2]]); 
    
    for j in np.arange(np.shape(indices)[1]):
        indq  =  int( np.abs( np.log10(qwrf[indices[0][j],indices[1][j],indices[2][j]])  - np.log10(lut_q[0]))/ qdelta )
        indn =   int( np.abs( np.log10(qnwrf[indices[0][j],indices[1][j],indices[2][j]]) - np.log10(lut_qn[0])) / qndelta )
        #for jj in np.arange(12): 
        Zh_out[indices[0][j],indices[1][j],indices[2][j]]   = Zh_RAIN[indq,indn];
        Zdr_out[indices[0][j],indices[1][j],indices[2][j]]  = Zdr_RAIN[indq,indn];

    return Zh_out, Zdr_out
    

start_time = time.time()                            

#ncfile = "/home/victoria.galligani/Work/Studies/TEPEMAI_01132011/WDM6/wrfout_d01_2011-01-13_22:00:00"
print('Enter mp_physics number: i.e., Morrison is mp=10 and WDM6 is mp=16')
mp     = int(raw_input())
ncfile = "/home/victoria.galligani/Work/Studies/WRF_radar_simulator/TEST_CRSIM_TPRADARPAU/wrfout_d01_2010-01-11_19:40:00"

#-----------------------------------------------------------------------------
# To avoid memory error: START WITH ANGUIL 
radar_lon     = -64.0103
radar_lat     = -36.5257 
anguil_window = [[radar_lon-5, radar_lat-5], [radar_lon-5, radar_lat+5], [radar_lon+5, radar_lat+5], [radar_lon+5, radar_lat-5]]
anguil_mask   = regionmask.Regions_cls('RADARES', [0], ['radar_anguil'], ['anguil'], [anguil_window])


# TEST A VERY SMALL WINDOW 
#anguil_window = [[radar_lon-0.5, radar_lat-0.5], [radar_lon-0.5, radar_lat+0.5], [radar_lon+0.5, radar_lat+0.5], [radar_lon+0.5, radar_lat-0.5]]
#anguil_mask   = regionmask.Regions_cls('RADARES', [0], ['radar_anguil'], ['anguil'], [anguil_window])

# Plot mask 
#ax = anguil_mask.plot()
#ax.set_extent([-94,-34, -56, -11], crs=ccrs.PlateCarree());
             

const=fun.return_constants()

             
if (mp == 16):

    # Load lookup tables
    f = open('WDM6_LOOKUPTABLE.pckl', 'rb')
    Zh_RAIN, Zdr_RAIN, Zh_SNOW, Zdr_SNOW, Zh_GRAU, Zdr_GRAU, Zh_CLOUD, Zdr_CLOUD = pickle.load(f)
    f.close() 
    
    from generate_lookups_WDM6 import const, create_logarithmic_scaled_vectors 
    
    [q_rain_vec]  = fun.create_logarithmic_scaled_vectors(const['q_rain_min'], const['q_rain_max'], const['dim'])
    [q_snow_vec]  = fun.create_logarithmic_scaled_vectors(const['q_snow_min'], const['q_snow_max'], const['dim'])
    [q_grau_vec]  = fun.create_logarithmic_scaled_vectors(const['q_grau_min'], const['q_grau_max'], const['dim'])
    [q_clou_vec]  = fun.create_logarithmic_scaled_vectors(const['q_clou_min'], const['q_clou_max'], const['dim'])
    [qn_rain_vec] = fun.create_logarithmic_scaled_vectors(const['qn_rain_min'], const['qn_rain_max'], const['dim']) 
    [qn_clou_vec] = fun.create_logarithmic_scaled_vectors(const['qn_clou_min'], const['qn_clou_max'], const['dim']) 
    q_rain_grid, qn_rain_grid = np.meshgrid(q_rain_vec, qn_rain_vec) 
    q_clou_grid, qn_clou_grid = np.meshgrid(q_clou_vec, qn_clou_vec) 

    [z_level, lat, lon, u, v, w, qr, qs, qc, qg, qi, qnr, qnc] = readWRF.readWRFvariables(ncfile, mp)

    ncfile2 = "/home/victoria.galligani/Work/Studies/TEPEMAI_01132011/WDM6/wrfout_d01_2011-01-13_22:00:00"
    [z_level2, lat2, lon2, u2, v2, w2, qr2, qs2, qc2, qg2, qi2, qnr2, qnc2] = readWRF.readWRFvariables(ncfile2, 16)

        
    themask       = anguil_mask.mask(lon,lat)             # Mask has nans and zeros 
    dstacked      = np.rollaxis(np.stack([themask]*qs.shape[0]),0)  # Mask has nans and zeros 

    qnr_restrict = np.ma.masked_array(qnr, dstacked)
    qnc_restrict = np.ma.masked_array(qnc, dstacked)

    
elif (mp == 10):
    print('ok changed')
    # Load lookup tables
    f = open('MORRISON_LOOKUPTABLE.pckl', 'rb')
    #Zh_RAIN, Zdr_RAIN = pickle.load(f)
    Zh_RAIN, Zdr_RAIN, Zh_SNOW, Zdr_SNOW, Zh_GRAU, Zdr_GRAU = pickle.load(f)
    f.close() 
    
    [z_level, lat, lon, u, v, w, qr, qs, qc, qg, qi, qnr, qns, qng] = readWRF.readWRFvariables(ncfile, mp)
    [q_rain_vec]  = fun.create_logarithmic_scaled_vectors(const['q_rain_min'], const['q_rain_max'], const['dim'])
    [q_snow_vec]  = fun.create_logarithmic_scaled_vectors(const['q_snow_min'], const['q_snow_max'], const['dim'])
    [q_grau_vec]  = fun.create_logarithmic_scaled_vectors(const['q_grau_min'], const['q_grau_max'], const['dim'])
    [q_clou_vec]  = fun.create_logarithmic_scaled_vectors(const['q_clou_min'], const['q_clou_max'], const['dim'])
    [qn_rain_vec] = fun.create_logarithmic_scaled_vectors(const['qn_rain_min'], const['qn_rain_max'], const['dim']) 
    [qn_snow_vec] = fun.create_logarithmic_scaled_vectors(const['qn_snow_min'], const['qn_snow_max'], const['dim']) 
    [qn_grau_vec] = fun.create_logarithmic_scaled_vectors(const['qn_grau_min'], const['qn_grau_max'], const['dim']) 
    q_rain_grid, qn_rain_grid = np.meshgrid(q_rain_vec, qn_rain_vec) 
    q_snow_grid, qn_snow_grid = np.meshgrid(q_snow_vec, qn_snow_vec) 
    q_grau_grid, qn_grau_grid = np.meshgrid(q_grau_vec, qn_grau_vec) 
    
    themask       = anguil_mask.mask(lon,lat)                       # Mask has nans and zeros 
    dstacked      = np.rollaxis(np.stack([themask]*qs.shape[0]),0)  # Mask has nans and zeros 
    ddstacked     = np.rollaxis(np.stack([dstacked]*12),0)  # Mask has nans and zeros 

    
    qnr_restrict = np.ma.masked_array(qnr, dstacked)
    qns_restrict = np.ma.masked_array(qns, dstacked)
    qng_restrict = np.ma.masked_array(qng, dstacked)    


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
      
#------------------------------------------------------------------------------                                  
                    
if (mp == 16):

    # RAIN  
    [Zh_r_out, Zdr_r_out] = find_indx_regular(indices_r, q_rain_vec, qn_rain_vec, qr_restrict, qnr_restrict, Zh_RAIN, Zdr_RAIN)
    Zh_r_out_restrict   = np.ma.masked_array(Zh_r_out,  dstacked)
    Zdr_r_out_restrict  = np.ma.masked_array(Zdr_r_out,  dstacked)    

    # SNOW
    Zh_s_out  = np.zeros(np.shape(qr)); 
    Zdr_s_out = np.zeros(np.shape(qr)); 
    grid_Zh_s  = intp.griddata(q_snow_vec, np.reshape(Zh_SNOW, const['dim']),  qs.values[indices_s], method='linear')
    grid_Zdr_s = intp.griddata(q_snow_vec, np.reshape(Zdr_SNOW, const['dim']), qs.values[indices_s], method='linear')
    Zh_s_out[indices_s]  = grid_Zh_s
    Zdr_s_out[indices_s] = grid_Zdr_s
    Zh_s_out_restrict    = np.ma.masked_array(Zh_s_out,  dstacked)
    Zdr_s_out_restrict   = np.ma.masked_array(Zdr_s_out,  dstacked)

    # GRAUPEL
    Zh_g_out  = np.zeros(np.shape(qr));                    
    Zdr_g_out = np.zeros(np.shape(qr));                  
    grid_Zh_g  = intp.griddata(q_grau_vec, np.reshape(Zh_GRAU, const['dim']),  qg.values[indices_g], method='linear')
    grid_Zdr_g = intp.griddata(q_grau_vec, np.reshape(Zdr_GRAU, const['dim']), qg.values[indices_g], method='linear')
    Zh_g_out[indices_g]   = grid_Zh_g
    Zdr_g_out[indices_g]  = grid_Zdr_g
    Zh_g_out_restrict     = np.ma.masked_array(Zh_g_out,  dstacked)
    Zdr_g_out_restrict    = np.ma.masked_array(Zdr_g_out,  dstacked)
    
    ##f = open('WDM6_Z.pckl', 'wb')
    ##pickle.dump([Zh_r_out, Zdr_r_out, Zh_s_out, Zdr_s_out, Zh_g_out, Zdr_g_out], f)
    ##f.close()

elif (mp == 10):
    
    # Parallelization test 
    # from pathos.multiprocessing import ProcessingPool as Pool
    # def find_LUT(i, j, k, q_wrf = qr, qn_wrf = qnr, q_vec = q_rain_vec, qn_vec = qn_rain_vec, Zh_LUT = Zh_RAIN, Zdr_LUT = Zh_RAIN):
        # Zh_out  = np.zeros(np.shape(q_wrf)); 
        # Zdr_out = np.zeros(np.shape(q_wrf));                 
        # diffq = np.abs(q_wrf[i,j,k]  - q_vec)
        # diffn = np.abs(qn_wrf[i,j,k] - qn_vec)
        # indq = diffq.argmin()
        # indn = diffn.argmin()   
        # Zh_out[i,j,k]  = Zh_LUT[indq,indn]
        # Zdr_out[i,j,k] = Zdr_LUT[indq,indn]   
        # return Zh_out, Zdr_out
    # indices_rnonzero = np.ma.nonzero(qr)
    # start_time = time.time()                            
    # cores = 2
    # p     = Pool(cores)
    # p.clear() 
    # res = p.map(find_LUT, indices_rnonzero[0], indices_rnonzero[1],indices_rnonzero[2])
    # p.close()
    # elapsed_time = time.time() - start_time
    # print(elapsed_time/60)    

    # RAIN  
    [Zh_r_out, Zdr_r_out] = find_indx_regular(indices_r, q_rain_vec, qn_rain_vec, qr_restrict, qnr_restrict, Zh_RAIN, Zdr_RAIN)
    Zh_r_out_restrict   = np.ma.masked_array(Zh_r_out,  dstacked)
    Zdr_r_out_restrict  = np.ma.masked_array(Zdr_r_out,  dstacked)    

    # SNOW  
    [Zh_s_out, Zdr_s_out] = find_indx_regular(indices_s, q_snow_vec, qn_snow_vec, qs_restrict, qns_restrict, Zh_SNOW, Zdr_SNOW)
    Zh_s_out_restrict   = np.ma.masked_array(Zh_s_out,  dstacked)
    Zdr_s_out_restrict  = np.ma.masked_array(Zdr_s_out,  dstacked)    

    # GRAU                    
    [Zh_g_out, Zdr_g_out] = find_indx_regular(indices_g, q_grau_vec, qn_grau_vec, qg_restrict, qng_restrict, Zh_GRAU, Zdr_GRAU)
    Zh_g_out_restrict   = np.ma.masked_array(Zh_g_out,  dstacked)
    Zdr_g_out_restrict  = np.ma.masked_array(Zdr_g_out,  dstacked)    
   
    colmax_Zh_r = np.nanmax(Zh_r_out_restrict,axis=0)
    colmax_Zh_s = np.nanmax(Zh_s_out_restrict,axis=0)
    colmax_Zh_g = np.nanmax(Zh_g_out_restrict,axis=0)
    
#==============================================================================
#==============================================================================
# ax = anguil_mask.plot()
# ax.set_extent([-70,-58, -42, -30], crs=ccrs.PlateCarree())
# plt.contourf(lon,lat, colmax_Zh_r); plt.colorbar()
# # 
# ax = anguil_mask.plot()
# ax.set_extent([-70,-58, -42, -30], crs=ccrs.PlateCarree())
# plt.contourf(lon,lat, colmax_Zh_s); plt.colorbar()
# # 
# ax = anguil_mask.plot()
# ax.set_extent([-70,-58, -42, -30], crs=ccrs.PlateCarree())
# plt.contourf(lon,lat, colmax_Zh_g); plt.colorbar()

#colmax_Zh_r1 = np.nanmax(Zh_r_out_restrict[0,:,:,:],axis=0)
#colmax_Zh_r2 = np.nanmax(Zh_r_out_restrict[11,:,:,:],axis=0)
    
#ax = anguil_mask.plot()
#ax.set_extent([-70,-58, -42, -30], crs=ccrs.PlateCarree())
#plt.contourf(lon,lat, (colmax_Zh_r1-colmax_Zh_r2) ); plt.colorbar()


#==============================================================================
#==============================================================================

#==============================================================================
#------------------------------------------------------------------------------
#                       radar de Anguil config
#------------------------------------------------------------------------------
#==============================================================================

# Initial radar elevation angles
theta_radar  = (0.5, 0.9, 1.3, 1.9, 2.3, 3, 3.5, 5, 6.9, 9.1, 11.8, 15.1)
# Radio de alcanze max   
max_range        = 240*1E3   
# Radar bandwidth                                                    
radar_antenna_bw = 1.0;  


#==============================================================================
#------------------------------------------------------------------------------
#   Transformation of WRF coordinates to cartesian coordinates for Anguil 
#------------------------------------------------------------------------------
#==============================================================================

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

r              = np.empty((dim1,dim2,dim3));      #r[:] = np.NAN  
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
azimuth_angle = np.rad2deg( np.arctan2(y_restrict, x_restrict) )
azimuth_angle[np.where(azimuth_angle<0)] = azimuth_angle[np.where(azimuth_angle<0)] + 360
# plt.scatter(lon,lat, s=10, c=azimuth_angle[:,:],vmin=0, vmax=360);plt.colorbar();plt.grid(True)







# Observation from anguil only at theta_radar, so for each (x,y) find the index in z that 
# corresponds to the zenith angle of interest.
# If abs(array-value) > 0.01, return NaN
# Add gaussian approach from Xue et al. (2006, An OSSE Framework
# Based on the Ensemble Square Root Kalman Filter for Evaluating the Impact of 
# Data from Radar Networks on Thunderstorm Analysis and Forecasting) 
zindeces        = np.zeros((len(theta_radar),dim2,dim3));        zindeces[:] = np.nan
value           = np.zeros((len(theta_radar),dim2,dim3));           value[:] = np.nan
zindeces_max_bw = np.zeros((len(theta_radar),dim2,dim3)); zindeces_max_bw[:] = np.nan
zindeces_min_bw = np.zeros((len(theta_radar),dim2,dim3)); zindeces_min_bw[:] = np.nan
value_max_bw    = np.zeros((len(theta_radar),dim2,dim3));           value_max_bw[:] = np.nan
value_min_bw    = np.zeros((len(theta_radar),dim2,dim3));           value_min_bw[:] = np.nan
gridlevels      = np.zeros((len(theta_radar),dim2,dim3));    gridlevels[:]   = np.nan

vr_doppler_weighted    = np.zeros((len(theta_radar),dim2,dim3));        vr_doppler_weighted[:] = np.nan                          
vradial_weighted_intep = np.zeros((len(theta_radar),dim2,dim3));        vradial_weighted_intep[:] = np.nan       
                                 
Zh_rain_weighted       = np.zeros((len(theta_radar),dim2,dim3));     Zh_rain_weighted[:] = np.nan
Zh_rain_weighted_intep = np.zeros((len(theta_radar),dim2,dim3));     Zh_rain_weighted_intep[:] = np.nan
                                 
Zh_snow_weighted       = np.zeros((len(theta_radar),dim2,dim3));     Zh_snow_weighted[:] = np.nan
Zh_snow_weighted_intep = np.zeros((len(theta_radar),dim2,dim3));     Zh_snow_weighted_intep[:] = np.nan      
                                 
Zh_grau_weighted       = np.zeros((len(theta_radar),dim2,dim3));     Zh_grau_weighted[:] = np.nan
Zh_grau_weighted_intep = np.zeros((len(theta_radar),dim2,dim3));     Zh_grau_weighted_intep[:] = np.nan                                          
                                 
Zh_r_theta_radar = np.zeros((len(theta_radar),dim2,dim3));           Zh_r_theta_radar[:] = np.nan
      
                           
                           
start_time = time.time()                            

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

        # Here apply also 0.1 to max and min gridlevels within bandwidth too 
        #if (np.abs(value[it,indices_lon[0][ix],indices_lon[1][ix]]-theta_radar[it])>0.1):
        #    value[it,indices_lon[0][ix],indices_lon[1][ix]] = np.nan 
        #    zindeces[it,indices_lon[0][ix],indices_lon[1][ix]] = np.nan 
                    
        # Get the right zindex Zh_rain 
        if (np.isnan(zindeces[it,indices_lon[0][ix],indices_lon[1][ix]])!=1):
            Zh_r_theta_radar[it,indices_lon[0][ix],indices_lon[1][ix]] =  Zh_r_out_restrict[int(zindeces[it,indices_lon[0][ix],indices_lon[1][ix]]),indices_lon[0][ix],indices_lon[1][ix]]

            gridpoints = gridlevels[it,indices_lon[0][ix],indices_lon[1][ix]]  # This is the number of gridpoints between bwtop and bwbottom
            
            Zh_riii     = np.empty(int(gridpoints));       Zh_riii[:] = np.nan    # This is the empty array that needs to be interpolated 
            Zh_siii     = np.empty(int(gridpoints));       Zh_riii[:] = np.nan    # This is the empty array that needs to be interpolated 
            Zh_giii     = np.empty(int(gridpoints));       Zh_riii[:] = np.nan    # This is the empty array that needs to be interpolated 

            elev_iii    = np.empty(int(gridpoints));       elev_iii[:] = np.nan  
            Gain_g      = np.empty(int(gridpoints));    
            layerdepth  = np.empty(int(gridpoints));    
            zgrid_iii   = np.empty(int(gridpoints));
            vradial_iii = np.empty(int(gridpoints));
                                  
            azimuth_angle_iii = np.empty(int(gridpoints));  
                                        
            if (gridpoints != 0): 
                # ------------------------------------------------------------------ 
                # OPTION 1: BUILD GAUSSIAN AROUND ZINDEX_MIN : ZINDEX_MAX DISREGARDING VERTICAL RESOLUTION/DIFFERENCE BETWEEN THETA_E AND BW/2
                n=0
                for iii in xrange(int(zindeces_min_bw[it,indices_lon[0][ix],indices_lon[1][ix]]), int(zindeces_max_bw[it,indices_lon[0][ix],indices_lon[1][ix]])):
                    Zh_riii[n]           = Zh_r_out_restrict[int(iii),indices_lon[0][ix],indices_lon[1][ix]]
                    Zh_siii[n]           = Zh_s_out_restrict[int(iii),indices_lon[0][ix],indices_lon[1][ix]]
                    Zh_giii[n]           = Zh_g_out_restrict[int(iii),indices_lon[0][ix],indices_lon[1][ix]]

                #    elev_iii[n]          = theta_e[int(iii),indices_lon[0][ix],indices_lon[1][ix]]
                #    zgrid_iii[n]         = z_restrict[int(iii),indices_lon[0][ix],indices_lon[1][ix]]
                #    elev_center          = value[it,indices_lon[0][ix],indices_lon[1][ix]]
                #    top_sq               = ((elev_iii[n]-elev_center)/radar_antenna_bw)**2
                    ## Gain_g[n]            = np.exp(-4*np.log(4)*top_sq)
                    ## layerdepth[n]        = z_restrict[int(iii)+1,indices_lon[0][ix],indices_lon[1][ix]]- z_restrict[int(iii),indices_lon[0][ix],indices_lon[1][ix]]
                #    azimuth_angle_iii[n] = azimuth_angle[indices_lon[0][ix],indices_lon[1][ix]]
                    # u,v,w conventions w/ respecto to radar
        
                #    vr1            = u_restrict[int(iii),indices_lon[0][ix],indices_lon[1][ix]]*np.cos(np.deg2rad(elev_iii[n]))*np.sin( np.deg2rad(azimuth_angle_iii[n]) )
                #    vr2            = v_restrict[int(iii),indices_lon[0][ix],indices_lon[1][ix]]*np.cos(np.deg2rad(elev_iii[n]))*np.cos( np.deg2rad(azimuth_angle_iii[n]) )
                #    vr3            = w_restrict[int(iii),indices_lon[0][ix],indices_lon[1][ix]]*np.sin(np.deg2rad(elev_iii[n]))
                #    vradial_iii[n] = vr1 + vr2 + vr3

                    n += 1
                #Zh_rain_weighted[it,indices_lon[0][ix],indices_lon[1][ix]]    = np.nansum(Gain_g*Zh_riii*layerdepth)/np.nansum(Gain_g*layerdepth)  
                #Zh_snow_weighted[it,indices_lon[0][ix],indices_lon[1][ix]]    = np.nansum(Gain_g*Zh_siii*layerdepth)/np.nansum(Gain_g*layerdepth)  
                #Zh_grau_weighted[it,indices_lon[0][ix],indices_lon[1][ix]]    = np.nansum(Gain_g*Zh_giii*layerdepth)/np.nansum(Gain_g*layerdepth)  
                #vr_doppler_weighted[it,indices_lon[0][ix],indices_lon[1][ix]] = np.nansum(Gain_g*vradial_iii*layerdepth)/np.nansum(Gain_g*layerdepth) 
                #del gridpoints, Gain_g, layerdepth

                # ------------------------------------------------------------------ 
                # OPTION 2: INTERPOLATE 
                # a) define a higher resolution grid between zindex_min and zindex_max and interpolate Zr and z_height 
                theta_interp_start = theta_e[int(zindeces_min_bw[it,indices_lon[0][ix],indices_lon[1][ix]]),indices_lon[0][ix],indices_lon[1][ix]]
                theta_interp_end   = theta_e[int(zindeces_max_bw[it,indices_lon[0][ix],indices_lon[1][ix]]),indices_lon[0][ix],indices_lon[1][ix]]
                theta_interp_vals  = np.linspace(theta_interp_start.values, theta_interp_end.values, 20)

                Zr_interp_indx  = np.interp( theta_interp_vals, elev_iii, Zh_riii)
                Zs_interp_indx  = np.interp( theta_interp_vals, elev_iii, Zh_siii)
                Zg_interp_indx  = np.interp( theta_interp_vals, elev_iii, Zh_giii)

                vradial_interp_indx  = np.interp( theta_interp_vals, elev_iii, vradial_iii)

                zgrid_interp_start = z_restrict[int(zindeces_min_bw[it,indices_lon[0][ix],indices_lon[1][ix]]),indices_lon[0][ix],indices_lon[1][ix]]
                zgrid_interp_end   = z_restrict[int(zindeces_max_bw[it,indices_lon[0][ix],indices_lon[1][ix]]),indices_lon[0][ix],indices_lon[1][ix]]                 
                zgrid_interp_vals  = np.linspace(zgrid_interp_start, zgrid_interp_end, 20)
                layerdepth         = zgrid_interp_vals[1]-zgrid_interp_vals[0]             
                
                #plt.plot(Zr_interp_indx, theta_interp_vals, '-oc'); plt.plot(Zh_riii, elev_iii, 'xr')
                #plt.plot(Zr_interp_indx, zgrid_interp_vals/1e3, '-oc'); plt.plot(Zh_riii, zgrid_iii/1e3, 'xr')

                # b) make nans those theta_interp_elev_vals that fall outisde the theta_radar +- bw/2
                theta_interp_vals[np.where(np.logical_or( theta_interp_vals[:] < (theta_radar[it]-(radar_antenna_bw/2)), theta_interp_vals[:] > (theta_radar[it]+(radar_antenna_bw/2))))] = np.nan
                Zr_interp_indx[np.where(np.isnan(theta_interp_vals) == 1)] = np.nan 
                Zs_interp_indx[np.where(np.isnan(theta_interp_vals) == 1)] = np.nan 
                Zg_interp_indx[np.where(np.isnan(theta_interp_vals) == 1)] = np.nan 
               
                vradial_interp_indx[np.where(np.isnan(theta_interp_vals) == 1)] = np.nan 
                               
                # c) apply gaussian equation here too 
                Gain_g     = np.empty(20);    
                for iii in range(20):                   
                    top_sq           = ((theta_interp_vals[iii]-elev_center)/radar_antenna_bw)**2
                    Gain_g[iii]      = np.exp(-4*np.log(4)*top_sq)

                Zh_rain_weighted_intep[it,indices_lon[0][ix],indices_lon[1][ix]] = np.nansum(Gain_g*Zr_interp_indx*layerdepth)/np.nansum(Gain_g*layerdepth)  
                Zh_snow_weighted_intep[it,indices_lon[0][ix],indices_lon[1][ix]] = np.nansum(Gain_g*Zs_interp_indx*layerdepth)/np.nansum(Gain_g*layerdepth)  
                Zh_grau_weighted_intep[it,indices_lon[0][ix],indices_lon[1][ix]] = np.nansum(Gain_g*Zg_interp_indx*layerdepth)/np.nansum(Gain_g*layerdepth)  

                vradial_weighted_intep[it,indices_lon[0][ix],indices_lon[1][ix]] = np.nansum(Gain_g*vradial_interp_indx*layerdepth)/np.nansum(Gain_g*layerdepth)  
                del Gain_g, Zh_riii, elev_iii

elapsed_time = time.time() - start_time
print('END')
print(elapsed_time/60)        
        

f = open('radarSim_output_MORR.pckl', 'wb')
pickle.dump([Zh_rain_weighted_intep, Zh_snow_weighted_intep, Zh_grau_weighted_intep, 
             vradial_weighted_intep, theta_e, r, masked_r, z_FOR_theta_radar, lon_restrict, lat_restrict, Zh_r_out_restrict, 
             Zh_s_out_restrict, Zh_g_out_restrict, z_restrict], f)
f.close()   



#check_Zhr_out    = np.ma.masked_where( r >= max_range+50E3, Zh_r_theta_radar)


#==============================================================================
# for it in range(len(theta_radar)):
#     print(theta_radar[it])
#     fig = plt.figure()
#     ax = anguil_mask.plot()
#     ax.set_extent([-70,-58, -42, -30], crs=ccrs.PlateCarree())
#     plt.contourf(lon,lat,Zh_r_theta_radar[it,:,:],vmin=-60,vmax=60); plt.colorbar();           
#     plt.title('Zh_rain')               
# 
#     fig = plt.figure()
#     ax = anguil_mask.plot()
#     ax.set_extent([-70,-58, -42, -30], crs=ccrs.PlateCarree());
#     plt.contourf(lon,lat,Zh_rain_weighted[it,:,:],vmin=-60,vmax=60); plt.colorbar();
#     plt.title('Zh_rain weigthed w/ Gaussian')
#     
#     fig = plt.figure()
#     ax = anguil_mask.plot()
#     ax.set_extent([-70,-58, -42, -30], crs=ccrs.PlateCarree());
#     plt.contourf(lon,lat,Zh_rain_weighted_intep[it,:,:],vmin=-60,vmax=60); plt.colorbar();
#     plt.title('Zh_rain weigthed w/ Gaussian + INTERP')    
#     
# 
#     fig = plt.figure()
#     ax = anguil_mask.plot()
#     ax.set_extent([-70,-55, -42, -30], crs=ccrs.PlateCarree());
#     diff = Zh_rain_weighted[it,:,:]-Zh_rain_weighted_intep[it,:,:]
#     plt.contourf(lon,lat,diff,vmin=-10,vmax=10); plt.colorbar();
#     plt.title('Zh_rain weigthed w/ Gaussian - Zh_rain weigthed w/ Gaussian + INTERP')
# 
#     fig = plt.figure()
#     ax = anguil_mask.plot()
#     ax.set_extent([-70,-55, -42, -30], crs=ccrs.PlateCarree());
#     diff = Zh_rain_weighted_intep[it,:,:]-Zh_r_theta_radar[it,:,:]
#     plt.contourf(lon,lat,diff,vmin=-20,vmax=20); plt.colorbar();
#     plt.title('Zh_rain weigthed w/ Gaussian INTERP - Zh_rain')
#       
#     
#     fig = plt.figure()
#     ax = anguil_mask.plot()
#     ax.set_extent([-70,-55, -42, -30], crs=ccrs.PlateCarree());
#     plt.contourf(lon,lat,vr_doppler_weighted[0,:,:]); plt.colorbar();
#     plt.title('test vr doppler')
#     
#     
#     fig = plt.figure()
#     ax = anguil_mask.plot()
#     ax.set_extent([-70,-55, -42, -30], crs=ccrs.PlateCarree());
#     plt.contourf(lon,lat,(vradial_weighted_intep[0,:,:])-vr_doppler_weighted[0,:,:]); plt.colorbar();
#     plt.title('test vr doppler')  
#     
#     
#     fig = plt.figure()
#     ax = anguil_mask.plot()
#     ax.set_extent([-70,-55, -42, -30], crs=ccrs.PlateCarree());
#     plt.contourf(lon,lat,(Zh_snow_weighted_intep[0,:,:])); plt.colorbar();
#         
#               
#     fig = plt.figure()
#     ax = anguil_mask.plot()
#     ax.set_extent([-70,-55, -42, -30], crs=ccrs.PlateCarree());
#     plt.contourf(lon,lat,(np.nansum(qs_restrict,0))); plt.colorbar();
#==============================================================================
                
                
# So, we have now x,y,z[zindex] which determine the pixels observed by the radar we are modelling. 
# For example plot volume scan at fixed theta_radar in polar coordinate system. 
# First define z_FOR_theta_radar which holds the height z for each ix,iy pair for
# each of the theta_radar angles 
#==============================================================================
# # poner lo de los indices sino es re lento!
z_FOR_theta_radar      =  np.zeros((len(theta_radar),dim2,dim3));       z_FOR_theta_radar[:]=np.nan
z_FOR_theta_radar_GEOM =  np.zeros((len(theta_radar),dim2,dim3));       z_FOR_theta_radar_GEOM[:]=np.nan

for ix in np.arange(np.shape(indices_lon)[1]):
     for it in range(len(theta_radar)):
         z_FOR_theta_radar[it,indices_lon[0][ix], indices_lon[1][ix]] = np.nan
         z_FOR_theta_radar_GEOM[it,indices_lon[0][ix], indices_lon[1][ix]] = np.nan
         if (zindeces[it,indices_lon[0][ix], indices_lon[1][ix]] != 0):
             if (np.isnan(value[it,indices_lon[0][ix], indices_lon[1][ix]] )!= 1):
                 integer_index=int(zindeces[it,indices_lon[0][ix], indices_lon[1][ix]])
                 z_FOR_theta_radar[it,indices_lon[0][ix], indices_lon[1][ix]]      = z_restrict[integer_index,indices_lon[0][ix], indices_lon[1][ix]]
                 #z_FOR_theta_radar_GEOM[it,indices_lon[0][ix], indices_lon[1][ix]] = z_restrict[geom_integer_index,indices_lon[0][ix], indices_lon[1][ix]]
# 
#==============================================================================
#==============================================================================
#Check for example for theta_radar[0] meaning 0.5 degrees observation angle! 
#For each [ix,iy] pair it searched for the iz that corresponds to the closesest 
#observation angle. 
#NOTE: First column in plots is z_index. Not altitude!!! 
# for it in range(len(theta_radar)):
#     print(theta_radar[it])
#     fig = plt.figure()
#     #xx = np.linspace(0, len(zindeces[it,:,:].ravel()), len(zindeces[it,:,:].ravel()))  # 100 evenly-spaced values for [ix,iy] for each angle it
#  
#     # Plot iz index values ! 
#     plt.subplot(131)
#     plt.plot( zindeces[it,indices_lon[0][:],indices_lon[1][:]], 'xr')
#     plt.axis([5000, 31000, 0, 60])
#     #plt.xticks(np.arange(xmin, xmax, 40))
#     plt.ylabel('Selected z index')
#  
#     # Plot theta values
#     plt.subplot(132)
#     plt.plot( value[it,indices_lon[0][:],indices_lon[1][:]], 'xr')
#     plt.ylabel('Selected Radar Angle (Theta)')
#     
#     plt.subplot(133)
#     plt.plot( value_min_bw[it,indices_lon[0][:],indices_lon[1][:]], 'xr')
#     plt.ylabel('Min bw Selected Radar Angle (Theta)')
#     plt.tight_layout()
#     plt.show()  
#==============================================================================


#==============================================================================
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import pyplot as plt
# import numpy as np
# # Plot altitudes for each radar angle elevation  (play with difference treshold when looking for angle diff above)
# fig = plt.figure()
# for it in range(len(theta_radar)):
#     print(theta_radar[it])
#     jj = np.linspace(0, 1000, len(z_FOR_theta_radar[it,:,:].ravel()))  # 100 evenly-spaced values for [ix,iy] for each angle it
#     # Plot z values ! 
#     plt.subplot(221)
#     plt.plot(jj.ravel(),z_FOR_theta_radar[it,:,:].ravel()/1E3, 'xr')
#     plt.ylabel('Altitude')
#     plt.axis([400, 600, 0, 40])
#     plt.show()
# 
# # Plot surface plots for propagation of radar rays for each theta observation
# 
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
# import numpy as np
# 
# for it in range(len(theta_radar)):
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
#     # Plot the surface.
#     ax.plot_wireframe(x/1000, y/1000, z_FOR_theta_radar[it,:,:]/1000, rstride=10, cstride=10)
#     ax.set_zlim(0, 20)
#     ax.set_xlim(-300, 300)
#     ax.set_ylim(-300, 300)
#     ax.set_xlabel('x (km)')
#     ax.set_ylabel('y (km)')
#     ax.set_zlabel('Altitude (km)')
#     ax.set_title("Radar zenith angle =%2.1f\n" % (theta_radar[it]) )            
#     plt.show()
# 
# # Compare ray propagation with and without curvature 
# # Need to find index of ix where radar is
# 
# XX = x.ravel()
# YY = y.ravel()
# ZZ_curvature = (z_FOR_theta_radar[1,:,:]/1000).ravel()
# ZZ = (z_FOR_theta_radar_GEOM[1,:,:]/1000).ravel()
# 
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(XX/1000, YY/1000, ZZ_curvature, c='r', marker='o')
# ax.scatter(XX/1000, YY/1000, ZZ, c='b', marker='x')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.view_init(elev=10., azim=50)
# plt.show()
# 
#==============================================================================





