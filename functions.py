def ncdump(nc_fid, verb=True):
    '''
    ncdump outputs dimensions, variables and their attribute information.
    The information is similar to that of NCAR's ncdump utility.
    ncdump requires a valid instance of Dataset.

    Parameters
    ----------
    nc_fid : netCDF4.Dataset
        A netCDF4 dateset object
    verb : Boolean
        whether or not nc_attrs, nc_dims, and nc_vars are printed

    Returns
    -------
    nc_attrs : list
        A Python list of the NetCDF file global attributes
    nc_dims : list
        A Python list of the NetCDF file dimensions
    nc_vars : list
        A Python list of the NetCDF file variables
    '''
    def print_ncattr(key):
        """
        Prints the NetCDF file attributes for a given key

        Parameters
        ----------
        key : unicode
            a valid netCDF4.Dataset.variables key
        """
        try:
            print ("\t\ttype:", repr(nc_fid.variables[key].dtype))
            for ncattr in nc_fid.variables[key].ncattrs():
                print ('\t\t%s:' % ncattr,\
                      repr(nc_fid.variables[key].getncattr(ncattr)))
        except KeyError:
            print ("\t\tWARNING: %s does not contain variable attributes" % key)

    # NetCDF global attributes
    nc_attrs = nc_fid.ncattrs()
    if verb:
        print ("NetCDF Global Attributes:")
        for nc_attr in nc_attrs:
            print ('\t%s:' % nc_attr, repr(nc_fid.getncattr(nc_attr)))
    nc_dims = [dim for dim in nc_fid.dimensions]  # list of nc dimensions
    # Dimension shape information.
    if verb:
        print ("NetCDF dimension information:")
        for dim in nc_dims:
            print ("\tName:", dim)
            print ("\t\tsize:", len(nc_fid.dimensions[dim]))
            print_ncattr(dim)
    # Variable information.
    nc_vars = [var for var in nc_fid.variables]  # list of nc variables
    if verb:
        print ("NetCDF variable information:")
        for var in nc_vars:
            if var not in nc_dims:
                print ('\tName:', var)
                print ("\t\tdimensions:", nc_fid.variables[var].dimensions)
                print ("\t\tsize:", nc_fid.variables[var].size)
                print_ncattr(var)
    return nc_attrs, nc_dims, nc_vars


def create_logarithmic_scaled_vectors(vec_min,vec_max, dim):
    
    import numpy as np

    vec = np.logspace(np.log10(vec_min), np.log10(vec_max), num=dim, endpoint=True, base=10.0);

    return [vec]

def interpLevel(grid,value,data,interp):
    """
    Interpolate 3d data to a common z coordinate.

    Can be used to calculate the wind/pv/whatsoever values for a common
    potential temperature / pressure level.

    grid : numpy.ndarray
       The grid. For example the potential temperature values for the whole 3d
       grid.

    value : float
       The common value in the grid, to which the data shall be interpolated.
       For example, 350.0

    data : numpy.ndarray
       The data which shall be interpolated. For example, the PV values for
       the whole 3d grid.

    kind : str
       This indicates which kind of interpolation will be done. It is directly
       passed on to scipy.interpolate.interp1d().

    returns : numpy.ndarray
       A 2d array containing the *data* values at *value*.

    """
    import numpy as np
    from scipy import interpolate
    import numpy.ma as ma

    ret = np.zeros_like(data[0, :, :])
    for yIdx in range(grid.shape[1]):
        for xIdx in range(grid.shape[2]):
            # check if we need to flip the column
            if grid[0,yIdx,xIdx] > grid[-1,yIdx,xIdx]:
                ind = -1
            else:
                ind = 1
            f = interpolate.interp1d(grid[::ind,yIdx,xIdx], \
                data[::ind,yIdx,xIdx], \
                kind=interp)
            ret[yIdx,xIdx] = f(value)

    #ret1 = ma.masked_where(ret <=10, ret)
    mask_ret = ma.masked_invalid(ret)

    return mask_ret

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    import matplotlib.pyplot as plt
    import numpy as np

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def return_constants():

    const = dict([('q_rain_min', 1e-8),      # input 1
                  ('q_rain_max', 0.013),               # input 2
                  ('qn_rain_min', 1E-8),               # input 3
                  ('qn_rain_max', 2.6e6),              # input 4
                  ('q_snow_min',  1e-8),      # input 5
                  ('q_snow_max',  0.02),     # input 6
                  ('qn_snow_min', 1E-8),      # input 3
                  ('qn_snow_max', 2.6e6),     # input 4   
                  ('q_grau_min',  1e-8),      # input 5
                  ('q_grau_max',  0.02),     # input 6
                  ('qn_grau_min', 1E-8),      # input 
                  ('qn_grau_max', 2.6e6),     # input 4   
                  ('q_ice_min',  1E-8),      # input 5
                  ('q_ice_max',  0.1),     # input 6 0.02
                  ('qn_ice_min', 1E-8),      # input 
                  ('qn_ice_max', 2.6e6),     # input 4   
                  ('q_clou_min',  1e-10),
                  ('q_clou_max',  0.013),
                  ('dim_ice',  200),
                  ('dim', 200)])          
    return const


def colormaps(variable):
    """
    Choose colormap for a radar variable

    variable : str
       Radar variable to define which colormap to use. (e.g. ref,
       dv, zdr..) More to come

    returns : matplotlib ColorList
       A Color List for the specified *variable*.

    """
    import matplotlib.colors as colors

    # Definicion de las disntitas paletas de colores:
    nws_ref_colors =([ [ 1.0/255.0, 159.0/255.0, 244.0/255.0],
                    [  3.0/255.0,   0.0/255.0, 244.0/255.0],
                    [  2.0/255.0, 253.0/255.0,   2.0/255.0],
                    [  1.0/255.0, 197.0/255.0,   1.0/255.0],
                    [  0.0/255.0, 142.0/255.0,   0.0/255.0],
                    [253.0/255.0, 248.0/255.0,   2.0/255.0],
                    [229.0/255.0, 188.0/255.0,   0.0/255.0],
                    [253.0/255.0, 149.0/255.0,   0.0/255.0],
                    [253.0/255.0,   0.0/255.0,   0.0/255.0],
                    #[212.0/255.0,   0.0/255.0,   0.0/255.0],
                    [188.0/255.0,   0.0/255.0,   0.0/255.0],
                    [248.0/255.0,   0.0/255.0, 253.0/255.0],
                    [152.0/255.0,  84.0/255.0, 198.0/255.0]
                    ])

    nws_zdr_colors = ([ [  1.0/255.0, 159.0/255.0, 244.0/255.0],
                    [  3.0/255.0,   0.0/255.0, 244.0/255.0],
                    [  2.0/255.0, 253.0/255.0,   2.0/255.0],
                    [  1.0/255.0, 197.0/255.0,   1.0/255.0],
                    [  0.0/255.0, 142.0/255.0,   0.0/255.0],
                    [253.0/255.0, 248.0/255.0,   2.0/255.0],
                    [229.0/255.0, 188.0/255.0,   2.0/255.0],
                    [253.0/255.0, 149.0/255.0,   0.0/255.0],
                    [253.0/255.0,   0.0/255.0,   0.0/255.0],
                    [188.0/255.0,   0.0/255.0,   0.0/255.0],
                    [152.0/255.0,  84.0/255.0, 198.0/255.0]
                    ])

    nws_dv_colors = ([  [0,  1,  1],
                    [0,  0.966666638851166,  1],
                    [0,  0.933333337306976,  1],
                    [0,  0.899999976158142,  1],
                    [0,  0.866666674613953,  1],
                    [0,  0.833333313465118,  1],
                    [0,  0.800000011920929,  1],
                    [0,  0.766666650772095,  1],
                    [0,  0.733333349227905,  1],
                    [0,  0.699999988079071,  1],
                    [0,  0.666666686534882,  1],
                    [0,  0.633333325386047,  1],
                    [0,  0.600000023841858,  1],
                    [0,  0.566666662693024,  1],
                    [0,  0.533333361148834,  1],
                    [0,  0.5,  1],
                    [0,  0.466666668653488,  1],
                    [0,  0.433333337306976,  1],
                    [0,  0.400000005960464,  1],
                    [0,  0.366666674613953,  1],
                    [0,  0.333333343267441,  1],
                    [0,  0.300000011920929,  1],
                    [0,  0.266666680574417,  1],
                    [0,  0.233333334326744,  1],
                    [0,  0.200000002980232,  1],
                    [0,  0.16666667163372,   1],
                    [0,  0.133333340287209,  1],
                    [0,  0.100000001490116,  1],
                    [0,  0.0666666701436043, 1],
                    [0,  0.0333333350718021, 1],
                    [0,  0,  1],
                    [0,  0,  0],
                    [0,  0,  0],
                    [0,  0,  0],
                    [0,  0,  0],
                    [1,  0,  0],
                    [1,  0.0322580635547638, 0],
                    [1,  0.0645161271095276, 0],
                    [1,  0.0967741906642914, 0],
                    [1,  0.129032254219055,  0],
                    [1,  0.161290317773819,  0],
                    [1,  0.193548381328583,  0],
                    [1,  0.225806444883347,  0],
                    [1,  0.25806450843811,   0],
                    [1,  0.290322571992874,  0],
                    [1,  0.322580635547638,  0],
                    [1,  0.354838699102402,  0],
                    [1,  0.387096762657166,  0],
                    [1,  0.419354826211929,  0],
                    [1,  0.451612889766693,  0],
                    [1,  0.483870953321457,  0],
                    [1,  0.516129016876221,  0],
                    [1,  0.548387110233307,  0],
                    [1,  0.580645143985748,  0],
                    [1,  0.612903237342834,  0],
                    [1,  0.645161271095276,  0],
                    [1,  0.677419364452362,  0],
                    [1,  0.709677398204803,  0],
                    [1,  0.74193549156189,   0],
                    [1,  0.774193525314331,  0],
                    [1,  0.806451618671417,  0],
                    [1,  0.838709652423859,  0],
                    [1,  0.870967745780945,  0],
                    [1,  0.903225779533386,  0],
                    [1,  0.935483872890472,  0],
                    [1,  0.967741906642914,  0],
                    [1,  1,  0]   ])


    cmap_nws_ref = colors.ListedColormap(nws_ref_colors)
    cmap_nws_zdr = colors.ListedColormap(nws_zdr_colors)
    cmap_nws_dv = colors.ListedColormap(nws_dv_colors)

    if variable == 'ref':
       return cmap_nws_ref

    if variable == 'zdr':
       return cmap_nws_zdr

    if variable == 'dv':
       return cmap_nws_dv

def pyplot_rings(lat_radar,lon_radar,radius):
    """
    Calculate lat-lon of the maximum range ring

    lat_radar : float32
       Radar latitude. Positive is north.

    lon_radar : float32
       Radar longitude. Positive is east.

    radius : float32
       Radar range in kilometers.

    returns : numpy.array
       A 2d array containing the 'radius' range latitudes (lat) and longitudes (lon)

    """
    import numpy as np

    R=12742./2.
    m=2.*np.pi*R/360.
    alfa=np.arange(-np.pi,np.pi,0.0001)

    nazim  = 360.0
    nbins  = 480.0
    binres = 0.5

    #azimuth = np.transpose(np.tile(np.arange(0,nazim,1), (int(nbins),1)))
    #rangos  = np.tile(np.arange(0,nbins,1)*binres, (int(nazim),1))
    #lats    = lat_radar + (rangos/m)*np.cos((azimuth)*np.pi/180.0)
    #lons    = lon_radar + (rangos/m)*np.sin((azimuth)*np.pi/180.0)/np.cos(lats*np.pi/180)

    lat_radius = lat_radar + (radius/m)*np.sin(alfa)
    lon_radius = lon_radar + ((radius/m)*np.cos(alfa)/np.cos(lat_radius*np.pi/180))

    return lat_radius, lon_radius

def rmaxRing(lat_radar,lon_radar,Rmax):
    """
    Calculate lat-lon of the maximum range ring

    lat_radar : float32
       Radar latitude. Positive is north.

    lon_radar : float32
       Radar lngitude. Positive is east.

    rmax : float32
       Radar maximum range in kilometers.

    returns : numpy.array
       A 2d array containing the maximum range latitudes (lat_Rmax) and longitudes (lon_Rmax)

    """
    import numpy as np

    R=12742./2.
    m=2.*np.pi*R/360.
    alfa=np.arange(-np.pi,np.pi,0.0001)

    nazim  = 360.0
    nbins  = 480.0
    binres = 0.5

    # A continuacion calculo las lats y lons alrededor del radar
    azimuth = np.transpose(np.tile(np.arange(0,nazim,1), (nbins,1)))
    rangos  = np.tile(np.arange(0,nbins,1)*binres, (nazim,1))
    lats    = lat_radar + (rangos/m)*np.cos((azimuth)*np.pi/180.0)
    lons    = lon_radar + (rangos/m)*np.sin((azimuth)*np.pi/180.0)/np.cos(lats*np.pi/180)

    lat_Rmax = lat_radar + (Rmax/m)*np.sin(alfa)
    lon_Rmax = lon_radar + ((Rmax/m)*np.cos(alfa)/np.cos(lat_Rmax*np.pi/180))

    return lat_Rmax, lon_Rmax
    
    
    
    
    
    
    
    
    
