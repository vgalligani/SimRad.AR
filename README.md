# SimRad.AR


RESUMEN
Se presenta un simulador de radar meteorológico polorimétrico robusto, rápido y de uso libre para ser utilizado en conjunto con el modelo Weather Research and Forecasting model (WRF). Este operador observacional es capaz de transformar las variables meteorológicas simuladas por el modelo WRF a las variables observadas por un radar meteorológico de superficie de banda C, banda S y banda X incluyendo la Reflectividad Horizontal, Reflectividad Diferencial, Diferencial de Fase Específico, atenuación y la Velocidad Doppler. El simulador cuenta con una agenda para la simulación de los radares de la red del Sistema Nacional de Radares Meteorológicos (SINARAME) de la Argentina. El presente trabajo presenta además una comparación con otro modelo similar, el Cloud Resolving model Radar SIM-ulator (CR-SIM), para casos de estudio donde los radares meteorológicos existentes en la Argentina observen fenómenos convectivos. 

DESCRIPCION DETALLADA
Herramienta capaz de simular las variables observadas por los radares de banda C polarimétricos doppler del Sistema Nacional de Radares Meteorológicos (SINERAME) de Argentina, a partir de las salidas de modelos numéricos en escala de nubes, particularmente del Weather Research and Forecasting model (WRF) el cual es utilizado operativamente en el Servicio Meteorológico Nacional. Asimismo, puede simular las observaciones de otros radares como INTA-Anguil (La Pampa), INTA-Paraná (Entre Rıos). Asi como tambien de radares "sinteticos" definiendo su ubicacion (lat, lon), bandwidth del haz, y frecuencia en la que opera. 

El concepto central del simulador es la transformación de las variables meteorológicas simuladas por el modelo WRF (e.g., perfiles de temperatura, perfiles de rain mixing ratio, etc) a las variables observadas por un radar meteorológico (e.g., Reflectividad Horizontal Zh, Reflectividad Diferencial Zdr=(Zh-Zv), Diferencial de Fase Específico Kdp, atenuación Ai, Velocidad Doppler). El modelo desarrollado en python toma en cuenta los siguientes puntos:
* Transformación de las variables de WRF en coordenadas geográficas (latitud, longitud, altura vertical) a coordenadas cartesianas (x, y, z) centradas en el radar meteorológico a ser modelado (el modelo cuenta con la información necesaria para simular los radares disponibles en la red del Sistema Nacional de Radares Meteorológicos, Sinarame)
* Modelar la propagación del haz del radar en la atmósfera teniendo en cuenta los efectos de la refracción atmosférica.
* Modelar la ganancia de la antena y el ancho del haz de los radares meteorológicos modelados.
* Transformar las variables atmosféricas de WRF (e.g., Qr - rain mixing ratio en kg/kg) en las variables polarimétricas observables de radares meteorológicos (e.g., Zh - reflectividad horizontal en dBz). En este último punto es clave incorporar de manera consistente con WRF las parametrizaciones de la microfísica correspondiente y un cálculo adecuado de las propiedades ópticas de las especies de la misma. El modelo actualmente puede utilizar las salidas del WRF con las parametrizaciones microfisicas WRF-WSM6 (wrf_microphysics_scheme = 6, Hong and Lim, 2006), WRF-Morrison (wrf_microphysics_scheme = 10, Morrison et al., 2009) y WRF-Thompson (wrf_microphysics_scheme = 8, Thompson et al., 2009) 
* Todas las salidas polarimetricas son para un radar de banda S, banda C y banda X.

Para correr adecuandamente este simulador, las siguientes herramientas son necesarias: 
* PyTMatrix (https://github.com/jleinonen/pytmatrix) 
* Py-ARTS (http://arm-doe.github.io/pyart/)
* wrf-python (https://wrf-python.readthedocs.io/en/latest/) 


Los radares en la base de datos son: 

| radar_no  | Radar  (location) |
| ------------- | ----------------- | 
|  0 |     (Anguil)  |
|  1 | AR7 (Parana)    |
|  2 | RMA5 (Bernardo de Irigoyen)  |
|  3 | RMA1 (CORDOBA)   |
|  4 | RMA10 (ESPORA)  |
|  5 | RMA2 (EZEIZA)  |
|  6 | RMA3 (LAS LOMITAS)  |
|  7 | RMA6 (MAR DEL PLATA)  |
|  8 | RMA8 (MERCEDES)  |
|  9 | RMA7 (NEUQUEN)  |
|  10 | AR5 (PERGAMINO)   |
|  11 | RMA4 (RESISTENCIA)  |
|  12 | RMA9 (RIO GRANDE)  |
|  13 | RMA11 (TERMAS RIO HONDO)  |
|  14 | RMA0 (BARILOCHE)  |
TABLE I

La funcion central de este simulador es wrf2radar que requiere de:

wrf2radar(radar_no, wrf_microphysics_scheme, max_range, mode, ncfile, ftable) donde, 

max_range               = ver tabla 1 
wrf_microphysics_scheme = 6 (WSM6), 10 (MORR), 8 (THOM)
max_range               = rango del radar, sugeridos 120 km o 240 km
mode                    = 0 (central haz) , 1 (gaussiana del ancho del haz)
ncfile                  = wrf output file
ftable                  = lookuptable file (generada con generate_lookups_WSM6, generate_lookups_MORR, etc)









