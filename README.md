# SimRad.AR
Herramienta capaz de simular las variables observadas por los radares de banda C polarimétricos doppler del Sistema Nacional de Radares Meteorológicos (SINERAME) de Argentina, a partir de las salidas de modelos numéricos en escala de nubes, particularmente del Weather Research and Forecasting model (WRF) el cual es utilizado operativamente en el Servicio Meteorológico Nacional. Asimismo, puede simular las observaciones de otros radares como INTA-Anguil (La Pampa), INTA-Paraná (Entre Rıos). 
El concepto central del simulador es la transformación de las variables meteorológicas simuladas por el modelo WRF (e.g., perfiles de temperatura, perfiles de rain mixing ratio, etc) a las variables observadas por un radar meteorológico (e.g., Reflectividad Horizontal Zh, Reflectividad Diferencial Zdr=(Zh-Zv), Specific Differential Phase Kdp, atenuación Ai, Velocidad Doppler). El modelo desarrollado en python toma en cuenta los siguientes puntos:
* Transformación de las variables de WRF en coordenadas geográficas (latitud, longitud, altura vertical) a coordenadas cartesianas (x, y, z) centradas en el radar meteorológico a ser modelado (el modelo cuenta con la información necesaria para simular los radares disponibles en la red del Sistema Nacional de Radares Meteorológicos, Sinarame)
* Modelar la propagación del haz del radar en la atmósfera teniendo en cuenta los efectos de la refracción atmosférica.
* Modelar la ganancia de la antena y el ancho del haz de los radares meteorológicos modelados.
* Transformar las variables atmosféricas de WRF (e.g., Qr - rain mixing ratio en kg/kg) en las variables polarimétricas observables de radares meteorológicos (e.g., Zh - reflectividad horizontal en dBz). En este último punto es clave incorporar de manera consistente con WRF las parametrizaciones de la microfísica correspondiente y un cálculo adecuado de las propiedades ópticas de las especies de la misma. El modelo actualmente puede utilizar las salidas del WRF con las parametrizaciones microfisicas WRF-WSM6 (Hong and Lim, 2006) y WRF-Morrison (Morrison et al., 2009). 
Para correr adecuandamente este simulador, las siguientes herramientas son necesarias: 
* PyTMatrix (https://github.com/jleinonen/pytmatrix) 
* Py-ARTS (http://arm-doe.github.io/pyart/)
* wrf-python (https://wrf-python.readthedocs.io/en/latest/) 


Los radares en la base de datos son: 

| Input number  | Radar  (location) |
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
