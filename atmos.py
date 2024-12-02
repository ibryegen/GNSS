# -*- coding: utf-8 -*-
"""
Created on Fri May 10 22:08:43 2024

@author: win10
"""

from math import radians
import numpy as np
from sat_pos import emist, sat_pos
from blhandlocal import global2local, xyz2blh
from Ion_Klobuchar import Ion_Klobuchar
from trop_SPP import trop_SPP




'''

3.03           OBSERVATION DATA    M: MIXED           


4239146.6414  2886967.1245  3778874.4800                  APPROX POSITION XYZ





2024 03 01 05 48  0.0000000  0 39
G05  23040745.421   121079888.30407  23040746.414    94347957.97307  23040745.835    94347950.97207
'''

'''

SP3

4.45 PG05  20182.196059  -7288.695155  15538.190449   -163.705736  5  4  4  61       

5.00 PG05  21819.288854  -6884.883905  13392.942554   -163.706702  4  5  4  52       

5.15 PG05  23218.081895  -6590.924045  11015.916850   -163.708521  4  5  4  77       

5.30 PG05  24348.690126  -6377.598372   8448.649676   -163.709459  3  5  4  76       

5.45 PG05  25188.961266  -6211.209684   5735.789326   -163.710561  3  4  4  78       

5.48 - EPOCH

6.00 PG05  25725.208827  -6055.085040   2924.296419   -163.712115  3  4  5  77       

6.15 PG05  25952.624873  -5871.189141     62.617636   -163.713157  3  3  5  83       

6.30 PG05  25875.360816  -5621.782578  -2800.151763   -163.714706  3  3  4  92       

6.45 PG05  25506.277419  -5271.060284  -5615.102084   -163.715970  2  3  4  76       

7.00 PG05  24866.377608  -4786.707510  -8334.326397   -163.717122  3  3  4  83       

'''

'''
             NAVIGATION DATA                         

    0.2794D-07  0.7451D-08 -0.1192D-06  0.5960D-07          ION ALPHA           
    0.1372D+06 -0.3277D+05 -0.6554D+05 -0.5898D+06          ION BETA            

'''















def atmos(doy, trec, trecw, c1, rec, sp3, alpha, beta, fpos):
    
    
    fpos = sat_pos(trec, c1, sp3, rec)
    
    # Receiver coordinates
    x = rec[0]
    y = rec[1]
    z = rec[2]
    
    # Convert receiver coordinates from XYZ to BLH (latitude, longitude, height)
    blh = xyz2blh(x, y, z)
    
    # Define latitude and longitude in radians
    lon = radians(blh[0])
    lat = radians(blh[1])
    
    # Convert global satellite position to local coordinates
    result = global2local(fpos, rec)
    
    # Azimuth, zenith, and elevation angles in radians
    azm = radians(result[0])
    zen = radians(result[1])
    elv = radians(90) - zen
    
    # Calculate ionospheric delay
    IonD = Ion_Klobuchar(lat, lon, elv, azm, alpha, beta, trecw)
    
    # Use latitude and ellipsoidal height for tropospheric delay calculation
    l1 = blh[1]
    H = blh[2]
    TrD, TrW, ME = trop_SPP(l1, doy, H, elv)
    TrD = ME * TrD
    TrW = ME* TrW
    
    # Convert azimuth and zenith angles back to degrees
    az = result[0]
    zen = result[1]
    slantd = result[2]
    
    
    return az, zen, slantd, IonD, TrD, TrW
    
    
    
    
    

    
    
    
    
    
    
    


