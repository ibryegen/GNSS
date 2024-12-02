# -*- coding: utf-8 -*-

# GMT312 - REFERENCE COORDINATE SYSTEMS - ASSIGNMENT I
# Author: @IBRAHIM YEGEN 05.04.2024

#  The purpose of the "xyz2blh" function is to convert x, y and z cartesian coordinate inputs 
# to ellipsoid coordinates with iterative solution. The iterations are repeated until the change
# between two successive values of 𝜑("fi") is smaller than the precision of 10−12 in degrees.
# "fi" is here meaning to "latitude" and 10^-12 will be threshold value. We need them because 
# we solve repeatedly until the change between two value smaller than threshold. We do not know
# how many times will it repedeat so  here "while loop" will be useful. Some variables dependent
# to "fi" so should be they updated with the new value of 𝜑["fi"] at each iteration.

# Function inputs are:                                 Function output is:
    # x [cartesian coordinate system]                   # 𝜑 - ellipsoidal latitude
    # y [cartesian coordinate system]                   # 𝜆 - ellipsoidal longitude
    # z [cartesian coordinate system]                   # ℎ - ellipsoidal height

import math
import numpy as np #for matrices and multiplications
import math #for trigonometric functions
from math import radians #for trigonometric functions

# WGS84 ELLIPSOID PARAMETERS
a = 6378137.0 # semi-major axis
b = 6356752.314245 # semi-minor-axis


def xyz2blh(x,y,z):#inputs are: x, y, and z.
    f = (a-b) / a # f is the flattening of the ellipsoid
    e2 = ((a**2) - (b**2)) / a**2 #  e2 is the square of the eccentricity of the ellipsoid
    f_s = f * (2 - f) # f_s is the square of the flattening.
    
    lambda0 = math.atan2(y,x) #  it calculates the longitude lambda0 using the atan2 function.
    xy = math.sqrt(x**2 + y**2) # x**2 + y**2 will be using multiple times so this is a useful variable.
    fi = math.atan2(z, xy * (1-e2)) # initial estimate of the latitude if h=0
   
    
    delta_fi = 1 # h=0, h=1, h=2, h=3, .... h=n.
    fi_d = 0 # new fi variable actually.
     
    while abs(delta_fi) > 10**-12: #That's the loop which is required to iterative solution. 
        
        if 1-e2 * (math.sin(fi)**2) < 0: # checks if the value less than zero. We must use this here 
                                         #because after some calculations it comes less than zero. 
                                         # The reason was square root inside must be equal or larger than 0
                                         # then it cannot be calculated in real numbers if value < 0. [f*(2-f) will be used.]
           
            N =  a / math.sqrt(1 - f_s * math.sin(fi)**2) #calculates N for each fi value. N is adius of curvature in the prime vertical.
            h = (xy / math.cos(fi) ) - N  #calculates h (ellipsoidal height), for each N and fi values. 
            
            
            fi_d = math.atan2(z, xy *(1 - f_s * (N / (N+h) ) )) #calculates new fi variable. fi(ellipsoidal latitude).
            delta_fi = fi_d - fi #calculates change between fi and fi_d.
            fi = fi_d # After many loop, last loop was the succesive, so last equivalence for fi.
            
            
            
            

        else: #the standard iterative process.
            N = a / math.sqrt(1- e2 * (math.sin(fi)**2) ) 
            h = (xy / math.cos(fi) ) - N
            
            
            fi_d = math.atan2(z, xy * (1 - e2 *  (N / (N+h) )))
            delta_fi = fi_d - fi
            fi = fi_d
            
            
            
            
    return [math.degrees(lambda0) , math.degrees(fi) ,h] #the function converts the latitude and longitude from 
                                                                             #radians to degrees and returns the latitude, longitude,
                                                                             #and height(rounded) as a tuple.
                                                                             
                                                                             



#   The purpose of the code below is to define a function that returns coordinates from global ellipsoidal system to local ellipsoidal system.
# The inputs of the functions are the P (target point) and R (topocenter) points whose coordinates are known in the global.
# Matrix multiplications are needed for operations, and thus angles are needed to rotate the axes. These angles are lambda 
# and fi angles, and these angles can be obtained with the xyz2blh function. The azimuth and zenith angles can be found from 
# the x, y and z coordinates of the ΔX′ coordinate matrix ([x,y,z]) obtained after matrix multiplications.
#                                                         Function output is:
                                                        # AZIMUTH, ZENITH, SLANT RANGE
 

def global2local(P,R):#inputs are P and R
    x = R[0]
    y = R[1]
    z = R[2]
    blh = xyz2blh(x, y, z) #Topocenter point in ellipsoidal system(𝜑, 𝜆, ℎ)
    
    lambda0 = radians(blh[0]) #define lambda angle
    fi0 = radians(blh[1]) # define phi angle
    
    A = np.array([[-math.sin(fi0) * math.cos(lambda0), -math.sin(lambda0), math.cos(fi0) *  math.cos(lambda0)], # A matrix is results of multiplication S2, R3(180-𝜆) and R2(90-𝜑).
                  [-math.sin(fi0)*math.sin(lambda0), math.cos(lambda0), math.cos(fi0) * math.sin(lambda0)],
                  [math.cos(fi0), 0, math.sin(fi0)]])
    

    A_inv = np.linalg.inv(A) #from global to local we need tranpose or inverse of A matrix. This method takes A matrix inverse.
  
    delta = P - R # delta x
    
    prime = np.dot(A_inv, delta) # multiplication of A' and delta x 
    
    slant_range = np.linalg.norm(delta) # slant range is distance between P and R.
    
    
    # In the Surveying, in the 2nd Preliminary Computation we know the calculation has valid calculations.
    # The calculated azimuth angle is added 180 or 360 degrees (200 or 400 grads) according to the y and x signs.
    # Here the conditions for the x and y signs are defined
    
    if prime[0]>0 and prime[1]>0: # If X>0 and Y>0 no addition.
        tana = prime[1]/prime[0]
        azimuth = math.atan(tana)
        azim = math.degrees(azimuth)
        
    elif prime[0]<0 and prime[1]>0: # If X<0 and Y>0 180 degree(200 grad) addition.
        tana = prime[1]/prime[0]
        azimuth = math.atan(tana)
        azim = math.degrees(azimuth) + 180
        
    elif prime[0]<0 and prime[1]<0: # If X<0 and Y<0 180 degree(200 grad) addition.
        tana = prime[1]/prime[0]
        azimuth = math.atan(tana)
        azim = math.degrees(azimuth) + 180
        
    elif prime[0]>0 and prime[1]<0: # If X>0 and Y<0 360 degree(400 grad) addition.
        tana = prime[1]/prime[0]
        azimuth = math.atan(tana)
        azim = math.degrees(azimuth) + 360
        
    zenith = math.atan(prime[2]/math.sqrt(prime[0]**2 + prime[1]**2)) #Compute the complement of the zenith angle.
    zen = 90 - math.degrees(zenith) # find zenith angle substracted from 90 degree.
    
    return [azim, zen, slant_range]


