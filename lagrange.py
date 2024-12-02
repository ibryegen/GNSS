# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 12:49:46 2024

@author: win10
"""

import numpy as np
from numpy import mod
from math import cos, sin, sqrt, atan
import matplotlib.pyplot as plt


# Additionally, a function for 9th order Lagrange interpolation is required to interpolate any variable
# based on ten data points for a given epoch. This function should be defined as follows:

# Function Name: lagrange
# Inputs:

# eph - The epoch for which interpolation is required.
# dat - A 10x2 matrix (or corresponding vector), including time tags (t) for the grid points 
#(1st column) and variable values at these grid points (2nd column).

# Output:

# out - The interpolated value for the variable."
# This description sets the groundwork for coding these functions, focusing on calculating satellite
# positions using precise ephemerides and interpolating data points using the Lagrange method.

"""


PG03   9628.949997  21814.881582 -11969.942958    224.670647  6  7  5  94       2.30

PG03   8481.123026  20878.991916 -14297.485729    224.691765  6  7  5  95       2.45

PG03   7096.256174  19844.779369 -16382.503957    224.712887  5  7  5  79       3.00

PG03   5481.955199  18756.228894 -18189.801158    224.734028  5  7  5 101       3.15

PG03   3654.395151  17656.347758 -19688.852155    224.755158  5  7  5  88       3.30

3.37 (13020 -seconds of day)

PG03   1637.895801  16585.533203 -20854.289284    224.776388  5  7  4  96       3.45

PG03   -535.806397  15580.051671 -21666.309540    224.797548  6  7  4  75       4.00

PG03  -2828.563847  14670.684859 -22110.997700    224.818685  6  6  5  68       4.15

PG03  -5197.065206  13881.592002 -22180.561141    224.839862  6  6  5  74       4.30

PG03  -7594.280478  13229.430245 -21873.472736    224.861018  6  6  5  72       4.45

"""




"""
 form a function for the Lagrange interpolation which is designed as follows:
[out]=lagrange(eph, dat)
This function should interpolate any variable with 10 grid points (9th order interpolation) for a
given epoch with Lagrange method.
Inputs:
eph - Interpolation epoch
dat - 10x2 matrix (or corresponding vector), including time tags (t) for the grid points (1st column)
and variable values at grid points (2nd column)
Output:
out - Interpolated value for the variable
"""



def lagrange(eph, dat):
    # Verileri ayırma ve saniyeye dönüştürme
    t = dat[:, 0]

    y = dat[:, 1]
    
    # İnterpolasyonun derecesi
    n = len(t)  # n = 10
    
    # İnterpolasyon değeri
    out = 0
    
    for i in range(n):
        # Lagrange polinomları
        L = 1
        for j in range(n):
            if j != i:
                L *= (eph - t[j]) / (t[i] - t[j])
        
        # İnterpolasyon değerini güncelleme
        out += y[i] * L
    
    return out


# Örnek veri



def lagrange_interpolation_plot(ephemeris_time, data):
    interpolated_value = lagrange(ephemeris_time, data)

    print("Interpolated value at ephemeris time:", interpolated_value)
    
    time = (ephemeris_time//3600) + ((ephemeris_time%3600) //60)/100 + (ephemeris_time%60)/1000

    # Lagrange interpolasyonunu hesaplama
    t_values = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 1000)
    lagrange_values = np.array([lagrange(t * 3600, data) for t in t_values])  # Zamanı saniyeye çevirme

    # Plot
    plt.plot(data[:, 0], data[:, 1], 'ro', label='Veri Noktaları')
    plt.plot(time, interpolated_value, 'bo', label='Interpolasyon Noktası') # Saati tekrar saat cinsine dönüştürme
    plt.xlabel('Zaman (saat)')
    plt.ylabel('Değer')
    plt.title('Lagrange Interpolasyonu')
    plt.legend()
    plt.grid(True)
    plt.show()







