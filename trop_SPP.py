import math

"""
The tropospheric model developed by[Collins, 1999] and adopted by the
Satellite - Based Augmentation System(SBAS).Details of the model are
presented in: GNSS Data Processing.J.Sanz Subirana, J.M.Juan Zornoza
and M.Hernández - Pajares, 2013, Vol.1, (ESA TM-23 / 1, May 2013),pp.123 - 124
 
 Inputs
 lat: Latitude of the receiver[deg]
 D: Day of year[integer]
 H: Height of the station above mean sea level[m]
 E: Elevation of the signal[rad]
 Outputs
 Trzd: Zenith dry(hydrostatic) delay[m]
 Trzw: Zenith wet delay[m]
 ME: Mapping function for both dry and wet delays
"""


def trop_SPP(lat, D, H, E):
    # Average meteorological parameters for the tropospheric delay pressure [P(mbar)], temperature [T(K)],
    # water vapour pressure [e(mbar)], temperature rate [Beta(K/m)] and water vapour rate [Alpha (Dimensionless)]
    Met15 = [1013.25, 299.65, 26.31, 6.30e-3, 2.77]
    Met30 = [1017.25, 294.15, 21.79, 6.05e-3, 3.15]
    Met45 = [1015.75, 283.15, 11.66, 5.58e-3, 2.57]
    Met60 = [1011.75, 272.15, 6.78, 5.39e-3, 1.81]
    Met75 = [1013.00, 263.65, 4.11, 4.53e-3, 1.55]

    # Seasonal variations for the meteorological parameters
    dMet15 = [0.00, 0.00, 0.00, 0.00e-3, 0.00]
    dMet30 = [-3.75, 7.00, 8.85, 0.25e-3, 0.33]
    dMet45 = [-2.25, 11.00, 7.24, 0.32e-3, 0.46]
    dMet60 = [-1.75, 15.00, 5.36, 0.81e-3, 0.74]
    dMet75 = [-0.50, 14.50, 3.39, 0.62e-3, 0.30]

    A = [0] * 5
    B = [0] * 5

    if lat <= 15:
        A = Met15
        B = dMet15
    elif lat > 15 and lat <= 30:
        for i in range(5):
            A[i] = Met15[i] + ((lat - 15) / 15) * (Met30[i] - Met15[i])
            B[i] = dMet15[i] + ((lat - 15) / 15) * (dMet30[i] - dMet15[i])
    elif lat > 30 and lat <= 45:
        for i in range(5):
            A[i] = Met30[i] + ((lat - 30) / 15) * (Met45[i] - Met30[i])
            B[i] = dMet30[i] + ((lat - 30) / 15) * (dMet45[i] - dMet30[i])
    elif lat > 45 and lat <= 60:
        for i in range(5):
            A[i] = Met45[i] + ((lat - 45) / 15) * (Met60[i] - Met45[i])
            B[i] = dMet45[i] + ((lat - 45) / 15) * (dMet60[i] - dMet45[i])
    elif lat > 60 and lat < 75:
        for i in range(5):
            A[i] = Met60[i] + ((lat - 60) / 15) * (Met75[i] - Met60[i])
            B[i] = dMet60[i] + ((lat - 60) / 15) * (dMet75[i] - dMet60[i])
    elif lat >= 75:
        A = Met75
        B = dMet75

    ME = 1.001 / math.sqrt(0.002001 + math.sin(E) ** 2)

    k1 = 77.604
    k2 = 382000
    Rd = 287.054
    gm = 9.784
    g = 9.80665

    if lat > 0:
        Dmin = 28
    else:
        Dmin = 211

    P = A[0] + B[0] * math.cos((2 * math.pi * (D - Dmin)) / 365.25)
    T = A[1] + B[1] * math.cos((2 * math.pi * (D - Dmin)) / 365.25)
    e = A[2] + B[2] * math.cos((2 * math.pi * (D - Dmin)) / 365.25)
    Beta = A[3] + B[3] * math.cos((2 * math.pi * (D - Dmin)) / 365.25)
    Alpha = A[4] + B[4] * math.cos((2 * math.pi * (D - Dmin)) / 365.25)

    Trz0d = (1e-6 * k1 * Rd * P) / gm
    Trz0w = ((1e-6 * k2 * Rd) / (((Alpha + 1) * gm) - Beta * Rd)) * (e / T)

    Trzd = ((1 - ((Beta * H) / T)) ** (g / (Rd * Beta))) * Trz0d
    Trzw = ((1 - ((Beta * H) / T)) ** ((((Alpha + 1) * g) / (Rd * Beta)) - 1)) * Trz0w

    return Trzd, Trzw, ME
