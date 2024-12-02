import math
"""
 This function compute the ionospheric slant delay for GPS code measeurement in L1 signal
 Inputs
 lat  : geodetic latitude (radian) from approximate coordinates in the observation file
 lon  : geodetic longitude(radian) from approximate coordinates in the observation file
 elv  : elevation angle   (radian) for the related satellite
 azm  : azimuth angle     (radian) for the related satellite
 alfa : Klobuchar coefficients (sec sec/semicircle sec/semicircle2 sec/semicircle3) 
 beta : Klobuchar coefficients (sec sec/semicircle sec/semicircle2 sec/semicircle3)
 tgps : gps time (seconds of week) for the observation
 Output
 dion : ionospheric slant delay (meter) for the observation
 Reference
 GNSS Data Processing, Vol. I: Fundamentals and Algorithms (ESA TM-23/1, May 2013)
"""

def Ion_Klobuchar(lat, lon, elv, azm, alfa, beta, tgps):
    # velocity of light
    c = 299792458  # m/s

    # calculate the Earth-centred angle
    Re = 6378  # km
    h = 350  # km
    cns = (Re / (Re + h)) * math.cos(elv)
    eca = math.pi / 2 - elv - math.asin(cns)

    # compute the latitude of IPP
    ax = math.sin(lat) * math.cos(eca) + math.cos(lat) * math.sin(eca) * math.cos(azm)
    lat_ipp = math.asin(ax)

    # compute the longitude of IPP
    lon_ipp = lon + (eca * math.sin(azm)) / (math.cos(lat_ipp))

    # Find the geomagnetic latitude of the IPP
    f_pol = math.radians(78.3)
    l_pol = math.radians(291)
    as_ = math.sin(lat_ipp) * math.sin(f_pol) + math.cos(lat_ipp) * math.cos(f_pol) * math.cos(lon_ipp - l_pol)
    geo = math.asin(as_)

    # Find the local time at the IPP
    t = 43200 * (lon_ipp / math.pi) + tgps
    t = t % 86400  # Seconds of day
    if t >= 86400:
        t = t - 86400
    elif t <= 0:
        t = t + 86400

    tsd = geo / math.pi
    AI = alfa[0] + alfa[1] * tsd + alfa[2] * (tsd ** 2) + alfa[3] * (tsd ** 3)  # seconds
    PI = beta[0] + beta[1] * tsd + beta[2] * (tsd ** 2) + beta[3] * (tsd ** 3)  # seconds
    if AI < 0:
        AI = 0
    if PI < 72000:
        PI = 72000

    # Compute the phase of ionospheric delay
    XI = (2 * math.pi * (t - 50400)) / PI  # radian

    # Compute the slant factor (ionospheric mapping function)
    F = (1 - cns ** 2) ** (-1 / 2)

    # Compute the ionospheric time delay
    if abs(XI) < (math.pi / 2):
        I1 = (5 * (10 ** (-9)) + AI * math.cos(XI)) * F
    elif abs(XI) >= (math.pi / 2):
        I1 = (5 * (10 ** (-9))) * F

    dion = I1 * c

    return dion

