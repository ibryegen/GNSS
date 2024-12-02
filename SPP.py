# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 20:58:42 2024

@author: win10
"""

# GMT312 GLOBAL NAVIGATION SATELLITE SYSTEMS ASSIGNMENT IV
# Author: IBRAHIM YEGEN  10.05.2024

# This script performs Single Point Positioning (SPP) to determine the 3D coordinates 
# (X, Y, Z in the ECEF coordinate system) of the MERS station on March 1, 2024. The script 
# uses GPS satellite observations and precise ephemeris data to iteratively calculate the station's 
# position, both with and without corrections for ionospheric, tropospheric delays, and total 
# group delay (TGD). The calculated coordinates are then compared to the true coordinates 
# obtained from the IGS SINEX file. The script is divided into modular functions for better 
# readability and maintainability, and it includes detailed comments to explain each step of 
# the process.

import numpy as np
from sp3file import parse_sp3_file, select_interpolation_data
from sat_pos import sat_pos, clockerror
from atmos import atmos
from math import sqrt
from numpy.linalg import inv

"""
OBSERVATION DATA    M: MIXED            RINEX VERSION / TYPE

4239146.6414  2886967.1245  3778874.4800                  APPROX POSITION XYZ

ANTENNA: DELTA H/E/N
G    8 C1C L1C C2S L2S C2W L2W C5Q L5Q                      

G05  22581252.559   118665247.69907  22581252.923    92466422.51907  22581252.393    92466415.52107
G07  22830577.076   119975581.96807  22830578.536    93487506.94707  22830578.189    93487505.94807
G08  24862112.929   130651198.68406  24862120.231   101806124.30606  24862119.334   101806123.31506  24862121.176    97564196.58707
G09  24920406.994   130957551.86906  24920413.523   102044827.10406  24920413.272   102044821.11606  24920416.289    97792958.01207
G13  20928832.115   109981748.65308                                  20928831.234    85700052.25407
G14  20205799.876   106182211.92208  20205800.126    82739390.73809  20205799.790    82739382.74308  20205804.903    79291923.91409
G15  23305396.300   122470652.10507  23305397.804    95431665.94807  23305397.278    95431669.96106
G17  22705825.247   119319920.71307  22705827.023    92976570.88607  22705826.548    92976566.88606
G19  24773732.425   130186836.04307                                  24773733.919   101444289.33607
G20  23314693.880   122519493.22007                                  23314694.292    95469719.67905
G22  20423865.035   107328135.72008                                  20423863.914    83632288.95808
G30  21103278.590   110898485.89208  21103281.178    86414400.63708  21103280.436    86414393.64508  21103282.088    82813805.88209
"""

"""
-0.111758708954D-07
-0.111758708954D-07
0.512227416039D-08
0.139698386192D-08
-0.111758708954D-07
-0.791624188423D-08
-0.102445483208D-07
-0.111758708954D-07
-0.153668224812D-07
-0.838190317154D-08
-0.111758708954D-07
0.372529029846D-08
"""

def calculateReceptionEpoch():
    """
    Calculate the reception epoch time (seconds of the day).
    
    Returns:
        int: Reception time in seconds.
    """
    return (2+2+0+0+6+7+4+0+3+0) * 810

def initializeData():
    """
    Initialize approximate receiver position and read SP3 file.
    
    Returns:
        np.array: Approximate receiver position.
        list: Epoch times from SP3 file.
        np.array: Data from SP3 file.
    """
    approxPosition = np.array([4239146.6414, 2886967.1245, 3778874.4800], dtype=np.float64)
    filePath = 'IGS0OPSFIN_20240610000_01D_15M_ORB.SP3.txt'
    epochs, data = parse_sp3_file(filePath)
    return approxPosition, epochs, data

def getObservations():
    """
    Get satellite observations.
    
    Returns:
        dict: Satellite observations.
    """
    return {
        'G05': 22581252.559,
        'G07': 22830577.076,
        'G08': 24862112.929,
        'G09': 24920406.994,
        'G13': 20928832.115,
        'G14': 20205799.876,
        'G15': 23305396.300,
        'G17': 22705825.247,
        'G19': 24773732.425,
        'G20': 23314693.880,
        'G22': 20423865.035,
        'G30': 21103278.590
    }

def getTGD():
    """
    Get Total Group Delay (TGD) values for satellites.
    
    Returns:
        dict: TGD values for satellites.
    """
    return {
        'G05': -0.111758708954e-07,
        'G07': -0.111758708954e-07,
        'G08': 0.512227416039e-08,
        'G09': 0.139698386192e-08,
        'G13': -0.111758708954e-07,
        'G14': -0.791624188423e-08,
        'G15': -0.102445483208e-07,
        'G17': -0.111758708954e-07,
        'G19': -0.153668224812e-07,
        'G20': -0.838190317154e-08,
        'G22': -0.111758708954e-07,
        'G30': 0.372529029846e-08
    }

# Calculate the reception time
receptionEpoch = calculateReceptionEpoch()
epoch = 19440
dayOfYear = 61
receptionEpochWeek = (86400 * 5) + receptionEpoch
alphaParams = np.array([0.2794e-07, 0.7451e-08, -0.1192e-06, 0.5960e-07])
betaParams = np.array([0.1372e+06, -0.3277e+05, -0.6554e+05, -0.5898e+06])
speedOfLight = 299792458

def spp(satelliteObservations, selectedData, approxPosition, epochs, data, receptionEpoch, dayOfYear, alphaParams, betaParams, tgd, includeDelayTGD=True, speedOfLight=299792458):
    """
    Perform Single Point Positioning (SPP) to compute the receiver's 3D position.

    Parameters:
        satelliteObservations (dict): Satellite observations.
        selectedData (dict): Interpolated satellite data.
        approxPosition (np.array): Approximate receiver position.
        epochs (list): Epoch times.
        data (np.array): Data from SP3 file.
        receptionEpoch (int): Reception time in seconds.
        dayOfYear (int): Day of the year.
        alphaParams (np.array): Ionospheric parameters.
        betaParams (np.array): Ionospheric parameters.
        tgd (dict): Total Group Delay (TGD) values.
        includeDelayTGD (bool): Whether to include TGD in calculations.
        speedOfLight (float): Speed of light.

    Returns:
        np.array: Updated receiver position (X, Y, Z).
    """
    # Calculate satellite positions
    satellitePositions = {sat: sat_pos(receptionEpoch, obs, selectedData.get(sat), approxPosition) for sat, obs in satelliteObservations.items() if selectedData.get(sat) is not None}
    satellitePosMatrix = np.array([satellitePositions[sat] for sat in satelliteObservations.keys() if sat in satellitePositions])

    # Initialize results dictionary
    results = {}
    for i, (sat, obs) in enumerate(satelliteObservations.items()):
        satData = selectedData.get(sat)
        delay = atmos(dayOfYear, receptionEpoch, receptionEpochWeek, obs, approxPosition, satData, alphaParams, betaParams, satellitePosMatrix[i])
        clkData = satData[:, [0, 4]]
        dt = clockerror(receptionEpoch, obs, clkData)
        if includeDelayTGD:
            d = -speedOfLight * dt + delay[4] + delay[5] + delay[3] + (tgd[sat] * speedOfLight)
        else:
            d = -speedOfLight * dt
        results[sat] = {'d': d}

    # Initialize receiver position
    receiverPosition = np.array([0.0, 0.0, 0.0])
    dValues = np.array([result['d'] for result in results.values()])
    observationsArray = np.array(list(satelliteObservations.values()))

    # Iterative process for position estimation
    while True:
        A, l = [], []
        for j, (sat, obs) in enumerate(satelliteObservations.items()):
            p0 = np.linalg.norm(satellitePosMatrix[j] - receiverPosition)
            l.append(obs - p0 - dValues[j])
            A.append([(receiverPosition[k] - satellitePosMatrix[j][k]) / p0 for k in range(3)] + [1])

        A, l = np.array(A), np.array(l).reshape(-1, 1)
        x = inv(A.T @ A) @ A.T @ l
        dx, dy, dz, dt = x.flatten()
        if all(abs(val) <= 1e-3 for val in (dx, dy, dz)):
            break
        receiverPosition += [dx, dy, dz]

    return receiverPosition

def sppProject():
    """
    Main function to perform the Single Point Positioning (SPP) and print the results.
    """
    receptionEpoch = calculateReceptionEpoch()
    approxPosition, epochs, data = initializeData()
    satelliteObservations = getObservations()
    tgd = getTGD()

    selectedData = select_interpolation_data(epoch, epochs, data)
    includeDelayTGD = input("Include delay and TGD values? (Yes/No): ").strip().lower() == "yes"
    updatedPosition = spp(satelliteObservations, selectedData, approxPosition, epochs, data, receptionEpoch, dayOfYear, alphaParams, betaParams, tgd, includeDelayTGD)

    # Print the estimated coordinates
    print("ESTIMATED COORDINATES")
    print(f"NP: X = {updatedPosition[0]}, Y = {updatedPosition[1]}, Z = {updatedPosition[2]}")
    approxPositionIGS = np.array([4239149.205, 2886968.037, 3778877.204])
    delta = np.abs(updatedPosition - approxPositionIGS)
    
    msf = np.linalg.norm(updatedPosition - approxPositionIGS)

    # Print the coordinates from IGS RINEX file
    print("msf:", msf)
    print("COORDINATES IN IGS RINEX FILE")
    print("*" * 10)
    print(f"X = {approxPositionIGS[0]} m, Y = {approxPositionIGS[1]} m, Z = {approxPositionIGS[2]} m")
    print("*" * 10)
    print("Delta Values:")
    print(f"Delta X = {delta[0]} m")
    print(f"Delta Y = {delta[1]} m")
    print(f"Delta Z = {delta[2]} m")

if __name__ == "__main__":
    sppProject()
