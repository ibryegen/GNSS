import numpy as np
import pandas as pd
import math

# Constants
c = 299792458  # Speed of light (m/s)
mu = 3.986005 * 10**14  # Earth's gravitational constant (m^3/s^2)
w_E = 7.2921151467 * 10**-5  # Earth's rotation rate (rad/s)

navigation_file = "brdc0600.23n"
rinex_file = "MERS00TUR_R_20230600000_01D_30S_MO.rnx"
sp3_file = "IGS0OPSFIN_20230600000_01D_15M_ORB.SP3"

student_id = 2200674002
treception = (sum(map(int, str(student_id)))) * 750
if treception % 900 == 0:
    treception += 750
print(treception)





# trec= treception
def compute_emission_time(trec , pc, clk):
    # Step 1:  trec is assigned by default into function

    # Step 2: Load observation file (RINEX)
    rinex_data = pd.read_csv(rinex_file, skiprows=32, delim_whitespace=True)
    r_apr = np.array([rinex_data["APPROX_POSITION_X"][0],
                      rinex_data["APPROX_POSITION_Y"][0],
                      rinex_data["APPROX_POSITION_Z"][0]])

    # Step 3: Load precise ephemeris file (SP3)
    sp3_data = pd.read_csv(sp3_file, skiprows=22, delim_whitespace=True, header=None)

    # Step 4: Compute the emission time and final satellite coordinates for all  GPS satellites
    satellite_ids = rinex_data["PRN"].unique()[:len(r_apr)]
    for sat_id in satellite_ids:
        # Filter data for the specific satellite
        sat_data = rinex_data[rinex_data["PRN"] == sat_id]

        # Extract code pseudorange (C/A observation)
        pc = sat_data["C1"]

        # Extract satellite clock corrections for 10 consecutive epochs
        clk = sp3_data[sp3_data[0] == sat_id].iloc[:, 2:12].values

    tems = trec - (pc / c) - clk
    return tems
trec = 37260

def compute_final_coordinates(trec, pc, sp3, r_apr):

    rinex_data = pd.read_csv(rinex_file, skiprows=32, delim_whitespace=True)
    r_apr = np.array([rinex_data["APPROX_POSITION_X"][0],
                      rinex_data["APPROX_POSITION_Y"][0],
                      rinex_data["APPROX_POSITION_Z"][0]])

    satellite_ids = rinex_data["PRN"].unique()[:2]
    sp3_data = pd.read_csv(sp3_file, skiprows=22, delim_whitespace=True, header=None)

    for sat_id in satellite_ids:
        # Filter data for the specific satellite
        sat_data = rinex_data[rinex_data["PRN"] == sat_id]

        # Extract code pseudorange (C/A observation)
        pc = sat_data["C1"]
        sp3 = sp3_data[sp3_data[0] == sat_id].iloc[:, 2:12].values

    # Extract required variables from sp3
    t = sp3[:, 0]  # Time tags for epochs
    X = sp3[:, 1]  # Satellite coordinates (X) for epochs
    Y = sp3[:, 2]  # Satellite coordinates (Y) for epochs
    Z = sp3[:, 3]  # Satellite coordinates (Z) for epochs
    clk = sp3[:, 4:]  # Satellite clock corrections for epochs

    # Find the index of the reception time in the time tags
    idx = t.tolist().index(trec)

    # Compute the satellite coordinates at the reception time
    x_sat = X[idx]
    y_sat = Y[idx]
    z_sat = Z[idx]
    clk_corr = clk[idx]

    # Compute the satellite position at the signal emission time
    t_diff = trec - t[idx]  # Time difference in seconds
    x_dot = (X[idx + 1] - X[idx - 1]) / (t[idx + 1] - t[idx - 1])  # Approximate rate of change of X
    y_dot = (Y[idx + 1] - Y[idx - 1]) / (t[idx + 1] - t[idx - 1])  # Approximate rate of change of Y
    z_dot = (Z[idx + 1] - Z[idx - 1]) / (t[idx + 1] - t[idx - 1])  # Approximate rate of change of Z

    x_corr = x_sat + x_dot * t_diff  # Corrected X coordinate
    y_corr = y_sat + y_dot * t_diff  # Corrected Y coordinate
    z_corr = z_sat + z_dot * t_diff  # Corrected Z coordinate

    # Compute the final satellite coordinates in ECEF
    fpos = [x_corr, y_corr, z_corr]

    return fpos


def atmos(doy, trec, trecw, C1, rec, sp3, alpha, beta, r_vec, rng, elev, exp, R, k1, k2):
    # Load observation file
    obs_data = pd.read_csv("MERS00TUR_R_20230600000_01D_30S_MO.rnx")

    # Load navigation data file
    nav_data = pd.read_csv("brdc0600.23n")

    az = math.atan2(r_vec[1], r_vec[0])
    zen = math.atan2(r_vec[2], math.sqrt(r_vec[0]**2 + r_vec[1]**2))

    # Convert azimuth angle and zenith angle from radians to degrees
    az = math.degrees(az)
    zen = math.degrees(zen)

    # Calculate slant distance
    slantd = rng / 1000  # Convert range from meters to kilometers

    # Calculate ionospheric delay using Klobuchar model
    ion_alpha = alpha[doy]
    ion_beta = beta[doy]
    phi_u = math.radians(rec[0])  # User latitude in radians
    lambda_u = math.radians(rec[1])  # User longitude in radians
    phi_i = math.radians(ion_alpha[0])  # Ionospheric grid latitude in radians
    lambda_i = math.radians(ion_alpha[1])  # Ionospheric grid longitude in radians
    psi_i = ion_alpha[2] + ion_alpha[3] * (lambda_u - lambda_i)
    phi_i_prime = phi_i + psi_i * math.cos(phi_u)
    if phi_i_prime > 0.416:
        phi_i_prime = 0.416
    elif phi_i_prime < -0.416:
        phi_i_prime = -0.416
    lambda_i_prime = lambda_i + psi_i * math.sin(phi_u) / math.cos(phi_i_prime)
    phi_r = math.radians(rec[0])  # Receiver latitude in radians
    lambda_r = math.radians(rec[1])  # Receiver longitude in radians
    psi_r = ion_beta[2] + ion_beta[3] * (lambda_u - lambda_r)
    phi_r_prime = phi_r + psi_r * math.cos(phi_u)
    if phi_r_prime > 0.416:
        phi_r_prime = 0.416
    elif phi_r_prime < -0.416:
        phi_r_prime = -0.416
    lambda_r_prime = lambda_r + psi_r * math.sin(phi_u) / math.cos(phi_r_prime)
    F = 1.0 + 16.0 * (0.53 - elev)**3  # Obliquity factor
    t = 1.0 + 16.0 * (0.53 - zen)**3  # Zenith delay mapping function
    iono_delay = F * (k1 * ion_alpha[0] + k2) * t

    # Calculate tropospheric delays
    pressure = exp(-0.0006396 * rec[2] / (R * (0.003773 * rec[2] + 0.0224)))
    humidity = 0.01  # Assume a fixed humidity value of 0.01
    e = pressure * humidity
    tropo_dry = 0.002277 / (1 - 0.00266 * math.cos(2 * phi_u) - 0.00028 * rec[2] / 1000) * pressure
    tropo_wet = 0.002277 * (1255 / (273.15 ))

    return az , zen, slantd, iono_delay, tropo_dry, tropo_wet




