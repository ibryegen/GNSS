# GNSS

This repository contains Python implementations for calculating satellite positions, ionospheric and tropospheric delays, and various GNSS-related transformations. The project is designed for educational purposes, providing tools for precise GNSS computations.

Overview
This project focuses on:

Calculating satellite positions using broadcast and precise ephemeris data.
Correcting GNSS signal errors caused by ionospheric and tropospheric delays.
Transforming coordinates between global and local reference systems.
Visualizing Lagrange interpolation for precise satellite positioning.
The computations rely on Python scripts developed as part of GNSS coursework.

Features
Satellite Positioning:

Compute satellite positions using precise (SP3) and broadcast (RINEX) ephemeris data.
Apply Lagrange interpolation for enhanced accuracy.
Atmospheric Corrections:

Calculate ionospheric delays using the Klobuchar model.
Compute tropospheric delays (dry and wet components) based on station location and meteorological parameters.
Coordinate Transformations:

Convert between Cartesian (XYZ) and geodetic (latitude, longitude, height) systems.
Transform global coordinates to local topocentric coordinates.
Visualization:

Plot Lagrange interpolation results.


File Descriptions
assignment2.py: Calculates GNSS satellite positions using broadcast ephemeris data.
blhandlocal.py: Functions for coordinate transformations and local system conversions.
cal_sp3.py: Implements satellite position calculation using precise ephemeris data and Lagrange interpolation.
lagrange.py: Handles 9th-degree Lagrange interpolation for satellite coordinates.
rotation.py: Performs coordinate rotations for transformations.
sat_pos.py: Calculates emission times and final satellite positions.
Ion_Klobuchar.py: Computes ionospheric delays using Klobuchar model.
trop_SPP.py: Estimates tropospheric delays (dry and wet).
atmos.py: Main script integrating all components for delay calculations and coordinate transformations.
