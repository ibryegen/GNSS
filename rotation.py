# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 21:32:55 2024

@author: win10
"""
import numpy as np

def rotation(vec, ang, ax):
    # inputs are: coordinate vector, angle, and axis.
    
    # Precaution to ensure that the angle is a real number.
    if not isinstance(ang, (float, int)):
        print("You should enter an angle!")
        return None
    
    if ax not in [1, 2, 3]:
        print("Error! Axis must be 1, 2, or 3.")
        return None
    
    # Convert angle to radians
    ang_rad = ang 
    
    if ax == 1:
        # R1(α) rotation matrix here;
        r1 = np.array([[1, 0, 0],
                       [0, np.cos(ang_rad), np.sin(ang_rad)],
                       [0, -np.sin(ang_rad), np.cos(ang_rad)]])
        new_vec = np.dot(r1, vec)  # .dot performs matrix multiplication. The result of the operation is assigned to a new variable.
        return new_vec
    
    elif ax == 2:
        # R2(β) rotation matrix here;
        r2 = np.array([[np.cos(ang_rad), 0, -np.sin(ang_rad)],
                       [0, 1, 0],
                       [np.sin(ang_rad), 0, np.cos(ang_rad)]])
        new_vec = np.dot(r2, vec)  # .dot performs matrix multiplication. The result of the operation is assigned to a new variable.
        return new_vec
    
    elif ax == 3:
        # R3(γ) rotation matrix here;
        r3 = np.array([[np.cos(ang_rad), np.sin(ang_rad), 0],
                       [-np.sin(ang_rad), np.cos(ang_rad), 0],
                       [0, 0, 1]])
        new_vec = np.dot(r3, vec)  # .dot performs matrix multiplication. The result of the operation is assigned to a new variable.
        return new_vec