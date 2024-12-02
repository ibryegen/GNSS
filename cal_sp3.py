# -*- coding: utf-8 -*-
"""
Created on Fri May  3 22:34:42 2024

@author: win10
"""
import numpy as np
from lagrange import lagrange
def cal_sp3(eph, sp3):
    
    # 10x2 matrix formed by the first column and the second column
    # [time tag, x]
    dat1 = sp3[:, [0, 1]]

    # 10x2 matrix formed by the first column and the third column
    # [time tag, y]
    dat2 = sp3[:, [0, 2]]


    # 10x2 matrix formed by the first column and the fourth column
    # [time tag, z]
    dat3 = sp3[:, [0, 3]]
    
    
    # Computing each coordinates with Lagrange Interpolation
    x = lagrange(eph, dat1)
    y = lagrange(eph, dat2)
    z = lagrange(eph, dat3)
    
    spos = [x,y,z]
    
    return spos