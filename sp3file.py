# -*- coding: utf-8 -*-
"""
Created on Fri May 17 14:26:52 2024

@author: win10
"""

import numpy as np

# Dosyayı okuyup gerekli verileri parse etme fonksiyonu
def parse_sp3_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    epochs = []
    data = {}

    for line in lines:
        if line.startswith('*'):
            # Yeni bir epoch başlangıcı
            year = int(line[3:7])
            month = int(line[8:10])
            day = int(line[11:13])
            hour = int(line[14:16])
            minute = int(line[17:19])
            second = float(line[20:31])
            epoch_time = (hour * 3600) + (minute * 60) + second
            epochs.append(epoch_time)
            current_epoch = epoch_time
        elif line.startswith('PG'):
            satellite = line[1:4]
            x = float(line[4:18])
            y = float(line[18:32])
            z = float(line[32:46])
            clock_error = float(line[46:60])
            if satellite not in data:
                data[satellite] = []
            data[satellite].append((current_epoch, x, y, z, clock_error))

    return epochs, data

# Lagrange interpolasyonu için gerekli 11 veriyi seçme fonksiyonu
def select_interpolation_data(epoch, epochs, data):
    closest_epoch = min(epochs, key=lambda x: abs(x - epoch))
    selected_data = {}
    for satellite, values in data.items():
        satellite_data = [v for v in values if v[0] in epochs]
        epoch_index = epochs.index(closest_epoch)
        if epoch_index >= 5 and epoch_index < len(epochs) - 5:
            selected_values = satellite_data[epoch_index-5:epoch_index+6]
            selected_data[satellite] = np.array(selected_values)
        else:
            selected_data[satellite] = None
    return selected_data