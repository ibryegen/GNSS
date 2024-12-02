# -*- coding: utf-8 -*-
"""
Created on Thu May 23 23:34:13 2024

@author: win10
"""

import numpy as np
from sp3file import parse_sp3_file, select_interpolation_data
from sat_pos import sat_pos, emist, clockerror
from atmos import atmos
from math import sqrt
import pandas as pd
from numpy.linalg import inv

trec = (2+2+0+0+6+7+4+0+3+0) * 810
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

r_apr = np.array([4239146.6414, 2886967.1245, 3778874.4800], dtype=np.float64)

# SP3 dosyasını oku
file_path = 'IGS0OPSFIN_20240610000_01D_15M_ORB.SP3.txt'
epochs, data = parse_sp3_file(file_path)

# İlgili epoch'u belirle (19440 saniye)
epoch = 19440

doy = 61
trecw = (86400 * 5) + trec
alpha = np.array([0.2794e-07, 0.7451e-08, -0.1192e-06, 0.5960e-07])

beta = np.array([0.1372e+06, -0.3277e+05, -0.6554e+05, -0.5898e+06])
c = 299792458 

# İlgili epoch etrafındaki 11 veriyi seç
selected_data = select_interpolation_data(epoch, epochs, data)

# Uyduların gözlemleri ve tanımlamaları
satellites_observations = {
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

tgd = {
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

# Sonuçları saklamak için bir sözlük
sat_fpos = {}

# Döngü ile her uydu için pozisyon hesaplama
for sat, obs in satellites_observations.items():
    sp3_data = selected_data.get(sat)
    if sp3_data is not None:
        sat_fpos[sat] = sat_pos(trec, obs, sp3_data, r_apr)
        


# Tüm uydu pozisyonlarını saklamak için bir matris oluştur
sat_pos_matrix = np.zeros((len(satellites_observations), 3))

# Otomatik olarak ilgili uydu pozisyonlarını değişkenlere atama ve matrise ekleme
for idx, sat in enumerate(satellites_observations.keys()):
    globals()[f"{sat}f"] = sat_fpos[sat]
    sat_pos_matrix[idx] = sat_fpos[sat]
    
    


results = {}

for i, (sat, obs) in enumerate(satellites_observations.items()):
    sat_data = selected_data.get(sat)
    
    # Calculate atmospheric delay
    delay = atmos(doy, trec, trecw, obs, r_apr, sat_data, alpha, beta, sat_pos_matrix[i])
    
    # Extract clock data
    clk_data = sat_data[:, [0, 4]]
    
    # Calculate emission time
    dt = clockerror(trec, obs, clk_data)
    
    # Calculate range
    d = -c * (dt) + delay[4] + delay[5] + delay[3]
    

    
    # Store results
    results[sat] = {'d': d}




r0 = np.array([0.0, 0.0, 0.0])

# Access 'd' values as an array
d_values = np.array([result['d'] for result in results.values()])





observations_array = np.array(list(satellites_observations.values()))

def spp(satellites_observations, selected_data, r_apr, epochs, data, trec, doys, alpha, beta, tgd, include_delay_tgd=True, c=299792458):
    """
    Tek noktadan konumlama (SPP) işlemi yapar.

    Parametreler:
    satellites_observations (dict): Uydu gözlemlerini içeren sözlük.
    selected_data (dict): Seçilen epoch etrafındaki uydu verilerini içeren sözlük.
    r_apr (numpy array): Yaklaşık alıcı konumu.
    epochs (list): Epoch zamanları.
    data (numpy array): SP3 dosyasından okunan veri.
    trec (int): Alım zamanı.
    doys (int): Yılın günü.
    alpha (numpy array): Ionospheric parametreler.
    beta (numpy array): Ionospheric parametreler.
    tgd (dict): Uydu group delay (TGD) değerlerini içeren sözlük.
    include_delay_tgd (bool): delay ve tgd değerlerinin hesaba katılıp katılmayacağını belirtir.
    c (float): Işık hızı (varsayılan: 299792458 m/s).

    Döndürür:
    numpy array: Güncellenmiş alıcı konumu (X, Y, Z).
    """

    # Sonuçları saklamak için bir sözlük
    sat_fpos = {}

    # Döngü ile her uydu için pozisyon hesaplama
    for sat, obs in satellites_observations.items():
        sp3_data = selected_data.get(sat)
        if sp3_data is not None:
            sat_fpos[sat] = sat_pos(trec, obs, sp3_data, r_apr)

    # Tüm uydu pozisyonlarını saklamak için bir matris oluştur
    sat_pos_matrix = np.zeros((len(satellites_observations), 3))

    # Otomatik olarak ilgili uydu pozisyonlarını değişkenlere atama ve matrise ekleme
    for idx, sat in enumerate(satellites_observations.keys()):
        sat_pos_matrix[idx] = sat_fpos[sat]


    results = {}

    for i, (sat, obs) in enumerate(satellites_observations.items()):
        sat_data = selected_data.get(sat)
        
        # Atmosferik gecikme hesaplaması
        delay = atmos(doy, trec, trecw, obs, r_apr, sat_data, alpha, beta, sat_pos_matrix[i])
        
        # Saat verilerini ayıkla
        clk_data = sat_data[:, [0, 4]]
        
        # Yayın zamanı hesapla
        dt = clockerror(trec, obs, clk_data)
        
        # Mesafe hesapla
        if include_delay_tgd:
            d = -c * (dt) + delay[4] + delay[5] + delay[3] + (tgd[sat] * c)
        else:
            d = -c * dt
        
        # Sonuçları sakla
        results[sat] = {'d': d}

    # Başlangıç pozisyonu
    r0 = np.array([0.0, 0.0, 0.0])

    # 'd' değerlerine bir dizi olarak eriş
    d_values = np.array([result['d'] for result in results.values()])

    observations_array = np.array(list(satellites_observations.values()))

    dx, dy, dz = np.inf, np.inf, np.inf  # Başlangıçta büyük değerler
    print(r0)

    # İterasyon döngüsü
    while abs(dx) > 1e-3 and abs(dy) > 1e-3 and abs(dz) > 1e-3:
        # Boş matrisler oluşturuluyor
        A = []
        l = []
        
        for j, (sat, obs) in enumerate(satellites_observations.items()):
            # p0 hesaplanması
            p0 = sqrt((sat_pos_matrix[j][0] - r0[0])**2 + (sat_pos_matrix[j][1] - r0[1])**2 + (sat_pos_matrix[j][2] - r0[2])**2)
            
            # Reduced observation vector (l)
            l_j = obs - p0 - d_values[j]
            l.append(l_j)
            
            # Design matrix (A)
            A_j = [
                (r0[0] - sat_pos_matrix[j][0]) / p0,
                (r0[1] - sat_pos_matrix[j][1]) / p0,
                (r0[2] - sat_pos_matrix[j][2]) / p0,
                1  # c*dt terimi için
            ]
            A.append(A_j)
        
        # Matrisleri numpy dizilerine dönüştür
        A = np.array(A)
        l = np.array(l).reshape(-1, 1)
        
        # En küçük kareler çözümü
        A_transpose = A.T
        x = inv(A_transpose @ A) @ A_transpose @ l
        
        # dx, dy, dz ve c*dt değerlerini ayıkla
        dx, dy, dz, dt = x.flatten()
        
        # Eğer dx, dy, dz 1-mm'den küçükse döngüden çık
        if abs(dx) <= 1e-3 and abs(dy) <= 1e-3 and abs(dz) <= 1e-3:
            break
        
        # r0'ı güncelle
        r0[0] += dx
        r0[1] += dy
        r0[2] += dz
        print(r0)
        print("*"*10)

    return r0



# Kullanıcıdan input alalım
include_delay_tgd = input("Delay ve TGD değerleri hesaba katılsın mı? (Evet/Hayır): ").strip().lower() == "evet"
print("*"*10)

updated_position = spp(satellites_observations, selected_data, r_apr, epochs, data, trec, doy, alpha, beta, tgd, include_delay_tgd)
print("ESTIMATED COORD")
print(f"NP: X = {updated_position[0]}, Y = {updated_position[1]}, Z = {updated_position[2]}")

print("*"*10)

# APR ESTIMATED COORD
# APR ESTIMATED COORD
approx_position = np.array([4239149.205, 2886968.037, 3778877.204])

# Hesaplanan pozisyon ile yaklaşık pozisyon arasındaki delta değerleri
delta_x = abs(updated_position[0] - approx_position[0])
delta_y = abs(updated_position[1] - approx_position[1])
delta_z = abs(updated_position[2] - approx_position[2])

# Sonuçları yazdırma
print("COORD IN IGS RINEX FILE")
print("*"*10)
print(f"X = {approx_position[0]} m, Y = {approx_position[1]} m, Z = {approx_position[2]} m")
print("*"*10)


# Delta değerlerini yazdırma
print("Delta Values:")
print(f"Delta X = {delta_x} m")
print(f"Delta Y = {delta_y} m")
print(f"Delta Z = {delta_z} m")
