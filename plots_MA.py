import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
import time as time_1
import pandas as pd
import os
import pickle
import random

path = r"C:\Users\dauserml\Documents\Regression_all_01"  #C:\Users\dauserml\Documents\Regression_all_01
filename_training = "\\0_284_2020_11_16_Manuel.csv"#"\\all_Files_C_R.csv"  #Regression_all_01

# Trainings Daten
all_or_range_selection_Data = False  # all=True, range selection=False
training_Data_from = 0  # Only "all_or_range_selection_Data = False"
training_Data_to = 3100  # Only "all_or_range_selection_Data = False"
time = 1  # 1= Timesteps
X_Signal_number = 0  # 0= Signal delimitation
Input_from = 9  # Input Signal from 9 to 40 (9=frame1_real,10=frame1_imag,11=frame2_real,....,40=main8_imag)
Input_to = 41
# Signal (5=Yaw,6=Roll,7=Pitch,8=Residual) (41= For Classification, witch shelf)
Output_from = 2  # Ouput Signal from 41 -> Class
Output_to = 5

time_xyz_antennen_Signal_Komplex_all_Files = np.loadtxt(path+filename_training,delimiter=';')

X_Signal_number=(time_xyz_antennen_Signal_Komplex_all_Files[0:,X_Signal_number:(X_Signal_number+1)].astype(int))
Class = (time_xyz_antennen_Signal_Komplex_all_Files[0:,41:42].astype(int))
X_1 = time_xyz_antennen_Signal_Komplex_all_Files[0:,Input_from:Input_to]*1000 #[:,9:]#
Y_1 = (time_xyz_antennen_Signal_Komplex_all_Files[0:,Output_from:Output_to])#[:,2:5]#


y_achse =np.arange(0,((len(X_1))*0.00715),0.00715)
plt.figure(93)
plt.title("32 Antennensignale")
plt.ylabel("Antennensignale in mV")
plt.xlabel("Zeit in s")
#plt.legend(["Main 1 reel","Main 1 imag","Frame 1 reel","Frame 1 imag","Main 2 reel","Main 2 imag","Frame 2 reel","Frame 2 imag","Main 3 reel","Main 3 imag","Frame 3 reel","Frame 3 imag",
#            "Main 4 reel","Main 4 imag","Frame 4 reel","Frame 4 imag","Main 5 reel","Main 5 imag","Frame 5 reel","Frame 5 imag","Main 6 reel","Main 6 imag","Frame 6 reel","Frame 6 imag",
#            "Main 7 reel","Main 7 imag","Frame 7 reel","Frame 7 imag","Main 8 reel","Main 8 imag","Frame 8 reel","Frame 8 imag"])
plt.plot(y_achse,X_1[:,0:],y_achse,Y_1[:,2])

#plt.figure(94)
figure,ax=plt.subplots(5,1)

ax[0].plot(y_achse,X_1[:,0:2])
ax[0].set_ylabel("Main 1 \n Spannung in mV")
ax[0].set_xlabel("Zeit in s")
ax[0].set_title("Antennensignal")

ax[1].plot(y_achse,X_1[:,16:18])
ax[1].set_ylabel("Frame 1 \n Spannung in mV")
ax[1].set_xlabel("Zeit in s")
#ax[1].set_title("Antennensignal Frame 1")

ax[2].plot(y_achse,X_1[:,14:16])
ax[2].set_ylabel("Main 8 \n Spannung in mV")
ax[2].set_xlabel("Zeit in s")
#ax[2].set_title("Antennensignal Main 8")

ax[3].plot(y_achse,X_1[:,30:32])
ax[3].set_ylabel("Frame 8 \n Spannung in mV")
ax[3].set_xlabel("Zeit in s")
#ax[3].set_title("Antennensignal Frame 8")

ax[4].plot(y_achse,Y_1[:,0])
ax[4].set_ylabel("X Position in mm")
ax[4].set_xlabel("Zeit in s")
#ax[4].set_title("Z Position")

#plt.tight_layout()

#plt.subplot(5,1,1)
#plt.title("Antennensignal Main 1")
#plt.plot(y_achse,X_1[:,0:2])
#plt.ylabel("mV")
#plt.xlabel("Zeit in s")
#plt.subplot(5, 1, 2)
#plt.title("Antennensignal Frame 1")
#plt.plot(y_achse,X_1[:, 2:4])
#plt.ylabel("mV")
#plt.xlabel("Zeit in s")
#plt.subplot(5, 1, 3)
#plt.title("Antennensignal Main 8")
#plt.plot(y_achse,X_1[:, 28:30])
#plt.ylabel("mV")
#plt.xlabel("Zeit in s")
#plt.subplot(5, 1, 4)
#plt.title("Antennensignal Frame 8")
#plt.plot(y_achse,X_1[:, 30:32])
#plt.ylabel("mV")
#plt.xlabel("Zeit in s")
#plt.subplot(5, 1, 5)
#plt.plot(y_achse,Y_1[:,2])
#plt.ylabel("mm")
#plt.xlabel("Zeit in s")
#plt.tight_layout()

plt.figure(95)
plt.subplot(3, 1, 1)
plt.title("X Achse")
plt.plot(y_achse,Y_1[:,0])
plt.ylabel("mm")
plt.xlabel("Zeit in s")
plt.subplot(3, 1, 2)
plt.title("Y Achse")
plt.plot(y_achse,Y_1[:,1])
plt.ylabel("mm")
plt.xlabel("Zeit in s")
plt.subplot(3, 1, 3)
plt.title("Z Achse")
plt.plot(y_achse,Y_1[:,2])
plt.ylabel("mm")
plt.xlabel("Zeit in s")
plt.tight_layout()
plt.show()