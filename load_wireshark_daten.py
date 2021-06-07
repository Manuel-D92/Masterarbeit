import matplotlib.pyplot as plt
import numpy as np
import math

def convert_Signal_wireshark(self, Signal_1):
   XX = [];
   YY = []
   # sample_rate =  1/((np.shape(Signal_1)[0]-1)/np.real(Signal_1[-1,0,0]))
   # lengh_Singal = len(Signal_1)
   # time_s = np.linspace(0,lengh_Singal*sample_rate,lengh_Singal)
   for i in range(0, len(Signal_1)):  # Signal_1.shape[0]): len(Signal_1)
       X = []
       for j in range(4, Signal_1.shape[2]):
           X.append(Signal_1.real[i, 3, j])
           X.append(Signal_1.imag[i, 3, j])
       XX.append(X)
   for i in range(0, len(Signal_1)):  # len(Signal_1)
       YY.append(0)
   return XX  # ,YY,time_s

ff = np.load('C:\\Users\\dauserml\\Desktop\\dauserml_Messungen_2020_07_22\\meas1.pcapng.cal.npy')
timestamp_wireshark = np.load('C:\\Users\\dauserml\\Desktop\\dauserml_Messungen_2020_07_22\\meas1.pcapng.timestamp_wireshark.npy')
ff = convert_Signal_wireshark(self=0,Signal_1=ff)
ff = np.array(ff)
plt.figure(5)
plt.subplot(4,2,1);plt.plot(timestamp_wireshark[0],ff[:,[0]]);plt.subplot(4,2,2);plt.plot(timestamp_wireshark[0],ff[:,[2]]);plt.subplot(4,2,3);plt.plot(timestamp_wireshark[0],ff[:,[4]]);plt.subplot(4,2,4);plt.plot(timestamp_wireshark[0],ff[:,[6]]);plt.subplot(4,2,5);plt.plot(timestamp_wireshark[0],ff[:,[8]]);plt.subplot(4,2,6);plt.plot(timestamp_wireshark[0],ff[:,[10]]);plt.subplot(4,2,7);plt.plot(timestamp_wireshark[0],ff[:,[12]]);plt.subplot(4,2,8);plt.plot(timestamp_wireshark[0],ff[:,[14]])
plt.figure(6)
plt.subplot(4,2,1);plt.plot(timestamp_wireshark[0],ff[:,[16]]);plt.subplot(4,2,2);plt.plot(timestamp_wireshark[0],ff[:,[18]]);plt.subplot(4,2,3);plt.plot(timestamp_wireshark[0],ff[:,[20]]);plt.subplot(4,2,4);plt.plot(timestamp_wireshark[0],ff[:,[22]]);plt.subplot(4,2,5);plt.plot(timestamp_wireshark[0],ff[:,[24]]);plt.subplot(4,2,6);plt.plot(timestamp_wireshark[0],ff[:,[26]]);plt.subplot(4,2,7);plt.plot(timestamp_wireshark[0],ff[:,[28]]);plt.subplot(4,2,8);plt.plot(timestamp_wireshark[0],ff[:,[30]])
plt.show()