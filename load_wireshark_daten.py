import matplotlib.pyplot as plt
import numpy as np
import qualisys as quali
import math

File= "messung_cali_1"
path= "C:\\Users\\dauserml\\Desktop\\dauserml_Messungen_2020_07_22\\cali_2020_09_01"

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

ff = np.load('C:\\Users\\dauserml\\Desktop\\dauserml_Messungen_2020_07_22\\cali_2020_09_01\\Used_Data\\messung_cali_1.pcapng.cal.npy')
timestamp_wireshark = np.load('C:\\Users\\dauserml\\Desktop\\dauserml_Messungen_2020_07_22\\cali_2020_09_01\\Used_Data\\messung_cali_1.pcapng.timestamp_wireshark.npy')
ff = convert_Signal_wireshark(self=0,Signal_1=ff)
ff = np.array(ff)
plt.figure(5)
plt.gcf().subplots_adjust(bottom=0.2)
plt.subplot(4,2,1);plt.plot(timestamp_wireshark[0],ff[:,[0,1]]);plt.subplot(4,2,2);plt.plot(timestamp_wireshark[0],ff[:,[2,3]]);plt.subplot(4,2,3);plt.plot(timestamp_wireshark[0],ff[:,[4,5]]);plt.subplot(4,2,4);plt.plot(timestamp_wireshark[0],ff[:,[6,7]]);plt.subplot(4,2,5);plt.plot(timestamp_wireshark[0],ff[:,[8,9]]);plt.subplot(4,2,6);plt.plot(timestamp_wireshark[0],ff[:,[10,11]]);plt.subplot(4,2,7);plt.plot(timestamp_wireshark[0],ff[:,[12,13]]);plt.subplot(4,2,8);plt.plot(timestamp_wireshark[0],ff[:,[14,15]])
plt.figure(6)
plt.gcf().subplots_adjust(bottom=0.2)
plt.subplot(4,2,1);plt.plot(timestamp_wireshark[0],ff[:,[16]]);plt.subplot(4,2,2);plt.plot(timestamp_wireshark[0],ff[:,[18]]);plt.subplot(4,2,3);plt.plot(timestamp_wireshark[0],ff[:,[20]]);plt.subplot(4,2,4);plt.plot(timestamp_wireshark[0],ff[:,[22]]);plt.subplot(4,2,5);plt.plot(timestamp_wireshark[0],ff[:,[24]]);plt.subplot(4,2,6);plt.plot(timestamp_wireshark[0],ff[:,[26]]);plt.subplot(4,2,7);plt.plot(timestamp_wireshark[0],ff[:,[28]]);plt.subplot(4,2,8);plt.plot(timestamp_wireshark[0],ff[:,[30]])

Signal_1, Kopf, first_timestamp_wireshark_qualisys = quali.qualisys.load_convert_qulisys_Data(
    '%s\\' % path + 'Used_Data\\%s_6D.tsv' % File,
    delet_Null=0)
timestamp_qualisys = Signal_1[:, 1]
Signal_mean = quali.qualisys.x_y_z_translation_and_rotation(Signal_1, 1)


#plt.figure(94)
figure,ax=plt.subplots(5,1)

ax[0].plot(timestamp_wireshark[0,47400:50000],ff[47400:50000,24:26])
ax[0].set_ylabel("Main 7 \n Spannung in V")
ax[0].set_xlabel("Zeit in s")
ax[0].set_title("Antennensignal")
ax[0].legend(["Realteil","Imaginärteil"])

ax[1].plot(timestamp_wireshark[0,47400:50000],ff[47400:50000,26:28])
ax[1].set_ylabel("Frame 7 \n Spannung in mV")
ax[1].set_xlabel("Zeit in s")
#ax[1].set_title("Antennensignal Frame 1")

ax[2].plot(timestamp_wireshark[0,47400:50000],ff[47400:50000,28:30])
ax[2].set_ylabel("Main 8 \n Spannung in V")
ax[2].set_xlabel("Zeit in s")
ax[2].legend(["Realteil","Imaginärteil"])
#ax[2].set_title("Antennensignal Main 8")

ax[3].plot(timestamp_wireshark[0,47400:50000],ff[47400:50000,30:32])
ax[3].set_ylabel("Frame 8 \n Spannung in mV")
ax[3].set_xlabel("Zeit in s")
#ax[3].set_title("Antennensignal Frame 8")

ax[4].plot(timestamp_qualisys[1890:2100],Signal_mean[1890:2100,0])
ax[4].set_ylabel("X Position \n in mm")
ax[4].set_xlabel("Zeit in s")


plt.figure()
plt.subplot(2,1,1)
plt.plot(timestamp_wireshark[0,47400:50000],np.sqrt((ff[47400:50000,24])**2+(ff[47400:50000,25])**2),timestamp_wireshark[0,47400:50000],np.sqrt((ff[47400:50000,26])**2+(ff[47400:50000,27])**2))
print(str(timestamp_wireshark[0,47400+np.argmax(np.sqrt((ff[47400:50000,24])**2+(ff[47400:50000,25])**2))])+" "+ str(np.max(np.sqrt((ff[47400:50000,24])**2+(ff[47400:50000,25])**2))))
plt.legend(["Main 7","Frame 7"])
plt.subplot(2,1,2)
plt.plot(timestamp_wireshark[0,47400:50000],np.sqrt((ff[47400:50000,28])**2+(ff[47400:50000,29])**2),timestamp_wireshark[0,47400:50000],np.sqrt((ff[47400:50000,30])**2+(ff[47400:50000,31])**2))
print(str(timestamp_wireshark[0,47400+np.argmax(np.sqrt((ff[47400:50000,28])**2+(ff[47400:50000,29])**2))])+" "+ str(np.max(np.sqrt((ff[47400:50000,28])**2+(ff[47400:50000,29])**2))))
plt.legend(["Main 8","Frame 8"])

index_m = abs(np.array(Signal_mean[1890:2100,0])-600).argmin()
print("400")
print(str(timestamp_qualisys[1890+(index_m)]) + " " + str(Signal_mean[1890+(index_m)]) )
print(str(timestamp_qualisys[1890+(index_m)+1]) + " " + str(Signal_mean[1890+(index_m)+1]))

index_b = abs(np.array(Signal_mean[1890:2100,0])-375).argmin()
print("375")
print(str(timestamp_qualisys[1890+(index_b)]) + " " + str(Signal_mean[1890+(index_b)]) )
print(str(timestamp_qualisys[1890+(index_b)+1]) + " " + str(Signal_mean[1890+(index_b)+1]))

index_e = abs(np.array(Signal_mean[1890:2100,0])-845).argmin()
print("845")
print(str(timestamp_qualisys[1890+(index_e)]) + " " + str(Signal_mean[1890+(index_e)]) )
print(str(timestamp_qualisys[1890+(index_e)+1]) + " " + str(Signal_mean[1890+(index_e)+1]))

diff_wireshark = timestamp_wireshark[0,47400+np.argmax(np.sqrt((ff[47400:50000,24])**2+(ff[47400:50000,25])**2))]-((timestamp_wireshark[0,47400+np.argmax(np.sqrt((ff[47400:50000,24])**2+(ff[47400:50000,25])**2))]-timestamp_wireshark[0,47400+np.argmax(np.sqrt((ff[47400:50000,28])**2+(ff[47400:50000,29])**2))])/2)
diff = diff_wireshark-(timestamp_qualisys[1890+(index_m)])

print("diff= "+str(diff))

diff_qaulisys = timestamp_qualisys[1890+(index_b)]+((timestamp_qualisys[1890+(index_b)]-timestamp_qualisys[1890+(index_e)])/2)
print("diff_2= "+str(diff_wireshark-diff_qaulisys))

plt.show()