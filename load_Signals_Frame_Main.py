
import matplotlib.pyplot as plt
import numpy as np


Signal= np.load('C:\\Users\\dauserml\\Desktop\\dauserml_Messungen_2020_07_22\\test_Feld_1.npy');

fig = plt.figure(1)
plt.subplot(4,2,1)
plt.plot(Signal[:,1,0])
plt.subplot(4,2,2)
plt.plot(Signal[:,1,1])
plt.subplot(4,2,3)
plt.plot(Signal[:,1,2])
plt.subplot(4,2,4)
plt.plot(Signal[:,1,3])
plt.subplot(4,2,5)
plt.plot(Signal[:,1,4])
plt.subplot(4,2,6)
plt.plot(Signal[:,1,5])
plt.subplot(4,2,7)
plt.plot(Signal[:,1,6])
plt.subplot(4,2,8)
plt.plot(Signal[:,1,7])
plt.suptitle('Antennen 1 - 8')

plt.figure(2)
plt.subplot(4,2,1)
plt.plot(Signal[:,1,8])
plt.subplot(4,2,2)
plt.plot(Signal[:,1,9])
plt.subplot(4,2,3)
plt.plot(Signal[:,1,10])
plt.subplot(4,2,4)
plt.plot(Signal[:,1,11])
plt.subplot(4,2,5)
plt.plot(Signal[:,1,12])
plt.subplot(4,2,6)
plt.plot(Signal[:,1,13])
plt.subplot(4,2,7)
plt.plot(Signal[:,1,14])
plt.subplot(4,2,8)
plt.plot(Signal[:,1,15])
plt.suptitle('Antennen 9 -16')

plt.figure(3)
plt.subplot(2,1,1)
plt.plot(Signal[:,1,16])
plt.subplot(2,1,2)
plt.plot(Signal[:,1,17])
plt.suptitle('Strom Feedback')
plt.show()