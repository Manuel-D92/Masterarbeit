'''
Created on 26.01.2017

@author: muellead
'''

from goalref.reader_interface import ReaderInterface
from time import sleep
import numpy as np
import matplotlib.pylab as p

antenna = 'C'

antenna = ord(antenna)-ord('A')
rawdata = None
received = 0

def callback(data):
    global rawdata, received
    rawdata = data
    received += 1

print('Setting up reader connection...')
reader = ReaderInterface(num_antennas=7,
    channel_permutation=[ 1,  0,  3,  2,  5,  4,  7,  6,  9,  8, 
                      11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 
                      20])
reader.setFrequency(1, 119000)
reader.setExciterGain(1, 3665)
reader.setExciterEnabled(1, True)
reader.enableConfiguration()

print('Requesting data...')
reader.requestData(callback, 10000, 2)

while received < 2:
    print('Waiting for data...')
    sleep(1)

print('Plotting...')
main = np.array([s.getSampleVal(2*antenna+1, 1) for s in rawdata])
frame = np.array([s.getSampleVal(2*antenna, 1) for s in rawdata])

print('Main means: Real %e, Imag %e' % (np.average(np.real(main)), np.average(np.imag(main))))
print('Frame means: Real %e, Imag %e' % (np.average(np.real(frame)), np.average(np.imag(frame))))

p.figure()
p.subplot(121, polar=True)
avg = np.average(np.real(main)) + 1j*np.average(np.imag(main))
p.plot([0, np.angle(avg)], [0, np.abs(avg)] , c='r')
avg = np.average(np.real(frame)) + 1j*np.average(np.imag(frame))
p.plot([0, np.angle(avg)], [0, np.abs(avg)] , c='b')
p.grid(True)
p.legend()

p.subplot(122)
p.scatter(np.real(main)-np.average(np.real(main)), np.imag(main)-np.average(np.imag(main)), c='r', label='Main')
p.scatter(np.real(frame)-np.average(np.real(frame)), np.imag(frame)-np.average(np.imag(frame)), c='b', label='Frame')
p.legend()
p.xlim((-0.0003, 0.0003))
p.ylim((-0.0003, 0.0003))
p.grid(True)
p.show()