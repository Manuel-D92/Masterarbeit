'''
Created on 26.01.2017

@author: muellead
'''

from goalref.reader_interface import ReaderInterface
from numpy.fft import fft, fftshift
from scipy.signal import get_window
import numpy as np
import matplotlib.pyplot as p
from matplotlib import animation
# Use application config
import config

# Local config
frequency = 1
channel = 1
fft_length = 1024
fft_window = get_window('blackman', 1024)

values = np.zeros(fft_length)

def callback(data):
    global frequency, channel, values
    tmp = np.array([sample.getSampleVal(channel, frequency) for sample in data])
    values = 20 * np.log10(np.abs(fftshift(fft(tmp * fft_window, fft_length)))/fft_length)

print('Setting up reader connection...')
# Connect to reader and configure it
reader = ReaderInterface(resistance=config.resistance, 
                         inductance=config.inductance, 
                         load_factor=config.loadFactor,
                         num_antennas=config.numAntennas,
                         channel_permutation=config.channelPermutation)
for idx, c in enumerate(config.channels):
    reader.setFrequency(idx, c[0])
    if c[1] > 0:
        reader.setExciterCurrent(idx, c[1])
        reader.setExciterEnabled(idx, True)
reader.setChannel20Switch(3) # Set channel 20 to current feedback
reader.setMainRelay(True)
reader.enableConfiguration()

print('Opening plot...')
fig = p.figure()
ax = fig.add_subplot(111)
ax.set_ylim((-160, 20))
ax.set_xlim((-1250, 1250))
ax.grid(True)
line, = ax.plot(np.linspace(-1250, 1250, fft_length), np.zeros(fft_length))

def animate(i):
    global values, line, reader
    if i % 30 == 0:
        reader.setMainRelay(not reader.getMainRelay())
    line.set_ydata(values)
    return line,

def init():
    global line
    line.set_ydata(np.ma.array(values, mask=True))
    return line,

print('Requesting data...')
reader.requestData(callback, len(fft_window), -1)

ani = animation.FuncAnimation(fig, animate, init_func=init, interval=(fft_length*1000/2500), blit=True)
p.show()
    