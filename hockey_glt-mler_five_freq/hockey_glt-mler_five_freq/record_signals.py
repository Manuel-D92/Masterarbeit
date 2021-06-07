'''
Created on 26.01.2017

@author: muellead
'''

from goalref.reader_interface import ReaderInterface
from time import sleep
# Use application config
import config

# Local config
outputPath = 'recorded.csv'
frequency = 1
ignoreSamples = 12500
everyNth = 25



print('Opening output file...')
received = 0
nextSample = everyNth
of = open(outputPath, 'w')

def callback(data):
    global ignoreSamples, received, nextSample, of, frequency
    if ignoreSamples > 0:
        ignoreSamples -= len(data)
        return
    
    for sample in data:
        nextSample -= 1
        if nextSample == 0:
            val = sample.getFrequency(frequency)
            of.write(';'.join(['%.10e;%.10e' % (x.real, x.imag) for x in val]) + '\n')
            received += 1
            nextSample = everyNth
    of.flush()

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
reader.enableConfiguration()

print('Requesting data...')
reader.requestData(callback, 1000, -1)

while True:
    print('Collecting data... wrote %d samples up to now.' % received)
    sleep(3)
