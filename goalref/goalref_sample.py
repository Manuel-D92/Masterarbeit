'''
******************************************************************************
* (c) 2015 Fraunhofer Institut Integrierte Schaltungen, Nuernberg
*          All Rights Reserved
******************************************************************************

Represents one sample from the GoalRef frontend and provides accessors
to each channel's data.

@author: muellead
'''

import numpy

class GoalrefSample:
    
    def __init__(self, sample, numAntennas = None):
        '''
        Creates a sample based on a numpy array constructed
        from raw received data.
        To use antenna oriented accessors the number of antennas
        must be given.
        '''
        self._sample = sample
        self._numAntennas = numAntennas
        
    def applyPermutation(self, channelPermutation):
        '''
        Applies the given array as a channel permutation to this sample's values.
        The array must contain NUM_CHANNELS elements as this sample's data will
        otherwise be incomplete.
        '''
        self._sample = self._sample[:,channelPermutation]
        
    def getRawData(self):
        return self._sample
    
    def getChannel(self, chan):
        '''
        Gets the sample data for all frequencies of the given channel number.
        Returns a numpy array of NUM_FREQ complex sample values.
        '''
        return numpy.array([self._sample[f][chan] for f in range(self._sample.shape[0])])
    
    def getSampleVal(self, chan, freq):
        '''
        Gets a single sample value for the given channel and frequency.
        '''
        return self._sample[freq][chan]
    
    def getFrequencyMain(self, freq):
        '''
        Gets all main antenna sample values for the given frequency.
        '''
        if not self._numAntennas:
            raise Exception('Antenna count unknown')
        return self._sample[freq][1:self._numAntennas*2:2]
    
    def getFrequencyFrame(self, freq):
        '''
        Gets all frame antenna sample values for the given frequency.
        '''
        if not self._numAntennas:
            raise Exception('Antenna count unknown')
        return self._sample[freq][0:self._numAntennas*2:2]
    
    def getFrequency(self, freq):
        '''
        Gets all samples values (main and frame in raw channel order) for the given frequency.
        '''
        if not self._numAntennas:
            raise Exception('Antenna count unknown')
        return self._sample[freq][0:self._numAntennas*2]
    
    def printSample(self):
        print('Sample data:')
        print(self._sample)
        