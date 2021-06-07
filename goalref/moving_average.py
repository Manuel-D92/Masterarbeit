'''
Created on 15.01.2017

@author: muellead
'''

import numpy as np
from collections import deque

class MovingAverage(object):

    def __init__(self, length=100):
        self._length = length
        self._buffer = None
        self.clear()
        
    def processSamples(self, samples):
        """!
        @brief Add a vector of samples at once and update the average value.
        @param samples Vector of samples to add (first axis is assumed to be sample index).
        @return New average value.
        """
        # Cut off excess samples
        if len(samples) > self._length:
            self._buffer.clear()
            samples = samples[-self._length:]
        # Add samples to buffer
        for s in samples:
            self._buffer.append(s)
        # Calculate new average value
        self._averaged = np.average(self._buffer, axis=0)
        return self._averaged
        
    def processSample(self, sample):
        """!
        @brief Add the given sample to the average and return its new value.
        @param sample Sample to add.
        @return New average value.
        """
        # Add sample (possibly dropping the oldest from the buffer)
        self._buffer.append(sample)
        # Calculate new average value
        self._averaged = np.average(self._buffer, axis=0)
        return self._averaged
    
    def getSamples(self):
        """!
        @brief Get the current averaging buffer contents.
        @return Numpy array with first dimension being the sample index.
        """
        return np.array(self._buffer)
    
    def getCurrentAverage(self):
        """!
        @brief Gets the last calculated value of the average.
        @return Current average value.
        """
        return self._averaged
    
    def getCurrentLength(self):
        """!
        @brief Gets the number of samples currently in the buffer.
        @return Current averaging buffer length.
        """
        return len(self._buffer)
    
    def getNoisePower(self):
        """!
        @brief Gets the variance of the samples in the buffer.
        @return Sample variance.
        """
        # Iterate over buffer and sum over squared signal-mean-difference
        # This gives the noise energy in the buffered signal
        # Divide by length of buffer to get noise power (energy per sample, standard deviation)
        return np.sum(np.abs((np.array(self._buffer) - self._averaged) ** 2), axis=0) / (len(self._buffer) - 1)
    
    def clear(self):
        """!
        @brief Resets the averaging buffer contents and the average value.
        """
        if self._buffer is not None and len(self._buffer) > 0:
            self._averaged = np.zeros(self._buffer[0].shape)
        else:
            self._averaged = 0.0
        self._buffer = deque([], maxlen=self._length)
        