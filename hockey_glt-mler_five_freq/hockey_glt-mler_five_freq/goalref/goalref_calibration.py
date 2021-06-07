'''
******************************************************************************
* (c) 2015 Fraunhofer Institut Integrierte Schaltungen, Nuernberg
*          All Rights Reserved
******************************************************************************

Goalref calibration.
Simple python class for representing a calibration of the GoalRef system.
It contains offset values and reference values for calibration.
Offset is subtracted from a samples signal values.
Samples are divided by the reference (normalizes amplitude and corrects phase).

@author: muellead
'''

import numpy as np
from copy import deepcopy
from collections import deque
from .reader_interface import NUM_FREQUENCIES, NUM_CHANNELS
from .goalref_sample import GoalrefSample

class GoalrefCalibration:
    
    def __init__(self, currentChannel=None, noiseCancellation=[], bufferLen=100):
        '''
        Creates a new and empty (neutral) calibration object.
        Add offsets and reference values afterwards.
        @param currentChannel The index of the current sensing channel in the raw sample arrays 
            (if None, noise cancellation is disabled).
        @param noiseCancellation Noise cancellation configurations. For each frequency to process
            specify an tuple of (<target freq>, <ref freq 1>, [<ref freq 2>, [...]]).
        @param bufferLen Length of sample buffers to use for h-factor calculation. 
        '''
        self._offset = np.zeros((NUM_FREQUENCIES,NUM_CHANNELS), dtype=np.complex)
        self._reference = np.ones((NUM_FREQUENCIES,NUM_CHANNELS), dtype=np.complex)
        self._weights = [np.ones((len(x)-1, NUM_CHANNELS), dtype=np.complex)/(len(x)-1) for x in noiseCancellation]
        
        self._currentChannel = currentChannel
        self._noiseCancellation = np.array(noiseCancellation)
        self._bufferLen = bufferLen
        
        self._primaryFrequencies = set([x[0] for x in noiseCancellation])
        self._sampleBuffer = deque([], maxlen=bufferLen)
    
    def apply(self, sample, offset=True, reference=True, current=False):
        '''
        Applies the calibration to the given sample.
        Offset and reference calibration can be selectively enabled and disabled.
        '''
        data = sample.getRawData() if isinstance(sample, GoalrefSample) else sample
        if current and self._currentChannel is not None:
            # Apply noise cancellation
            # Take middle sample from buffer for actual processing
            self._sampleBuffer.append(deepcopy(data))
            data[:] = self._sampleBuffer[len(self._sampleBuffer)//2]
            
            if offset:
                current_feedback = data[:, self._currentChannel]
                h = self._calc_h(np.array(self._sampleBuffer), skipPrimary=True)
                for i, ncd in enumerate(self._noiseCancellation):
                    h_pred = np.sum(h[ncd[1:]] * self._weights[i], axis=0)
                    data[ncd[0]] -= h_pred * current_feedback[ncd[0]]
        else:
            # Apply conventional calibration
            if offset:
                data -= self._offset
            self._sampleBuffer.clear()
        # Reference calibration is the same for both approaches
        if reference:
            data /= self._reference
        
    def updateCurrentWeights(self, samples, updateOffset=False):
        '''!
        @brief Updates the stored weighting factors for noise cancellation channel prediction.
        @param samples A vector of samples taken with no localization objects in the field.
        @param updateOffset Automatically also update offset vector (set to average of given samples).
        '''
        if np.iscomplexobj(samples):
            data = samples
        else:
            data = np.array([sample.getRawData() for sample in samples])
        if updateOffset:
            self._offset = np.average(data, axis=0)
        h = self._calc_h(data)
        self._weights = []
        for ncd in self._noiseCancellation:
            self._weights.append(h[ncd[0]] / h[ncd[1:]] / (len(ncd)-1))
        
    def setOffset(self, offset):
        '''!
        @brief Set the signal offset. The offset is computed through the average of the sample raw data.
        @param offset: The offset vector to use for compensation.
        '''
        self._offset = offset
        
    def getOffset(self):
        return self._offset
        
    def setChannelReference(self, channel, reference):
        '''!
        @brief Set the phase/amplitude reference value.
        @param channel Channel to set the value for.
        @param reference Reference value to set.
        '''
        for f in range(NUM_FREQUENCIES):
            self._reference[f][channel] = reference[f]
            
    def getChannelReference(self, channel):
        out = np.zeros(NUM_FREQUENCIES, dtype=np.complex)
        for f in range(NUM_FREQUENCIES):
            out[f] = self._reference[f][channel]
        return np.array(out)
            
    def save(self, filename):
        '''!
        @brief Save the calibration represented by this object in the given file.
        @param filename Path of target file.
        '''
        f = open(filename, 'wb')
        np.save(f, np.array([self._offset, self._reference]))
        f.close()
        
    def load(self, filename):
        f = open(filename, 'rb')
        data = np.load(f, None)
        f.close()
        self._offset = data[0]
        self._reference = data[1]
        
    def printCalibration(self):
        print('Offsets:')
        print(self._offset)
        print('Reference values:')
        print(self._reference)
        
        
    def _calc_h(self, samples, skipPrimary=False):
        current_feedback = samples[:, :, self._currentChannel]
        h = np.empty((NUM_FREQUENCIES, NUM_CHANNELS), dtype=np.complex128)
        antennas_signal = np.transpose(samples, (1,2,0))
        for i in range(NUM_FREQUENCIES):
            if skipPrimary and i in self._primaryFrequencies:
                continue
            scaled_current_feedback = np.conjugate(current_feedback[:, i]) / np.vdot(current_feedback[:, i], current_feedback[:, i])
            h[i] = np.dot(antennas_signal[i], scaled_current_feedback)
        return h
        