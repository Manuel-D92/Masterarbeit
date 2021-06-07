'''
Created on 25.11.2016

@author: muellead
'''

import numpy as np
from .moving_average import MovingAverage

class GoalDetector(object):
    '''!
    @brief
    '''

    def __init__(self, reader, calibration,
                 goal_queue=None, main_sum_queue=None, frame_sum_queue=None,
                 ddft_pairs=[], moving_average_len=1,
                 invert_main=False, invert_frame=False,
                 main_live_threshold=10, frame_inside_threshold=10,
                 main_offset_factor=0):
        '''!
        @brief
        @param reader
        @param calibration Calibration which is created with the goalref_calibration tool. Contains ...
        @param goal_queue
        @param main_sum_queue
        @param frame_sum_queue
        @param ddft_pairs
        @param moving_average_len
        @param invert_main
        @param invert_frame
        @param main_live_threshold
        @param frame_inside_threshold
        @param main_offset_factor=0
        '''
        self._reader = reader
        self._calibration = calibration
        
        self._goal_queue = goal_queue
        self._main_sum_queue = main_sum_queue
        self._frame_sum_queue = frame_sum_queue
        
        self._ddft_pairs = ddft_pairs
        self._invert_main = invert_main
        self._invert_frame = invert_frame
        self._main_live_threshold = main_live_threshold
        self._frame_inside_threshold = frame_inside_threshold
        self._main_offset_factor = main_offset_factor
        
        self._moving_average = MovingAverage(moving_average_len)
        
        self._sample_count = 0
        self._detection_live = False
        self._frame_sum_positive_samples = 0
        
        reader.requestData(self._processSamples, blockSize=100, blocks=-1)
        
    def setThresholds(self, main_live_threshold, frame_inside_threshold):
        """!
        @brief Set the thresholds ...
        @param main_live_threshold
        @param frame_inside_threshold
        """
        self._main_live_threshold = main_live_threshold
        self._frame_inside_threshold = frame_inside_threshold
        
    def getThresholds(self):
        """!
        @brief Returns the thresholds ...
        @return main_live_threshold
        @return frame_inside_threshold
        """
        return self._main_live_threshold, self._frame_inside_threshold

    def _processSamples(self, samples):
        """!
        @brief
        @param samples Samples of the reader
        """
        filteredSignal = []
        mainSumSignal = []
        frameSumSignal = []
        
        # For each sample apply the calibration and perform moving average calculation
        for sample in samples:
            self._calibration.apply(sample)

            # Initialize signal values
            signal = np.zeros((2,self._reader.getNumAntennas()), dtype=np.complex)
            
            # Process all defined DDFTs
            for pos, neg, weight in self._ddft_pairs:
                # Calculate main antenna DDFT
                positive = sample.getFrequencyMain(pos)
                if neg >= 0:
                    negative = sample.getFrequencyMain(neg)
                    signal[1] += (positive - negative) * weight
                else:
                    signal[1] += positive * weight
                
                # Calculate frame antenna DDFT
                positive = sample.getFrequencyFrame(pos)
                if neg >= 0:
                    negative = sample.getFrequencyFrame(neg)
                    signal[0] += (positive - negative) * weight
                else:
                    signal[0] += positive * weight
            
            # Apply moving average
            averaged = self._moving_average.processSample(signal)
            filteredSignal.append(averaged)
            
            # Adaptive main offset
            mainOffset = 1j*np.imag(np.sum(averaged[0]) * self._main_offset_factor)
            
            # Sum over all antennas and append the result to the sum signal vector
            frameSumSignal.append(np.sum(averaged[0]))
            mainSumSignal.append(np.sum(averaged[1]) - mainOffset)
            
            
        # Filter the signals and forward them through the corresponding queues
        mainSumSignal = np.array(mainSumSignal)
        if self._invert_main:
            mainSumSignal = -mainSumSignal
        if self._main_sum_queue is not None:
            self._main_sum_queue.put(mainSumSignal)
        
        frameSumSignal = np.array(frameSumSignal)
        if self._invert_frame:
            frameSumSignal = -frameSumSignal
        if self._frame_sum_queue is not None:
            self._frame_sum_queue.put(frameSumSignal)

        # Loop over single samples
        for i in range(len(mainSumSignal)):
            main = np.imag(mainSumSignal[i])
            frame = np.imag(frameSumSignal[i])
            
            # Check if main signal imaginary part exceeds live threshold
            if main < -self._main_live_threshold:
                self._detection_live = True
                
            # Check if frame signal imaginary part is positive
            if frame > 0:
                self._frame_sum_positive_samples += 1
            else:
                self._frame_sum_positive_samples = 0
                
            # Check if frame signal imaginary part exceeds inside threshold
            inside = False
            if (frame > self._frame_inside_threshold and
                self._frame_sum_positive_samples > 20):
                inside = True
        
            # Check for positive main sum
            if main > 0 and self._detection_live:
                self._detection_live = False
                
                # Goal, only if ball was inside
                if inside:
                    self._goal_queue.put(self._sample_count)
                    print('Goal at %d' % self._sample_count)
                    
            # Count processed samples
            self._sample_count += 1
    