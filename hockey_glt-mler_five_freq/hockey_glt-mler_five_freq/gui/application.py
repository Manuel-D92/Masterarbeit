'''
Created on Nov 17, 2016

@author: muellead
'''

from threading import Condition
from tkinter.ttk import Frame, Notebook
import numpy as np

from .view_calibration import ViewCalibration
from .view_debug import ViewDebug
from .view_goal import ViewGoal
from .view_noise import ViewNoise
from .view_vectors import ViewVectors
from goalref.goalref_sample import GoalrefSample
from goalref.reader_interface import NUM_CHANNELS

class Application(Frame):
    def _createWidgets(self):
        self._tabs = Notebook(self, width=800, height=600)
        self._tabs.pack(side="top", expand=True, fill="both")
        
        self._viewGoal = ViewGoal(self._tabs, self, self._reader, self._calibration, self._detectorConfig)
        self._tabs.add(self._viewGoal, text="Goal Detection")
        
        self._viewCalibration = ViewCalibration(self._tabs, self, self._reader, self._calibration)
        self._tabs.add(self._viewCalibration, text="Calibration")
        
        self._viewDebug = ViewDebug(self._tabs, self, self._reader, self._calibration, self._debugNoiseWindowLen)
        self._tabs.add(self._viewDebug, text="Debug")
        
        self._viewNoise = ViewNoise(self._tabs, self, self._reader, self._calibration, self._noiseNoiseWindowLen)
        self._tabs.add(self._viewNoise, text="Noise")
        
        self._viewVectors = ViewVectors(self._tabs, self, self._reader, self._calibration)
        self._tabs.add(self._viewVectors, text="Vectors")
        
        self._tabs.bind('<<NotebookTabChanged>>', self._tabChanged)
        
        self._calCondition = Condition()
        self._calPhaseChannel = 0
        
    def __init__(self, master, reader, calibration, detectorConfig, debugNoiseWindowLen, noiseNoiseWindowLen):
        Frame.__init__(self, master)
        '''!
        @brief
        @param master
        @param reader
        @param calibration Calibration which is created with the goalref_calibration tool.
        @param detectorConfig
        @paramdebugNoiseWindowLen
        @param noiseNoiseWindowLen
        '''
        
        self._reader = reader
        self._calibration = calibration
        self._detectorConfig = detectorConfig
        self._debugNoiseWindowLen = debugNoiseWindowLen
        self._noiseNoiseWindowLen = noiseNoiseWindowLen
        
        self._createWidgets()
        
    def _tabChanged(self, evt):
        tab_idx = self._tabs.index(self._tabs.select())
        tabs = self._tabs.winfo_children()
        
        for i in range(len(tabs)):
            tabs[i].activate(i == tab_idx)
            
    def doOffsetRecalibration(self):
        with self._calCondition:
            request = self._reader.requestData(self._processSamplesOffset, blockSize=1000, blocks=1)
            if not self._calCondition.wait(timeout=3):
                self._reader.cancelRequest(request)
                
    def doPhaseRecalibration(self, channel):
        with self._calCondition:
            self._calPhaseChannel = channel
            request = self._reader.requestData(self._processSamplesPhase, blockSize=1000, blocks=1)
            if not self._calCondition.wait(timeout=3):
                self._reader.cancelRequest(request)
        
    def _processSamplesOffset(self, samples):
        with self._calCondition:
            self._calibration.updateCurrentWeights(samples, updateOffset=True)
            self._calCondition.notify_all()
        
    def _processSamplesPhase(self, samples):
        average = None
        for sample in samples:
            if average is None:
                average = sample.getRawData()
            else:
                average += sample.getRawData()
        average /= len(samples)
        
        sample = GoalrefSample(average)
        phase_ref = sample.getChannel(self._calPhaseChannel)
        phase_ref /= np.abs(phase_ref)
        
        old_ref = self._calibration.getChannelReference(self._calPhaseChannel)
        old_ref /= np.abs(old_ref)
        
        phase_shift = phase_ref/old_ref
        
        with self._calCondition:
            for c in range(NUM_CHANNELS):
                old = self._calibration.getChannelReference(c)
                new = old * phase_shift
                self._calibration.setChannelReference(c, new)
            self._calCondition.notify_all()
    