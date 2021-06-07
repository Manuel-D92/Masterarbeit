'''
Created on 13. Nov. 2016

@author: amueller
'''

from tkinter import N, S, W, E, IntVar, StringVar
from tkinter.ttk import Frame, Label, LabelFrame, OptionMenu, Checkbutton, Button, Entry
import numpy as np
from queue import Queue

from .plot_widgets import PolarWidget
from goalref.reader_interface import NUM_FREQUENCIES

class ViewVectors(Frame):
    '''
    classdocs
    '''

    def __init__(self, master, application, reader, calibration):
        '''
        Constructor
        '''
        Frame.__init__(self, master)
        self._application = application
        self._reader = reader
        self._calibration = calibration
        
        self.calibModes = ['None', 'Main', 'Frame']
        
        self.frequencies = ['All']
        for i in range(NUM_FREQUENCIES):
            self.frequencies.append("%d - %.1f kHz" % (i, reader.getFrequency(i)/1000.0))
        
        self._calOffset = True
        self._calReference = True
        self._calCurrent = False
        self._freq = 0
        
        self._createWidgets()
        self._visible = True
        
        self._queue = Queue()
        self._dataRequest = reader.requestData(self._processSamples, blockSize=100, blocks=-1)
        self._checkQueue()
        
    def activate(self, active):
        self._visible = active
        
    def _createWidgets(self):
        self.rowconfigure(3, weight=1)
        self.columnconfigure(0, weight=1)
        
        frequencyFrame = LabelFrame(self, text="Frequency selection")
        frequencyFrame.grid(row=0, column=0, sticky=W+E, padx=10, pady=(10,0))
        
        self._varFreq = StringVar(self, value=self.frequencies[1])
        self._lblFreqTitle = Label(frequencyFrame, text="Frequency:")
        self._lblFreqTitle.grid(row=0, column=0, sticky=W+E, padx=(10, 5), pady=5)
        self._menuFreq = OptionMenu(frequencyFrame, self._varFreq, self.frequencies[2], *self.frequencies)
        self._menuFreq.grid(row=0, column=1, sticky=W+E, pady=5)
        
        self._btnScale = Button(frequencyFrame, text="Auto-scale all", command=self._doAutoScale)
        self._btnScale.grid(row=0, column=2, sticky=E, padx=10, pady=5)
        
        calibrationFrame = LabelFrame(self, text="Calibration control")
        calibrationFrame.grid(row=1, column=0, sticky=W+E, padx=10, pady=(5,0))
        calibrationFrame.columnconfigure(0, weight=1)
        calibrationFrame.columnconfigure(1, weight=1)
        
        self._varCalOffset = IntVar(self, value=1)
        self._checkCalOffset = Checkbutton(calibrationFrame, variable=self._varCalOffset, text="Apply offset cal.")
        self._checkCalOffset.grid(row=0, column=0, sticky=W+E, padx=10, pady=5)
        self._varCalReference = IntVar(self, value=1)
        self._checkCalReference = Checkbutton(calibrationFrame, variable=self._varCalReference, text="Apply reference cal.")
        self._checkCalReference.grid(row=0, column=1, sticky=W+E, padx=10, pady=5)
        self._varCalCurrent = IntVar(self, value=0)
        self._checkCalCurrent = Checkbutton(calibrationFrame, variable=self._varCalCurrent, text="Apply noise cancellation")
        self._checkCalCurrent.grid(row=0, column=2, sticky=W+E, padx=10, pady=5)
        self._btnRecalibrateOffset = Button(calibrationFrame, text="Recalibrate offsets", command=self._application.doOffsetRecalibration)
        self._btnRecalibrateOffset.grid(row=1, column=0, columnspan=3, sticky=W+E, padx=10, pady=(0,5))
        
        antennaCtrlFrame = LabelFrame(self, text="Antenna control")
        antennaCtrlFrame.grid(row=2, column=0, sticky=W+E, padx=10, pady=5)
        antennaCtrlFrame.columnconfigure(1, weight=1)
        
        self._varGain = StringVar(self, value="32")
        self._lblGain = Label(antennaCtrlFrame, text="Antenna gain:")
        self._lblGain.grid(row=0, column=0, sticky=W+E, padx=10, pady=5)
        self._txtGain = Entry(antennaCtrlFrame, width=10, textvariable=self._varGain)
        self._txtGain.grid(row=0, column=1, sticky=W+E, padx=10, pady=5)
        self._btnSetGain = Button(antennaCtrlFrame, text="Set gain", command=self._setGain)
        self._btnSetGain.grid(row=0, column=2, sticky=W+E, padx=10, pady=5)
        
        self._varCalib = StringVar(self, value=self.calibModes[0])
        self._lblCalib = Label(antennaCtrlFrame, text="Cal. loop selection:")
        self._lblCalib.grid(row=1, column=0, sticky=W+E, padx=(10,5), pady=5)
        self._menuCalib = OptionMenu(antennaCtrlFrame, self._varCalib, self.calibModes[0], *self.calibModes)
        self._menuCalib.grid(row=1, column=1, sticky=W+E, padx=10, pady=5)
        self._btnSetGain = Button(antennaCtrlFrame, text="Set cal. loops", command=self._setCalib)
        self._btnSetGain.grid(row=1, column=2, sticky=W+E, padx=10, pady=5)
        
        vectorFrame = LabelFrame(self, text="Signal vectors")
        vectorFrame.grid(row=3, column=0, sticky=N+S+W+E, padx=10, pady=(5,10))
        vectorFrame.rowconfigure(0, weight=1)
        
        self._plots = []
        perRow = int(np.sqrt(self._reader.getNumAntennas()*2))
        for i in range(perRow):
            vectorFrame.columnconfigure(i, weight=1)
        row = -1
        for i in range(self._reader.getNumAntennas()*2):
            if i%perRow == 0:
                row += 1
                vectorFrame.rowconfigure(row, weight=1)
            frame = Frame(vectorFrame)
            frame.grid(row=row, column=(i%perRow), sticky=W+E+S+N)
            plot = PolarWidget(frame, rMax=5, pointerMode=True, pointers=NUM_FREQUENCIES, logarithmic=True)
            plot.pack(side="top", expand=True, fill="both")
            plot.grid_propagate(False)
            self._plots.append(plot)
        
    def _checkQueue(self):
        # Retrieve data from queue keeping only the last bunch of samples
        data = None
        while not self._queue.empty():
            data = self._queue.get()
        
        # Update plots with that sample
        if self._visible and data is not None:
            self._max = 0
            for i, plot in enumerate(self._plots):
                vals = np.average([sample.getChannel(i) for sample in data], axis=0)
                if self._freq != 0:
                    vals[:self._freq-1] = 0
                    vals[self._freq:] = 0
                self._max = max(np.max(np.abs(vals)), self._max)
                plot.updateData(vals)
                plot.updatePlot()
        
        # Update configuration data from GUI elements
        self._calOffset = self._varCalOffset.get() == 1
        self._calReference = self._varCalReference.get() == 1
        self._calCurrent = self._varCalCurrent.get() == 1
        self._freq = self.frequencies.index(self._varFreq.get())
        
        # Reschedule method execution
        self.after(25, self._checkQueue)

    def _processSamples(self, samples):
        out = []
        for sample in samples:
            self._calibration.apply(sample, offset=self._calOffset, reference=self._calReference, current=self._calCurrent)
            out.append(sample)
        
        self._queue.put(np.array(out))
        
    def _setGain(self):
        try:
            gain = int(self._varGain.get())
            for i in range(12):
                self._reader.setMainGain(i, gain)
                self._reader.setFrameGain(i, gain)
        except:
            pass
        
    def _setCalib(self):
        mode = self._varCalib.get()
        if mode == 'Main':
            for i in range(12):
                self._reader.setMainCalib(i, 1)
        elif mode == 'Frame':
            for i in range(12):
                self._reader.setFrameCalib(i, 1)
        else:
            for i in range(12):
                self._reader.setMainCalib(i, 0)
        
    def _doAutoScale(self):
        for plot in self._plots:
            plot.setScale(rMax=self._max*1.5)
