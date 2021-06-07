'''
Created on 13. Nov. 2016

@author: amueller
'''

from tkinter import N, S, W, E, IntVar, StringVar, VERTICAL
from tkinter.ttk import Frame, Label, LabelFrame, Separator, OptionMenu, Checkbutton, Button
import numpy as np
from queue import Queue

from .plot_widgets import ScopeWidget
from .plot_widgets import PolarWidget
from goalref.moving_average import MovingAverage
from goalref.reader_interface import NUM_FREQUENCIES

class ViewDebug(Frame):
    '''
    classdocs
    '''
    
    main_frame = ("Main", "Frame")
    scope_polar = ("Scope", "Polar", "Both", "None")
    clock_modes = ('AUTO', 'MASTER', 'SLAVE', 'FAILSAFE')

    def __init__(self, master, application, reader, calibration, noiseWindowLen):
        '''
        Constructor
        '''
        Frame.__init__(self, master)
        self._application = application
        self._reader = reader
        self._calibration = calibration
        
        self.antennas = []
        for i in range(reader.getNumAntennas()):
            self.antennas.append(chr(ord('A') + i))
        self.frequencies = []
        for i in range(NUM_FREQUENCIES):
            self.frequencies.append("%d - %.1f kHz" % (i, reader.getFrequency(i)/1000.0))
        
        self._movingAverage = MovingAverage(noiseWindowLen)
        self._movingAverageUpdateCnt = 0
        
        self._scopePolarIdx = 0
        self._calOffset = True
        self._calReference = True
        self._calCurrent = False
        self._freq = 0
        self._channel = 0
        self._switchDelay = 0
        
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
        
        antennaFrame = LabelFrame(self, text="Antenna selection")
        antennaFrame.grid(row=0, column=0, sticky=W+E, padx=10, pady=(10,0))
        antennaFrame.columnconfigure(0, weight=0)
        antennaFrame.columnconfigure(1, weight=1)
        antennaFrame.columnconfigure(2, weight=0)
        antennaFrame.columnconfigure(3, weight=0)
        antennaFrame.columnconfigure(4, weight=1)
        antennaFrame.columnconfigure(5, weight=0)
        antennaFrame.columnconfigure(6, weight=0)
        antennaFrame.columnconfigure(7, weight=1)
        
        self._varFreq = StringVar(self, value=self.frequencies[1])
        self._lblFreqTitle = Label(antennaFrame, text="Frequency:")
        self._lblFreqTitle.grid(row=0, column=0, sticky=W+E, padx=(10, 5), pady=5)
        self._menuFreq = OptionMenu(antennaFrame, self._varFreq, self.frequencies[1], *self.frequencies)
        self._menuFreq.grid(row=0, column=1, sticky=W+E, pady=5)
        
        sep1 = Separator(antennaFrame, orient=VERTICAL)
        sep1.grid(row=0, column=2, padx=10, pady=5, sticky=N+S)
        
        self._varMainFrame = StringVar(self, value=self.main_frame[0])
        self._lblMainFrameTitle = Label(antennaFrame, text="Main/Frame:")
        self._lblMainFrameTitle.grid(row=0, column=3, sticky=W+E, padx=(0, 5), pady=5)
        self._menuMainFrame = OptionMenu(antennaFrame, self._varMainFrame, self.main_frame[0], *self.main_frame)
        self._menuMainFrame.grid(row=0, column=4, sticky=W+E, pady=5)
        
        sep2 = Separator(antennaFrame, orient=VERTICAL)
        sep2.grid(row=0, column=5, padx=10, pady=5, sticky=N+S)
        
        self._varAntenna = StringVar(self, value=self.antennas[0])
        self._lblAntennaTitle = Label(antennaFrame, text="Antenna:")
        self._lblAntennaTitle.grid(row=0, column=6, sticky=W+E, padx=(0, 5), pady=5)
        self._menuAntenna = OptionMenu(antennaFrame, self._varAntenna, self.antennas[0], *self.antennas)
        self._menuAntenna.grid(row=0, column=7, sticky=W+E, padx=(0, 10), pady=5)
        
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
        
        clockFrame = LabelFrame(self, text="Clock control")
        clockFrame.grid(row=2, column=0, sticky=W+E, padx=10, pady=(5,0))
        clockFrame.columnconfigure(0, weight=1)
        clockFrame.columnconfigure(1, weight=1)
        clockFrame.columnconfigure(2, weight=1)
        clockFrame.columnconfigure(3, weight=1)
        
        self._varLFSY = IntVar(self, value=0)
        self._checkLFSY = Checkbutton(clockFrame, variable=self._varLFSY, text="LFSync enable", command=self._clockCtrl)
        self._checkLFSY.grid(row=0, column=0, sticky=W+E, padx=10, pady=5)
        self._varSATA = IntVar(self, value=0)
        self._checkSATA = Checkbutton(clockFrame, variable=self._varSATA, text="SATA enable", command=self._clockCtrl)
        self._checkSATA.grid(row=0, column=1, sticky=W+E, padx=10, pady=5)
        self._varFITX = IntVar(self, value=0)
        self._checkFITX = Checkbutton(clockFrame, variable=self._varFITX, text="Fiber TX enable", command=self._clockCtrl)
        self._checkFITX.grid(row=0, column=2, sticky=W+E, padx=10, pady=5)
        self._varFIRX = IntVar(self, value=0)
        self._checkFIRX = Checkbutton(clockFrame, variable=self._varFIRX, text="Fiber RX enable", command=self._clockCtrl)
        self._checkFIRX.grid(row=0, column=3, sticky=W+E, padx=10, pady=5)
        self._varCLO1 = IntVar(self, value=0)
        self._checkCLO1 = Checkbutton(clockFrame, variable=self._varCLO1, text="CLO1", command=self._clockCtrl)
        self._checkCLO1.grid(row=1, column=0, sticky=W+E, padx=10, pady=5)
        self._varCLO0 = IntVar(self, value=0)
        self._checkCLO0 = Checkbutton(clockFrame, variable=self._varCLO0, text="CLO0", command=self._clockCtrl)
        self._checkCLO0.grid(row=1, column=1, sticky=W+E, padx=10, pady=5)
        self._varCLI1 = IntVar(self, value=0)
        self._checkCLI1 = Checkbutton(clockFrame, variable=self._varCLI1, text="CLI1", command=self._clockCtrl)
        self._checkCLI1.grid(row=1, column=2, sticky=W+E, padx=10, pady=5)
        self._varCLI0 = IntVar(self, value=0)
        self._checkCLI0 = Checkbutton(clockFrame, variable=self._varCLI0, text="CLI0", command=self._clockCtrl)
        self._checkCLI0.grid(row=1, column=3, sticky=W+E, padx=10, pady=5)
        self._varClockMode = StringVar(self, value=self.clock_modes[0])
        self._lblFreqTitle = Label(clockFrame, text="Clock mode:")
        self._lblFreqTitle.grid(row=2, column=0, columnspan=2, sticky=W+E, padx=(10, 5), pady=5)
        self._menuFreq = OptionMenu(clockFrame, self._varClockMode, self.clock_modes[0], *self.clock_modes, command=self._clockMode)
        self._menuFreq.grid(row=2, column=2, columnspan=2, sticky=W+E, pady=5)
        
        plotFrame = LabelFrame(self, text="Antenna signal")
        plotFrame.grid(row=3, column=0, sticky=N+S+W+E, padx=10, pady=(5,10))
        plotFrame.rowconfigure(0, weight=0)
        plotFrame.rowconfigure(1, weight=1)
        plotFrame.columnconfigure(0, weight=1)
        plotFrame.columnconfigure(1, weight=1)
        
        plotConfig = Frame(plotFrame)
        plotConfig.grid(row=0, column=0, sticky=W+E, padx=10)
        self._varScopePolar = StringVar(self, value=self.scope_polar[self._scopePolarIdx])
        self._lblScopePolarTitle = Label(plotConfig, text="Plot type:")
        self._lblScopePolarTitle.grid(row=0, column=0, sticky=W+E, padx=(0, 5), pady=5)
        self._menuScopePolar = OptionMenu(plotConfig, self._varScopePolar, self.scope_polar[self._scopePolarIdx], *self.scope_polar)
        self._menuScopePolar.grid(row=0, column=1, sticky=W+E, pady=5)
        
        noisePower = Frame(plotFrame)
        noisePower.grid(row=0, column=1, sticky=W+E, padx=10)
        self._varNoisePower = StringVar(self, value="")
        self._lblNoisePowerTitle = Label(noisePower, text="Noise std. dev.:")
        self._lblNoisePowerTitle.grid(row=0, column=0, sticky=W+E, padx=(0,5), pady=5)
        self._lblNoisePower = Label(noisePower, textvariable=self._varNoisePower)
        self._lblNoisePower.grid(row=0, column=1, sticky=W+E, pady=5)
        
        self._scopeFrame = Frame(plotFrame)
        self._scopeFrame.grid(row=1, column=0, columnspan=2, sticky=W+E+S+N)
        self._plotScope = ScopeWidget(self._scopeFrame, yMin=-10, yMax=10, samples=7500, complex=True)
        self._plotScope.pack(side="top", expand=True, fill="both")
        self._plotScope.grid_propagate(False)
        
        self._polarFrame = Frame(plotFrame)
        self._plotPolar = PolarWidget(self._polarFrame, rMax=5, pointerMode=True, pointers=1)
        self._plotPolar.pack(side="top", expand=True, fill="both")
        self._plotPolar.grid_propagate(False)
        
    def _checkQueue(self):
        # Check for updated plot type selection
        if self.scope_polar.index(self._varScopePolar.get()) != self._scopePolarIdx:
            self._scopePolarIdx = self.scope_polar.index(self._varScopePolar.get())
            # New plot type selected -> display corresponding plot(s)
            if self._scopePolarIdx == 0:
                self._polarFrame.grid_forget()
                self._scopeFrame.grid(row=1, column=0, columnspan=2, sticky=W+E+S+N)
            elif self._scopePolarIdx == 1:
                self._scopeFrame.grid_forget()
                self._polarFrame.grid(row=1, column=0, columnspan=2, sticky=W+E+S+N)
            elif self._scopePolarIdx == 2:
                self._scopeFrame.grid(row=1, column=0, sticky=W+E+S+N)
                self._polarFrame.grid(row=1, column=1, sticky=W+E+S+N)
            else:
                self._scopeFrame.grid_forget()
                self._polarFrame.grid_forget()
        
        # Retrieve data from queue and forward to processing
        while not self._queue.empty():
            data = self._queue.get()
            self._plotScope.updateData(data)
            self._plotPolar.updateData(np.array([data[-1]]))
            self._movingAverage.processSamples(data)
            
        # Update plots if visible
        if self._visible and self._scopePolarIdx == 0:
            self._plotScope.updatePlot()
        elif self._visible and self._scopePolarIdx == 1:
            self._plotPolar.updatePlot()
        elif self._visible and self._scopePolarIdx == 2:
            self._plotScope.updatePlot()
            self._plotPolar.updatePlot()
            
        # Update noise power value
        self._movingAverageUpdateCnt += 1
        if self._movingAverageUpdateCnt >= 25 and self._visible:
            self._varNoisePower.set("%.5e (from %d samples with mean %.5e)" % (np.sqrt(self._movingAverage.getNoisePower()), self._movingAverage.getCurrentLength(), np.abs(self._movingAverage.getCurrentAverage())))
            self._movingAverageUpdateCnt = 0
        
        # Update configuration data from GUI elements
        before = [self._calOffset, self._calReference, self._calCurrent, self._freq, self._channel]
        cc = self.antennas.index(self._varAntenna.get()) * 2
        if self.main_frame.index(self._varMainFrame.get()) == 0:
            cc += 1
        after = [self._varCalOffset.get() == 1, self._varCalReference.get() == 1, self._varCalCurrent.get() == 1, 
                 self.frequencies.index(self._varFreq.get()), cc]
        # Add switching delay to work around bug in newer Tkinter versions, which sometimes glitch out and return wrong Option menu selections
        if before != after:
            self._switchDelay += 1
            if self._switchDelay >= 3:
                self._calOffset, self._calReference, self._calCurrent, self._freq, self._channel = after
                while not self._queue.empty():
                    self._queue.get()
                self._movingAverage.clear()
                self._switchDelay = 0
        else:
            self._switchDelay = 0
        
        # Reschedule method execution
        self.after(25, self._checkQueue)

    def _processSamples(self, samples):
        out = []
        for sample in samples:
            self._calibration.apply(sample, offset=self._calOffset, reference=self._calReference, current=self._calCurrent)
            out.append(sample.getSampleVal(chan=self._channel, freq=self._freq))
        
        self._queue.put(np.array(out))
        
    def _clockCtrl(self):
        self._reader.setClockControl(
            self._varLFSY.get() == 1,
            self._varSATA.get() == 1,
            self._varFITX.get() == 1,
            self._varFIRX.get() == 1,
            self._varCLO1.get() == 1,
            self._varCLO0.get() == 1,
            self._varCLI1.get() == 1,
            self._varCLI0.get() == 1)
            
    def _clockMode(self, _):
        self._reader.setClockMode(self.clock_modes.index(self._varClockMode.get()))
