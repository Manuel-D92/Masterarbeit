'''
Created on 13. Nov. 2016

@author: amueller
'''

from tkinter import N, S, W, E, IntVar, StringVar, CENTER, VERTICAL
from tkinter.ttk import Frame, Label, LabelFrame, Style, Button, Entry, Separator
from tkinter.font import nametofont
from tkinter.messagebox import showerror
import time
from queue import Queue
import numpy as np

from goalref.goal_detector import GoalDetector
from .plot_widgets import ScopeWidget

class ViewGoal(Frame):
    '''
    classdocs
    '''
    
    BLINK_DURATION = 0.5

    def __init__(self, master, application, reader, calibration, detectorConfig):
        '''
        Constructor
        '''
        Frame.__init__(self, master)
        self._application = application
        self._reader = reader
        self._calibration = calibration
        
        self._createWidgets()
        self._visible = True
        self._blinkUntil = time.time()-1
        
        self._main_sum_queue = Queue()
        self._frame_sum_queue = Queue()
        self._goal_queue = Queue()
        self._goalDetector = GoalDetector(reader, calibration, 
                                          self._goal_queue, self._main_sum_queue, self._frame_sum_queue,
                                          **detectorConfig)
        
        th = self._goalDetector.getThresholds()
        self._varMainThreshold.set(th[0])
        self._varMainCurrent.set(th[0])
        self._varFrameThreshold.set(th[1])
        self._varFrameCurrent.set(th[1])
        self._checkQueue()
        
    def activate(self, active):
        self._visible = active
        
    def _createWidgets(self):
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)
        self.rowconfigure(2, weight=2)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        
        style = Style()
        style.configure("Green.TLabel", background="green")
        style.configure("Red.TLabel", background="red")
        
        counterFrame = LabelFrame(self, text="Goals")
        counterFrame.grid(row=0, column=0, columnspan=2, sticky=W+E+N+S, padx=10, pady=(10,0))
        counterFrame.rowconfigure(0, weight=1)
        counterFrame.columnconfigure(0, weight=2)
        #counterFrame.columnconfigure(1, weight=5)
        #counterFrame.columnconfigure(2, weight=2)
        self._varCounter = IntVar(self, value=0)
        self._lblBarLeft = Label(counterFrame)
        self._lblBarLeft.grid(row=0, column=0, sticky=W+E+N+S, padx=10, pady=(10,0))
        self._lblCounter = Label(counterFrame, textvariable=self._varCounter, anchor=CENTER)
        #self._lblCounter.grid(row=0, column=1, sticky=W+E+N+S, pady=(10,0))
        font = nametofont("TkDefaultFont").copy()
        font.config(weight="bold", size=100)
        self._lblCounter["font"] = font
        self._lblBarRight = Label(counterFrame)
        #self._lblBarRight.grid(row=0, column=2, sticky=W+E+N+S, padx=10, pady=(10,0))
        self._btnResetCount = Button(counterFrame, text="Reset counter", command=self._resetCount)
        #self._btnResetCount.grid(row=1, column=0, columnspan=3, sticky=W+E, padx=10, pady=(5,10))
        
        thresholdFrame = LabelFrame(self, text="Threshold configuration")
        thresholdFrame.grid(row=1, column=0, columnspan=2, sticky=W+E, padx=10, pady=(5,0))
        thresholdFrame.columnconfigure(1, weight=1)
        thresholdFrame.columnconfigure(4, weight=1)
        
        titleMainThreshold = Label(thresholdFrame, text="Main threshold:")
        titleMainThreshold.grid(row=0, column=0, pady=(5,2), padx=(10,5))
        self._varMainThreshold = StringVar(self, value="")
        self._entryMainThreshold = Entry(thresholdFrame, textvariable=self._varMainThreshold)
        self._entryMainThreshold.grid(row=0, column=1, sticky=W+E, pady=(5,2))

        titleFrameThreshold = Label(thresholdFrame, text="Frame threshold:")
        titleFrameThreshold.grid(row=1, column=0, pady=2, padx=(10,5))
        self._varFrameThreshold = StringVar(self, value="")
        self._entryFrameThreshold = Entry(thresholdFrame, textvariable=self._varFrameThreshold)
        self._entryFrameThreshold.grid(row=1, column=1, sticky=W+E, pady=2)
        
        thresholdSeparator = Separator(thresholdFrame, orient=VERTICAL)
        thresholdSeparator.grid(row=0, column=2, rowspan=2, sticky=N+S, padx=10, pady=5)
        
        titleMainCurrent = Label(thresholdFrame, text="Current:")
        titleMainCurrent.grid(row=0, column=3, pady=(5,2))
        self._varMainCurrent = StringVar(self, value="")
        self._lblMainCurrent = Label(thresholdFrame, textvariable=self._varMainCurrent)
        self._lblMainCurrent.grid(row=0, column=4, sticky=W+E, padx=(5,10), pady=(5,2))
        
        titleFrameCurrent = Label(thresholdFrame, text="Current:")
        titleFrameCurrent.grid(row=1, column=3, pady=2)
        self._varFrameCurrent = StringVar(self, value="")
        self._lblFrameCurrent = Label(thresholdFrame, textvariable=self._varFrameCurrent)
        self._lblFrameCurrent.grid(row=1, column=4, sticky=W+E, padx=(5,10), pady=2)
        
        self._btnApplyThresholds = Button(thresholdFrame, text="Apply thresholds", command=self._updateThresholds)
        self._btnApplyThresholds.grid(row=2, column=0, columnspan=5, sticky=W+E, padx=10, pady=2)
        
        self._btnRecalibrateOffset = Button(thresholdFrame, text="Recalibrate offsets", command=self._application.doOffsetRecalibration)
        self._btnRecalibrateOffset.grid(row=3, column=0, columnspan=5, sticky=W+E, padx=10, pady=(2,5))
        
        mainSignalFrame = LabelFrame(self, text="Main antenna sum signal")
        mainSignalFrame.grid(row=2, column=0, sticky=W+E+N+S, padx=(10,5), pady=(5,10))
        self._plotMain = ScopeWidget(mainSignalFrame, yMin=-5, yMax=5, samples=5000, complex=False)
        self._plotMain.pack(side="top", expand=True, fill="both")
        self._plotMain.grid_propagate(False)
        
        frameSignalFrame = LabelFrame(self, text="Frame antenna sum signal")
        frameSignalFrame.grid(row=2, column=1, sticky=W+E+N+S, padx=(5,10), pady=(5,10))
        self._plotFrame = ScopeWidget(frameSignalFrame, yMin=-5, yMax=5, samples=5000, complex=False)
        self._plotFrame.pack(side="top", expand=True, fill="both")
        self._plotFrame.grid_propagate(False)
        
    def _resetCount(self):
        self._varCounter.set(0)
        
    def _checkQueue(self):
        while not self._main_sum_queue.empty():
            self._plotMain.updateData(np.imag(self._main_sum_queue.get()))
        while not self._frame_sum_queue.empty():
            self._plotFrame.updateData(np.imag(self._frame_sum_queue.get()))
        if self._visible:
            self._plotMain.updatePlot()
            self._plotFrame.updatePlot()
            
        while not self._goal_queue.empty():
            self._goal_queue.get()
            self._varCounter.set(self._varCounter.get() + 1)
            self._blinkUntil = time.time() + self.BLINK_DURATION
        
        if self._blinkUntil > time.time():
            self._lblBarLeft["style"] = "Green.TLabel"
            self._lblBarRight["style"] = "Green.TLabel"
        else:
            self._lblBarLeft["style"] = ""
            self._lblBarRight["style"] = ""
        
        self.after(50, self._checkQueue)
        
    def _updateThresholds(self):
        th = [self._varMainThreshold.get(), self._varFrameThreshold.get()]
        try:
            th = list(map(float, th))
        except ValueError:
            showerror('Invalid input', 'Failed to convert to floats: %s / %s' % tuple(th), parent=self._application)
            return
        
        self._goalDetector.setThresholds(th[0], th[1])
        th = self._goalDetector.getThresholds()
        
        self._varMainThreshold.set(th[0])
        self._varMainCurrent.set(th[0])
        self._varFrameThreshold.set(th[1])
        self._varFrameCurrent.set(th[1])
        

