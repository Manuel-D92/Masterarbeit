'''
Created on 13. Nov. 2016

@author: amueller
'''

from tkinter import N, S, W, E, StringVar, DISABLED, NORMAL
from tkinter.ttk import Frame, Label, Button, LabelFrame
from tkinter.filedialog import askopenfilename, asksaveasfilename
from tkinter.font import nametofont
from tkinter.messagebox import askquestion, QUESTION, YES
from threading import Lock
import os.path
import numpy as np

from .plot_widgets import PolarWidget
from goalref.goalref_calibration import GoalrefCalibration
from goalref.reader_interface import NUM_CHANNELS, NUM_FREQUENCIES

class ViewCalibration(Frame):
    '''
    classdocs
    '''
    
    POLAR_PLOT_FACTOR = 5e2

    def __init__(self, master, application, reader, calibration):
        '''!
        @brief
        @param master
        @param application
        @param reader
        @param calibration
        '''
        Frame.__init__(self, master)
        self._application = application
        self._reader = reader
        self._calibration = calibration
        
        self._calDir = "/"
        self._calFile = "calibration.npy"
        
        self._createStepDefinitions()
        self._createWidgets()
        
        self._newCal = GoalrefCalibration()
        self._currentStep = -1
        self._samples = None
        self._samplesReady = False
        self._sampleLock = Lock()
        self.after(200, self._checkSamples)
        
    def activate(self, active):
        pass
        
    def _createStepDefinitions(self):
        '''!
        @brief  In the following the steps necessary for calibration of CommonConf.numAntennas are generated.
                The resulting data structure is then used to provide the user with a step-by-step experience
                during calibration.
        '''

        self._calSteps = [
            {
                'message': 'Remove all interfering elements from the exciter field and press "Next step".',
                'buttonActive': True,
            },
            {
                'message': 'Collecting samples. Please wait...',
                'buttonActive': False,
                'dismissSamples': 100,
                'collectSamples': 1000,
                'finishAction': self._calibrateOffset
            },
        ]
        self._frameStepOffset = len(self._calSteps)
        for i in range(self._reader.getNumAntennas()):
            self._calSteps.append({
                        'message': 'Place ferrite besides FRAME antenna %s and press "Next step".' % chr(65 + i),
                        'buttonActive': True,
                    })
            self._calSteps.append({
                        'message': 'Collecting samples. Please wait...',
                        'buttonActive': False,
                        'dismissSamples': 100,
                        'collectSamples': 1000,
                        'finishAction': self._calibrateFrameAntennaReference
                    })
            self._calSteps.append({
                        'message': 'Remove all interfering elements from the exciter field and press "Next step".',
                        'buttonActive': True,
                    })
            self._calSteps.append({
                    'message': 'Collecting samples. Please wait...',
                    'buttonActive': False,
                    'dismissSamples': 100,
                    'collectSamples': 1000,
                    'finishAction': self._calibrateOffset
                    })
        self._mainStepOffset = len(self._calSteps)
        for i in range(self._reader.getNumAntennas()):
            self._calSteps.append({
                        'message': 'Place ferrite besides MAIN antenna %s and press "Next step".' % chr(65 + i),
                        'buttonActive': True,
                    })
            self._calSteps.append({
                        'message': 'Collecting samples. Please wait...',
                        'buttonActive': False,
                        'dismissSamples': 100,
                        'collectSamples': 1000,
                        'finishAction': self._calibrateMainAntennaReference
                    })
            self._calSteps.append({
                        'message': 'Remove all interfering elements from the exciter field and press "Next step".',
                        'buttonActive': True,
                    })
            self._calSteps.append({
                    'message': 'Collecting samples. Please wait...',
                    'buttonActive': False,
                    'dismissSamples': 100,
                    'collectSamples': 1000,
                    'finishAction': self._calibrateOffset
                    })
        self._calSteps.append({
                    'message': 'Calibration procedure finished. Press "Next step" to complete the calibration.',
                    'buttonActive': True,
                    'finishAction': self._store
                })
        
    def _createWidgets(self):
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        
        plotFrame = LabelFrame(self, text="Antenna response")
        plotFrame.grid(row=0, column=0, sticky=N+S+W+E, padx=10, pady=(10,0))
        self._plot = PolarWidget(plotFrame, rMax=1, pointerMode=True, pointers=NUM_FREQUENCIES)
        self._plot.pack(side="top", expand=True, fill="both")
        self._plot.grid_propagate(False)
        
        controlFrame = LabelFrame(self, text="Controls")
        controlFrame.grid(row=1, column=0, sticky=W+E, padx=10, pady=(5,10))
        controlFrame.columnconfigure(0, weight=0)
        controlFrame.columnconfigure(1, weight=1)
        controlFrame.columnconfigure(2, weight=0)
        
        self._varMsg = StringVar(value="No calibration in progress.")
        self._lblMsgTitle = Label(controlFrame, text="Message:")
        self._lblMsgTitle.grid(row=0, column=0, sticky=E, padx=10, pady=2)
        self._lblMsg = Label(controlFrame, textvariable=self._varMsg)
        font = nametofont("TkDefaultFont").copy()
        font.config(weight="bold")
        self._lblMsg["font"] = font
        self._lblMsg.grid(row=0, column=1, sticky=W)
        self._btnNext = Button(controlFrame, text="Start calibration", command=self._nextStep)
        self._btnNext.grid(row=0, column=2, sticky=W+E, padx=10, pady=2)
        
        self._varStep = StringVar(value=("0 / %d" % len(self._calSteps)))
        self._lblStepTitle = Label(controlFrame, text="Step:")
        self._lblStepTitle.grid(row=1, column=0, sticky=E, padx=10, pady=2)
        self._lblStep = Label(controlFrame, textvariable=self._varStep)
        self._lblStep.grid(row=1, column=1, sticky=W)
        self._btnCancel = Button(controlFrame, text="Cancel calibration", command=self._cancel, state=DISABLED)
        self._btnCancel.grid(row=1, column=2, sticky=W+E, padx=10, pady=2)
        
        self._btnRecalibrateOffset = Button(controlFrame, text="Recalibrate offsets", command=self._application.doOffsetRecalibration)
        self._btnRecalibrateOffset.grid(row=2, column=0, columnspan=3, sticky=W+E, padx=10, pady=2)
        
        self._btnRecalibratePhase = Button(controlFrame, text="Recalibrate phase (using frame antenna A)", command=lambda: self._application.doPhaseRecalibration(0)) 
        self._btnRecalibratePhase.grid(row=3, column=0, columnspan=3, sticky=W+E, padx=10, pady=2)
        
        saveLoadFrame = Frame(controlFrame)
        saveLoadFrame.grid(row=4, column=0, columnspan=3, sticky=W+E, padx=10, pady=5)
        saveLoadFrame.columnconfigure(0, weight=1)
        saveLoadFrame.columnconfigure(1, weight=1)
        
        self._btnLoad = Button(saveLoadFrame, text="Load from file...", command=self._loadCalibration)
        self._btnLoad.grid(row=0, column=0, sticky=W+E)
        
        self._btnSave = Button(saveLoadFrame, text="Save to file...", command=self._saveCalibration)
        self._btnSave.grid(row=0, column=1, sticky=W+E)

    def _nextStep(self):
        if self._currentStep >= 0:
            step = self._calSteps[self._currentStep]
            if 'finishAction' in step:
                step['finishAction']()
        
        self._currentStep += 1
        if self._currentStep == len(self._calSteps):
            self._cancel()
        else:
            if self._currentStep == 0:
                self._btnNext["text"] = "Next step"
                self._btnLoad["state"] = DISABLED
                self._btnSave["state"] = DISABLED
                self._btnRecalibrateOffset["state"] = DISABLED
                self._btnRecalibratePhase["state"] = DISABLED
                self._btnCancel["state"] = NORMAL
                
            step = self._calSteps[self._currentStep]
            self._varStep.set(("%d / %d" % (self._currentStep, len(self._calSteps))))
            self._varMsg.set(step["message"])
            
            if not step["buttonActive"]:
                self._btnNext["state"] = DISABLED
                self._btnCancel["state"] = DISABLED
            else:
                self._btnNext["state"] = NORMAL
                self._btnCancel["state"] = NORMAL
                
            if "collectSamples" in step:
                collect = step["collectSamples"]
                dismiss = step["dismissSamples"]
                self._reader.requestData(self._processSamples, collect+dismiss, 1)
                
    def _cancel(self):
        self._currentStep = -1
        
        self._btnNext["state"] = NORMAL
        self._btnNext["text"] = "Start calibration"
        self._varMsg.set("No calibration in progress.")
        self._varStep.set("0 / %d" % len(self._calSteps))
        
        self._btnLoad["state"] = NORMAL
        self._btnSave["state"] = NORMAL
        self._btnRecalibrateOffset["state"] = NORMAL
        self._btnRecalibratePhase["state"] = NORMAL
        self._btnCancel["state"] = DISABLED
        
    def _store(self):
        ok = askquestion("Calibration", "Do you want to apply the new calibration?", icon=QUESTION)
        if ok == YES:
            self._calibration.setOffset(self._newCal.getOffset())
            for i in range(NUM_CHANNELS):
                self._calibration.setChannelReference(i, self._newCal.getChannelReference(i))
            self._calibration.printCalibration()
    
    def _loadCalibration(self):
        fn = askopenfilename(initialdir=self._calDir, title="Load calibration...", parent=self,
                        filetypes=(("NumPy data files","*.npy"),("All files","*.*")))
        if fn == '':
            return
        self._calibration.load(fn)
        self._calibration.printCalibration()
        
        head, tail = os.path.split(fn)
        self._calDir = head
        self._calFile = tail
    
    def _saveCalibration(self):
        '''!
        @brief
        '''
        fn = asksaveasfilename(initialdir=self._calDir, initialfile=self._calFile, title="Save calibration...", parent=self,
                        defaultextension="npy", filetypes=(("NumPy data files","*.npy"),("All files","*.*")))
        if fn == '':
            return
        self._calibration.save(fn)
        
        head, tail = os.path.split(fn)
        self._calDir = head
        self._calFile = tail
        
    def _processSamples(self, samples):
        step = self._calSteps[self._currentStep]
        
        dismiss = step["dismissSamples"]
        with self._sampleLock:
            self._samples = samples[dismiss:]
            self._samplesReady = True
            
    def _checkSamples(self):
        with self._sampleLock:
            if self._samplesReady:
                self._nextStep()
                self._samplesReady = False
        self.after(200, self._checkSamples)
        
    def _calcAverage(self):
        '''!
        @brief Calculate the average of the sample raw data.
        '''
        print('Calculating average...')
        average = None
        for sample in self._samples:
            if average is None:
                average = sample.getRawData()
            else:
                average += sample.getRawData()
        average /= len(self._samples)
        return average
        
    def _calibrateOffset(self):
        '''!
        @brief Get the average of the sample raw data and set it as the signal offset.
        '''
        # Calculate average
        average = self._calcAverage()
        
        # Save in Calibration object
        print('Storing offset calibration...')
        self._newCal.setOffset(average)
        
    def _calibrateMainAntennaReference(self):
        '''!
        @brief
        '''
        # Calculate average
        average = self._calcAverage()
        
        # Calculate antenna indices
        antenna = (self._currentStep - self._mainStepOffset - 1) / 4
        channel = int(antenna*2+1)
        
        # Apply offset calibration and store result
        print('Storing reference calibration for channels %d...' % channel)
        average -= self._newCal.getOffset()
        self._newCal.setChannelReference(channel, np.array([average[f][channel] for f in range(NUM_FREQUENCIES)]))
        # Write result to polar chart
        self._plot.updateData(np.array([average[f][channel] for f in range(NUM_FREQUENCIES)]) * self.POLAR_PLOT_FACTOR)
        self._plot.updatePlot()
        
    def _calibrateFrameAntennaReference(self):
        '''!
        @brief
        '''
        # Calculate average
        average = self._calcAverage()
        
        # Calculate antenna indices
        antenna = (self._currentStep - self._frameStepOffset - 1) / 4
        channel = int(antenna*2)
        
        # Apply offset calibration and store result
        print('Storing reference calibration for channels %d...' % channel)
        average -= self._newCal.getOffset()
        self._newCal.setChannelReference(channel, np.array([average[f][channel] for f in range(NUM_FREQUENCIES)]))
        # Write result to polar chart
        self._plot.updateData(np.array([average[f][channel] for f in range(NUM_FREQUENCIES)]) * self.POLAR_PLOT_FACTOR)
        self._plot.updatePlot()
