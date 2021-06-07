'''
Created on Nov 12, 2016

@author: muellead
'''

from tkinter import Frame, N, S, W, E
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Matplotlib 2.2.0 harmonized Tk backends and removed NavigationToolbar2TkAgg in favor of NavigationToolbar2Tk
# To keep compatible to older versions, resort to old name if import fails
try:
    from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
except ImportError:
    from matplotlib.backends.backend_tkagg import NavigationToolbar2TkAgg as NavigationToolbar2Tk

class PlotWidget(Frame):
    def __init__(self, master=None, figsize=(5,3), dpi=72):
        Frame.__init__(self, master)
        
        self._fig = Figure(figsize=figsize, dpi=dpi)
        self._fig.set_facecolor([x/65536.0 for x in self.winfo_rgb(self.cget("bg"))])
        
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self._createWidgets()

        self._fig.canvas.mpl_connect('draw_event', self._onDraw)
        self._fig.canvas.mpl_connect('resize_event', self._onResize)
        
    def _onResize(self, evt):
        self._initPlot()
        
    def _onDraw(self, evt):
        self._background = self._fig.canvas.copy_from_bbox(self._ax.bbox)
        self.updatePlot()
        
    def _initPlot(self):
        pass
    
    def updateData(self):
        pass
    
    def updatePlot(self):
        pass
    
    def _getDataLim(self):
        return None
        
    def _showToolbar(self, evt):
        self._toolbarShown =  not self._toolbarShown
        self._canvasWidget.pack_forget()
        if self._toolbarShown:
            self._toolbarContainer.pack(side="top", fill="x")
        else:
            self._toolbarContainer.pack_forget()
        self._canvasWidget.pack(side="bottom", expand=True, fill="both")
        self._canvasWidget["borderwidth"] = 6
        self._canvasWidget["highlightthickness"] = 0
        
    def _createWidgets(self):        
        self._plotContainer = Frame(self)
        self._plotContainer.grid(row=1, column=0, columnspan=2, sticky=N+S+W+E)
        self._toolbarShown = True
        self._toolbarContainer = Frame(self._plotContainer)
        
        canvas = FigureCanvasTkAgg(self._fig, self._plotContainer)
        canvas.draw()
        toolbar = NavigationToolbar2TkAggAutoScale(canvas, self._toolbarContainer, self)
        toolbar.update()
        self._canvasWidget = canvas.get_tk_widget()
        self._canvasWidget.bind("<Double-1>", self._showToolbar)
        self._showToolbar(None)
        
class ScopeWidget(PlotWidget):
    def __init__(self, master=None, samples=1000, yMin=-1, yMax=1, complex=False, grid=True):
        self._complex = complex
        if self._complex:
            self._sampleBuffer = [np.inf + 1j*np.inf] * samples
        else:
            self._sampleBuffer = [np.inf] * samples
        self._sampleBufferIdx = 0
        self._initialized = False
        
        self._ylim = (yMin, yMax)
        self._grid = grid
        
        PlotWidget.__init__(self, master)
        
    def _initPlot(self):
        self._fig.clf()
        self._ax = self._fig.add_subplot(111)
        self._ax.set_xlim((0, len(self._sampleBuffer)))
        self._ax.set_ylim(self._ylim)
        self._ax.grid(self._grid)
        
        if self._complex:
            self._lines = self._ax.plot(np.arange(0, len(self._sampleBuffer)), np.real(self._sampleBuffer), c='b', animated=True)
            self._lines.extend(self._ax.plot(np.arange(0, len(self._sampleBuffer)), np.imag(self._sampleBuffer), c='g', animated=True))
        else:
            self._lines = self._ax.plot(np.arange(0, len(self._sampleBuffer)), self._sampleBuffer, c='b', animated=True)
        self._marker, = self._ax.plot((0, 0), self._ylim, c='r', animated=True)
        self._initialized = True
        
    def _getDataLim(self):
        if self._complex:
            return (min(np.min(np.real(self._sampleBuffer)), np.min(np.imag(self._sampleBuffer))),
                    max(np.max(np.real(self._sampleBuffer)), np.max(np.imag(self._sampleBuffer))))
        else:
            return (np.min(self._sampleBuffer), np.max(self._sampleBuffer))
        
    def updateData(self, samples):
        begin = 0
        length = len(samples)
        if length > len(self._sampleBuffer):
            begin = len(samples) - len(self._sampleBuffer)
            length = len(self._sampleBuffer)
            self._sampleBufferIdx = (self._sampleBufferIdx + len(samples)) % len(self._sampleBuffer)
        if length > len(self._sampleBuffer) - self._sampleBufferIdx:
            split = len(self._sampleBuffer) - self._sampleBufferIdx
            self._sampleBuffer[self._sampleBufferIdx:] = samples[begin:begin+split]
            self._sampleBuffer[:length-split] = samples[begin+split:]
            self._sampleBufferIdx = length-split
        else:
            self._sampleBuffer[self._sampleBufferIdx:self._sampleBufferIdx+length] = samples[begin:]
            self._sampleBufferIdx = (self._sampleBufferIdx + length) % len(self._sampleBuffer)
        
    def updatePlot(self):
        if not self._initialized:
            return
        
        if not self._complex:
            self._lines[0].set_ydata(self._sampleBuffer)
        else:
            self._lines[0].set_ydata(np.real(self._sampleBuffer))
            self._lines[1].set_ydata(np.imag(self._sampleBuffer))
        self._marker.set_xdata((self._sampleBufferIdx, self._sampleBufferIdx))
        
        self._fig.canvas.restore_region(self._background)
        for l in self._lines:
            self._ax.draw_artist(l)
        self._ax.draw_artist(self._marker)
        self._fig.canvas.blit()
        
        
class PolarWidget(PlotWidget):
    def __init__(self, master=None, rMax=1, grid=True, pointerMode=False, pointers=1, logarithmic=False):
        PlotWidget.__init__(self, master)
        
        self._initialized = False
        
        self._rMax = rMax if not logarithmic else np.log1p(rMax)
        self._grid = grid
        self._pointerMode = pointerMode
        self._pointers = pointers
        self._logarithmic = logarithmic
        
    def _initPlot(self):
        self._fig.clf()
        self._ax = self._fig.add_subplot(111, projection='polar')
        self._ax.set_ylim(0, self._rMax)
        self._ax.grid(self._grid)
        
        if not self._pointerMode:
            self._lines = self._ax.plot((0), (0), c='b', animated=True)
            self._samples = [[0]]
        else:
            self._lines = []
            for i in range(self._pointers):
                self._lines.extend(self._ax.plot((0), (0), animated=True))
            self._samples = np.zeros((self._pointers, 2))
        self._initialized = True
        
    def setScale(self, rMax=1):
        self._rMax = rMax if not self._logarithmic else np.log1p(rMax)
        self._initPlot()
        self._fig.canvas.draw()
        self._background = self._fig.canvas.copy_from_bbox(self._ax.bbox)
        self.updatePlot()
        
    def updateData(self, samples):
        if not self._pointerMode:
            self._samples = np.array([samples])
        else:
            p = min(self._pointers, len(samples))
            self._samples = np.zeros((self._pointers, 2), dtype=np.complex)
            self._samples[:p,1] = samples[:p]
        
    def updatePlot(self):
        if not self._initialized:
            return
        
        if not self._logarithmic:
            for i in range(len(self._lines)):
                self._lines[i].set_data(np.angle(self._samples[i]), np.abs(self._samples[i]))
        else:
            for i in range(len(self._lines)):
                self._lines[i].set_data(np.angle(self._samples[i]), np.log1p(np.abs(self._samples[i])))
        
        self._fig.canvas.restore_region(self._background)
        for l in self._lines:
            self._ax.draw_artist(l)
        self._fig.canvas.blit()
        
class NavigationToolbar2TkAggAutoScale (NavigationToolbar2Tk):
    def __init__(self, canvas, window, plotWidget):
        if plotWidget._getDataLim() is not None:
            ti = list(self.toolitems)
            ti.append((None, None, None, None))
            ti.append(('Auto Scale', 'Zoom in to current signal levels', 'hand', 'auto_scale'))
            self.toolitems = tuple(ti)
            self.plotWidget = plotWidget
        NavigationToolbar2Tk.__init__(self, canvas, window)
        
    def auto_scale(self):
        # Matplotlib <= 2.1.0
        if hasattr(self, '_views') and self._views.empty():
            self.push_current()
        # Matplotlib >= 2.1.1
        if hasattr(self, '_nav_stack') and self._nav_stack() is None:
            self.push_current()
            
        last_a = []
        for a in self.canvas.figure.get_axes():
            if not (a.get_navigate() and a.can_zoom()):
                continue
            
            bbox = a.bbox.get_points()
            ax_min, ax_max = a.get_ylim()
            data_min, data_max = self.plotWidget._getDataLim()
            disp_min, disp_max = bbox[:,1]
            
            if data_min == -np.inf or data_max == np.inf:
                continue
            
            # detect twinx,y axes and avoid double zooming
            twinx, twiny = False, False
            if last_a:
                for la in last_a:
                    if a.get_shared_x_axes().joined(a, la):
                        twinx = True
                    if a.get_shared_y_axes().joined(a, la):
                        twiny = True
            last_a.append(a)
            
            height = (data_max-data_min)*2.0 / (ax_max-ax_min) * (disp_max-disp_min)
            top = (ax_max-data_max) / (ax_max-ax_min) * (disp_max-disp_min) - 0.25*height
            
            a._set_view_from_bbox((bbox[0,0], disp_max-top, bbox[1,0], disp_max-top-height), 
                                  'in', None, twinx, twiny)
            
        self.draw()
        self.push_current()
