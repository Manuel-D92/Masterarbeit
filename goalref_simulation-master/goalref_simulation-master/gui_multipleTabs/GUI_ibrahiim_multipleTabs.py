
import sys
import tkinter as ttk
from tkinter import *
from tkinter.ttk import *
import tkinter.filedialog as fileDialog
import getpass
import os, signal
import math
import numpy as np
import subprocess
import psutil
import time
import csv
import matplotlib.pyplot as plt
import class_conductor as c
from PIL import Image, ImageTk
from threading import Thread
import queue
__author__ = 'ihm'

## @package GUI_multipleTabs
# @brief A Graphical User Interface for configuring
# the parameters of the simulation tool originally made by "artizaaa".
# This GUI is made using python Tkinter library and it requires Python 3.6.
# @details This GUI contains several classes and methods that manage the tabs, labels, entries and widgets layout.
# It contains a field for each parameter that can be configured in the original simualtion tool.
# To make it easier for the user, each tab in that GUI contains the default values for
# the desired application. The user can also change and modify any of these default values.
# New applications can be added to this GUI by creating new class (tab) and adding it to
# the main notebook (self.nb).
#@author ihm
#@version 2.0
#@date Created on Wed Dec 6 09:00:00 2017



## class GUI is responsible for starting the main (root) window (instance of Tkinter class),
# getting and setting the correct scaling factor depending on the used screen size,
# configuring the root window's rows and columns, then it calls the other tabs to run them all
# in the mainloop()
class GUI:

    def __init__(self):
        ## Instance of Tkinter class
        self.root = ttk.Tk()
        self.set_scale()
        self.configure_root()
        GUI_inducedVoltageTab(self.scale_factor, self.screenWidth, self.screenHeight, self.paramFontSize,
                           self.titleFontSize, self.nb, self.root)
        GUI_couplingFactorTab(self.scale_factor, self.screenWidth, self.screenHeight, self.paramFontSize,
                        self.titleFontSize, self.nb)
        self.tab_magneticFieldSimulation = GUI_magneticFieldSimulation(self.nb)
        self.nb.add(self.tab_magneticFieldSimulation, text="Magn. Field Simulation")
        GUI_aboutTab(self.scale_factor, self.screenWidth, self.screenHeight, self.paramFontSize,
                     self.titleFontSize, self.nb)
        self.root.mainloop()
    ## This method calculates the scale_factor using a scaling algorithm,
    # then a new screen width, height, parameters and titles font size
    # are adjusted using the calculated scale_factor throughout the GUI.
    def get_scale(self):
        ## Base screen size
        normal_width = 1280
        normal_height = 720
        ## For HD screens (1280x720)
        if self.root.winfo_screenwidth() <= 1280 or self.root.winfo_screenheight() <= 720:
            ##sides threshold
            sideThreshold = 50
            screen_width = 1280 - sideThreshold
            screen_height = 720 - sideThreshold
            ## Get percentage of screen size from Base size
            percentage_width = screen_width / (normal_width / 100)
            percentage_height = screen_height / (normal_height / 100)
            ## Make a scaling factor, this is bases on average percentage from
            # width and height.
            scale_factor = ((percentage_width + percentage_height) / 2) / 100
            ## screen size after scaling
            screenWidth = int(normal_width * scale_factor)
            screenHeight = int(normal_height * scale_factor)
            ## Set the fontsize based on scale_factor,
            # if the fontsize is less than minimum_size
            # it is set to the minimum size
            # parameters' labels font size
            paramFontSize = int(8 * scale_factor)
            paramFont_minimum_size = 5
            if paramFontSize < paramFont_minimum_size:
                paramFontSize = paramFont_minimum_size
            ## Titles or headlines' font size
            titleFontSize = int(12 * scale_factor)
            titleFont_minimum_size = 9
            if paramFontSize < titleFont_minimum_size:
                titleFontSize = titleFont_minimum_size
        ## For FHD screens (1920x1080)
        elif self.root.winfo_screenwidth() == 1920 or self.root.winfo_screenheight() == 1080:
            ## sides threshold
            sideThreshold = 0.2
            screen_width = self.root.winfo_screenwidth() - (self.root.winfo_screenwidth() * sideThreshold)
            screen_height = self.root.winfo_screenheight() - (self.root.winfo_screenheight() * sideThreshold)
            ## Get percentage of screen size from Base size
            percentage_width = screen_width / (normal_width / 100)
            percentage_height = screen_height / (normal_height / 100)
            ## Make a scaling factor, this is bases on average percentage from
            ## width and height.
            scale_factor = ((percentage_width + percentage_height) / 2) / 100
            ## screen size after scaling
            screenWidth = int(normal_width * scale_factor)
            screenHeight = int(normal_height * scale_factor)
            ## Set the fontsize based on scale_factor,
            # if the fontsize is less than minimum_size
            # it is set to the minimum size
            # parameters' labels font size
            paramFontSize = int(8 * scale_factor)
            paramFont_minimum_size = 5
            if paramFontSize < paramFont_minimum_size:
                paramFontSize = paramFont_minimum_size
            ## Titles or headlines' font size
            titleFontSize = int(12 * scale_factor)
            titleFont_minimum_size = 9
            if paramFontSize < titleFont_minimum_size:
                titleFontSize = titleFont_minimum_size
        ## For QHD screens (2560x1440)
        elif self.root.winfo_screenwidth() == 2560 and self.root.winfo_screenheight() == 1440:
            ## sides threshold
            sideThreshold = 0.4
            screen_width = self.root.winfo_screenwidth() - (self.root.winfo_screenwidth() * sideThreshold)
            screen_height = self.root.winfo_screenheight() - (self.root.winfo_screenheight() * sideThreshold)
            ## Get percentage of screen size from Base size
            percentage_width = screen_width / (normal_width / 100)
            percentage_height = screen_height / (normal_height / 100)
            ## Make a scaling factor, this is bases on average percentage from
            ## width and height.
            scale_factor = ((percentage_width + percentage_height) / 2) / 100
            ## screen size after scaling
            screenWidth = int(normal_width * scale_factor)
            screenHeight = int(normal_height * scale_factor)
            ## Set the fontsize based on scale_factor,
            # if the fontsize is less than minimum_size
            # it is set to the minimum size
            # parameters' labels font size
            paramFontSize = int(11 * scale_factor)
            paramFont_minimum_size = 7
            if paramFontSize < paramFont_minimum_size:
                paramFontSize = paramFont_minimum_size
            ## Titles or headlines' font size
            titleFontSize = int(13 * scale_factor)
            titleFont_minimum_size = 9
            if paramFontSize < titleFont_minimum_size:
                titleFontSize = titleFont_minimum_size
        ## For UHD and higher (3840x2160)
        elif self.root.winfo_screenwidth() == 3840 and self.root.winfo_screenheight() == 2160:
            ## sides threshold
            sideThreshold = 0.5
            screen_width = self.root.winfo_screenwidth() - (self.root.winfo_screenwidth() * sideThreshold)
            screen_height = self.root.winfo_screenheight() - (self.root.winfo_screenheight() * sideThreshold)
            ## Get percentage of screen size from Base size
            percentage_width = screen_width / (normal_width / 100)
            percentage_height = screen_height / (normal_height / 100)
            ## Make a scaling factor, this is bases on average percentage from
            ## width and height.
            scale_factor = ((percentage_width + percentage_height) / 2) / 100
            ## screen size after scaling
            screenWidth = int(normal_width * scale_factor)
            screenHeight = int(normal_height * scale_factor)
            ## Set the fontsize based on scale_factor,
            # if the fontsize is less than minimum_size
            # it is set to the minimum size
            # parameters' labels font size
            paramFontSize = int(13 * scale_factor)
            paramFont_minimum_size = 9
            if paramFontSize < paramFont_minimum_size:
                paramFontSize = paramFont_minimum_size
            ## Titles or headlines' font size
            titleFontSize = int(15 * scale_factor)
            titleFont_minimum_size = 11
            if paramFontSize < titleFont_minimum_size:
                titleFontSize = titleFont_minimum_size
        ## For other non-standard screens
        else:
            ## sides threshold
            sideThreshold = 0.1
            screen_width = self.root.winfo_screenwidth() - (self.root.winfo_screenwidth() * sideThreshold)
            screen_height = self.root.winfo_screenheight() - (self.root.winfo_screenheight() * sideThreshold)
            ## Get percentage of screen size from Base size
            percentage_width = screen_width / (normal_width / 100)
            percentage_height = screen_height / (normal_height / 100)
            ## Make a scaling factor, this is bases on average percentage from
            ## width and height.
            scale_factor = ((percentage_width + percentage_height) / 2) / 100
            ## screen size after scaling
            screenWidth = int(normal_width * scale_factor)
            screenHeight = int(normal_height * scale_factor)
            ## Set the fontsize based on scale_factor,
            # if the fontsize is less than minimum_size
            # it is set to the minimum size
            # parameters' labels font size
            paramFontSize = int(7 * scale_factor)
            paramFont_minimum_size = 5
            if paramFontSize < paramFont_minimum_size:
                paramFontSize = paramFont_minimum_size
            ## Titles or headlines' font size
            titleFontSize = int(11 * scale_factor)
            titleFont_minimum_size = 9
            if paramFontSize < titleFont_minimum_size:
                titleFontSize = titleFont_minimum_size
        return scale_factor, screenWidth, screenHeight, paramFontSize, titleFontSize
    def set_scale(self):
        self.scale_factor,self.screenWidth,self.screenHeight,self.paramFontSize,self.titleFontSize = self.get_scale()
    ## This method configure the root window's rows, columns,
    # geometrical dimesniosn, logo and title.
    def configure_root(self):
        ## Setting the width and hight of the main (root) GUI widnow
        self.root.geometry('{}x{}'.format(self.screenWidth, self.screenHeight))
        ## GUI window title
        self.root.title("Fraunhofer-iis: GUI_Simulation Tool")
        ## GUI window logo
        self.root.iconbitmap(default='iis.ico')
        ## Configuring the rows and columns of the root window
        rows = 0
        while rows < 10:
            self.root.rowconfigure(rows, weight=1)
            self.root.columnconfigure(rows, weight=1)
            rows += 1
        ## Setting the resizing option for the GUI window
        self.root.resizable(width=False, height=False)
        ## creating a notebook instance
        self.nb = Notebook(self.root)
        ## placing out the notebook at the first row and the first column
        self.nb.grid(row=0, column=0,columnspan=rows, rowspan=rows, sticky='NESW')
    ## This method creates a new notebook, then grid it to the given
    # row and column values.
    def create_grid_notebook(self, parentFrame, row, column):
        nb = Notebook(parentFrame)
        nb.grid(row=row, column=column)
        return nb
    ## This method creates a new tab, then add it to the given
    # parent notebook.
    def create_add_tab(self, parentNotebook, tabWidth, tabHeight, tabName):
        if tabWidth != None and tabHeight !=None:
            newTab = ttk.Frame(parentNotebook, width=tabWidth, height=tabHeight)
        else:
            newTab = ttk.Frame(parentNotebook)
        ## adding the tab to the notebook
        parentNotebook.add(newTab, text=tabName)
        return newTab
## class GUI_inducedVoltageTab displays the parameters that can be configured by the user
# it also contains the initial default values for the hockey setup
class GUI_inducedVoltageTab(GUI):
    def __init__(self, scale_factor, screenWidth, screenHeight, paramFontSize, titleFontSize, frame, root):
        self.scale_factor = scale_factor
        self.screenWidth = screenWidth
        self.screenHeight = screenHeight
        self.paramFontSize = paramFontSize
        self.titleFontSize = titleFontSize
        self.frame = frame
        self.root = root
        self.inducedVoltageTab= self.create_add_tab(self.frame, None, None, 'Induced Voltage')
        ## creating a notebook instance
        self.nb_inducedVoltage = self.create_grid_notebook(self.inducedVoltageTab, 0,0)
        ## creating a new tab for exciter(s) parameters
        self.exciterParametersTab = self.create_add_tab(self.nb_inducedVoltage,
                                                        self.screenWidth - int(100 * self.scale_factor),
                                                        self.screenHeight - int(150 * self.scale_factor),
                                                        'Exciter(s) parameters')
        ## creating a new tab for antenna(s) parameters
        self.antennaParametersTab = self.create_add_tab(self.nb_inducedVoltage,
                                                        self.screenWidth - int(100 * self.scale_factor),
                                                        self.screenHeight - int(150 * self.scale_factor),
                                                        'Antenna(s) parameters')
        ## creating a new tab for object(s) parameters
        self.objectParametersTab = self.create_add_tab(self.nb_inducedVoltage,
                                                       self.screenWidth - int(100 * self.scale_factor),
                                                       self.screenHeight - int(150 * self.scale_factor),
                                                       'Object(s) parameters')
        ## creating a new tab for table parameters
        self.tableParametersTab = self.create_add_tab(self.nb_inducedVoltage,
                                                      self.screenWidth - int(100 * self.scale_factor),
                                                      self.screenHeight - int(150 * self.scale_factor),
                                                      'Table parameters')
        self.plotTab = self.create_add_tab(self.nb_inducedVoltage,
                                                      self.screenWidth - int(100 * self.scale_factor),
                                                      self.screenHeight - int(150 * self.scale_factor),
                                                      'Plot')
        self.generateDifferentFrequencyTab = self.create_add_tab(self.nb_inducedVoltage,
                                           self.screenWidth - int(100 * self.scale_factor),
                                           self.screenHeight - int(150 * self.scale_factor),
                                           'Generate different frequency tables')

        ## Starting the initial GUI window configuration and layout
        self.createStartUpElments()
    ## This function is trigered when the user check or uncheck  the Polygon check-box
    # If the Polygon object was chosen, the other three (Ellipse, Puk, Ball) check-boxes are set to zero,
    # then it displays the Polygon object parameters labes and entries.
    def exciterCornersFrameDestroy(self):
        if self.exciterCornersFrame_register != 0:
            self.exciterCornersFrame_register.destroy()
            self.exciterCornersFrame_register = 0
    def addExciterCorners(self):
        if self.numberOfExciterCornersBuffer != self.numberOfExciterCorners.get():
            if self.exciterCornersFrame_register != 0:
                self.exciterCornersFrameDestroy()
            exciterCornersFrame = Frame(self.exciterParametersTab, relief='groove', borderwidth=5)
            exciterCornersFrame.place(x=0, y=int(130*self.scale_factor))
            self.exciterCornersFrame_register = exciterCornersFrame
            self.exciterCornersEntriesDict = {}
            self.numberOfExciterCornersBuffer = self.numberOfExciterCorners.get()
            self.registeredExciterCorners = self.numberOfExciterCorners.get()
            for i in range(1,self.numberOfExciterCorners.get()+1):
                if i < 6:
                    self.exciterCornersEntriesDict["Corner {0}".format(i)] = Label(self.exciterCornersFrame_register, text="Corner {0} [x(m),y(m),z(m)]: ".format(i),
                                                                      font=("Arial", self.paramFontSize-1),justify='left')
                    self.exciterCornersEntriesDict["Corner {0}".format(i)].grid(row=i-1, column=0)

                    self.exciterCornersEntriesDict["Corner {0} entry".format(i)] = Entry(self.exciterCornersFrame_register,width=int(12*self.scale_factor))
                    self.exciterCornersEntriesDict["Corner {0} entry".format(i)].grid(row=i-1, column=1)
                    self.exciterCornersEntriesDict["Corner {0} entry".format(i)].insert(0, str(self.exciterParam_PolygonExciterCorners[i - 1]))
                else:
                    self.exciterCornersEntriesDict["Corner {0}".format(i)] = Label(self.exciterCornersFrame_register,
                                                                                   text="Corner {0} [x(m),y(m),z(m)]: ".format(
                                                                                       i),
                                                                                   font=("Arial", self.paramFontSize-1),
                                                                                   justify='left')
                    self.exciterCornersEntriesDict["Corner {0}".format(i)].grid(row=i-6, column=2)

                    self.exciterCornersEntriesDict["Corner {0} entry".format(i)] = Entry(
                        self.exciterCornersFrame_register, width=int(12*self.scale_factor))
                    self.exciterCornersEntriesDict["Corner {0} entry".format(i)].grid(row=i-6, column=3)
                    self.exciterCornersEntriesDict["Corner {0} entry".format(i)].insert(0,
                                                                                        str(self.exciterParam_PolygonExciterCorners[
                                                                                            i - 1]))
    def displayObjectTypePolygon(self):
        if self.objectTypePolygon.get()==1:
            self.objectTypeEllipse.set(0)
            self.objectTypePuk.set(0)
            self.objectTypeBall.set(0)
            self.objectTypeWearable.set(0)
            ## Remove the previous object entries and labels from the GUI if any.
            self.objectParamFrameDestroy()
            ## Create and place a frame that will contain the Polygon object labels and entries.
            objectTypeParamFramePolygon = Frame(self.objectParametersTab, relief='groove', borderwidth=5)
            objectTypeParamFramePolygon.place(x=0, y=int(110*self.scale_factor))
            ## This regesiter contains the current object type displayed frame.
            # This is important when the function objectParamFrameDestroy() is called.
            self.ObjectParamTypeFrame_register=objectTypeParamFramePolygon

            #Create object type parameters labels and entries
            label_ObjectParamType_Title = Label(objectTypeParamFramePolygon, text="Polygon coil parameters", font=("Arial", self.titleFontSize),
                                             justify='left')

            label_ObjectParamType_CoilDimensions = Label(objectTypeParamFramePolygon, text="Coil dimensions Width, Length (m): ",
                                               font=("Arial", self.paramFontSize))
            label_ObjectParamType_CoilDimensionsWidth = Label(objectTypeParamFramePolygon,
                                                         text="W",
                                                         font=("Arial", self.paramFontSize))
            label_ObjectParamType_CoilDimensionsLength = Label(objectTypeParamFramePolygon,
                                                         text="L",
                                                         font=("Arial", self.paramFontSize))
            label_ObjectParamType_OrientaionAngles = Label(objectTypeParamFramePolygon, text="Coil orientation angles (deg): ",
                                                           font=("Arial", self.paramFontSize))
            label_ObjectParamType_Alpha = Label(objectTypeParamFramePolygon,
                                                           text=u'\u03b1',
                                                           font=("Arial", self.paramFontSize))
            label_ObjectParamType_Beta = Label(objectTypeParamFramePolygon,
                                                text=u'\u03b2',
                                                font=("Arial", self.paramFontSize))
            label_ObjectParamType_Gamma = Label(objectTypeParamFramePolygon,
                                                text=u'\u03b3',
                                                font=("Arial", self.paramFontSize))
            label_ObjectParamType_DisplayObjectPosition = Label(objectTypeParamFramePolygon,
                                                text='Display coil at (m): ',
                                                font=("Arial", self.paramFontSize))
            label_ObjectParamType_X = Label(objectTypeParamFramePolygon,
                                                text='x',
                                                font=("Arial", self.paramFontSize))
            label_ObjectParamType_Y = Label(objectTypeParamFramePolygon,
                                               text='y',
                                               font=("Arial", self.paramFontSize))
            label_ObjectParamType_Z = Label(objectTypeParamFramePolygon,
                                                text='z',
                                                font=("Arial", self.paramFontSize))
            label_ObjectParamType_AnglesMessage = ttk.Label(objectTypeParamFramePolygon,
                                                      text="Only for plotting. not included in the final table.",
                                                      font=("Arial", int(8*self.scale_factor)), fg='blue')
            label_ObjectParamType_DisplayObjectPositionMessage = ttk.Label(objectTypeParamFramePolygon,
                                                            text="Only for plotting. not included in the final table.",
                                                            font=("Arial", int(8*self.scale_factor)), fg='blue')
            label_ObjectParamType_DisplayObjectPositionMessage1 = ttk.Label(objectTypeParamFramePolygon,
                                                                            text="if any of x,y,z is empty, "+ '\n' + "coil will be plotted at center of exciter.",
                                                                            font=("Arial", int(8*self.scale_factor)), fg='blue')
            self.entry_ObjectParamType_CoilDimensionsWidth = Entry(objectTypeParamFramePolygon,width=int(5*self.scale_factor))
            self.entry_ObjectParamType_CoilDimensionsWidth.insert(0,str(self.objectParam_PolygonCoilDimensionsWidth))
            self.entry_ObjectParamType_CoilDimensionsLength = Entry(objectTypeParamFramePolygon, width=int(5*self.scale_factor))
            self.entry_ObjectParamType_CoilDimensionsLength.insert(0, str(self.objectParam_PolygonCoilDimensionsLength))
            self.entry_ObjectParamType_Alpha = Entry(objectTypeParamFramePolygon,width=int(5*self.scale_factor))
            self.entry_ObjectParamType_Alpha.insert(0, str(self.objectParam_PolygonCoilOrientationAlpha))
            self.entry_ObjectParamType_Beta = Entry(objectTypeParamFramePolygon,width=int(5*self.scale_factor))
            self.entry_ObjectParamType_Beta.insert(0, str(self.objectParam_PolygonCoilOrientationBeta))
            self.entry_ObjectParamType_Gamma = Entry(objectTypeParamFramePolygon,width=int(5*self.scale_factor))
            self.entry_ObjectParamType_Gamma.insert(0, str(self.objectParam_PolygonCoilOrientationGamma))

            self.entry_ObjectParamType_X = Entry(objectTypeParamFramePolygon, width=int(5*self.scale_factor))
            self.entry_ObjectParamType_Y = Entry(objectTypeParamFramePolygon, width=int(5*self.scale_factor))
            self.entry_ObjectParamType_Z = Entry(objectTypeParamFramePolygon, width=int(5*self.scale_factor))

            ## Griding the labels and entries into the frame.
            label_ObjectParamType_Title.grid(row=0)
            label_ObjectParamType_CoilDimensions.grid(row=1, column=0)
            label_ObjectParamType_CoilDimensionsWidth.grid(row=1, column=1)
            label_ObjectParamType_CoilDimensionsLength.grid(row=1, column=3)
            label_ObjectParamType_OrientaionAngles.grid(row=2, column=0)
            label_ObjectParamType_Alpha.grid(row=2, column=1)
            self.entry_ObjectParamType_Alpha.grid(row=2, column=2)
            label_ObjectParamType_Beta.grid(row=2, column=3)
            self.entry_ObjectParamType_Beta.grid(row=2, column=4)
            label_ObjectParamType_Gamma.grid(row=2, column=5)
            self.entry_ObjectParamType_Gamma.grid(row=2, column=6)
            self.entry_ObjectParamType_CoilDimensionsWidth.grid(row=1, column=2)
            self.entry_ObjectParamType_CoilDimensionsLength.grid(row=1, column=4)

            label_ObjectParamType_DisplayObjectPosition.grid(row=3, column=0)
            label_ObjectParamType_X.grid(row=3, column=1)
            label_ObjectParamType_Y.grid(row=3, column=3)
            label_ObjectParamType_Z.grid(row=3, column=5)
            self.entry_ObjectParamType_X.grid(row=3, column=2)
            self.entry_ObjectParamType_Y.grid(row=3, column=4)
            self.entry_ObjectParamType_Z.grid(row=3, column=6)
            label_ObjectParamType_AnglesMessage.grid(row=2, column=7)
            label_ObjectParamType_DisplayObjectPositionMessage.grid(row=3, column=7)
            label_ObjectParamType_DisplayObjectPositionMessage1.grid(row=4, column=0)


        else:
            self.objectParamFrameDestroy()
    ## This function is trigered when the user check or uncheck  the Ellipse check-box
    # If the Ellipse object was chosen, the other three (Polygon, Puk, Ball) check-boxes are set to zero,
    # then it displays the Elliptical object parameters labes and entries.
    def displayObjectTypeEllipse(self):

        if self.objectTypeEllipse.get()==1:
            self.objectTypePolygon.set(0)
            self.objectTypePuk.set(0)
            self.objectTypeBall.set(0)
            self.objectTypeWearable.set(0)
            ## Remove the previous object entries and labels from the GUI if any.
            self.objectParamFrameDestroy()
            ## Create and place a frame that will contain the Ellipse object labels and entries.
            objectTypeParamFrameEllipse = Frame(self.objectParametersTab, relief='groove', borderwidth=5)
            objectTypeParamFrameEllipse.place(x=0, y = int(110*self.scale_factor))
            self.ObjectParamTypeFrame_register=objectTypeParamFrameEllipse
            # Create object type parameters labels and entries
            label_ObjectParamType_Title = Label(objectTypeParamFrameEllipse, text="Ellipse coil parameters",
                                                font=("Arial", self.titleFontSize),
                                                justify='left')

            label_ObjectParamType_MajorAxis = Label(objectTypeParamFrameEllipse, text="Major axis length (m): ",
                                                  font=("Arial", self.paramFontSize))
            label_ObjectParamType_MinorAxis = Label(objectTypeParamFrameEllipse, text="Minor axis length (m): ",
                                                    font=("Arial", self.paramFontSize))
            label_ObjectParamType_OrientaionAngles = Label(objectTypeParamFrameEllipse, text="Coil orientation angles (deg): ",
                                                  font=("Arial", self.paramFontSize))
            label_ObjectParamType_Alpha = Label(objectTypeParamFrameEllipse,
                                                text=u'\u03b1',
                                                font=("Arial", self.paramFontSize))
            label_ObjectParamType_Beta = Label(objectTypeParamFrameEllipse,
                                               text=u'\u03b2',
                                               font=("Arial", self.paramFontSize))
            label_ObjectParamType_Gamma = Label(objectTypeParamFrameEllipse,
                                                text=u'\u03b3',
                                                font=("Arial", self.paramFontSize))
            label_ObjectParamType_DisplayObjectPosition = Label(objectTypeParamFrameEllipse,
                                                                text='Display coil at (m): ',
                                                                font=("Arial", self.paramFontSize))
            label_ObjectParamType_X = Label(objectTypeParamFrameEllipse,
                                            text='x',
                                            font=("Arial", self.paramFontSize))
            label_ObjectParamType_Y = Label(objectTypeParamFrameEllipse,
                                            text='y',
                                            font=("Arial", self.paramFontSize))
            label_ObjectParamType_Z = Label(objectTypeParamFrameEllipse,
                                            text='z',
                                            font=("Arial", self.paramFontSize))
            label_ObjectParamType_AnglesMessage = ttk.Label(objectTypeParamFrameEllipse,
                                                            text="Only for plotting. not included in the final table.",
                                                            font=("Arial", int(8*self.scale_factor)), fg='blue')
            label_ObjectParamType_DisplayObjectPositionMessage = ttk.Label(objectTypeParamFrameEllipse,
                                                                           text="Only for plotting. not included in the final table.",
                                                                           font=("Arial", int(8*self.scale_factor)), fg='blue')
            label_ObjectParamType_DisplayObjectPositionMessage1 = ttk.Label(objectTypeParamFrameEllipse,
                                                                           text="if any of x,y,z is empty,"+'\n'+" coil will be plotted at center of exciter.",
                                                                           font=("Arial", int(7*self.scale_factor)), fg='blue')
            self.entry_ObjectParamType_MajorAxis = Entry(objectTypeParamFrameEllipse, width=int(5*self.scale_factor))
            self.entry_ObjectParamType_MajorAxis.insert(0, str(self.objectParam_EllipseCoilMajorAxis))
            self.entry_ObjectParamType_MinorAxis = Entry(objectTypeParamFrameEllipse, width=int(5*self.scale_factor))
            self.entry_ObjectParamType_MinorAxis.insert(0, str(self.objectParam_EllipseCoilMinorAxis))
            self.entry_ObjectParamType_Alpha = Entry(objectTypeParamFrameEllipse, width=int(5*self.scale_factor))
            self.entry_ObjectParamType_Alpha.insert(0, str(self.objectParam_EllipseCoilOrientationAlpha))
            self.entry_ObjectParamType_Beta = Entry(objectTypeParamFrameEllipse, width=int(5*self.scale_factor))
            self.entry_ObjectParamType_Beta.insert(0, str(self.objectParam_EllipseCoilOrientationBeta))
            self.entry_ObjectParamType_Gamma = Entry(objectTypeParamFrameEllipse, width=int(5*self.scale_factor))
            self.entry_ObjectParamType_Gamma.insert(0, str(self.objectParam_EllipseCoilOrientationGamma))

            self.entry_ObjectParamType_X = Entry(objectTypeParamFrameEllipse, width=int(5*self.scale_factor))
            self.entry_ObjectParamType_Y = Entry(objectTypeParamFrameEllipse, width=int(5*self.scale_factor))
            self.entry_ObjectParamType_Z = Entry(objectTypeParamFrameEllipse, width=int(5*self.scale_factor))
            ## Griding the labels and entries into the frame.
            label_ObjectParamType_Title.grid(row=0)
            label_ObjectParamType_MajorAxis.grid(row=1, column=0)
            label_ObjectParamType_MinorAxis.grid(row=2, column=0)
            label_ObjectParamType_OrientaionAngles.grid(row=3, column=0)
            label_ObjectParamType_Alpha.grid(row=3, column=1)
            self.entry_ObjectParamType_Alpha.grid(row=3, column=2)
            label_ObjectParamType_Beta.grid(row=3, column=3)
            self.entry_ObjectParamType_Beta.grid(row=3, column=4)
            label_ObjectParamType_Gamma.grid(row=3, column=5)
            self.entry_ObjectParamType_Gamma.grid(row=3, column=6)
            self.entry_ObjectParamType_MajorAxis.grid(row=1, column=2)
            self.entry_ObjectParamType_MinorAxis.grid(row=2, column=2)

            label_ObjectParamType_DisplayObjectPosition.grid(row=4, column=0)
            label_ObjectParamType_X.grid(row=4, column=1)
            label_ObjectParamType_Y.grid(row=4, column=3)
            label_ObjectParamType_Z.grid(row=4, column=5)
            self.entry_ObjectParamType_X.grid(row=4, column=2)
            self.entry_ObjectParamType_Y.grid(row=4, column=4)
            self.entry_ObjectParamType_Z.grid(row=4, column=6)
            label_ObjectParamType_AnglesMessage.grid(row=3, column=7)
            label_ObjectParamType_DisplayObjectPositionMessage.grid(row=4, column=7)
            label_ObjectParamType_DisplayObjectPositionMessage1.grid(row=5, column=0)
        else: self.objectParamFrameDestroy()
    ## This function is trigered when the user check or uncheck  the Puk check-box
    # If the Puk object was chosen, the other three (Polygon, Ellipse, Ball) check-boxes are set to zero,
    # then it displays the Puk object parameters labes and entries.
    def displayObjectTypePuk(self):
        if self.objectTypePuk.get()==1:
            self.objectTypePolygon.set(0)
            self.objectTypeEllipse.set(0)
            self.objectTypeBall.set(0)
            self.objectTypeWearable.set(0)
            ## Remove the previous object entries and labels from the GUI if any.
            self.objectParamFrameDestroy()
            ## Create and place a frame that will contain the Puk object labels and entries.
            objectTypeParamFramePuk = Frame(self.objectParametersTab, relief='groove', borderwidth=5)
            objectTypeParamFramePuk.place(x=0, y = int(110*self.scale_factor))
            self.ObjectParamTypeFrame_register=objectTypeParamFramePuk
            # Create object type parameters labels and entries
            label_ObjectParamType_Title = Label(objectTypeParamFramePuk, text="Puk coils parameters",
                                                font=("Arial", self.titleFontSize),
                                                justify='left')

            label_ObjectParamType_RectangularCoilsDimensions = Label(objectTypeParamFramePuk, text="Puk height (m): ",
                                                  font=("Arial", self.paramFontSize))
            label_ObjectParamType_CircularCoilRadius = Label(objectTypeParamFramePuk, text="Circular coil radius (m): ",
                                                  font=("Arial", self.paramFontSize))
            label_ObjectParamType_OrientaionAngles = Label(objectTypeParamFramePuk, text="Coil orientation angles (deg): ",
                                                           font=("Arial", self.paramFontSize))
            label_ObjectParamType_Alpha = Label(objectTypeParamFramePuk,
                                                text=u'\u03b1',
                                                font=("Arial", self.paramFontSize))
            label_ObjectParamType_Beta = Label(objectTypeParamFramePuk,
                                               text=u'\u03b2',
                                               font=("Arial", self.paramFontSize))
            label_ObjectParamType_Gamma = Label(objectTypeParamFramePuk,
                                                text=u'\u03b3',
                                                font=("Arial", self.paramFontSize))

            label_ObjectParamType_DisplayObjectPosition = Label(objectTypeParamFramePuk,
                                                                text='Display puk at (m): ',
                                                                font=("Arial", self.paramFontSize))
            label_ObjectParamType_X = Label(objectTypeParamFramePuk,
                                            text='x',
                                            font=("Arial", self.paramFontSize))
            label_ObjectParamType_Y = Label(objectTypeParamFramePuk,
                                            text='y',
                                            font=("Arial", self.paramFontSize))
            label_ObjectParamType_Z = Label(objectTypeParamFramePuk,
                                            text='z',
                                            font=("Arial", self.paramFontSize))
            label_ObjectParamType_AnglesMessage = ttk.Label(objectTypeParamFramePuk,
                                                            text="Only for plotting. not included in the final table.",
                                                            font=("Arial", int(8*self.scale_factor)), fg='blue')
            label_ObjectParamType_DisplayObjectPositionMessage = ttk.Label(objectTypeParamFramePuk,
                                                                           text="Only for plotting. not included in the final table.",
                                                                           font=("Arial", int(8*self.scale_factor)), fg='blue')
            label_ObjectParamType_DisplayObjectPositionMessage1 = ttk.Label(objectTypeParamFramePuk,
                                                                            text="if any of x,y,z is empty,"+'\n'+" puk will be plotted at center of exciter.",
                                                                            font=("Arial", int(7*self.scale_factor)), fg='blue')
            self.entry_ObjectParamType_RectangularCoilsDimensions = Entry(objectTypeParamFramePuk, width=int(5*self.scale_factor))
            self.entry_ObjectParamType_RectangularCoilsDimensions.insert(0, str(self.objectParam_PukCoilDimensions))
            self.entry_ObjectParamType_CircularCoilRadius = Entry(objectTypeParamFramePuk, width=int(5*self.scale_factor))
            self.entry_ObjectParamType_CircularCoilRadius.insert(0, str(self.objectParam_PukCoilRadius))
            self.entry_ObjectParamType_Alpha = Entry(objectTypeParamFramePuk, width=int(5*self.scale_factor))
            self.entry_ObjectParamType_Alpha.insert(0, str(self.objectParam_PukCoilOrientationAlpha))
            self.entry_ObjectParamType_Beta = Entry(objectTypeParamFramePuk, width=int(5*self.scale_factor))
            self.entry_ObjectParamType_Beta.insert(0, str(self.objectParam_PukCoilOrientationBeta))
            self.entry_ObjectParamType_Gamma = Entry(objectTypeParamFramePuk, width=int(5*self.scale_factor))
            self.entry_ObjectParamType_Gamma.insert(0, str(self.objectParam_PukCoilOrientationGamma))

            self.entry_ObjectParamType_X = Entry(objectTypeParamFramePuk, width=int(5*self.scale_factor))
            self.entry_ObjectParamType_Y = Entry(objectTypeParamFramePuk, width=int(5*self.scale_factor))
            self.entry_ObjectParamType_Z = Entry(objectTypeParamFramePuk, width=int(5*self.scale_factor))
            ## Griding the labels and entries into the frame.
            label_ObjectParamType_Title.grid(row=0)
            label_ObjectParamType_OrientaionAngles.grid(row=3, column=0)
            label_ObjectParamType_Alpha.grid(row=3, column=1)
            self.entry_ObjectParamType_Alpha.grid(row=3, column=2)
            label_ObjectParamType_Beta.grid(row=3, column=3)
            self.entry_ObjectParamType_Beta.grid(row=3, column=4)
            label_ObjectParamType_Gamma.grid(row=3, column=5)
            self.entry_ObjectParamType_Gamma.grid(row=3, column=6)
            label_ObjectParamType_RectangularCoilsDimensions.grid(row=1, column=0)
            label_ObjectParamType_CircularCoilRadius.grid(row=2, column=0)

            self.entry_ObjectParamType_RectangularCoilsDimensions.grid(row=1, column=2)
            self.entry_ObjectParamType_CircularCoilRadius.grid(row=2, column=2)
            label_ObjectParamType_DisplayObjectPosition.grid(row=4, column=0)
            label_ObjectParamType_X.grid(row=4, column=1)
            label_ObjectParamType_Y.grid(row=4, column=3)
            label_ObjectParamType_Z.grid(row=4, column=5)
            self.entry_ObjectParamType_X.grid(row=4, column=2)
            self.entry_ObjectParamType_Y.grid(row=4, column=4)
            self.entry_ObjectParamType_Z.grid(row=4, column=6)
            label_ObjectParamType_AnglesMessage.grid(row=3, column=7)
            label_ObjectParamType_DisplayObjectPositionMessage.grid(row=4, column=7)
            label_ObjectParamType_DisplayObjectPositionMessage1.grid(row=5, column=0)
        else: self.objectParamFrameDestroy()
    ## This function is trigered when the user check or uncheck  the Ball check-box
    # If the Ball object was chosen, the other three (Polygon, Puk, Ellipse) check-boxes are set to zero,
    # then it displays the Ball object parameters labes and entries.
    def displayObjectTypeBall(self):
        if self.objectTypeBall.get()==1:
            self.objectTypePolygon.set(0)
            self.objectTypePuk.set(0)
            self.objectTypeEllipse.set(0)
            self.objectTypeWearable.set(0)
            ## Remove the previous object entries and labels from the GUI if any.
            self.objectParamFrameDestroy()
            ## Create and place a frame that will contain the Ellipse object labels and entries.
            objectTypeParamFrameBall = Frame(self.objectParametersTab, relief='groove', borderwidth=5)
            objectTypeParamFrameBall.place(x=0, y = int(110*self.scale_factor))
            self.ObjectParamTypeFrame_register=objectTypeParamFrameBall
            # Create object type parameters labels and entries
            label_ObjectParamType_Title = Label(objectTypeParamFrameBall, text="Ball coils parameters",
                                                font=("Arial", self.titleFontSize),
                                                justify='left')

            label_ObjectParamType_Radius = Label(objectTypeParamFrameBall, text="Coils' radius (m): ",
                                                       font=("Arial", self.paramFontSize))
            label_ObjectParamType_OrientaionAngles = Label(objectTypeParamFrameBall, text="Coil orientation angles (deg): ",
                                                           font=("Arial", self.paramFontSize))
            label_ObjectParamType_Alpha = Label(objectTypeParamFrameBall,
                                                text=u'\u03b1',
                                                font=("Arial", self.paramFontSize))
            label_ObjectParamType_Beta = Label(objectTypeParamFrameBall,
                                               text=u'\u03b2',
                                               font=("Arial", self.paramFontSize))
            label_ObjectParamType_Gamma = Label(objectTypeParamFrameBall,
                                                text=u'\u03b3',
                                                font=("Arial", self.paramFontSize))
            label_ObjectParamType_DisplayObjectPosition = Label(objectTypeParamFrameBall,
                                                                text='Display ball at (m): ',
                                                                font=("Arial", self.paramFontSize))
            label_ObjectParamType_X = Label(objectTypeParamFrameBall,
                                            text='x',
                                            font=("Arial", self.paramFontSize))
            label_ObjectParamType_Y = Label(objectTypeParamFrameBall,
                                            text='y',
                                            font=("Arial", self.paramFontSize))
            label_ObjectParamType_Z = Label(objectTypeParamFrameBall,
                                            text='z',
                                            font=("Arial", self.paramFontSize))
            label_ObjectParamType_AnglesMessage = ttk.Label(objectTypeParamFrameBall,
                                                            text="Only for plotting. not included in the final table.",
                                                            font=("Arial", int(8*self.scale_factor)), fg='blue')
            label_ObjectParamType_DisplayObjectPositionMessage = ttk.Label(objectTypeParamFrameBall,
                                                                           text="Only for plotting. not included in the final table.",
                                                                           font=("Arial", int(8*self.scale_factor)), fg='blue')
            label_ObjectParamType_DisplayObjectPositionMessage1 = ttk.Label(objectTypeParamFrameBall,
                                                                            text="if any of x,y,z is empty,"+'\n'+" ball will be plotted at center of exciter.",
                                                                            font=("Arial", int(8*self.scale_factor)), fg='blue')
            self.entry_ObjectParamType_Radius = Entry(objectTypeParamFrameBall, width=int(5*self.scale_factor))
            self.entry_ObjectParamType_Radius.insert(0, str(self.objectParam_BallCoilRadius))
            label_ObjectParamType_Title.grid(row=0)
            self.entry_ObjectParamType_Alpha = Entry(objectTypeParamFrameBall, width=int(5*self.scale_factor))
            self.entry_ObjectParamType_Alpha.insert(0, str(self.objectParam_BallCoilOrientationAlpha))
            self.entry_ObjectParamType_Beta = Entry(objectTypeParamFrameBall, width=int(5*self.scale_factor))
            self.entry_ObjectParamType_Beta.insert(0, str(self.objectParam_BallCoilOrientationBeta))
            self.entry_ObjectParamType_Gamma = Entry(objectTypeParamFrameBall, width=int(5*self.scale_factor))
            self.entry_ObjectParamType_Gamma.insert(0, str(self.objectParam_BallCoilOrientationGamma))
            self.entry_ObjectParamType_X = Entry(objectTypeParamFrameBall, width=int(5*self.scale_factor))
            self.entry_ObjectParamType_Y = Entry(objectTypeParamFrameBall, width=int(5*self.scale_factor))
            self.entry_ObjectParamType_Z = Entry(objectTypeParamFrameBall, width=int(5*self.scale_factor))
            ## Griding the labels and entries into the frame.
            label_ObjectParamType_Radius.grid(row=1, column=0)
            self.entry_ObjectParamType_Radius.grid(row=1, column=2)
            label_ObjectParamType_OrientaionAngles.grid(row=2, column=0)
            label_ObjectParamType_Alpha.grid(row=2, column=1)
            self.entry_ObjectParamType_Alpha.grid(row=2, column=2)
            label_ObjectParamType_Beta.grid(row=2, column=3)
            self.entry_ObjectParamType_Beta.grid(row=2, column=4)
            label_ObjectParamType_Gamma.grid(row=2, column=5)
            self.entry_ObjectParamType_Gamma.grid(row=2, column=6)
            label_ObjectParamType_DisplayObjectPosition.grid(row=3, column=0)
            label_ObjectParamType_X.grid(row=3, column=1)
            label_ObjectParamType_Y.grid(row=3, column=3)
            label_ObjectParamType_Z.grid(row=3, column=5)
            self.entry_ObjectParamType_X.grid(row=3, column=2)
            self.entry_ObjectParamType_Y.grid(row=3, column=4)
            self.entry_ObjectParamType_Z.grid(row=3, column=6)
            label_ObjectParamType_AnglesMessage.grid(row=2, column=7)
            label_ObjectParamType_DisplayObjectPositionMessage.grid(row=3, column=7)
            label_ObjectParamType_DisplayObjectPositionMessage1.grid(row=4, column=0)
        else: self.objectParamFrameDestroy()

    ## This function is trigered when the user check or uncheck  the Ball check-box
    # If the Ball object was chosen, the other three (Polygon, Puk, Ellipse) check-boxes are set to zero,
    # then it displays the Ball object parameters labes and entries.
    def displayObjectTypeWearable(self):
        if self.objectTypeWearable.get() == 1:
            self.objectTypePolygon.set(0)
            self.objectTypePuk.set(0)
            self.objectTypeEllipse.set(0)
            self.objectTypeBall.set(0)
            ## Remove the previous object entries and labels from the GUI if any.
            self.objectParamFrameDestroy()
            ## Create and place a frame that will contain the Ellipse object labels and entries.
            objectTypeParamFrameWearable = Frame(self.objectParametersTab, relief='groove', borderwidth=5)
            objectTypeParamFrameWearable.place(x=0, y=int(110 * self.scale_factor))
            self.ObjectParamTypeFrame_register = objectTypeParamFrameWearable
            # Create object type parameters labels and entries
            label_ObjectParamType_Title = Label(objectTypeParamFrameWearable, text="Wearable coils parameters",
                                                font=("Arial", self.titleFontSize),
                                                justify='left')

            label_ObjectParamType_Coil1WidthLength = Label(objectTypeParamFrameWearable, text="Coil 1 Width, Length (m): ",
                                                 font=("Arial", self.paramFontSize))
            label_ObjectParamType_Coil2WidthLength = Label(objectTypeParamFrameWearable,
                                                           text="Coil 2 Width, Length (m): ",
                                                           font=("Arial", self.paramFontSize))
            label_ObjectParamType_Coil3WidthLength = Label(objectTypeParamFrameWearable,
                                                           text="Coil 3 Width, Length (m): ",
                                                           font=("Arial", self.paramFontSize))
            label_ObjectParamType_OrientaionAngles = Label(objectTypeParamFrameWearable,
                                                           text="Object orientation angles (deg): ",
                                                           font=("Arial", self.paramFontSize))
            label_ObjectParamType_Alpha = Label(objectTypeParamFrameWearable,
                                                text=u'\u03b1',
                                                font=("Arial", self.paramFontSize))
            label_ObjectParamType_Beta = Label(objectTypeParamFrameWearable,
                                               text=u'\u03b2',
                                               font=("Arial", self.paramFontSize))
            label_ObjectParamType_Gamma = Label(objectTypeParamFrameWearable,
                                                text=u'\u03b3',
                                                font=("Arial", self.paramFontSize))
            label_ObjectParamType_DisplayObjectPosition = Label(objectTypeParamFrameWearable,
                                                                text='Display wearable at (m): ',
                                                                font=("Arial", self.paramFontSize))
            label_ObjectParamType_X = Label(objectTypeParamFrameWearable,
                                            text='x',
                                            font=("Arial", self.paramFontSize))
            label_ObjectParamType_Y = Label(objectTypeParamFrameWearable,
                                            text='y',
                                            font=("Arial", self.paramFontSize))
            label_ObjectParamType_Z = Label(objectTypeParamFrameWearable,
                                            text='z',
                                            font=("Arial", self.paramFontSize))
            label_ObjectParamType_AnglesMessage = ttk.Label(objectTypeParamFrameWearable,
                                                            text="Only for plotting. not included in the final table.",
                                                            font=("Arial", int(8 * self.scale_factor)), fg='blue')
            label_ObjectParamType_DisplayObjectPositionMessage = ttk.Label(objectTypeParamFrameWearable,
                                                                           text="Only for plotting. not included in the final table.",
                                                                           font=(
                                                                           "Arial", int(8 * self.scale_factor)),
                                                                           fg='blue')
            label_ObjectParamType_DisplayObjectPositionMessage1 = ttk.Label(objectTypeParamFrameWearable,
                                                                            text="if any of x,y,z is empty," + '\n' + " ball will be plotted at center of exciter.",
                                                                            font=(
                                                                            "Arial", int(8 * self.scale_factor)),
                                                                            fg='blue')
            self.entry_ObjectParamType_coil1Width = Entry(objectTypeParamFrameWearable, width=int(5 * self.scale_factor))
            self.entry_ObjectParamType_coil1Width.insert(0, str(self.objectParam_PolygonCoilDimensionsLength))
            self.entry_ObjectParamType_coil1Length = Entry(objectTypeParamFrameWearable,
                                                          width=int(5 * self.scale_factor))
            self.entry_ObjectParamType_coil1Length.insert(0, str(self.objectParam_PolygonCoilDimensionsLength))
            self.entry_ObjectParamType_coil2Width = Entry(objectTypeParamFrameWearable,
                                                          width=int(5 * self.scale_factor))
            self.entry_ObjectParamType_coil2Width.insert(0, str(self.objectParam_PolygonCoilDimensionsLength))
            self.entry_ObjectParamType_coil2Length = Entry(objectTypeParamFrameWearable,
                                                           width=int(5 * self.scale_factor))
            self.entry_ObjectParamType_coil2Length.insert(0, str(self.objectParam_PolygonCoilDimensionsLength))
            self.entry_ObjectParamType_coil3Width = Entry(objectTypeParamFrameWearable,
                                                          width=int(5 * self.scale_factor))
            self.entry_ObjectParamType_coil3Width.insert(0, str(self.objectParam_PolygonCoilDimensionsLength))
            self.entry_ObjectParamType_coil3Length = Entry(objectTypeParamFrameWearable,
                                                           width=int(5 * self.scale_factor))
            self.entry_ObjectParamType_coil3Length.insert(0, str(self.objectParam_PolygonCoilDimensionsLength))
            label_ObjectParamType_Title.grid(row=0)
            self.entry_ObjectParamType_Alpha = Entry(objectTypeParamFrameWearable, width=int(5 * self.scale_factor))
            self.entry_ObjectParamType_Alpha.insert(0, str(self.objectParam_BallCoilOrientationAlpha))
            self.entry_ObjectParamType_Beta = Entry(objectTypeParamFrameWearable, width=int(5 * self.scale_factor))
            self.entry_ObjectParamType_Beta.insert(0, str(self.objectParam_BallCoilOrientationBeta))
            self.entry_ObjectParamType_Gamma = Entry(objectTypeParamFrameWearable, width=int(5 * self.scale_factor))
            self.entry_ObjectParamType_Gamma.insert(0, str(self.objectParam_BallCoilOrientationGamma))
            self.entry_ObjectParamType_X = Entry(objectTypeParamFrameWearable, width=int(5 * self.scale_factor))
            self.entry_ObjectParamType_Y = Entry(objectTypeParamFrameWearable, width=int(5 * self.scale_factor))
            self.entry_ObjectParamType_Z = Entry(objectTypeParamFrameWearable, width=int(5 * self.scale_factor))
            ## Griding the labels and entries into the frame.
            label_ObjectParamType_Coil1WidthLength.grid(row=1, column=0)
            label_ObjectParamType_Coil2WidthLength.grid(row=2, column=0)
            label_ObjectParamType_Coil3WidthLength.grid(row=3, column=0)
            self.entry_ObjectParamType_coil1Width.grid(row=1, column=2)
            self.entry_ObjectParamType_coil1Length.grid(row=1, column=3)
            self.entry_ObjectParamType_coil2Width.grid(row=2, column=2)
            self.entry_ObjectParamType_coil2Length.grid(row=2, column=3)
            self.entry_ObjectParamType_coil3Width.grid(row=3, column=2)
            self.entry_ObjectParamType_coil3Length.grid(row=3, column=3)
            label_ObjectParamType_OrientaionAngles.grid(row=4, column=0)
            label_ObjectParamType_Alpha.grid(row=4, column=1)
            self.entry_ObjectParamType_Alpha.grid(row=4, column=2)
            label_ObjectParamType_Beta.grid(row=4, column=3)
            self.entry_ObjectParamType_Beta.grid(row=4, column=4)
            label_ObjectParamType_Gamma.grid(row=4, column=5)
            self.entry_ObjectParamType_Gamma.grid(row=4, column=6)
            label_ObjectParamType_DisplayObjectPosition.grid(row=5, column=0)
            label_ObjectParamType_X.grid(row=5, column=1)
            label_ObjectParamType_Y.grid(row=5, column=3)
            label_ObjectParamType_Z.grid(row=5, column=5)
            self.entry_ObjectParamType_X.grid(row=5, column=2)
            self.entry_ObjectParamType_Y.grid(row=5, column=4)
            self.entry_ObjectParamType_Z.grid(row=5, column=6)
            label_ObjectParamType_AnglesMessage.grid(row=4, column=7)
            label_ObjectParamType_DisplayObjectPositionMessage.grid(row=5, column=7)
            label_ObjectParamType_DisplayObjectPositionMessage1.grid(row=6, column=0)
        else:
            self.objectParamFrameDestroy()
    ## This function destroys the current object's parameters displayed frame
    # and sets the new value to zero
    def objectParamFrameDestroy(self):
        if self.ObjectParamTypeFrame_register!= 0:
            self.ObjectParamTypeFrame_register.destroy()
            self.ObjectParamTypeFrame_register=0
    ## This function destroys the current main antennas' parameters displayed frame
    # and sets the new value to zero
    def antFrameMainDestroy(self):
        if self.AntParamTypeFrameMain_register!= 0:
            self.AntParamTypeFrameMain_register.destroy()
            self.AntParamTypeFrameMain_register = 0
    ## This function destroys the current frame antennas' parameters displayed frame
    # and sets the new value to zero
    def antFrameFrameDestroy(self):
        if self.AntParamTypeFrameFrame_register!= 0:
            self.AntParamTypeFrameFrame_register.destroy()
            self.AntParamTypeFrameFrame_register = 0
    ## This function destroys the current dictionary that contains
    # the displayed main antennas parameters variables' names, label objects, entries objects
    # and sets the new dictionary value to {}
    def mainParamLabelsEntriesDictDestroy(self):
        if len(self.mainParamLabelsEntriesDict) != 0:
            for elment in self.mainParamLabelsEntriesDict:
                self.mainParamLabelsEntriesDict[elment].destroy()
            self.mainParamLabelsEntriesDict ={}
    ## This function destroys the current dictionary that contains
    # the displayed frame antennas parameters variables' names, label objects, entries objects
    # and sets the new dictionary value to {}
    def frameParamLabelsEntriesDictDestroy(self):
        if len(self.frameParamLabelsEntriesDict) != 0:
            for elment in self.frameParamLabelsEntriesDict:
                self.frameParamLabelsEntriesDict[elment].destroy()
            self.frameParamLabelsEntriesDict = {}
            #self.antFrameFrameDestroy()
    ## This function creates and displays the necessary
    # labels and entries for each main antenna position
    # and orientation
    def addMainAnt(self):
        if self.numberOfMainAntennas.get() != self.numberOfMainAntennasBuffer:
            self.mainParamLabelsEntriesDictDestroy()
            self.numberOfMainAntennasBuffer = self.numberOfMainAntennas.get()
            self.registeredMainAntennas = self.numberOfMainAntennas.get()
            for i in range(1,self.numberOfMainAntennas.get()+1):
                self.mainParamLabelsEntriesDict["Ant {0} position".format(i)] = Label(self.AntParamTypeFrameMain_register, text="Ant. {0} pos. [x(m),y(m),z(m)]: ".format(i),
                                                                  font=("Arial", self.paramFontSize),justify='left')
                self.mainParamLabelsEntriesDict["Ant {0} position".format(i)].grid(row=i + 4, column=0)

                self.mainParamLabelsEntriesDict["Ant {0} position entry".format(i)] = Entry(self.AntParamTypeFrameMain_register,width=int(15*self.scale_factor))
                self.mainParamLabelsEntriesDict["Ant {0} position entry".format(i)].grid(row=i + 4, column=1)
                self.mainParamLabelsEntriesDict["Ant {0} position entry".format(i)].insert(0, str(self.mainAntennasDefaultPositions[i-1]))
                self.mainParamLabelsEntriesDict["Ant {0} orientaion".format(i)] = Label(
                    self.AntParamTypeFrameMain_register, text="orient. (deg): ".format(i),
                    font=("Arial", self.paramFontSize), justify='left')


                self.mainParamLabelsEntriesDict["Ant {0} alpha".format(i)] = Label(
                    self.AntParamTypeFrameMain_register, text=u'\u03b1',
                    font=("Arial", self.paramFontSize), justify='left')
                self.mainParamLabelsEntriesDict["Ant {0} beta".format(i)] = Label(
                    self.AntParamTypeFrameMain_register, text=u'\u03b2',
                    font=("Arial", self.paramFontSize), justify='left')
                self.mainParamLabelsEntriesDict["Ant {0} gamma".format(i)] = Label(
                    self.AntParamTypeFrameMain_register, text=u'\u03b3',
                    font=("Arial", self.paramFontSize), justify='left')
                self.mainParamLabelsEntriesDict["Ant {0} alpha entry".format(i)] = Entry(
                    self.AntParamTypeFrameMain_register, width=int(5*self.scale_factor))
                self.mainParamLabelsEntriesDict["Ant {0} beta entry".format(i)] = Entry(
                    self.AntParamTypeFrameMain_register, width=int(5*self.scale_factor))
                self.mainParamLabelsEntriesDict["Ant {0} gamma entry".format(i)] = Entry(
                    self.AntParamTypeFrameMain_register, width=int(5*self.scale_factor))


                self.mainParamLabelsEntriesDict["Ant {0} orientaion".format(i)].grid(row=i + 4, column=2)
                self.mainParamLabelsEntriesDict["Ant {0} alpha".format(i)].grid(row=i + 4, column=3)
                self.mainParamLabelsEntriesDict["Ant {0} alpha entry".format(i)].grid(row=i + 4, column=4)
                self.mainParamLabelsEntriesDict["Ant {0} beta".format(i)].grid(row=i + 4, column=5)
                self.mainParamLabelsEntriesDict["Ant {0} beta entry".format(i)].grid(row=i + 4, column=6)
                self.mainParamLabelsEntriesDict["Ant {0} gamma".format(i)].grid(row=i + 4, column=7)
                self.mainParamLabelsEntriesDict["Ant {0} gamma entry".format(i)].grid(row=i + 4, column=8)

                self.mainParamLabelsEntriesDict["Ant {0} alpha entry".format(i)].insert(0, self.mainAntennasDefaultOrientations[i-1][0])
                self.mainParamLabelsEntriesDict["Ant {0} beta entry".format(i)].insert(0,self.mainAntennasDefaultOrientations[i - 1][1])
                self.mainParamLabelsEntriesDict["Ant {0} gamma entry".format(i)].insert(0,self.mainAntennasDefaultOrientations[i - 1][2])
    ## This function creates and displays the necessary
    # labels and entries objects for each frame antenna position
    # and orientation
    def addFrameAnt(self):
        if self.numberOfFrameAntennas.get() != self.numberOfFrameAntennasBuffer:
            self.frameParamLabelsEntriesDictDestroy()
            self.numberOfFrameAntennasBuffer = self.numberOfFrameAntennas.get()
            self.registeredFrameAntennas = self.numberOfFrameAntennas.get()
            for i in range(1,self.numberOfFrameAntennas.get()+1):
                self.frameParamLabelsEntriesDict["Ant {0} position".format(i)] = Label(self.AntParamTypeFrameFrame_register, text="Ant. {0} pos. [x(m),y(m),z(m)]: ".format(i),
                                                                  font=("Arial", self.paramFontSize),justify='left')
                self.frameParamLabelsEntriesDict["Ant {0} position".format(i)].grid(row=i + 4, column=0)
                self.frameParamLabelsEntriesDict["Ant {0} position entry".format(i)] = Entry(self.AntParamTypeFrameFrame_register, width=int(15*self.scale_factor))
                self.frameParamLabelsEntriesDict["Ant {0} position entry".format(i)].grid(row=i + 4, column=1)
                self.frameParamLabelsEntriesDict["Ant {0} position entry".format(i)].insert(0, str(self.frameAntennasDefaultPositions[i-1]))

                self.frameParamLabelsEntriesDict["Ant {0} orientaion".format(i)] = Label(
                    self.AntParamTypeFrameFrame_register, text="orient. (deg): ".format(i),
                    font=("Arial", self.paramFontSize), justify='left')
                self.frameParamLabelsEntriesDict["Ant {0} alpha".format(i)] = Label(
                    self.AntParamTypeFrameFrame_register, text=u'\u03b1',
                    font=("Arial", self.paramFontSize), justify='left')
                self.frameParamLabelsEntriesDict["Ant {0} beta".format(i)] = Label(
                    self.AntParamTypeFrameFrame_register, text=u'\u03b2',
                    font=("Arial", self.paramFontSize), justify='left')
                self.frameParamLabelsEntriesDict["Ant {0} gamma".format(i)] = Label(
                    self.AntParamTypeFrameFrame_register, text=u'\u03b3',
                    font=("Arial", self.paramFontSize), justify='left')
                self.frameParamLabelsEntriesDict["Ant {0} alpha entry".format(i)] = Entry(
                    self.AntParamTypeFrameFrame_register, width=int(5*self.scale_factor))
                self.frameParamLabelsEntriesDict["Ant {0} beta entry".format(i)] = Entry(
                    self.AntParamTypeFrameFrame_register, width=int(5*self.scale_factor))
                self.frameParamLabelsEntriesDict["Ant {0} gamma entry".format(i)] = Entry(
                    self.AntParamTypeFrameFrame_register, width=int(5*self.scale_factor))

                self.frameParamLabelsEntriesDict["Ant {0} orientaion".format(i)].grid(row=i + 4, column=2)
                self.frameParamLabelsEntriesDict["Ant {0} alpha".format(i)].grid(row=i + 4, column=3)
                self.frameParamLabelsEntriesDict["Ant {0} alpha entry".format(i)].grid(row=i + 4, column=4)
                self.frameParamLabelsEntriesDict["Ant {0} beta".format(i)].grid(row=i + 4, column=5)
                self.frameParamLabelsEntriesDict["Ant {0} beta entry".format(i)].grid(row=i + 4, column=6)
                self.frameParamLabelsEntriesDict["Ant {0} gamma".format(i)].grid(row=i + 4, column=7)
                self.frameParamLabelsEntriesDict["Ant {0} gamma entry".format(i)].grid(row=i + 4, column=8)
                self.frameParamLabelsEntriesDict["Ant {0} alpha entry".format(i)].insert(0,self.frameAntennasDefaultOrientations[i - 1][0])
                self.frameParamLabelsEntriesDict["Ant {0} beta entry".format(i)].insert(0,self.frameAntennasDefaultOrientations[ i - 1][1])
                self.frameParamLabelsEntriesDict["Ant {0} gamma entry".format(i)].insert(0,self.frameAntennasDefaultOrientations[i - 1][2])
    ## This function is trigered when the user check or uncheck the Main check-box
    # it displays the Main antennas parameters labes and entries
    def displayAntTypeMain(self):
        if self.antTypeMain.get()== 1:
            if self.AntParamTypeFrameMain_register != 0:
                self.antFrameMainDestroy()
            antennaTypeFrameMain = Frame(self.antennaParametersTab, relief='groove', borderwidth=5)
            antennaTypeFrameMain.place(x=0, y=int(40*self.scale_factor))
            self.AntParamTypeFrameMain_register = antennaTypeFrameMain
            self.mainParamLabelsEntriesDict = {}
            self.numberOfMainAntennas = IntVar()
            self.numberOfMainAntennas.set(0)
            self.numberOfMainAntennasBuffer = self.numberOfMainAntennas.get()
            choices = {'',1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                       11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
            # Create object type parameters labels, dropdown menu and entries:
            label_mainAntennasWindings = Label(antennaTypeFrameMain, text="Number of windings: ", font=("Arial", self.paramFontSize),
                                                 justify='left')
            label_mainAntennasLength = Label(antennaTypeFrameMain, text="Length (m): ", font=("Arial", self.paramFontSize),
                                                 justify='left')
            label_mainAntennasHeight = Label(antennaTypeFrameMain, text="Height (m): ", font=("Arial", self.paramFontSize),
                                             justify='left')
            self.entry_mainAntennasWindings = Entry(antennaTypeFrameMain, width=int(5*self.scale_factor))
            self.entry_mainAntennasWindings.insert(0, str(self.antennaParam_MainNumberOfWindings))
            self.entry_mainAntennasLength = Entry(antennaTypeFrameMain, width=int(5*self.scale_factor))
            self.entry_mainAntennasLength.insert(0, str(self.antennaParam_MainLength))
            self.entry_mainAntennasHeight = Entry(antennaTypeFrameMain, width=int(5*self.scale_factor))
            self.entry_mainAntennasHeight.insert(0, str(self.antennaParam_MainHeight))
            label_mainAntennasWindings.grid(row= 0, column=0)
            label_mainAntennasLength.grid(row= 1, column=0)
            label_mainAntennasHeight.grid(row= 2, column=0)
            self.entry_mainAntennasWindings.grid(row= 0, column=1)
            self.entry_mainAntennasLength.grid(row= 1, column=1)
            self.entry_mainAntennasHeight.grid(row= 2, column=1)
            label_NumberOfMainAntennas = Label(antennaTypeFrameMain, text="Number of main ants. : ", font=("Arial", self.paramFontSize),
                                                 justify='left')
            dropDownList = OptionMenu(antennaTypeFrameMain, self.numberOfMainAntennas, *choices)
            dropDownList.grid(row=3, column=1)
            label_NumberOfMainAntennas.grid(row= 3, column=0)
            addMainAntButton = ttk.Button(antennaTypeFrameMain, text="Add", command=self.addMainAnt,height=int(1*self.scale_factor), width=int(6*self.scale_factor))
            addMainAntButton.grid(row=3, column=2)
        elif self.antTypeMain.get()== 0:
            self.mainParamLabelsEntriesDictDestroy()
            self.registeredMainAntennas = 0
            self.antFrameMainDestroy()
    ## This function is trigered when the user check or uncheck the Frame check-box
    # it displays the Frame antennas parameters labes and entries
    def displayAntTypeFrame(self):
        if  self.antTypeFrame.get()== 1:
            if self.AntParamTypeFrameFrame_register != 0:
                self.antFrameFrameDestroy()
            antennaTypeFrameFrame = Frame(self.antennaParametersTab, relief='groove', borderwidth=5)
            antennaTypeFrameFrame.place(x=int(600*self.scale_factor), y=int(40*self.scale_factor))
            self.AntParamTypeFrameFrame_register = antennaTypeFrameFrame
            self.frameParamLabelsEntriesDict = {}
            self.numberOfFrameAntennas = IntVar()
            self.numberOfFrameAntennas.set(0)
            self.numberOfFrameAntennasBuffer = self.numberOfFrameAntennas.get()
            choicesFrame = {'',1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                       11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
            # Create object type parameters labels, dropdown menu and entries:
            label_frameAntennasWindings = Label(antennaTypeFrameFrame, text="Number of windings: ",
                                               font=("Arial", self.paramFontSize),
                                               justify='left')
            label_frameAntennasLength = Label(antennaTypeFrameFrame, text="Length (m): ", font=("Arial", self.paramFontSize),
                                             justify='left')
            label_frameAntennasHeight = Label(antennaTypeFrameFrame, text="Height (m): ", font=("Arial", self.paramFontSize),
                                             justify='left')
            self.entry_frameAntennasWindings = Entry(antennaTypeFrameFrame, width=int(5*self.scale_factor))
            self.entry_frameAntennasWindings.insert(0, str(self.antennaParam_FrameNumberOfWindings))
            self.entry_frameAntennasLength = Entry(antennaTypeFrameFrame, width=int(5*self.scale_factor))
            self.entry_frameAntennasLength.insert(0, str(self.antennaParam_FrameLength))
            self.entry_frameAntennasHeight = Entry(antennaTypeFrameFrame, width=int(5*self.scale_factor))
            self.entry_frameAntennasHeight.insert(0, str(self.antennaParam_FrameHeight))
            label_frameAntennasWindings.grid(row=0, column=0)
            label_frameAntennasLength.grid(row=1, column=0)
            label_frameAntennasHeight.grid(row=2, column=0)
            self.entry_frameAntennasWindings.grid(row=0, column=1)
            self.entry_frameAntennasLength.grid(row=1, column=1)
            self.entry_frameAntennasHeight.grid(row=2, column=1)
            label_NumberOfFrameAntennas = Label(antennaTypeFrameFrame, text="Number of frame ants. : ", font=("Arial", self.paramFontSize),
                                                 justify='left')
            dropDownListFrame = OptionMenu(antennaTypeFrameFrame, self.numberOfFrameAntennas, *choicesFrame)
            dropDownListFrame.grid(row=3, column=1)
            label_NumberOfFrameAntennas.grid(row= 3, column=0)
            addFrameAntButton = ttk.Button(antennaTypeFrameFrame, text="Add", command=self.addFrameAnt,height=int(1*self.scale_factor), width=int(6*self.scale_factor))
            addFrameAntButton.grid(row=3, column=2)
        elif self.antTypeFrame.get()== 0:
            self.frameParamLabelsEntriesDictDestroy()
            self.registeredFrameAntennas = 0
            self.antFrameFrameDestroy()
    ## This function is trigered when the user check or uncheck the
    # xpos: Single/Multiple points checkbox
    # it displays the necessary labels and entries
    def displayXPosSingleMultiplePoints(self):
        if self.xpos_singleMultiplePoints.get()==1:
            self.xpos_sweep.set(0)
            if len(self.xPosOption_register) != 0:
                for elment in self.xPosOption_register:
                    elment.destroy()
            label_xpos_singleMultiple = Label(self.tableParamFrame, text="x-point(s) (m) [p1, p2,...]", font=("Arial", self.paramFontSize))
            self.entry_xpos_singleMultiple = Entry(self.tableParamFrame, width=int(15*self.scale_factor))
            self.entry_xpos_singleMultiple.insert(0,str([0.05]))
            label_xpos_singleMultiple.grid(row=3, column=0)
            self.entry_xpos_singleMultiple.grid(row=3, column=1)
            self.xPosOption_register.append(label_xpos_singleMultiple)
            self.xPosOption_register.append(self.entry_xpos_singleMultiple)
        else:
            self.xpos_sweep.set(1)
            self.displayXPosSweep()
    ## This function is trigered when the user check or uncheck the
    # xpos: Sweep checkbox
    # it displays the necessary labels and entries
    def displayXPosSweep(self):
        if self.xpos_sweep.get() == 1:
            self.xpos_singleMultiplePoints.set(0)
            if len(self.xPosOption_register) != 0:
                for elment in self.xPosOption_register:
                    elment.destroy()
            label_xpos_sweep = Label(self.tableParamFrame, text="x-sweep (m), end not included: ",
                                              font=("Arial", self.paramFontSize))
            label_xpos_sweepStart = Label(self.tableParamFrame, text="start",
                                     font=("Arial", self.paramFontSize))
            label_xpos_sweepEnd = Label(self.tableParamFrame, text="end",
                                          font=("Arial", self.paramFontSize))
            label_xpos_sweepStep = Label(self.tableParamFrame, text="step",
                                         font=("Arial", self.paramFontSize))

            self.entry_xpos_sweepStart = Entry(self.tableParamFrame, width=int(5*self.scale_factor))
            self.entry_xpos_sweepStart.insert(0, 0.05)
            self.entry_xpos_sweepEnd = Entry(self.tableParamFrame, width=int(5*self.scale_factor))
            self.entry_xpos_sweepEnd.insert(0, 1.85)
            self.entry_xpos_sweepStep = Entry(self.tableParamFrame, width=int(5*self.scale_factor))
            self.entry_xpos_sweepStep.insert(0, 0.01)
            label_xpos_sweep.grid(row=3, column=0)
            label_xpos_sweepStart.grid(row=3, column=2)
            self.entry_xpos_sweepStart.grid(row=3, column=3)
            label_xpos_sweepEnd.grid(row=3, column=4)
            self.entry_xpos_sweepEnd.grid(row=3, column=5)
            label_xpos_sweepStep.grid(row=3, column=6)
            self.entry_xpos_sweepStep.grid(row=3, column=7)
            self.xPosOption_register.append(label_xpos_sweep)
            self.xPosOption_register.append(label_xpos_sweepStart)
            self.xPosOption_register.append(label_xpos_sweepEnd)
            self.xPosOption_register.append(label_xpos_sweepStep)
            self.xPosOption_register.append(self.entry_xpos_sweepStart)
            self.xPosOption_register.append(self.entry_xpos_sweepEnd)
            self.xPosOption_register.append(self.entry_xpos_sweepStep)

        else:
            self.xpos_singleMultiplePoints.set(1)
            self.displayXPosSingleMultiplePoints()
    ## This function is trigered when the user check or uncheck the
    # ypos: Single/Multiple points checkbox
    # it displays the necessary labels and entries
    def displayYPosSingleMultiplePoints(self):
        if self.ypos_singleMultiplePoints.get()==1:
            self.ypos_sweep.set(0)
            if len(self.yPosOption_register) != 0:
                for elment in self.yPosOption_register:
                    elment.destroy()
            label_ypos_singleMultiple = Label(self.tableParamFrame, text="y-point(s) (m) [p1, p2,...]", font=("Arial", self.paramFontSize))
            self.entry_ypos_singleMultiple = Entry(self.tableParamFrame, width=int(15*self.scale_factor))
            self.entry_ypos_singleMultiple.insert(0,str([0.05]))
            label_ypos_singleMultiple.grid(row=5, column=0)
            self.entry_ypos_singleMultiple.grid(row=5, column=1)
            self.yPosOption_register.append(label_ypos_singleMultiple)
            self.yPosOption_register.append(self.entry_ypos_singleMultiple)
        else:
            self.ypos_sweep.set(1)
            self.displayYPosSweep()
    ## This function is trigered when the user check or uncheck the
    # ypos: Sweep checkbox
    # it displays the necessary labels and entries
    def displayYPosSweep(self):
        if self.ypos_sweep.get() == 1:
            self.ypos_singleMultiplePoints.set(0)
            if len(self.yPosOption_register) != 0:
                for elment in self.yPosOption_register:
                    elment.destroy()
            label_ypos_sweep = Label(self.tableParamFrame, text="y-sweep (m), end not included: ",
                                              font=("Arial", self.paramFontSize))
            label_ypos_sweepStart = Label(self.tableParamFrame, text="start",
                                     font=("Arial", self.paramFontSize))
            label_ypos_sweepEnd = Label(self.tableParamFrame, text="end",
                                          font=("Arial", self.paramFontSize))
            label_ypos_sweepStep = Label(self.tableParamFrame, text="step",
                                         font=("Arial", self.paramFontSize))

            self.entry_ypos_sweepStart = Entry(self.tableParamFrame, width=int(5*self.scale_factor))
            self.entry_ypos_sweepStart.insert(0, 0.05)
            self.entry_ypos_sweepEnd = Entry(self.tableParamFrame, width=int(5*self.scale_factor))
            self.entry_ypos_sweepEnd.insert(0, 1.2)
            self.entry_ypos_sweepStep = Entry(self.tableParamFrame, width=int(5*self.scale_factor))
            self.entry_ypos_sweepStep.insert(0, 0.01)
            label_ypos_sweep.grid(row=5, column=0)
            label_ypos_sweepStart.grid(row=5, column=2)
            self.entry_ypos_sweepStart.grid(row=5, column=3)
            label_ypos_sweepEnd.grid(row=5, column=4)
            self.entry_ypos_sweepEnd.grid(row=5, column=5)
            label_ypos_sweepStep.grid(row=5, column=6)
            self.entry_ypos_sweepStep.grid(row=5, column=7)
            self.yPosOption_register.append(label_ypos_sweep)
            self.yPosOption_register.append(label_ypos_sweepStart)
            self.yPosOption_register.append(label_ypos_sweepEnd)
            self.yPosOption_register.append(label_ypos_sweepStep)
            self.yPosOption_register.append(self.entry_ypos_sweepStart)
            self.yPosOption_register.append(self.entry_ypos_sweepEnd)
            self.yPosOption_register.append(self.entry_ypos_sweepStep)

        else:
            self.ypos_singleMultiplePoints.set(1)
            self.displayYPosSingleMultiplePoints()
    ## This function is trigered when the user check or uncheck the
    # zpos: Single/Multiple points checkbox
    # it displays the necessary labels and entries
    def displayZPosSingleMultiplePoints(self):
        if self.zpos_singleMultiplePoints.get()==1:
            self.zpos_sweep.set(0)
            if len(self.zPosOption_register) != 0:
                for elment in self.zPosOption_register:
                    elment.destroy()
            label_zpos_singleMultiple = Label(self.tableParamFrame, text="z-point(s) (m) [p1, p2,...]", font=("Arial", self.paramFontSize))
            self.entry_zpos_singleMultiple = Entry(self.tableParamFrame, width=int(15*self.scale_factor))
            self.entry_zpos_singleMultiple.insert(0,str([0]))
            label_zpos_singleMultiple.grid(row=7, column=0)
            self.entry_zpos_singleMultiple.grid(row=7, column=1)
            self.zPosOption_register.append(label_zpos_singleMultiple)
            self.zPosOption_register.append(self.entry_zpos_singleMultiple)
        else:
            self.zpos_sweep.set(1)
            self.displayZPosSweep()
    ## This function is trigered when the user check or uncheck the
    # zpos: Sweep checkbox
    # it displays the necessary labels and entries
    def displayZPosSweep(self):
        if self.zpos_sweep.get() == 1:
            self.zpos_singleMultiplePoints.set(0)
            if len(self.zPosOption_register) != 0:
                for elment in self.zPosOption_register:
                    elment.destroy()
            label_zpos_sweep = Label(self.tableParamFrame, text="z-sweep (m), end not included: ",
                                              font=("Arial", self.paramFontSize))
            label_zpos_sweepStart = Label(self.tableParamFrame, text="start",
                                     font=("Arial", self.paramFontSize))
            label_zpos_sweepEnd = Label(self.tableParamFrame, text="end",
                                          font=("Arial", self.paramFontSize))
            label_zpos_sweepStep = Label(self.tableParamFrame, text="step",
                                         font=("Arial", self.paramFontSize))

            self.entry_zpos_sweepStart = Entry(self.tableParamFrame, width=int(5*self.scale_factor))
            self.entry_zpos_sweepStart.insert(0, -1)
            self.entry_zpos_sweepEnd = Entry(self.tableParamFrame, width=int(5*self.scale_factor))
            self.entry_zpos_sweepEnd.insert(0, 1)
            self.entry_zpos_sweepStep = Entry(self.tableParamFrame, width=int(5*self.scale_factor))
            self.entry_zpos_sweepStep.insert(0, 0.1)
            label_zpos_sweep.grid(row=7, column=0)
            label_zpos_sweepStart.grid(row=7, column=2)
            self.entry_zpos_sweepStart.grid(row=7, column=3)
            label_zpos_sweepEnd.grid(row=7, column=4)
            self.entry_zpos_sweepEnd.grid(row=7, column=5)
            label_zpos_sweepStep.grid(row=7, column=6)
            self.entry_zpos_sweepStep.grid(row=7, column=7)
            self.zPosOption_register.append(label_zpos_sweep)
            self.zPosOption_register.append(label_zpos_sweepStart)
            self.zPosOption_register.append(label_zpos_sweepEnd)
            self.zPosOption_register.append(label_zpos_sweepStep)
            self.zPosOption_register.append(self.entry_zpos_sweepStart)
            self.zPosOption_register.append(self.entry_zpos_sweepEnd)
            self.zPosOption_register.append(self.entry_zpos_sweepStep)

        else:
            self.zpos_singleMultiplePoints.set(1)
            self.displayZPosSingleMultiplePoints()
    ## This function is trigered when the user check or uncheck the
    # alpha: Single/Multiple angles checkbox
    # it displays the necessary labels and entries
    def displayAlphaSingleMultipleAngles(self):
        if self.alpha_singleMultipleAngles.get()==1:
            self.alpha_sweep.set(0)
            if len(self.alphaOption_register) != 0:
                for elment in self.alphaOption_register:
                    elment.destroy()
            label_alpha_singleMultiple = Label(self.tableParamFrame, text=u'\u03b1'+"-angle(s) (deg) ["+u'\u03b1'+"1, "+u'\u03b1'+"2,...]", font=("Arial", self.paramFontSize))
            self.entry_alpha_singleMultiple = Entry(self.tableParamFrame, width=int(15*self.scale_factor))
            self.entry_alpha_singleMultiple.insert(0,str([0]))
            label_alpha_singleMultiple.grid(row=9, column=0)
            self.entry_alpha_singleMultiple.grid(row=9, column=1)
            self.alphaOption_register.append(label_alpha_singleMultiple)
            self.alphaOption_register.append(self.entry_alpha_singleMultiple)
        else:
            self.alpha_sweep.set(1)
            self.displayAlphaSweep()
    ## This function is trigered when the user check or uncheck the
    # alpha: Sweep checkbox
    # it displays the necessary labels and entries
    def displayAlphaSweep(self):
        if self.alpha_sweep.get() == 1:
            self.alpha_singleMultipleAngles.set(0)
            if len(self.alphaOption_register) != 0:
                for elment in self.alphaOption_register:
                    elment.destroy()
            label_alpha_sweep = Label(self.tableParamFrame, text=u'\u03b1'+"-sweep (deg), end not included: ",
                                              font=("Arial", self.paramFontSize))
            label_alpha_sweepStart = Label(self.tableParamFrame, text="start",
                                     font=("Arial", self.paramFontSize))
            label_alpha_sweepEnd = Label(self.tableParamFrame, text="end",
                                          font=("Arial", self.paramFontSize))
            label_alpha_sweepStep = Label(self.tableParamFrame, text="step",
                                         font=("Arial", self.paramFontSize))

            self.entry_alpha_sweepStart = Entry(self.tableParamFrame, width=int(5*self.scale_factor))
            self.entry_alpha_sweepStart.insert(0, 0)
            self.entry_alpha_sweepEnd = Entry(self.tableParamFrame, width=int(5*self.scale_factor))
            self.entry_alpha_sweepEnd.insert(0, 180)
            self.entry_alpha_sweepStep = Entry(self.tableParamFrame, width=int(5*self.scale_factor))
            self.entry_alpha_sweepStep.insert(0, 45)
            label_alpha_sweep.grid(row=9, column=0)
            label_alpha_sweepStart.grid(row=9, column=2)
            self.entry_alpha_sweepStart.grid(row=9, column=3)
            label_alpha_sweepEnd.grid(row=9, column=4)
            self.entry_alpha_sweepEnd.grid(row=9, column=5)
            label_alpha_sweepStep.grid(row=9, column=6)
            self.entry_alpha_sweepStep.grid(row=9, column=7)
            self.alphaOption_register.append(label_alpha_sweep)
            self.alphaOption_register.append(label_alpha_sweepStart)
            self.alphaOption_register.append(label_alpha_sweepEnd)
            self.alphaOption_register.append(label_alpha_sweepStep)
            self.alphaOption_register.append(self.entry_alpha_sweepStart)
            self.alphaOption_register.append(self.entry_alpha_sweepEnd)
            self.alphaOption_register.append(self.entry_alpha_sweepStep)

        else:
            self.alpha_singleMultipleAngles.set(1)
            self.displayAlphaSingleMultipleAngles()
    ## This function is trigered when the user check or uncheck the
    # beta: Single/Multiple angles checkbox
    # it displays the necessary labels and entries
    def displayBetaSingleMultipleAngles(self):
        if self.beta_singleMultipleAngles.get()==1:
            self.beta_sweep.set(0)
            if len(self.betaOption_register) != 0:
                for elment in self.betaOption_register:
                    elment.destroy()
            label_beta_singleMultiple = Label(self.tableParamFrame, text=u'\u03b2'+"-angle(s) (deg) ["+u'\u03b2'+"1, "+u'\u03b2'+"2,...]", font=("Arial", self.paramFontSize))
            self.entry_beta_singleMultiple = Entry(self.tableParamFrame, width=int(15*self.scale_factor))
            self.entry_beta_singleMultiple.insert(0,str([0]))
            label_beta_singleMultiple.grid(row=11, column=0)
            self.entry_beta_singleMultiple.grid(row=11, column=1)
            self.betaOption_register.append(label_beta_singleMultiple)
            self.betaOption_register.append(self.entry_beta_singleMultiple)
        else:
            self.beta_sweep.set(1)
            self.displayBetaSweep()
    ## This function is trigered when the user check or uncheck the
    # beta: Sweep checkbox
    # it displays the necessary labels and entries
    def displayBetaSweep(self):
        if self.beta_sweep.get() == 1:
            self.beta_singleMultipleAngles.set(0)
            if len(self.betaOption_register) != 0:
                for elment in self.betaOption_register:
                    elment.destroy()
            label_beta_sweep = Label(self.tableParamFrame, text=u'\u03b2'+"-sweep (deg), end not included: ",
                                              font=("Arial", self.paramFontSize))
            label_beta_sweepStart = Label(self.tableParamFrame, text="start",
                                     font=("Arial", self.paramFontSize))
            label_beta_sweepEnd = Label(self.tableParamFrame, text="end",
                                          font=("Arial", self.paramFontSize))
            label_beta_sweepStep = Label(self.tableParamFrame, text="step",
                                         font=("Arial", self.paramFontSize))

            self.entry_beta_sweepStart = Entry(self.tableParamFrame, width=int(5*self.scale_factor))
            self.entry_beta_sweepStart.insert(0, 0)
            self.entry_beta_sweepEnd = Entry(self.tableParamFrame, width=int(5*self.scale_factor))
            self.entry_beta_sweepEnd.insert(0, 180)
            self.entry_beta_sweepStep = Entry(self.tableParamFrame, width=int(5*self.scale_factor))
            self.entry_beta_sweepStep.insert(0, 45)
            label_beta_sweep.grid(row=11, column=0)
            label_beta_sweepStart.grid(row=11, column=2)
            self.entry_beta_sweepStart.grid(row=11, column=3)
            label_beta_sweepEnd.grid(row=11, column=4)
            self.entry_beta_sweepEnd.grid(row=11, column=5)
            label_beta_sweepStep.grid(row=11, column=6)
            self.entry_beta_sweepStep.grid(row=11, column=7)
            self.betaOption_register.append(label_beta_sweep)
            self.betaOption_register.append(label_beta_sweepStart)
            self.betaOption_register.append(label_beta_sweepEnd)
            self.betaOption_register.append(label_beta_sweepStep)
            self.betaOption_register.append(self.entry_beta_sweepStart)
            self.betaOption_register.append(self.entry_beta_sweepEnd)
            self.betaOption_register.append(self.entry_beta_sweepStep)

        else:
            self.beta_singleMultipleAngles.set(1)
            self.displayBetaSingleMultipleAngles()
    ## This function is trigered when the user check or uncheck the
    # gamma: Single/Multiple angles checkbox
    # it displays the necessary labels and entries
    def displayGammaSingleMultipleAngles(self):
        if self.gamma_singleMultipleAngles.get()==1:
            self.gamma_sweep.set(0)
            if len(self.gammaOption_register) != 0:
                for elment in self.gammaOption_register:
                    elment.destroy()
            label_gamma_singleMultiple = Label(self.tableParamFrame, text=u'\u03b3'+"-angle(s) (deg) ["+u'\u03b3'+"1, "+u'\u03b3'+"2,...]", font=("Arial", self.paramFontSize))
            self.entry_gamma_singleMultiple = Entry(self.tableParamFrame, width=int(15*self.scale_factor))
            self.entry_gamma_singleMultiple.insert(0,str([0]))
            label_gamma_singleMultiple.grid(row=13, column=0)
            self.entry_gamma_singleMultiple.grid(row=13, column=1)
            self.gammaOption_register.append(label_gamma_singleMultiple)
            self.gammaOption_register.append(self.entry_gamma_singleMultiple)
        else:
            self.gamma_sweep.set(1)
            self.displayGammaSweep()
    ## This function is trigered when the user check or uncheck the
    # gamma: Sweep checkbox
    # it displays the necessary labels and entries
    def displayGammaSweep(self):
        if self.gamma_sweep.get() == 1:
            self.gamma_singleMultipleAngles.set(0)
            if len(self.gammaOption_register) != 0:
                for elment in self.gammaOption_register:
                    elment.destroy()
            label_gamma_sweep = Label(self.tableParamFrame, text=u'\u03b3'+"-sweep (deg), end not included: ",
                                              font=("Arial", self.paramFontSize))
            label_gamma_sweepStart = Label(self.tableParamFrame, text="start",
                                     font=("Arial", self.paramFontSize))
            label_gamma_sweepEnd = Label(self.tableParamFrame, text="end",
                                          font=("Arial", self.paramFontSize))
            label_gamma_sweepStep = Label(self.tableParamFrame, text="step",
                                         font=("Arial", self.paramFontSize))

            self.entry_gamma_sweepStart = Entry(self.tableParamFrame, width=int(5*self.scale_factor))
            self.entry_gamma_sweepStart.insert(0, 0)
            self.entry_gamma_sweepEnd = Entry(self.tableParamFrame, width=int(5*self.scale_factor))
            self.entry_gamma_sweepEnd.insert(0, 180)
            self.entry_gamma_sweepStep = Entry(self.tableParamFrame, width=int(5*self.scale_factor))
            self.entry_gamma_sweepStep.insert(0, 45)
            label_gamma_sweep.grid(row=13, column=0)
            label_gamma_sweepStart.grid(row=13, column=2)
            self.entry_gamma_sweepStart.grid(row=13, column=3)
            label_gamma_sweepEnd.grid(row=13, column=4)
            self.entry_gamma_sweepEnd.grid(row=13, column=5)
            label_gamma_sweepStep.grid(row=13, column=6)
            self.entry_gamma_sweepStep.grid(row=13, column=7)
            self.gammaOption_register.append(label_gamma_sweep)
            self.gammaOption_register.append(label_gamma_sweepStart)
            self.gammaOption_register.append(label_gamma_sweepEnd)
            self.gammaOption_register.append(label_gamma_sweepStep)
            self.gammaOption_register.append(self.entry_gamma_sweepStart)
            self.gammaOption_register.append(self.entry_gamma_sweepEnd)
            self.gammaOption_register.append(self.entry_gamma_sweepStep)

        else:
            self.gamma_singleMultipleAngles.set(1)
            self.displayGammaSingleMultipleAngles()
    ## This function kills the given running process.
    def kill_proc_tree(self,pid, including_parent=True):
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            for child in children:
                child.kill()
            gone, still_alive = psutil.wait_procs(children, timeout=5)
            if including_parent:
                parent.kill()
                parent.wait(5)
        except:
            pass
    ## This function check if there is a running process appeneded to self.proc
    # and if so, it calls the kill_proc_tree() method to kill them all the appended processes
    # then sets the current self.proc to an empty list [].
    # else, it just displays a message.
    ## This function is trigered when the user press the Stop button
    # it calls the kill_proc_tree() if there is any process in the
    # self.proc list.
    # else it displays a message.
    def stop(self):
        if len(self.proc) !=0:
            try:
                for process in self.proc:
                    self.kill_proc_tree(process.pid,True)
                self.proc = []
                self.label_PlotGenerateStopMessages.configure(text="All"+'\n'+"running"+'\n'+"processes"+'\n'+"stopped.", fg='red')
            except:
                self.label_PlotGenerateStopMessages.configure(text="All"+'\n'+"running"+'\n'+"processes"+'\n'+"stopped.", fg='red')
        else:
            self.label_PlotGenerateStopMessages.configure(text="There is"+'\n'+"no running"+'\n'+"processes to"+'\n'+"be stopped!", fg='red')
    ## This function creates a new configFile.py in the given directory of fileName
    # if there is a file already exisiting it replaces it with a new one.
    # It writes all the displayed parameters on the GUI from the user
    # to the file.
    def generateConfigFile(self, fileName):
        with open(fileName, 'w') as config:
            config.write('import numpy as np' +'\n')
            config.write('import math' + '\n')
            year, month, day, hour, minute = time.localtime()[:5]
            config.write('Date_Time = ' + "'"+'%s.%s.%s_%s:%s' % (day, month, year, hour, minute) +"'"'\n')
            config.write('Author = ' + "'"+str(self.entry_TableParam_Author.get())+"'" + '\n')
            config.write('App = ' + "'" + 'Induced voltage' + "'" + '\n')
            config.write('exciterWindings = ' + str(self.entry_ExciterParam_NumberWindings.get()) + '\n')
            config.write('exciterCurrent = '+str(self.entry_ExciterParam_Current.get())+'\n')
            config.write('frequency = ' + str(self.entry_ExciterParam_Frequency.get()) + '\n')
            config.write('exciterPosition = ' + str(self.entry_ExciterParam_ExciterOrigin.get()) + '\n')
            config.write('registeredExciterCorners = ' + str(self.registeredExciterCorners) + '\n')
            for i in range(1, self.registeredExciterCorners + 1):
                config.write('e{0} = '.format(i) + self.exciterCornersEntriesDict[
                    "Corner {0} entry".format(i)].get() + '\n')


            config.write('coilWindings = ' + str(self.entry_ObjectParam_NumberWindings.get()) + '\n')
            config.write('coilResistance = ' + str(self.entry_ObjectParam_Resistance.get()) + '\n')
            if self.objectTypePolygon.get() == 1:
                config.write('objectType = \'Polygon\'' + '\n')
                config.write('coilLength = ' + str(self.entry_ObjectParamType_CoilDimensionsLength.get()) + '\n')
                config.write('coilWidth = ' + str(self.entry_ObjectParamType_CoilDimensionsWidth.get()) + '\n')
                config.write('coilOrientationAlpha = ' + str(self.entry_ObjectParamType_Alpha.get()) + '\n')
                config.write('coilOrientationBeta = ' + str(self.entry_ObjectParamType_Beta.get()) + '\n')
                config.write('coilOrientationGamma = ' + str(self.entry_ObjectParamType_Gamma.get()) + '\n')
                try:
                    if len(self.entry_ObjectParamType_X.get()) == 0 or len(self.entry_ObjectParamType_Z.get()) == 0 or len(self.entry_ObjectParamType_Y.get()) == 0:
                        config.write('coilPositionX = ' + "'" +  "'" + '\n')
                        config.write('coilPositionY = ' + "'" + "'" + '\n')
                        config.write('coilPositionZ = ' + "'" + "'" + '\n')
                    else:
                        config.write('coilPositionX = ' + str(self.entry_ObjectParamType_X.get()) + '\n')
                        config.write('coilPositionY = ' + str(self.entry_ObjectParamType_Y.get()) + '\n')
                        config.write('coilPositionZ = ' + str(self.entry_ObjectParamType_Z.get()) + '\n')

                except:
                    config.write('coilPositionX = ' + "'" +  "'" + '\n')
                    config.write('coilPositionY = ' + "'" +  "'" + '\n')
                    config.write('coilPositionZ = ' + "'" +  "'" + '\n')

            elif self.objectTypeEllipse.get() == 1:
                config.write('objectType = \'Ellipse\'' + '\n')
                config.write('majorAxisLength = ' + str(self.entry_ObjectParamType_MajorAxis.get()) + '\n')
                config.write('minorAxisLength =  ' + str(self.entry_ObjectParamType_MinorAxis.get()) + '\n')
                config.write('coilOrientationAlpha = ' + str(self.entry_ObjectParamType_Alpha.get()) + '\n')
                config.write('coilOrientationBeta = ' + str(self.entry_ObjectParamType_Beta.get()) + '\n')
                config.write('coilOrientationGamma = ' + str(self.entry_ObjectParamType_Gamma.get()) + '\n')
                try:
                    if len(self.entry_ObjectParamType_X.get()) == 0 or len(self.entry_ObjectParamType_Z.get()) == 0 or len(self.entry_ObjectParamType_Y.get()) == 0:
                        config.write('coilPositionX = ' + "'" +  "'" + '\n')
                        config.write('coilPositionY = ' + "'" + "'" + '\n')
                        config.write('coilPositionZ = ' + "'" + "'" + '\n')
                    else:
                        config.write('coilPositionX = ' + str(self.entry_ObjectParamType_X.get()) + '\n')
                        config.write('coilPositionY = ' + str(self.entry_ObjectParamType_Y.get()) + '\n')
                        config.write('coilPositionZ = ' + str(self.entry_ObjectParamType_Z.get()) + '\n')

                except:
                    config.write('coilPositionX = ' + "'" +  "'" + '\n')
                    config.write('coilPositionY = ' + "'" +  "'" + '\n')
                    config.write('coilPositionZ = ' + "'" +  "'" + '\n')

            elif self.objectTypePuk.get() == 1:
                config.write('objectType = \'Puk\'' + '\n')
                config.write('circularCoilRadius = ' + str(self.entry_ObjectParamType_CircularCoilRadius.get()) + '\n')
                config.write('pukHeight = ' + str(self.entry_ObjectParamType_RectangularCoilsDimensions.get()) + '\n')
                config.write('coilOrientationAlpha = ' + str(self.entry_ObjectParamType_Alpha.get()) + '\n')
                config.write('coilOrientationBeta = ' + str(self.entry_ObjectParamType_Beta.get()) + '\n')
                config.write('coilOrientationGamma = ' + str(self.entry_ObjectParamType_Gamma.get()) + '\n')
                try:
                    if len(self.entry_ObjectParamType_X.get()) == 0 or len(self.entry_ObjectParamType_Z.get()) == 0 or len(self.entry_ObjectParamType_Y.get()) == 0:
                        config.write('coilPositionX = ' + "'" +  "'" + '\n')
                        config.write('coilPositionY = ' + "'" + "'" + '\n')
                        config.write('coilPositionZ = ' + "'" + "'" + '\n')
                    else:
                        config.write('coilPositionX = ' + str(self.entry_ObjectParamType_X.get()) + '\n')
                        config.write('coilPositionY = ' + str(self.entry_ObjectParamType_Y.get()) + '\n')
                        config.write('coilPositionZ = ' + str(self.entry_ObjectParamType_Z.get()) + '\n')

                except:
                    config.write('coilPositionX = ' + "'" +  "'" + '\n')
                    config.write('coilPositionY = ' + "'" +  "'" + '\n')
                    config.write('coilPositionZ = ' + "'" +  "'" + '\n')
            elif self.objectTypeBall.get() == 1:
                config.write('objectType = \'Ball\'' + '\n')
                config.write('coilsRadius = ' + str(self.entry_ObjectParamType_Radius.get()) + '\n')
                config.write('coilOrientationAlpha = ' + str(self.entry_ObjectParamType_Alpha.get()) + '\n')
                config.write('coilOrientationBeta = ' + str(self.entry_ObjectParamType_Beta.get()) + '\n')
                config.write('coilOrientationGamma = ' + str(self.entry_ObjectParamType_Gamma.get()) + '\n')
                try:
                    if len(self.entry_ObjectParamType_X.get()) == 0 or len(self.entry_ObjectParamType_Z.get()) == 0 or len(self.entry_ObjectParamType_Y.get()) == 0:
                        config.write('coilPositionX = ' + "'" +  "'" + '\n')
                        config.write('coilPositionY = ' + "'" + "'" + '\n')
                        config.write('coilPositionZ = ' + "'" + "'" + '\n')
                    else:
                        config.write('coilPositionX = ' + str(self.entry_ObjectParamType_X.get()) + '\n')
                        config.write('coilPositionY = ' + str(self.entry_ObjectParamType_Y.get()) + '\n')
                        config.write('coilPositionZ = ' + str(self.entry_ObjectParamType_Z.get()) + '\n')

                except:
                    config.write('coilPositionX = ' + "'" +  "'" + '\n')
                    config.write('coilPositionY = ' + "'" +  "'" + '\n')
                    config.write('coilPositionZ = ' + "'" +  "'" + '\n')
            elif self.objectTypeWearable.get() == 1:
                config.write('objectType = \'Wearable\'' + '\n')
                config.write('coil1Width = ' + str(self.entry_ObjectParamType_coil1Width.get()) + '\n')
                config.write('coil2Width = ' + str(self.entry_ObjectParamType_coil2Width.get()) + '\n')
                config.write('coil3Width = ' + str(self.entry_ObjectParamType_coil3Width.get()) + '\n')
                config.write('coil1Length = ' + str(self.entry_ObjectParamType_coil1Length.get()) + '\n')
                config.write('coil2Length = ' + str(self.entry_ObjectParamType_coil2Length.get()) + '\n')
                config.write('coil3Length = ' + str(self.entry_ObjectParamType_coil3Length.get()) + '\n')

                config.write('coilOrientationAlpha = ' + str(self.entry_ObjectParamType_Alpha.get()) + '\n')
                config.write('coilOrientationBeta = ' + str(self.entry_ObjectParamType_Beta.get()) + '\n')
                config.write('coilOrientationGamma = ' + str(self.entry_ObjectParamType_Gamma.get()) + '\n')
                try:
                    if len(self.entry_ObjectParamType_X.get()) == 0 or len(self.entry_ObjectParamType_Z.get()) == 0 or len(self.entry_ObjectParamType_Y.get()) == 0:
                        config.write('coilPositionX = ' + "'" +  "'" + '\n')
                        config.write('coilPositionY = ' + "'" + "'" + '\n')
                        config.write('coilPositionZ = ' + "'" + "'" + '\n')
                    else:
                        config.write('coilPositionX = ' + str(self.entry_ObjectParamType_X.get()) + '\n')
                        config.write('coilPositionY = ' + str(self.entry_ObjectParamType_Y.get()) + '\n')
                        config.write('coilPositionZ = ' + str(self.entry_ObjectParamType_Z.get()) + '\n')

                except:
                    config.write('coilPositionX = ' + "'" +  "'" + '\n')
                    config.write('coilPositionY = ' + "'" +  "'" + '\n')
                    config.write('coilPositionZ = ' + "'" +  "'" + '\n')
            else:config.write('objectType = 0' + '\n')
            if self.antTypeMain.get() == 1:
                config.write('mainAntennaCheckBox = ' + str(self.antTypeMain.get()) + '\n')
                config.write('mainAntennaWindings = ' + str(self.entry_mainAntennasWindings.get()) + '\n')
                config.write('mainAntennaLength = ' + str(self.entry_mainAntennasLength.get()) + '\n')
                config.write('mainAntennaHeight = ' + str(self.entry_mainAntennasHeight.get()) + '\n')
                if self.registeredMainAntennas !=0:
                    config.write('registeredMainAntennas = ' + str(self.registeredMainAntennas) + '\n')
                    for i in range(1, self.registeredMainAntennas + 1):
                        config.write('mainAntennaPosition{0} = '.format(i) + self.mainParamLabelsEntriesDict[
                            "Ant {0} position entry".format(i)].get() + '\n')
                        config.write('mainAntennaOrientation{0} = '.format(i) + '[' + self.mainParamLabelsEntriesDict[
                            "Ant {0} alpha entry".format(i)].get() + ','+self.mainParamLabelsEntriesDict[
                            "Ant {0} beta entry".format(i)].get()+','+self.mainParamLabelsEntriesDict[
                            "Ant {0} gamma entry".format(i)].get()+']'+ '\n')
                else:
                    config.write('registeredMainAntennas = ' + str(self.registeredMainAntennas) + '\n')
            else:
                config.write('mainAntennaCheckBox = ' + str(self.antTypeMain.get()) + '\n')
            if self.antTypeFrame.get() == 1:
                config.write('frameAntennaCheckBox = ' + str(self.antTypeFrame.get()) + '\n')
                config.write('frameAntennaWindings = ' + str(self.entry_frameAntennasWindings.get()) + '\n')
                config.write('frameAntennaLength = ' + str(self.entry_frameAntennasLength.get()) + '\n')
                config.write('frameAntennaHeight = ' + str(self.entry_frameAntennasHeight.get()) + '\n')
                if self.registeredFrameAntennas != 0:
                    config.write('registeredFrameAntennas = ' + str(self.registeredFrameAntennas) + '\n')
                    for i in range(1, self.registeredFrameAntennas + 1):
                        config.write('frameAntennaPosition{0} = '.format(i) + self.frameParamLabelsEntriesDict["Ant {0} position entry".format(i)].get() + '\n')
                        config.write('frameAntennaOrientation{0} = '.format(i)  + '[' + self.frameParamLabelsEntriesDict[
                            "Ant {0} alpha entry".format(i)].get() + ','+self.frameParamLabelsEntriesDict[
                            "Ant {0} beta entry".format(i)].get()+','+self.frameParamLabelsEntriesDict[
                            "Ant {0} gamma entry".format(i)].get()+']'+ '\n')
                else:
                    config.write('registeredFrameAntennas = ' + str(self.registeredFrameAntennas) + '\n')
            else:
                config.write('frameAntennaCheckBox = ' + str(self.antTypeFrame.get()) + '\n')
            config.write('numberOfCores = ' + self.entry_TableParam_NumberOfCores.get() + '\n')
            if self.xpos_singleMultiplePoints.get() == 1:
                config.write('xPosOption = \'SingleMultiple\'' + '\n')
                config.write('xpos = ' + str(self.entry_xpos_singleMultiple.get()) + '\n')
            elif self.xpos_sweep.get() == 1:
                config.write('xPosOption = \'Sweep\'' + '\n')
                config.write('xposSweepStart = ' + str(self.entry_xpos_sweepStart.get()) + '\n')
                config.write('xposSweepEnd = ' + str(self.entry_xpos_sweepEnd.get()) + '\n')
                config.write('xposSweepStep = ' + str(self.entry_xpos_sweepStep.get()) + '\n')
            if self.ypos_singleMultiplePoints.get() == 1:
                config.write('yPosOption = \'SingleMultiple\'' + '\n')
                config.write('ypos = ' + str(self.entry_ypos_singleMultiple.get()) + '\n')
            elif self.ypos_sweep.get() == 1:
                config.write('yPosOption = \'Sweep\'' + '\n')
                config.write('yposSweepStart = ' + str(self.entry_ypos_sweepStart.get()) + '\n')
                config.write('yposSweepEnd = ' + str(self.entry_ypos_sweepEnd.get()) + '\n')
                config.write('yposSweepStep = ' + str(self.entry_ypos_sweepStep.get()) + '\n')
            if self.zpos_singleMultiplePoints.get() == 1:
                config.write('zPosOption = \'SingleMultiple\'' + '\n')
                config.write('zpos = ' + str(self.entry_zpos_singleMultiple.get()) + '\n')
            elif self.zpos_sweep.get() == 1:
                config.write('zPosOption = \'Sweep\'' + '\n')
                config.write('zposSweepStart = ' + str(self.entry_zpos_sweepStart.get()) + '\n')
                config.write('zposSweepEnd = ' + str(self.entry_zpos_sweepEnd.get()) + '\n')
                config.write('zposSweepStep = ' + str(self.entry_zpos_sweepStep.get()) + '\n')
            if self.alpha_singleMultipleAngles.get() == 1:
                config.write('alphaOption = \'SingleMultiple\'' + '\n')
                config.write('alpha = ' + str(self.entry_alpha_singleMultiple.get()) + '\n')
            elif self.alpha_sweep.get() == 1:
                config.write('alphaOption = \'Sweep\'' + '\n')
                config.write('alphaSweepStart = ' + str(self.entry_alpha_sweepStart.get()) + '\n')
                config.write('alphaSweepEnd = ' + str(self.entry_alpha_sweepEnd.get()) + '\n')
                config.write('alphaSweepStep = ' + str(self.entry_alpha_sweepStep.get()) + '\n')
            if self.beta_singleMultipleAngles.get() == 1:
                config.write('betaOption = \'SingleMultiple\'' + '\n')
                config.write('beta = ' + str(self.entry_beta_singleMultiple.get()) + '\n')
            elif self.beta_sweep.get() == 1:
                config.write('betaOption = \'Sweep\'' + '\n')
                config.write('betaSweepStart = ' + str(self.entry_beta_sweepStart.get()) + '\n')
                config.write('betaSweepEnd = ' + str(self.entry_beta_sweepEnd.get()) + '\n')
                config.write('betaSweepStep = ' + str(self.entry_beta_sweepStep.get()) + '\n')
            if self.gamma_singleMultipleAngles.get() == 1:
                config.write('gammaOption = \'SingleMultiple\'' + '\n')
                config.write('gamma = ' + str(self.entry_gamma_singleMultiple.get()) + '\n')
            elif self.gamma_sweep.get() == 1:
                config.write('gammaOption = \'Sweep\'' + '\n')
                config.write('gammaSweepStart = ' + str(self.entry_gamma_sweepStart.get()) + '\n')
                config.write('gammaSweepEnd = ' + str(self.entry_gamma_sweepEnd.get()) + '\n')
                config.write('gammaSweepStep = ' + str(self.entry_gamma_sweepStep.get()) + '\n')
            config.write('tableName = ' + "'"+str(self.entry_TableParam_TableName.get()) + "'" '\n')
    ## This function is trigered when the user clicks the 'Generate table' button
    # it excutes the python logic written in the file: start_fingerprinting_table.py
    def generateTable(self):
        try:
            self.generateConfigFile('GUI_configFile.py')
            self.label_PlotGenerateStopMessages.configure(text="Generating"+'\n'+"table.", fg='dark green')
            startFingerPrintingProcess = subprocess.Popen("python GUI_start_fingerprinting_table.py")
            self.proc.append(startFingerPrintingProcess)
        except:
            self.label_PlotGenerateStopMessages.configure(text="Error"+'\n'+"during"+'\n'+"generating"+'\n'+"table!", fg='red')
    ## This function plots the configured shapes from the configFile
    # into a seperate matplotlib figure, it can plot the exciter alone, exciter+ antnennas
    # exciter + object coil(s), exciter+ antennas+ object coil(s)
    def plotSetup(self):
        try:
            self.label_PlotGenerateStopMessages.configure(text="Plotting"+'\n'+"setup.", fg='blue')
            self.generateConfigFile('GUI_configFile.py')
            self.proc_plotSetup = subprocess.Popen("python GUI_plot_setup.py")
            self.proc.append(self.proc_plotSetup)
            self.label_PlotGenerateStopMessages.configure(text="Setup plotted.", fg='blue')
        except:
            self.label_PlotGenerateStopMessages.configure(text="Error"+'\n'+"during "+'\n'+"plotting"+'\n'+"setup!", fg='red')
    ## This function is responsible for calculating the total number of points
    # entered by the user. It estimates a maximum simulation time, a minimum
    # simulation time and an average simualtion time.
    # The calculated values are refereshed every 1 second and displayed.
    # The maximum estimated time is based on: a Puk object with the standard dimensions
    # poitioned at [0.05,0.05,0] m at a hockey goal, 10 main and  10 frame antennas, the object is then rotated
    # around the three axes.
    # The minimum estimated time is ased on: a Puk object with the standard dimensions
    # poitioned at the middle of a hockey goal m, 10 main and  10 frame antennas, the object is then rotated
    # around the three axes.
    # This function is not optimized for each application. More future work should be done to improve
    # it's performance.
    def estimateTimeRefresher(self):
        try:
            numberOfCores = int(self.entry_TableParam_NumberOfCores.get())
            if self.xpos_singleMultiplePoints.get() == 1:
                xpos = eval(self.entry_xpos_singleMultiple.get())
            elif self.xpos_sweep.get() == 1:
                xpos = np.arange(float(self.entry_xpos_sweepStart.get()), float(self.entry_xpos_sweepEnd.get()),float(self.entry_xpos_sweepStep.get()))


            if self.ypos_singleMultiplePoints.get() == 1:
                ypos = eval(self.entry_ypos_singleMultiple.get())
            elif self.ypos_sweep.get() == 1:
                ypos = np.arange(float(self.entry_ypos_sweepStart.get()), float(self.entry_ypos_sweepEnd.get()),float(self.entry_ypos_sweepStep.get()))

            if self.zpos_singleMultiplePoints.get() == 1:
                zpos = eval(self.entry_zpos_singleMultiple.get())
            elif self.zpos_sweep.get() == 1:
                zpos = np.arange(float(self.entry_zpos_sweepStart.get()),float(self.entry_zpos_sweepEnd.get() ),float(self.entry_zpos_sweepStep.get()))

            if self.alpha_singleMultipleAngles.get() == 1:
                xangles = eval(self.entry_alpha_singleMultiple.get())
            elif self.alpha_sweep.get() == 1:
                xangles = np.arange(float(self.entry_alpha_sweepStart.get()), float(self.entry_alpha_sweepEnd.get()) ,float(self.entry_alpha_sweepStep.get()))

            if self.beta_singleMultipleAngles.get() == 1:
                yangles = eval(self.entry_beta_singleMultiple.get())
            elif self.beta_sweep.get() == 1:
                yangles = np.arange(float(self.entry_beta_sweepStart.get()), float(self.entry_beta_sweepEnd.get() ),float(self.entry_beta_sweepStep.get()))

            if self.gamma_singleMultipleAngles.get() == 1:
                zangles = eval(self.entry_gamma_singleMultiple.get())
            elif self.gamma_sweep.get() == 1:
                zangles = np.arange(float(self.entry_gamma_sweepStart.get()), float(self.entry_gamma_sweepEnd.get() ),float(self.entry_gamma_sweepStep.get()))
            totalNumberOfPoints = len(xpos) * len(ypos) * len(zpos) * len(xangles) * len(yangles) * len(zangles)
            minTime = float(totalNumberOfPoints * 3.696 )/float(numberOfCores)
            maxTime = float(totalNumberOfPoints * 6.336 )/float(numberOfCores)
            avgTime = float(minTime +maxTime) / 2
            self.entry_totalPoints.delete(0, END)
            self.entry_maxTime.delete(0, END)
            self.entry_minTime.delete(0, END)
            self.entry_avgTime.delete(0, END)
            self.entry_totalPoints.configure(fg='black', width = int(10*self.scale_factor))
            self.entry_maxTime.configure(fg='black', width = int(10*self.scale_factor))
            self.entry_minTime.configure(fg='black', width = int(10*self.scale_factor))
            self.entry_avgTime.configure(fg='black', width = int(10*self.scale_factor))
            self.entry_totalPoints.insert(0, totalNumberOfPoints)
            self.entry_maxTime.insert(0, "%.2f" % (maxTime / 60))
            self.entry_minTime.insert(0, "%.2f" % (minTime / 60))
            self.entry_avgTime.insert(0, "%.2f" % (avgTime / 60))

            self.root.after(1000, self.estimateTimeRefresher)
        except:
            self.exceptionMessage()
            self.root.after(1000, self.estimateTimeRefresher)
    ## This function is trigerred when the user type an invalid syntax
    # in the fields of the table simualtion points.
    # Error messages will be displayed.
    def exceptionMessage(self):
        self.entry_totalPoints.delete(0, END)
        self.entry_maxTime.delete(0, END)
        self.entry_minTime.delete(0, END)
        self.entry_avgTime.delete(0, END)
        self.entry_totalPoints.configure(fg='red', width = int(40*self.scale_factor))
        self.entry_maxTime.configure(fg='red', width = int(40*self.scale_factor))
        self.entry_minTime.configure(fg='red', width = int(40*self.scale_factor))
        self.entry_avgTime.configure(fg='red', width = int(40*self.scale_factor))
        self.entry_totalPoints.insert(0, 'Invalid syntax! Please enter correct.')
        self.entry_maxTime.insert(0, 'Check the table parameters fields and make')
        self.entry_minTime.insert(0, 'sure no special charachters such as + - * ( / $ ')
        self.entry_avgTime.insert(0, 'or letters (a A) were typed in.  ')
    ## This function is trigerred when the user press the Save config. file button
    # it saves the current values of the displayed parameters on the GUI to a .py file.
    # The file will be saved at \configFiles directory. if the directory does not exist,
    # it creates a new directory. if the file with the same name exists, it creates a new
    # file name by adding a number to the end of the entered file name.
    # ex: configFile_(1).py, configFile_(2).py, configFile_(3).py...
    def saveConfigFile(self):

        path = '.\\configFiles'
        # Check, if directory exists and create new folder, if necessary:
        try:
            if os.path.isdir(path) == False:
                os.mkdir(path)
            filename_full = os.path.join(path, self.entry_configFilesSave_EnterConfigFileName.get()+ '.py')
        except IOError as exception:
            filename_full = self.entry_configFilesSave_EnterConfigFileName.get()
            print(exception, '\n--> ConfigFile is saved to source-directory')
        # Extend file name, if a file with the same name already exists:
        count = 0
        if (os.path.isfile(filename_full)):
            while (os.path.isfile(filename_full)):
                 count += 1
                 filename_full = os.path.join(path, self.entry_configFilesSave_EnterConfigFileName.get()+ '(%s)'% count+ '.py')
                 self.label_SaveConfigTextMessage.configure(text="File already exists. config. file renamed to: "+
                                                          self.entry_configFilesSave_EnterConfigFileName.get()+ '(%s)'% count+ '.py'+ ' and saved.', fg='blue')
        else:
            self.label_SaveConfigTextMessage.configure(text="File: " +
                                                            self.entry_configFilesSave_EnterConfigFileName.get() + '.py' + ' saved.',
                                                       fg='blue')
        self.generateConfigFile(filename_full)
    ## This function opens a browsing for a file window
    #  returns the chosen current file path from
    # the browsing window when the user press: Oeffnen.
    # returns 0 if the user pressed: Abbrechen or closed the browsing window.
    def getFilePath(self):
        toplevel = Tk()
        toplevel.withdraw()
        filePath = fileDialog.askopenfilename()
        if os.path.isfile(filePath):
            return filePath
        else:
            return 0

    def getDirectory(self):
        toplevel = Tk()
        toplevel.withdraw()
        filePath = fileDialog.askdirectory()
        if os.path.isdir(filePath):
            return filePath
        else:
            return 0

    def browseForSavingNewTable(self):
        directory = self.getDirectory()
        if directory != 0:
            self.entry_newTablePath.delete(0, END)
            self.entry_newTablePath.configure(fg='black')
            self.entry_newTablePath.insert(0, directory)
        else:
            self.entry_newTablePath.delete(0, END)
            self.entry_newTablePath.configure(fg='red')
            self.entry_newTablePath.insert(0, 'Directory was not chosen')

    ## This function is trigerred when the user press Browse button.
    # It displays the chosen file bath if any. it displays an error message if
    # the returned value from getConfigFilePath is 0.
    def browseForConfigFile(self):
        filePath = self.getFilePath()
        if filePath != 0:
            self.entry_configFilesLoad_BrowseConfigFilePath.delete(0, END)
            self.entry_configFilesLoad_BrowseConfigFilePath.configure(fg='black')
            self.entry_configFilesLoad_BrowseConfigFilePath.insert(0, filePath)
        else:
            self.entry_configFilesLoad_BrowseConfigFilePath.delete(0, END)
            self.entry_configFilesLoad_BrowseConfigFilePath.configure(fg='red')
            self.entry_configFilesLoad_BrowseConfigFilePath.insert(0, 'File was not chosen or it does not exist.')
    ## This function is trigerred when the user press: Load button.
    # It tries to read the parameters values saved in the given configFile path.
    # It sets all the read parameters entries and labels and displays them on he GUI.
    # It shows an error message if it could not read the confingFile.
    def loadConfigFile(self):
        filePath = self.entry_configFilesLoad_BrowseConfigFilePath.get()
        try:
            if os.path.isfile(filePath):
                #load the parameters values
                print('Loading parameters values')
                file = open(filePath, 'r')
                contents = file.readlines()
                paramDict = {}
                for elment in contents:

                    if '=' in elment:
                        key = ''

                        for character in elment:
                            if character != '=':
                                key += character
                            elif character == '=':
                                paramDict[key[0:-1]] = eval(elment[elment.index('=') + 1:-1])
                                break
                self.entry_ExciterParam_NumberWindings.delete(0, END)
                self.entry_ExciterParam_NumberWindings.insert(0,str(paramDict['exciterWindings']))
                self.entry_ExciterParam_Current.delete(0, END)
                self.entry_ExciterParam_Current.insert(0,str(paramDict['exciterCurrent']))
                self.entry_ExciterParam_Frequency.delete(0, END)
                self.entry_ExciterParam_Frequency.insert(0, str(paramDict['frequency']))
                self.entry_ExciterParam_ExciterOrigin.delete(0, END)
                self.entry_ExciterParam_ExciterOrigin.insert(0, str(paramDict['exciterPosition']))
                self.numberOfExciterCornersBuffer = 0
                self.numberOfExciterCorners.set(paramDict['registeredExciterCorners'])
                self.addExciterCorners()
                for i in range(1, self.registeredExciterCorners + 1):
                    self.exciterCornersEntriesDict["Corner {0} entry".format(i)].delete(0, END)
                    self.exciterCornersEntriesDict["Corner {0} entry".format(i)].insert(0, str(paramDict['e{0}'.format(i)]))
                if paramDict['mainAntennaCheckBox'] == 0:
                    self.antTypeMain.set(0)
                    self.displayAntTypeMain()
                elif paramDict['mainAntennaCheckBox'] == 1:
                    self.antTypeMain.set(1)
                    self.displayAntTypeMain()
                    self.entry_mainAntennasWindings.delete(0,END)
                    self.entry_mainAntennasWindings.insert(0, str(paramDict['mainAntennaWindings']))
                    self.entry_mainAntennasLength.delete(0, END)
                    self.entry_mainAntennasLength.insert(0, str(paramDict['mainAntennaLength']))
                    self.entry_mainAntennasHeight.delete(0, END)
                    self.entry_mainAntennasHeight.insert(0, str(paramDict['mainAntennaHeight']))
                    if paramDict['registeredMainAntennas'] != 0:
                        self.numberOfMainAntennas.set(paramDict['registeredMainAntennas'])
                        self.addMainAnt()
                        for i in range(1, self.numberOfMainAntennas.get() + 1):
                            self.mainParamLabelsEntriesDict["Ant {0} position entry".format(i)].delete(0, END)
                            self.mainParamLabelsEntriesDict["Ant {0} position entry".format(i)].insert(0, str(paramDict['mainAntennaPosition{0}'.format(i)]))
                            self.mainParamLabelsEntriesDict["Ant {0} alpha entry".format(i)].delete(0, END)
                            self.mainParamLabelsEntriesDict["Ant {0} alpha entry".format(i)].insert(0, str(paramDict['mainAntennaOrientation{0}'.format(i)][0]))
                            self.mainParamLabelsEntriesDict["Ant {0} beta entry".format(i)].delete(0, END)
                            self.mainParamLabelsEntriesDict["Ant {0} beta entry".format(i)].insert(0, str(paramDict[
                                'mainAntennaOrientation{0}'.format(i)][1]))
                            self.mainParamLabelsEntriesDict["Ant {0} gamma entry".format(i)].delete(0, END)
                            self.mainParamLabelsEntriesDict["Ant {0} gamma entry".format(i)].insert(0, str(paramDict[
                                'mainAntennaOrientation{0}'.format(i)][2]))
                    elif paramDict['registeredMainAntennas'] == 0:
                        self.displayAntTypeMain()
                        self.entry_mainAntennasWindings.delete(0, END)
                        self.entry_mainAntennasWindings.insert(0, str(paramDict['mainAntennaWindings']))
                        self.entry_mainAntennasLength.delete(0, END)
                        self.entry_mainAntennasLength.insert(0, str(paramDict['mainAntennaLength']))
                        self.entry_mainAntennasHeight.delete(0, END)
                        self.entry_mainAntennasHeight.insert(0, str(paramDict['mainAntennaHeight']))

                if paramDict['frameAntennaCheckBox'] == 0:
                    self.antTypeFrame.set(0)
                    self.displayAntTypeFrame()
                elif paramDict['frameAntennaCheckBox'] == 1:
                    self.antTypeFrame.set(1)
                    self.displayAntTypeFrame()
                    self.entry_frameAntennasWindings.delete(0,END)
                    self.entry_frameAntennasWindings.insert(0, str(paramDict['frameAntennaWindings']))
                    self.entry_frameAntennasLength.delete(0, END)
                    self.entry_frameAntennasLength.insert(0, str(paramDict['frameAntennaLength']))
                    self.entry_frameAntennasHeight.delete(0, END)
                    self.entry_frameAntennasHeight.insert(0, str(paramDict['frameAntennaHeight']))
                    if paramDict['registeredFrameAntennas'] != 0:
                        self.numberOfFrameAntennas.set(paramDict['registeredFrameAntennas'])
                        self.addFrameAnt()
                        for i in range(1, self.numberOfFrameAntennas.get() + 1):
                            self.frameParamLabelsEntriesDict["Ant {0} position entry".format(i)].delete(0, END)
                            self.frameParamLabelsEntriesDict["Ant {0} position entry".format(i)].insert(0, str(paramDict['frameAntennaPosition{0}'.format(i)]))
                            self.frameParamLabelsEntriesDict["Ant {0} alpha entry".format(i)].delete(0, END)
                            self.frameParamLabelsEntriesDict["Ant {0} alpha entry".format(i)].insert(0, str(paramDict['frameAntennaOrientation{0}'.format(i)][0]))
                            self.frameParamLabelsEntriesDict["Ant {0} beta entry".format(i)].delete(0, END)
                            self.frameParamLabelsEntriesDict["Ant {0} beta entry".format(i)].insert(0, str(paramDict[
                                'frameAntennaOrientation{0}'.format(i)][1]))
                            self.frameParamLabelsEntriesDict["Ant {0} gamma entry".format(i)].delete(0, END)
                            self.frameParamLabelsEntriesDict["Ant {0} gamma entry".format(i)].insert(0, str(paramDict[
                                'frameAntennaOrientation{0}'.format(i)][2]))
                    elif paramDict['registeredFrameAntennas'] == 0:
                        self.displayAntTypeFrame()
                        self.entry_frameAntennasWindings.delete(0, END)
                        self.entry_frameAntennasWindings.insert(0, str(paramDict['frameAntennaWindings']))
                        self.entry_frameAntennasLength.delete(0, END)
                        self.entry_frameAntennasLength.insert(0, str(paramDict['frameAntennaLength']))
                        self.entry_frameAntennasHeight.delete(0, END)
                        self.entry_frameAntennasHeight.insert(0, str(paramDict['frameAntennaHeight']))

                self.entry_ObjectParam_NumberWindings.delete(0, END)
                self.entry_ObjectParam_NumberWindings.insert(0, str(paramDict['coilWindings']))
                self.entry_ObjectParam_Resistance.delete(0, END)
                self.entry_ObjectParam_Resistance.insert(0, str(paramDict['coilResistance']))
                if paramDict['objectType'] == 'Polygon':
                    self.objectTypePolygon.set(1)
                    self.displayObjectTypePolygon()
                    self.entry_ObjectParamType_CoilDimensionsWidth.delete(0, END)
                    self.entry_ObjectParamType_CoilDimensionsWidth.insert(0, str(paramDict['coilWidth']))
                    self.entry_ObjectParamType_CoilDimensionsLength.delete(0, END)
                    self.entry_ObjectParamType_CoilDimensionsLength.insert(0, str(paramDict['coilLength']))
                    self.entry_ObjectParamType_Alpha.delete(0, END)
                    self.entry_ObjectParamType_Alpha.insert(0, str(paramDict['coilOrientationAlpha']))
                    self.entry_ObjectParamType_Beta.delete(0, END)
                    self.entry_ObjectParamType_Beta.insert(0, str(paramDict['coilOrientationBeta']))
                    self.entry_ObjectParamType_Gamma.delete(0, END)
                    self.entry_ObjectParamType_Gamma.insert(0, str(paramDict['coilOrientationGamma']))
                    self.entry_ObjectParamType_X.delete(0, END)
                    self.entry_ObjectParamType_X.insert(0, str(paramDict['coilPositionX']))
                    self.entry_ObjectParamType_Y.delete(0, END)
                    self.entry_ObjectParamType_Y.insert(0, str(paramDict['coilPositionY']))
                    self.entry_ObjectParamType_Z.delete(0, END)
                    self.entry_ObjectParamType_Z.insert(0, str(paramDict['coilPositionZ']))
                elif paramDict['objectType'] == 'Ellipse':
                    self.objectTypeEllipse.set(1)
                    self.displayObjectTypeEllipse()
                    self.entry_ObjectParamType_MajorAxis.delete(0, END)
                    self.entry_ObjectParamType_MajorAxis.insert(0, str(paramDict['majorAxisLength']))
                    self.entry_ObjectParamType_MinorAxis.delete(0, END)
                    self.entry_ObjectParamType_MinorAxis.insert(0, str(paramDict['minorAxisLength']))
                    self.entry_ObjectParamType_Alpha.delete(0, END)
                    self.entry_ObjectParamType_Alpha.insert(0, str(paramDict['coilOrientationAlpha']))
                    self.entry_ObjectParamType_Beta.delete(0, END)
                    self.entry_ObjectParamType_Beta.insert(0, str(paramDict['coilOrientationBeta']))
                    self.entry_ObjectParamType_Gamma.delete(0, END)
                    self.entry_ObjectParamType_Gamma.insert(0, str(paramDict['coilOrientationGamma']))
                    self.entry_ObjectParamType_X.delete(0, END)
                    self.entry_ObjectParamType_X.insert(0, str(paramDict['coilPositionX']))
                    self.entry_ObjectParamType_Y.delete(0, END)
                    self.entry_ObjectParamType_Y.insert(0, str(paramDict['coilPositionY']))
                    self.entry_ObjectParamType_Z.delete(0, END)
                    self.entry_ObjectParamType_Z.insert(0, str(paramDict['coilPositionZ']))
                elif paramDict['objectType'] == 'Puk':
                    self.objectTypePuk.set(1)
                    self.displayObjectTypePuk()
                    self.entry_ObjectParamType_RectangularCoilsDimensions.delete(0, END)
                    self.entry_ObjectParamType_RectangularCoilsDimensions.insert(0,  str(paramDict['pukHeight']))
                    self.entry_ObjectParamType_CircularCoilRadius.delete(0, END)
                    self.entry_ObjectParamType_CircularCoilRadius.insert(0,  str(paramDict['circularCoilRadius']))
                    self.entry_ObjectParamType_Alpha.delete(0, END)
                    self.entry_ObjectParamType_Alpha.insert(0, str(paramDict['coilOrientationAlpha']))
                    self.entry_ObjectParamType_Beta.delete(0, END)
                    self.entry_ObjectParamType_Beta.insert(0, str(paramDict['coilOrientationBeta']))
                    self.entry_ObjectParamType_Gamma.delete(0, END)
                    self.entry_ObjectParamType_Gamma.insert(0, str(paramDict['coilOrientationGamma']))
                    self.entry_ObjectParamType_X.delete(0, END)
                    self.entry_ObjectParamType_X.insert(0, str(paramDict['coilPositionX']))
                    self.entry_ObjectParamType_Y.delete(0, END)
                    self.entry_ObjectParamType_Y.insert(0, str(paramDict['coilPositionY']))
                    self.entry_ObjectParamType_Z.delete(0, END)
                    self.entry_ObjectParamType_Z.insert(0, str(paramDict['coilPositionZ']))
                elif paramDict['objectType'] == 'Ball':
                    self.objectTypeBall.set(1)
                    self.displayObjectTypeBall()
                    self.entry_ObjectParamType_Radius.delete(0, END)
                    self.entry_ObjectParamType_Radius.insert(0,  str(paramDict['coilsRadius']))
                    self.entry_ObjectParamType_Alpha.delete(0, END)
                    self.entry_ObjectParamType_Alpha.insert(0, str(paramDict['coilOrientationAlpha']))
                    self.entry_ObjectParamType_Beta.delete(0, END)
                    self.entry_ObjectParamType_Beta.insert(0, str(paramDict['coilOrientationBeta']))
                    self.entry_ObjectParamType_Gamma.delete(0, END)
                    self.entry_ObjectParamType_Gamma.insert(0, str(paramDict['coilOrientationGamma']))
                    self.entry_ObjectParamType_X.delete(0, END)
                    self.entry_ObjectParamType_X.insert(0, str(paramDict['coilPositionX']))
                    self.entry_ObjectParamType_Y.delete(0, END)
                    self.entry_ObjectParamType_Y.insert(0, str(paramDict['coilPositionY']))
                    self.entry_ObjectParamType_Z.delete(0, END)
                    self.entry_ObjectParamType_Z.insert(0, str(paramDict['coilPositionZ']))
                elif paramDict['objectType'] == 'Wearable':
                    self.objectTypeWearable.set(1)
                    self.displayObjectTypeWearable()
                    self.entry_ObjectParamType_coil1Width.delete(0, END)
                    self.entry_ObjectParamType_coil2Width.delete(0, END)
                    self.entry_ObjectParamType_coil3Width.delete(0, END)
                    self.entry_ObjectParamType_coil1Length.delete(0, END)
                    self.entry_ObjectParamType_coil2Length.delete(0, END)
                    self.entry_ObjectParamType_coil3Length.delete(0, END)
                    self.entry_ObjectParamType_coil1Width.insert(0,  str(paramDict['coil1Width']))
                    self.entry_ObjectParamType_coil2Width.insert(0,  str(paramDict['coil2Width']))
                    self.entry_ObjectParamType_coil3Width.insert(0,  str(paramDict['coil3Width']))
                    self.entry_ObjectParamType_coil1Length.insert(0,  str(paramDict['coil1Length']))
                    self.entry_ObjectParamType_coil2Length.insert(0,  str(paramDict['coil2Length']))
                    self.entry_ObjectParamType_coil3Length.insert(0,  str(paramDict['coil3Length']))
                    self.entry_ObjectParamType_Alpha.delete(0, END)
                    self.entry_ObjectParamType_Alpha.insert(0, str(paramDict['coilOrientationAlpha']))
                    self.entry_ObjectParamType_Beta.delete(0, END)
                    self.entry_ObjectParamType_Beta.insert(0, str(paramDict['coilOrientationBeta']))
                    self.entry_ObjectParamType_Gamma.delete(0, END)
                    self.entry_ObjectParamType_Gamma.insert(0, str(paramDict['coilOrientationGamma']))
                    self.entry_ObjectParamType_X.delete(0, END)
                    self.entry_ObjectParamType_X.insert(0, str(paramDict['coilPositionX']))
                    self.entry_ObjectParamType_Y.delete(0, END)
                    self.entry_ObjectParamType_Y.insert(0, str(paramDict['coilPositionY']))
                    self.entry_ObjectParamType_Z.delete(0, END)
                    self.entry_ObjectParamType_Z.insert(0, str(paramDict['coilPositionZ']))
                elif paramDict['objectType'] == 0:
                    self.objectTypePolygon.set(0)
                    self.objectTypeEllipse.set(0)
                    self.objectTypePuk.set(0)
                    self.objectTypeBall.set(0)
                    if self.ObjectParamTypeFrame_register != 0:
                        self.objectParamFrameDestroy()
                self.entry_TableParam_NumberOfCores.delete(0, END)
                self.entry_TableParam_NumberOfCores.insert(0, str(paramDict['numberOfCores']))
                if paramDict['xPosOption'] == 'SingleMultiple':
                    self.xpos_singleMultiplePoints.set(1)
                    self.displayXPosSingleMultiplePoints()
                    self.entry_xpos_singleMultiple.delete(0, END)
                    self.entry_xpos_singleMultiple.insert(0, str(paramDict['xpos']))
                elif paramDict['xPosOption'] == 'Sweep':
                    self.xpos_sweep.set(1)
                    self.displayXPosSweep()
                    self.entry_xpos_sweepStart.delete(0, END)
                    self.entry_xpos_sweepStart.insert(0, str(paramDict['xposSweepStart']))
                    self.entry_xpos_sweepEnd.delete(0, END)
                    self.entry_xpos_sweepEnd.insert(0, str(paramDict['xposSweepEnd']))
                    self.entry_xpos_sweepStep.delete(0, END)
                    self.entry_xpos_sweepStep.insert(0, str(paramDict['xposSweepStep']))
                if paramDict['yPosOption'] == 'SingleMultiple':
                    self.ypos_singleMultiplePoints.set(1)
                    self.displayYPosSingleMultiplePoints()
                    self.entry_ypos_singleMultiple.delete(0, END)
                    self.entry_ypos_singleMultiple.insert(0, str(paramDict['ypos']))
                elif paramDict['yPosOption'] == 'Sweep':
                    self.ypos_sweep.set(1)
                    self.displayYPosSweep()
                    self.entry_ypos_sweepStart.delete(0, END)
                    self.entry_ypos_sweepStart.insert(0, str(paramDict['yposSweepStart']))
                    self.entry_ypos_sweepEnd.delete(0, END)
                    self.entry_ypos_sweepEnd.insert(0, str(paramDict['yposSweepEnd']))
                    self.entry_ypos_sweepStep.delete(0, END)
                    self.entry_ypos_sweepStep.insert(0, str(paramDict['yposSweepStep']))
                if paramDict['zPosOption'] == 'SingleMultiple':
                    self.zpos_singleMultiplePoints.set(1)
                    self.displayZPosSingleMultiplePoints()
                    self.entry_zpos_singleMultiple.delete(0, END)
                    self.entry_zpos_singleMultiple.insert(0, str(paramDict['zpos']))
                elif paramDict['zPosOption'] == 'Sweep':
                    self.zpos_sweep.set(1)
                    self.displayZPosSweep()
                    self.entry_zpos_sweepStart.delete(0, END)
                    self.entry_zpos_sweepStart.insert(0, str(paramDict['zposSweepStart']))
                    self.entry_zpos_sweepEnd.delete(0, END)
                    self.entry_zpos_sweepEnd.insert(0, str(paramDict['zposSweepEnd']))
                    self.entry_zpos_sweepStep.delete(0, END)
                    self.entry_zpos_sweepStep.insert(0, str(paramDict['zposSweepStep']))
                if paramDict['alphaOption'] == 'SingleMultiple':
                    self.alpha_singleMultipleAngles.set(1)
                    self.displayAlphaSingleMultipleAngles()
                    self.entry_alpha_singleMultiple.delete(0, END)
                    self.entry_alpha_singleMultiple.insert(0, str(paramDict['alpha']))
                elif paramDict['alphaOption'] == 'Sweep':
                    self.alpha_sweep.set(1)
                    self.displayAlphaSweep()
                    self.entry_alpha_sweepStart.delete(0, END)
                    self.entry_alpha_sweepStart.insert(0, str(paramDict['alphaSweepStart']))
                    self.entry_alpha_sweepEnd.delete(0, END)
                    self.entry_alpha_sweepEnd.insert(0, str(paramDict['alphaSweepEnd']))
                    self.entry_alpha_sweepStep.delete(0, END)
                    self.entry_alpha_sweepStep.insert(0, str(paramDict['alphaSweepStep']))
                if paramDict['betaOption'] == 'SingleMultiple':
                    self.beta_singleMultipleAngles.set(1)
                    self.displayBetaSingleMultipleAngles()
                    self.entry_beta_singleMultiple.delete(0, END)
                    self.entry_beta_singleMultiple.insert(0, str(paramDict['beta']))
                elif paramDict['betaOption'] == 'Sweep':
                    self.beta_sweep.set(1)
                    self.displayBetaSweep()
                    self.entry_beta_sweepStart.delete(0, END)
                    self.entry_beta_sweepStart.insert(0, str(paramDict['betaSweepStart']))
                    self.entry_beta_sweepEnd.delete(0, END)
                    self.entry_beta_sweepEnd.insert(0, str(paramDict['betaSweepEnd']))
                    self.entry_beta_sweepStep.delete(0, END)
                    self.entry_beta_sweepStep.insert(0, str(paramDict['betaSweepStep']))
                if paramDict['gammaOption'] == 'SingleMultiple':
                    self.gamma_singleMultipleAngles.set(1)
                    self.displayGammaSingleMultipleAngles()
                    self.entry_gamma_singleMultiple.delete(0, END)
                    self.entry_gamma_singleMultiple.insert(0, str(paramDict['gamma']))
                elif paramDict['gammaOption'] == 'Sweep':
                    self.gamma_sweep.set(1)
                    self.displayGammaSweep()
                    self.entry_gamma_sweepStart.delete(0, END)
                    self.entry_gamma_sweepStart.insert(0, str(paramDict['gammaSweepStart']))
                    self.entry_gamma_sweepEnd.delete(0, END)
                    self.entry_gamma_sweepEnd.insert(0, str(paramDict['gammaSweepEnd']))
                    self.entry_gamma_sweepStep.delete(0, END)
                    self.entry_gamma_sweepStep.insert(0, str(paramDict['gammaSweepStep']))
                self.entry_TableParam_TableName.delete(0, END)
                self.entry_TableParam_TableName.insert(0, str(paramDict['tableName']))
                self.entry_TableParam_Author.delete(0, END)
                self.entry_TableParam_Author.insert(0, str(paramDict['Author']))
            else:
                self.entry_configFilesLoad_BrowseConfigFilePath.delete(0, END)
                self.entry_configFilesLoad_BrowseConfigFilePath.configure(fg='red')
                self.entry_configFilesLoad_BrowseConfigFilePath.insert(0, 'Chosen configuration file does not exist.')
        except:
            self.entry_configFilesLoad_BrowseConfigFilePath.delete(0, END)
            self.entry_configFilesLoad_BrowseConfigFilePath.configure(fg='red')
            self.entry_configFilesLoad_BrowseConfigFilePath.insert(0, 'Error while loading parameters values.')
    ## This function is trigerred when the user press Browse button for the table.
    # It displays the chosen file path if any. it displays an error message if
    # the returned value from getConfigFilePath is 0.
    def browseForTable(self):
        filePath = self.getFilePath()
        if filePath != 0:
            self.entry_simulatedTablePath.delete(0, END)
            self.entry_simulatedTablePath.configure(fg='black')
            self.entry_simulatedTablePath.insert(0, filePath)
        else:
            self.entry_simulatedTablePath.delete(0, END)
            self.entry_simulatedTablePath.configure(fg='red')
            self.entry_simulatedTablePath.insert(0, 'Table was not chosen or it does not exist.')
    def analyzeTable(self, tablePath):
        valuesDict = {}
        valuesDict['X-position'] = []
        valuesDict['Y-position'] = []
        valuesDict['Z-position'] = []
        valuesDict['X-angle'] = []
        valuesDict['Y-angle'] = []
        valuesDict['Z-angle'] = []
        valuesDict['coil X'] = []
        valuesDict['coil Y'] = []
        valuesDict['coil Z'] = []
        for i in range(1,11):
            valuesDict['frame{0}'.format(i)] = []
            valuesDict['main{0}'.format(i)] = []
        startingRow = float('Inf')
        with open(tablePath) as table:
            reader = csv.reader(table, delimiter=';')
            for i, row in enumerate(reader):
                if i > startingRow:
                    for ii, value in enumerate(row):
                        valuesDict[dictKeys[ii]].append(float(value))
                if i == 1:
                    self.tableDateCreated = row[0][4:]
                if i == 2:
                    self.tableAuthor = row[0][4:]
                if i == 3:
                    self.tableObjectType = row[0][2:]
                if row[0] == '# orientationsWise':
                    self.tableArrangment = 'orientationsWise'
                if row[0] == '# positionsWise':
                    self.tableArrangment = 'positionsWise'
                if row[0] == '# X-position':
                    startingRow = i
                    dictKeys = row
                    dictKeys[0] = 'X-position'
                    dictKeys = dictKeys[0:-1]
        uniqueValuesDict = {}
        for key in dictKeys:
            uniqueKey = key+'Unique'
            tempList = []
            tempList.append(valuesDict[key][0])
            for value in valuesDict[key]:
                if value not in tempList:
                    tempList.append(value)
            uniqueValuesDict[uniqueKey] = tempList
        self.dictKeys = dictKeys
        self.valuesDict = valuesDict
        self.uniqueValuesDict = uniqueValuesDict
    def displayTableInfo(self):
        label_thisTableContains = Label(self.loadSimulatedTableFrame, text="Table info: ",font=("Arial", self.paramFontSize))
        label_thisTableContains.grid(row=0, column=3)
        labelsEntriesDict = {}
        frameAntennaCounter = 0
        mainAntennaCounter = 0
        tempKeys = []
        for key in self.dictKeys:
            if key[0] != 'c' and key[0] != 'f' and key[0] != 'm':
                tempKeys.append(key)
                labelsEntriesDict[key+'Label']=Label(self.loadSimulatedTableFrame, text=str(key)+'(s): ',font=("Arial", self.paramFontSize-1))
                labelsEntriesDict[key +'Entry'] = Entry(self.loadSimulatedTableFrame, width=int(5*self.scale_factor))
                labelsEntriesDict[key + 'Entry'].insert(0, str(len(self.uniqueValuesDict[key+'Unique'])))
                labelsEntriesDict[key + 'Entry'].configure(state='readonly')
            elif key[0] == 'c':
                tempKeys.append(key)
                labelsEntriesDict[key+'Label'] = Label(self.loadSimulatedTableFrame,
                                        text=str(key)+': ',
                                        font=("Arial", self.paramFontSize-1))
                labelsEntriesDict[key+'Entry'] = Entry(self.loadSimulatedTableFrame,width=int(5*self.scale_factor))
            elif key[0] == 'f':

                frameAntennaCounter +=1

            elif key[0] == 'm':

                mainAntennaCounter += 1

        if self.tableObjectType == 'Puk' or self.tableObjectType == 'Ball':
            labelsEntriesDict['coil XEntry'].insert(0, str(1))
            labelsEntriesDict['coil XEntry'].configure(state='readonly')

            labelsEntriesDict['coil YEntry'].insert(0, str(1))
            labelsEntriesDict['coil YEntry'].configure(state='readonly')

            labelsEntriesDict['coil ZEntry'].insert(0, str(1))
            labelsEntriesDict['coil ZEntry'].configure(state='readonly')
        elif self.tableObjectType == 'Ellipse' or self.tableObjectType == 'Polygon':
            labelsEntriesDict['coil XEntry'].insert(0, str(0))
            labelsEntriesDict['coil XEntry'].configure(state='readonly')

            labelsEntriesDict['coil YEntry'].insert(0, str(0))
            labelsEntriesDict['coil YEntry'].configure(state='readonly')

            labelsEntriesDict['coil ZEntry'].insert(0, str(1))
            labelsEntriesDict['coil ZEntry'].configure(state='readonly')

        labelsEntriesDict["mainLabel"] = Label(self.loadSimulatedTableFrame,
                                       text="Main antenna(s): ",
                                       font=("Arial", self.paramFontSize - 1))
        labelsEntriesDict["mainEntry"] = Entry(self.loadSimulatedTableFrame, width = int(5 * self.scale_factor))
        labelsEntriesDict["mainEntry"].insert(0,str(mainAntennaCounter))
        labelsEntriesDict["mainEntry"].configure(state='readonly')
        labelsEntriesDict["frameLabel"] = Label(self.loadSimulatedTableFrame,
                                       text="Frame antenna(s): ",
                                       font=("Arial", self.paramFontSize - 1))
        labelsEntriesDict["frameEntry"] = Entry(self.loadSimulatedTableFrame, width=int(5*self.scale_factor))
        labelsEntriesDict["frameEntry"].insert(0,str(frameAntennaCounter))
        labelsEntriesDict["frameEntry"].configure(state='readonly')
        tempKeys.append('main')
        tempKeys.append('frame')
        labelRow = 0
        entryRow = 0
        labelColumn = 4
        entryColumn = 5
        internalI = 0
        for i,key in enumerate(tempKeys):
            if i!=0 and i%3==0:
                internalI =0
                labelRow+=1
                entryRow+=1
            labelsEntriesDict[key+'Label'].grid(row=labelRow, column=labelColumn+internalI,sticky='W')
            labelsEntriesDict[key + 'Entry'].grid(row=entryRow, column=entryColumn+internalI,sticky='W')
            internalI +=2

        label_object = Label(self.loadSimulatedTableFrame, text="Object: ",
                                       font=("Arial", self.paramFontSize))
        label_object.grid(row=3, column=8, sticky='W')
        entry_object = Entry(self.loadSimulatedTableFrame, width=int(8*self.scale_factor))
        entry_object.insert(0, str(self.tableObjectType))
        entry_object.configure(state='readonly')
        entry_object.grid(row=3, column=9, sticky='W')

        label_tableDateCreated = Label(self.loadSimulatedTableFrame, text="Date created: ",font=("Arial", self.paramFontSize))
        label_tableDateCreated.grid(row=4, column=4, sticky='W')
        entry_tableDateCreated = Entry(self.loadSimulatedTableFrame, width=int(8*self.scale_factor))
        entry_tableDateCreated.insert(0,str(self.tableDateCreated))
        entry_tableDateCreated.configure(state='readonly')
        entry_tableDateCreated.grid(row=4, column=5, sticky='W')
        label_tableAuthor = Label(self.loadSimulatedTableFrame, text="Author: ",
                                       font=("Arial", self.paramFontSize))
        label_tableAuthor.grid(row=4, column=6, sticky='W')
        entry_tableAuthor = Entry(self.loadSimulatedTableFrame, width=int(8 * self.scale_factor))
        entry_tableAuthor.insert(0,str(self.tableAuthor))
        entry_tableAuthor.configure(state='readonly')
        entry_tableAuthor.grid(row=4, column=7, sticky='W')
    def displayConfigure2DPlot(self):
        if self.configure2DPlotFrame != None:
            self.configure2DPlotFrame.destroy()
        self.configure2DPlotFrame = Frame(self.plotTab, relief='groove', borderwidth=3)
        self.configure2DPlotFrame.place(x=int(10 * self.scale_factor), y=int(120 * self.scale_factor))
        label_configurePlot = Label(self.configure2DPlotFrame, text="Configure a 2D plot: ",
                                         font=("Arial", int(self.paramFontSize)))
        label_configurePlot.grid(row=0, column=0,sticky='W')
        label_onTheYAxis = Label(self.configure2DPlotFrame, text="On the y-axis: induced voltage in ",
                                         font=("Arial", int(self.paramFontSize)))
        self.specificLabelsDict = {}
        self.specificCombosDict = {}
        inducedVoltagesList = self.dictKeys[6:len(self.dictKeys)+1]
        self.comboBox_onTheYAxis = Combobox(self.configure2DPlotFrame,
                                           values=inducedVoltagesList,width=int(10*self.scale_factor),state='readonly', height=6)

        label_onTheYAxis.grid(row=1, column=0,sticky='W')
        self.comboBox_onTheYAxis.grid(row=1, column=1,sticky='W')
        self.comboBox_onTheYAxis.set(inducedVoltagesList[0])



        label_onTheXAxis = Label(self.configure2DPlotFrame, text="On the x-axis: ",
                                 font=("Arial", int(self.paramFontSize)))

        positionsRotationsList = self.dictKeys[0:6]
        for i in range(0,len(positionsRotationsList)):
            positionsRotationsList[i] = positionsRotationsList[i] +'s'

        positionsRotationsVariable = StringVar()
        self.comboBox_onTheXAxis = Combobox(self.configure2DPlotFrame, textvariable=positionsRotationsVariable,
                                           values=positionsRotationsList,width=int(10*self.scale_factor),state='readonly', height=6)

        self.comboBox_onTheXAxis.bind('<<ComboboxSelected>>', self.generateSpecificComboBoxes)

        label_onTheXAxis.grid(row=2, column=0, sticky='W')
        self.comboBox_onTheXAxis.grid(row=2, column=1, sticky='W')

        label_forASpecific = ttk.Label(self.configure2DPlotFrame, text="For a specific: ",
                                 font=("Arial", int(self.paramFontSize)))
        label_forASpecific.grid(row=3, column=0, sticky='W')

        self.variable_checkbox_withWithoutLegend = IntVar()
        self.variable_checkbox_withWithoutLegend.set(1)
        checkbox_withWithoutLegend = ttk.Checkbutton(self.configure2DPlotFrame, text="With/Without legend.", variable = self.variable_checkbox_withWithoutLegend,
                                                     font=("Arial", int(8 * self.scale_factor)))
        checkbox_withWithoutLegend.grid(row=4, column=0, sticky='W')
        button_plot2D = ttk.Button(self.configure2DPlotFrame, text="Plot 2D", command=self.plot2D,
                                                  height=int(1 * self.scale_factor), width=int(8 * self.scale_factor),
                                                  font=("Arial", int(8 * self.scale_factor)),fg='blue')

        button_plot2D.grid(row=4, column=1, sticky='W')
    def plot2D(self):
        if len(self.plot2D_figureList) == 0:
            self.fig = plt.figure(num=self.plot2D_figureNumber)
            self.plot2D_figureList.append(self.fig)
            self.plot2D_figureNumber+=1
        else:
            if plt.fignum_exists(self.plot2D_figureList[-1].number):
                self.fig = self.plot2D_figureList[-1]
            else:
                self.fig = plt.figure(num=self.plot2D_figureNumber)
                self.plot2D_figureList.append(self.fig)
                self.plot2D_figureNumber += 1
        ## Starting of the extracting algorithm
        yAxis = []
        xAxis = self.uniqueValuesDict[self.comboBox_onTheXAxis.get()[0:-1] + 'Unique']
        index = 0
        stepSize = 1
        print (self.tableArrangment)
        #if self.tableArrangment == 'positionsWise':
        print (self.positionsRotationsList)
        print (self.positionsRotationsListForPlot2D)
        for elment in self.positionsRotationsList:
            for i in range(index, len(self.valuesDict[elment])):
                if float(self.specificCombosDict[elment].get()) == float(self.valuesDict[elment][i]):
                    index = i
                    break
        if self.positionsRotationsListForPlot2D.index(0) != len(self.positionsRotationsListForPlot2D) -1:
            for i in range(self.positionsRotationsListForPlot2D.index(0)+1,len(self.positionsRotationsListForPlot2D)):
                stepSize= stepSize*len(self.uniqueValuesDict[self.positionsRotationsListForPlot2D[i]+'Unique'])
        for i in range(index, index+len(self.uniqueValuesDict[self.comboBox_onTheXAxis.get()[0:-1]+'Unique'])*stepSize, stepSize):
            yAxis.append(self.valuesDict[self.comboBox_onTheYAxis.get()][i])
        print(xAxis)
        print(yAxis)
        # elif self.tableArrangment == 'orientationsWise':
        #
        #     self.positionsRotationsList = ['X-angle', 'Y-angle', 'Z-angle','X-position', 'Y-position','Z-position']
        #     selectedOnXAxis = str(self.comboBox_onTheXAxis.get())
        #     selectedOnXAxis = selectedOnXAxis[0:-1]
        #     self.positionsRotationsListForPlot2D = self.positionsRotationsList[:]
        #     self.positionsRotationsListForPlot2D[self.positionsRotationsListForPlot2D.index(selectedOnXAxis)] = 0
        #     self.positionsRotationsList.remove(selectedOnXAxis)
        #     print(self.positionsRotationsList)
        #     print(self.positionsRotationsListForPlot2D)
        #     for elment in self.positionsRotationsList:
        #         for i in range(index, len(self.valuesDict[elment])):
        #             if float(self.specificCombosDict[elment].get()) == float(self.valuesDict[elment][i]):
        #                 index = i
        #                 break
        #     if self.positionsRotationsListForPlot2D.index(0) != len(self.positionsRotationsListForPlot2D) - 1:
        #         for i in range(self.positionsRotationsListForPlot2D.index(0) + 1,len(self.positionsRotationsListForPlot2D)):
        #             stepSize = stepSize * len(self.uniqueValuesDict[self.positionsRotationsListForPlot2D[i] + 'Unique'])
        #     for i in range(index,index + len(self.uniqueValuesDict[self.comboBox_onTheXAxis.get()[0:-1] + 'Unique']) * stepSize,stepSize):
        #         yAxis.append(self.valuesDict[self.comboBox_onTheYAxis.get()][i])
        #     print (xAxis)
        #     print (yAxis)
        ## Plotting
        plt.figure(num=self.fig.number)
        if len(xAxis) < 2 and len(yAxis) < 2:
            plt.scatter(xAxis, yAxis, label=str(self.positionsRotationsList[0])+
                                        '='+self.specificCombosDict[self.positionsRotationsList[0]].get()+
                                        '\n'+str(self.positionsRotationsList[1])
                                        + '=' +self.specificCombosDict[self.positionsRotationsList[1]].get()
                                        +'\n'+
                                        str(self.positionsRotationsList[2])+'='+
                                        self.specificCombosDict[self.positionsRotationsList[2]].get()+
                                        '\n'+
                                        str(self.positionsRotationsList[3])+'='+
                                        self.specificCombosDict[self.positionsRotationsList[3]].get()+
                                        '\n'+
                                        str(self.positionsRotationsList[4])+'='+
                                        self.specificCombosDict[self.positionsRotationsList[4]].get())
        else:
            plt.plot(xAxis,yAxis, label=str(self.positionsRotationsList[0])+
                                        '='+self.specificCombosDict[self.positionsRotationsList[0]].get()+
                                        '\n'+str(self.positionsRotationsList[1])
                                        + '=' +self.specificCombosDict[self.positionsRotationsList[1]].get()
                                        +'\n'+
                                        str(self.positionsRotationsList[2])+'='+
                                        self.specificCombosDict[self.positionsRotationsList[2]].get()+
                                        '\n'+
                                        str(self.positionsRotationsList[3])+'='+
                                        self.specificCombosDict[self.positionsRotationsList[3]].get()+
                                        '\n'+
                                        str(self.positionsRotationsList[4])+'='+
                                        self.specificCombosDict[self.positionsRotationsList[4]].get())
        plt.xlabel(str(self.comboBox_onTheXAxis.get())+' (m)', fontsize=18)
        plt.ylabel('Induced voltage in: '+str(self.comboBox_onTheYAxis.get())+' (V)', fontsize=18)
        if self.variable_checkbox_withWithoutLegend.get() == 1:
            plt.legend()
        plt.show()
    def generateSpecificComboBoxes(self, event=None):
        self.plot2D_figureList = []
        selectedOnXAxis = str(self.comboBox_onTheXAxis.get())
        selectedOnXAxis = selectedOnXAxis[0:-1]
        self.positionsRotationsList = self.dictKeys[0:6]
        print(self.positionsRotationsList)
        self.positionsRotationsListForPlot2D = self.positionsRotationsList[:]
        self.positionsRotationsListForPlot2D[self.positionsRotationsListForPlot2D.index(selectedOnXAxis)] = 0
        self.positionsRotationsList.remove(selectedOnXAxis)
        if len(self.specificLabelsDict)!=0:
            for key in self.specificLabelsDict:
                self.specificLabelsDict[key].destroy()
        column=1
        for key in self.positionsRotationsList:
            self.specificLabelsDict[key] = ttk.Label(self.configure2DPlotFrame, text=key+': ',
                                font=("Arial", int(self.paramFontSize)))
            self.specificLabelsDict[key].grid(row=3, column=column, sticky='W')
            column+=2

        if len(self.specificCombosDict)!=0:
            for key in self.specificCombosDict:
                self.specificCombosDict[key].destroy()
        column =2
        for key in self.positionsRotationsList:
            values = self.uniqueValuesDict[key+'Unique']
            self.specificCombosDict[key] = Combobox(self.configure2DPlotFrame,
                                           values=values,width=int(10*self.scale_factor),state='readonly', height=6)
            self.specificCombosDict[key].grid(row=3, column=column, sticky='W')
            self.specificCombosDict[key].set(self.uniqueValuesDict[key+'Unique'][0])
            column+=2
    ## This function is trigerred when the user press: Load button for the table.
    def loadTable(self):
        tablePath = self.entry_simulatedTablePath.get()
        try:
            if os.path.isfile(tablePath):
                # load the parameters values
                print('Loading table.')
                self.analyzeTable(tablePath)
                self.displayTableInfo()
                self.displayConfigure2DPlot()
            else:
                self.entry_simulatedTablePath.delete(0, END)
                self.entry_simulatedTablePath.configure(fg='red')
                self.entry_simulatedTablePath.insert(0, 'Chosen table.csv does not exist.')
        except:
            self.entry_simulatedTablePath.delete(0, END)
            self.entry_simulatedTablePath.configure(fg='red')
            self.entry_simulatedTablePath.insert(0, 'Error while loading table.')

    def browseForTable1Hz(self):
        filePath = self.getFilePath()
        if filePath != 0:
            self.entry_tablePathWith1Hz.delete(0, END)
            self.entry_tablePathWith1Hz.configure(fg='black')
            self.entry_tablePathWith1Hz.insert(0, filePath)
        else:
            self.entry_tablePathWith1Hz.delete(0, END)
            self.entry_tablePathWith1Hz.configure(fg='red')
            self.entry_tablePathWith1Hz.insert(0, 'Table was not chosen or it does not exist.')
    def generateDifferentFrequencyTablesStart(self):
        self.label_tablePathWith1Hz = Label(self.generateDifferentFrequencyTab, text='Table with freq. 1 Hz: ')
        self.label_tablePathWith1Hz.grid(row=0, column=0, sticky=N + S + W + E, padx=10, pady=(10, 0))
        self.entry_tablePathWith1Hz = ttk.Entry(self.generateDifferentFrequencyTab, width=50)
        self.entry_tablePathWith1Hz.grid(row=0, column=1, sticky=N + S + W + E, padx=10, pady=(10, 0))
        self.button_browseTableWith1Hz = Button(self.generateDifferentFrequencyTab, text='Browse', command=self.browseForTable1Hz)
        self.button_browseTableWith1Hz.grid(row=0, column=2, sticky=N + S + W + E, padx=10, pady=(10, 0))
        self.var_includeObjectPositions = IntVar()
        self.var_includeObjectPositions.set(1)
        self.checkbutton_includeObjectPositions = Checkbutton(self.generateDifferentFrequencyTab,
                                                            text='include object\'s positions',
                                                            variable=self.var_includeObjectPositions,
                                                            onvalue=1, offvalue=0)
        self.checkbutton_includeObjectPositions.grid(row=1, column=0, sticky=N + S + W + E, padx=10, pady=(10, 0))
        self.var_includeObjectRotations = IntVar()
        self.var_includeObjectRotations.set(1)
        self.checkbutton_includeObjectRotations = Checkbutton(self.generateDifferentFrequencyTab,
                                                              text='include object\'s rotations',
                                                              variable=self.var_includeObjectRotations,
                                                              onvalue=1, offvalue=0)
        self.checkbutton_includeObjectRotations.grid(row=2, column=0, sticky=N + S + W + E, padx=10, pady=(10, 0))

        self.var_includeObjectCoils = IntVar()
        self.var_includeObjectCoils.set(1)
        self.checkbutton_includeObjectCoils = Checkbutton(self.generateDifferentFrequencyTab,
                                                              text='include object\'s coils x,y,z',
                                                              variable=self.var_includeObjectCoils,
                                                              onvalue=1, offvalue=0)
        self.checkbutton_includeObjectCoils.grid(row=3, column=0, sticky=N + S + W + E, padx=10, pady=(10, 0))

        self.var_includeFrameAntennas = IntVar()
        self.var_includeFrameAntennas.set(1)
        self.checkbutton_includeFrameAntennas = Checkbutton(self.generateDifferentFrequencyTab,
                                                          text='include frame antennas',
                                                          variable=self.var_includeFrameAntennas,
                                                          onvalue=1, offvalue=0)
        self.checkbutton_includeFrameAntennas.grid(row=4, column=0, sticky=N + S + W + E, padx=10, pady=(10, 0))

        self.var_includeMainAntennas = IntVar()
        self.var_includeMainAntennas.set(1)
        self.checkbutton_includeMainAntennas = Checkbutton(self.generateDifferentFrequencyTab,
                                                            text='include main antennas',
                                                            variable=self.var_includeMainAntennas,
                                                            onvalue=1, offvalue=0)
        self.checkbutton_includeMainAntennas.grid(row=5, column=0, sticky=N + S + W + E, padx=10, pady=(10, 0))

        self.var_saveTableAsNumpy = IntVar()
        self.var_saveTableAsNumpy.set(1)
        self.checkbutton_saveTableAsNumpy = Checkbutton(self.generateDifferentFrequencyTab,
                                                           text='save table as .npy',
                                                           variable=self.var_saveTableAsNumpy,
                                                           onvalue=1, offvalue=0)
        self.checkbutton_saveTableAsNumpy.grid(row=6, column=0, sticky=N + S + W + E, padx=10, pady=(10, 0))

        self.var_saveTableAsCsv = IntVar()
        self.var_saveTableAsCsv.set(0)
        self.checkbutton_saveTableAsCsv = Checkbutton(self.generateDifferentFrequencyTab,
                                                        text='save table as .csv',
                                                        variable=self.var_saveTableAsCsv,
                                                        onvalue=1, offvalue=0)
        self.checkbutton_saveTableAsCsv.grid(row=7, column=0, sticky=N + S + W + E, padx=10, pady=(10, 0))

        self.label_newFreq = Label(self.generateDifferentFrequencyTab, text='New frequency (Hz): ')
        self.label_newFreq.grid(row=8, column=0, sticky=N + S + W + E, padx=10, pady=(10, 0))
        self.entry_newFreq = Entry(self.generateDifferentFrequencyTab, width=10)
        self.entry_newFreq.grid(row=8, column=1, sticky=N + S + W + E, padx=10, pady=(10, 0))


        self.label_newTablePath = Label(self.generateDifferentFrequencyTab, text='Save new table to: ')
        self.label_newTablePath.grid(row=9, column=0, sticky=N + S + W + E, padx=10, pady=(10, 0))
        self.entry_newTablePath = ttk.Entry(self.generateDifferentFrequencyTab, width=50)
        self.entry_newTablePath.grid(row=9, column=1, sticky=N + S + W + E, padx=10, pady=(10, 0))
        self.button_browseNewTable = Button(self.generateDifferentFrequencyTab, text='Browse', command=self.browseForSavingNewTable)
        self.button_browseNewTable.grid(row=9, column=2, sticky=N + S + W + E, padx=10, pady=(10, 0))
        self.button_generateNewTable = Button(self.generateDifferentFrequencyTab, text='Generate', command=self.generateNewFreqTable)
        self.button_generateNewTable.grid(row=9, column=3, sticky=N + S + W + E, padx=10, pady=(10, 0))

    def generateNewFreqTable(self):
        table1Hz = np.loadtxt(self.entry_tablePathWith1Hz.get(), delimiter=';')
        print('Table with size' + str(np.shape(table1Hz)) + ' was loaded')
        newTable = np.empty(np.shape(table1Hz))
        newTable[:] = np.nan

        if self.var_includeObjectPositions.get() == 1:
            for i in range(3):
                for ii in range(np.shape(table1Hz)[0]):
                    newTable[ii, i] = table1Hz[ii, i]
        if self.var_includeObjectRotations.get() == 1:
            for i in range(3, 6):
                for ii in range(np.shape(table1Hz)[0]):
                    newTable[ii, i] = table1Hz[ii, i]
        if self.var_includeObjectCoils.get() == 1:
            for i in range(6, 9):
                for ii in range(np.shape(table1Hz)[0]):
                    newTable[ii, i] = table1Hz[ii, i] * float(self.entry_newFreq.get())
        if self.var_includeFrameAntennas.get() == 1:
            for i in range(9, 17):
                for ii in range(np.shape(table1Hz)[0]):
                    newTable[ii, i] = table1Hz[ii, i] * float(self.entry_newFreq.get()) * float(self.entry_newFreq.get())

        if self.var_includeMainAntennas.get() == 1:
            for i in range(17, 25):
                for ii in range(np.shape(table1Hz)[0]):
                    newTable[ii, i] = table1Hz[ii, i] * float(self.entry_newFreq.get()) * float(self.entry_newFreq.get())

        newTable = newTable[:,~np.all(np.isnan(newTable), axis=0)]

        if self.var_saveTableAsCsv.get() == 1:
            np.savetxt(self.entry_newTablePath.get() + '/table_' +str(self.entry_newFreq.get())+'.csv', newTable, fmt='%10.5E', delimiter=';')
            print('New table size: ' + str(np.shape(newTable)))
            print('table_freq_' + str(self.entry_newFreq.get()) + 'saved to: ' + self.entry_newTablePath.get())
        if self.var_saveTableAsNumpy.get() == 1:
            np.save(self.entry_newTablePath.get() + '/table_' +str(self.entry_newFreq.get()), newTable)
            print('New table size: ' + str(np.shape(newTable)))
            print('table_freq_' +str(self.entry_newFreq.get())+ 'saved to: ' + self.entry_newTablePath.get())
    ## This function creates and displays the initial main elments
    # of the GUI window
    def createStartUpElments(self):
        exciterParamFrame = Frame(self.exciterParametersTab)
        exciterParamFrame.place(x=0,y=int(10*self.scale_factor))
        objectParamFrame = Frame(self.objectParametersTab)
        objectParamFrame.place(x=0,y=int(10*self.scale_factor))
        antennaParamFrame= Frame(self.antennaParametersTab)
        antennaParamFrame.place(x=0,y=int(10*self.scale_factor))
        self.tableParamFrame = Frame(self.tableParametersTab)
        self.tableParamFrame.place(x=0, y=int(10*self.scale_factor))
        plotSetupFrame = Frame(self.inducedVoltageTab)
        plotSetupFrame.place(x=self.screenWidth-int(80*self.scale_factor), y=int(self.screenHeight/2)-int(200*self.scale_factor))
        generateTableFrame = Frame(self.inducedVoltageTab)
        generateTableFrame.place(x=self.screenWidth-int(80*self.scale_factor), y=int(self.screenHeight/2)-int(150*self.scale_factor))
        stopFrame = Frame(self.inducedVoltageTab)
        stopFrame.place(x=self.screenWidth-int(80*self.scale_factor), y=int(self.screenHeight/2)+int(200*self.scale_factor))
        estimateTimeFrame = Frame(self.tableParametersTab, relief='groove', borderwidth=5)
        estimateTimeFrame.place(x=int(900*self.scale_factor), y=int(10*self.scale_factor))
        configureFileSaveFrame = Frame(self.inducedVoltageTab)
        configureFileSaveFrame.place(x=0, y=(self.screenHeight)-int(110*self.scale_factor))
        configureFileSaveTextFrame= Frame(self.inducedVoltageTab)
        configureFileSaveTextFrame.place(x=0, y=(self.screenHeight)-int(60*self.scale_factor))
        configureFileLoadFrame = Frame(self.inducedVoltageTab)
        configureFileLoadFrame.place(x=int((self.screenWidth/2)+(10*self.scale_factor)), y=(self.screenHeight)-int(110*self.scale_factor))
        self.loadSimulatedTableFrame = Frame(self.plotTab, relief='groove', borderwidth=5)
        self.loadSimulatedTableFrame.place(x=int(10*self.scale_factor),y=int(10*self.scale_factor))
        ######################################################################################################################
        ## Initial values for GUI entries:
        ######################################################################################################################
        ## Number of windings of the exciter loop, default = 1
        self.exciterParam_NumberOfWindings = 1
        ## AC current amplitude in the exciter loop, default = 1 A
        self.exciterParam_Current = 1
        ## Frequency of the exciter's current in hz, default = 119000 hz
        self.exciterParam_Frequency = 119000
        ## Origin of the exciter coordinate system relative
        # to the intertial system, default = [0,0,0]
        self.exciterParam_ExciterOrigin = [0, 0, 0]
        ## Default Positions of the  corners of the exciter wire.
        self.exciterParam_PolygonExciterCorners = [[0, 0, 0], [1.9, 0, 0],[1.9, 1.26, 0],
                                                   [0, 1.26, 0],[0, 1, 0],[0, 0.75, 0],
                                                   [0, 0.5, 0], [0, 0.25, 0],[0, 0.1, 0]
                                                   ,[0, 0.05, 0]]

        ## Number of windings of each of the object's coil(s), default = 20
        self.objectParam_NumberOfWindings = 20
        ## Resistance of each of the object's coil(s), default = 10 Ohm
        self.objectParam_CoilResistance = 10
        ## Dimensions (if rectangle, width is smaller than length ) of a polygonal object's coil, default =0.02 m
        # in the y-axis
        self.objectParam_PolygonCoilDimensionsWidth = 0.02
        ## Dimensions (if rectangle, length is bigger than width) of a polygonal object's coil, default =0.02 m
        # in the x-axis
        self.objectParam_PolygonCoilDimensionsLength = 0.02
        ## Rotation angle alpha around the x-axis using left hand rule , default = [0] deg
        # Initial Orientation of the polygonal coil, default coil in x-y plane with MTX = [[1,0,0],[0,1,0],[0,0,1]]
        self.objectParam_PolygonCoilOrientationAlpha = 0
        ## Rotation angle alpha around the y-axis using left hand rule , default = [0] deg
        # Initial Orientation of the polygonal coil, default coil in x-y plane with MTX = [[1,0,0],[0,1,0],[0,0,1]]
        self.objectParam_PolygonCoilOrientationBeta = 0
        ## Rotation angle alpha around the z-axis using left hand rule , default = [0] deg
        # Initial Orientation of the polygonal coil, default coil in x-y plane with MTX = [[1,0,0],[0,1,0],[0,0,1]]
        self.objectParam_PolygonCoilOrientationGamma = 0
        ## Length of the major axis of the ellipse (coincides with the X-axis of it's coordinate system)
        # default = 3.81e-2 m, if EllipseCoilMajorAxis == EllipseCoilMajorAxis: this is a circular coil
        self.objectParam_EllipseCoilMajorAxis = 3.81e-2
        ## Length of the minor axis of the ellipse (coincides with the Y-axis of it's coordinate system)
        # default = 3.81e-2 m, if EllipseCoilMajorAxis == EllipseCoilMajorAxis: this is a circular coil
        self.objectParam_EllipseCoilMinorAxis = 3.81e-2
        ## Rotation angle alpha around the x-axis using left hand rule , default = [0] deg
        # Initial Orientation of the elliptical coil, default coil in x-y plane with MTX = [[1,0,0],[0,1,0],[0,0,1]]
        self.objectParam_EllipseCoilOrientationAlpha = 0
        ## Rotation angle alpha around the y-axis using left hand rule , default = [0] deg
        # Initial Orientation of the elliptical coil, default coil in x-y plane with MTX = [[1,0,0],[0,1,0],[0,0,1]]
        self.objectParam_EllipseCoilOrientationBeta = 0
        ## Rotation angle alpha around the z-axis using left hand rule , default = [0] deg
        # Initial Orientation of the elliptical coil, default coil in x-y plane with MTX = [[1,0,0],[0,1,0],[0,0,1]]
        self.objectParam_EllipseCoilOrientationGamma = 0
        ## Height of a hockey Puk, default = 0.015 m
        self.objectParam_PukCoilDimensions = 0.015
        ## Radius of the third hockey Puk's circular coil, default = 3.81e-2 m
        self.objectParam_PukCoilRadius = 3.81e-2
        ## Rotation angle alpha around the x-axis using left hand rule , default = [0] deg
        # Initial Orientation of the puk coils, default coil in x-y plane with MTX = [[1,0,0],[0,1,0],[0,0,1]]
        self.objectParam_PukCoilOrientationAlpha = 0
        ## Rotation angle alpha around the y-axis using left hand rule , default = [0] deg
        # Initial Orientation of the puk coils, default coil in x-y plane with MTX = [[1,0,0],[0,1,0],[0,0,1]]
        self.objectParam_PukCoilOrientationBeta = 0
        ## Rotation angle alpha around the z-axis using left hand rule , default = [0] deg
        # Initial Orientation of the puk coils, default coil in x-y plane with MTX = [[1,0,0],[0,1,0],[0,0,1]]
        self.objectParam_PukCoilOrientationGamma = 0
        ## Radius of the three circular coils of a Ball, default = 3.81e-2 m
        self.objectParam_BallCoilRadius = 3.81e-2
        ## Rotation angle alpha around the x-axis using left hand rule , default = [0] deg
        # Initial Orientation of the ball coils, default coil in x-y plane with MTX = [[1,0,0],[0,1,0],[0,0,1]]
        self.objectParam_BallCoilOrientationAlpha = 0
        ## Rotation angle alpha around the y-axis using left hand rule , default = [0] deg
        # Initial Orientation of the ball coils, default coil in x-y plane with MTX = [[1,0,0],[0,1,0],[0,0,1]]
        self.objectParam_BallCoilOrientationBeta = 0
        ## Rotation angle alpha around the z-axis using left hand rule , default = [0] deg
        # Initial Orientation of the ball coils, default coil in x-y plane with MTX = [[1,0,0],[0,1,0],[0,0,1]]
        self.objectParam_BallCoilOrientationGamma = 0
        ## Initial value for the antennas entries lables Frame
        # equals zero means there is no labels or entries to display
        self.AntParamTypeFrameFrame_register = 0
        self.AntParamTypeFrameMain_register = 0
        ##Initial values for the dictionaries that contains
        # the labels and entries for main and frame antennas fields
        self.mainParamLabelsEntriesDict = {}
        self.frameParamLabelsEntriesDict = {}
        ## Number of windings of each main antenna, default = 10
        self.antennaParam_MainNumberOfWindings = 10
        ## Length of each main antenna, default = 0.5 m
        self.antennaParam_MainLength = 0.5
        ## Height of each main antenna, default = 0.04 m
        self.antennaParam_MainHeight = 0.04
        ## Registered and displayed main antennas on the screen
        self.registeredMainAntennas = 0
        ## Registered and displayed frame antennas on the screen
        self.registeredFrameAntennas = 0
        ## Number of windings of each frame antenna, default = 10
        self.antennaParam_FrameNumberOfWindings = 10
        ## Length of each frame antenna, default = 0.5 m
        self.antennaParam_FrameLength = 0.5
        ## Height of each frame antenna, default = 0.02 m
        self.antennaParam_FrameHeight = 0.02
        ## list that contains the default positions for 10 main antennas
        self.mainAntennasDefaultPositions=[[0,0.33, 0],[0,0.91, 0],[0.35,1.26, 0],[0.95,1.26, 0],[1.55,1.26, 0],[1.9,0.91, 0],
                                           [1.9,0.33, 0],[1.55,0, 0],[0.95,0, 0],[0.35,0, 0],
                                           [0, 0.33, 0], [0, 0.91, 0], [0.35, 1.26, 0], [0.95, 1.26, 0],
                                           [1.55, 1.26, 0], [1.9, 0.91, 0],
                                           [1.9, 0.33, 0], [1.55, 0, 0], [0.95, 0, 0], [0.35, 0, 0]
                                           ]
        ## list that contains the default orientation of 10 main antennas
        self.mainAntennasDefaultOrientations = [[0,0,270],[0,0,270],
                                                [0,0,180],[0,0,180], [0,0,180],
                                                [0, 0, 90], [0, 0, 90],
                                                [0, 0, 0], [0, 0, 0],[0, 0, 0],
                                                [0, 0, 270], [0, 0, 270],
                                                [0, 0, 180], [0, 0, 180], [0, 0, 180],
                                                [0, 0, 90], [0, 0, 90],
                                                [0, 0, 0], [0, 0, 0], [0, 0, 0]
                                                ]
        ## list that contains the default positions for 10 frame antennas
        self.frameAntennasDefaultPositions = [[0,0.33, 0],[0,0.91, 0],[0.35,1.26, 0]
                                         ,[0.95,1.26, 0],[1.55,1.26, 0],[1.9,0.91, 0],
                                           [1.9,0.33, 0],[1.55,0, 0],[0.95,0, 0],[0.35,0, 0],
                                              [0, 0.33, 0], [0, 0.91, 0], [0.35, 1.26, 0]
            , [0.95, 1.26, 0], [1.55, 1.26, 0], [1.9, 0.91, 0],
                                              [1.9, 0.33, 0], [1.55, 0, 0], [0.95, 0, 0], [0.35, 0, 0]
                                              ]
        ## list that contains the default orientation of 10 frame antennas
        self.frameAntennasDefaultOrientations = [[0,0,270],[0,0,270],
                                                [0,0,180],[0,0,180], [0,0,180],
                                                [0, 0, 90], [0, 0, 90],
                                                [0, 0, 0], [0, 0, 0],[0, 0, 0],
                                                 [0, 0, 270], [0, 0, 270],
                                                 [0, 0, 180], [0, 0, 180], [0, 0, 180],
                                                 [0, 0, 90], [0, 0, 90],
                                                 [0, 0, 0], [0, 0, 0], [0, 0, 0]
                                                 ]
        ## Number of CPU cores to be used during the simulation, default = 6 cores
        self.excelTableParam_NumberOfCores = 6
        ## The desired object's x-position(s) to be simulated, it can be [start, end, step] m
        # or a single value [xposition] m, default = [0.05, 1.851, 0.001] m
        self.excelTableParam_XPos = [0.05, 0.06, 0.1]
        ## The desired object's y-position(s) to be simulated, it can be [start, end, step] m
        # or a single value [yposition] m, default = [0.05, 1.201, 0.001] m
        self.excelTableParam_YPos = [0.05, 0.06, 0.1]
        ## The desired object's z-position(s) to be simulated, it can be [start, end, step] m
        # or a single value [zposition] m, default = [-1, 1.01, 0.01] m
        self.excelTableParam_ZPos = [-1, 1.1, 0.1]
        ## The desired object's rotation(s) around the x-axis, using left hand rule,
        # to be simulated, it can be [start, end, step] degress
        # or a single value [xrotation] degrees, default = [0] degrees
        self.excelTableParam_XAngles = [0]
        ## The desired object's rotation(s) around the y-axis, using left hand rule,
        # to be simulated, it can be [start, end, step] degress
        # or a single value [yrotation] degrees, default = [0] degrees
        self.excelTableParam_YAngles = [0]
        ## The desired object's rotation(s) around the z-axis, using left hand rule,
        # to be simulated, it can be [start, end, step] degress
        # or a single value [zrotation] degrees, default = [0] degrees
        self.excelTableParam_ZAngles = [0]
        ## The desired excel table name, .csv will be added automatically
        # to the given name, default = testing
        self.excelTableParam_TableName = str("inducedVoltage")
        ## The desired directory path that the simulated table will be
        # saved to, default = Y:\myFolder\\
        # if there is already a file with the same name exists in the
        # desired directory, the table will be saved in a new name
        #self.excelTableParam_SaveTo = "\\tables"
        ## Table creator name, default=ibrahiim
        self.excelTableParam_Author = str(getpass.getuser())
        ## intitial process id
        self.proc = []
        ##
        self.configure2DPlotFrame = None
        self.plot2D_figureList = []
        self.plot2D_figureNumber = 1
        ######################################################################################################################
        ## create exciter parameters labels
        ######################################################################################################################
        # label_ExciterParam_Title = Label(exciterParamFrame, text="Exciter parameters" , font=("Arial", self.titleFontSize))
        label_ExciterParam_NumberWindings = Label(exciterParamFrame, text="Number of windings: " , font=("Arial", self.paramFontSize))
        label_ExciterParam_Current = Label(exciterParamFrame, text="Current (A): " , font=("Arial", self.paramFontSize))
        label_ExciterParam_Frequency = Label(exciterParamFrame, text="Frequency (Hz): ",  font=("Arial", self.paramFontSize))
        label_ExciterParam_ExciterOrigin = Label(exciterParamFrame, text="Origin [x(m), y(m), z(m)]: ", font=("Arial", self.paramFontSize))
        label_ExciterParam_NumberOfExciterCorners = Label(exciterParamFrame, text="Number of exciter corners: ", font=("Arial", self.paramFontSize),
                                                 justify='left')

        #######################################################################################################################################
        # ExciterParameters Dropdown list:
        #########################################################################################################################
        cornersChoices = {'', 2, 3, 4, 5, 6, 7, 8, 9, 10}
        self.numberOfExciterCorners = IntVar()
        self.numberOfExciterCorners.set(4)
        self.numberOfExciterCornersBuffer = 0
        exciterParam_dropDownList = OptionMenu(exciterParamFrame, self.numberOfExciterCorners, *cornersChoices)
        exciterParam_addExciterCornersButton = ttk.Button(exciterParamFrame, text="Add", command=self.addExciterCorners,
                                                          height=int(1*self.scale_factor), width=int(6*self.scale_factor))

        #######################################################################################################################################
        #ExciterParameters entries:
        #########################################################################################################################
        self.entry_ExciterParam_NumberWindings = Entry(exciterParamFrame, width=int(5*self.scale_factor))
        self.entry_ExciterParam_NumberWindings.insert(0, self.exciterParam_NumberOfWindings)
        self.entry_ExciterParam_Current = Entry(exciterParamFrame, width=int(5*self.scale_factor))
        self.entry_ExciterParam_Current.insert(0, str(self.exciterParam_Current))
        self.entry_ExciterParam_Frequency = Entry(exciterParamFrame, width=int(10*self.scale_factor))
        self.entry_ExciterParam_Frequency.insert(0, str(self.exciterParam_Frequency))
        self.entry_ExciterParam_ExciterOrigin = Entry(exciterParamFrame, width=int(10*self.scale_factor))
        self.entry_ExciterParam_ExciterOrigin.insert(0, str(self.exciterParam_ExciterOrigin))

        ########################################################################################################################
        ## Gridding ExciterParameters-labels and entries:
        ########################################################################################################################
        #label_ExciterParam_Title.grid(row=0, column=0)
        label_ExciterParam_NumberWindings.grid(row=0, column=0)
        label_ExciterParam_Current.grid(row=1, column=0)
        label_ExciterParam_Frequency.grid(row=2, column=0)
        label_ExciterParam_ExciterOrigin.grid(row=3, column=0)
        label_ExciterParam_NumberOfExciterCorners.grid(row=4, column=0)
        exciterParam_dropDownList.grid(row=4, column=1)
        exciterParam_addExciterCornersButton.grid(row=4, column=2)
        self.entry_ExciterParam_NumberWindings.grid(row=0, column=1)
        self.entry_ExciterParam_Current.grid(row=1, column=1)
        self.entry_ExciterParam_Frequency.grid(row=2, column=1)
        self.entry_ExciterParam_ExciterOrigin.grid(row=3, column=1)

        ########################################################################################################################
        ## Initialize exciter corners:
        ########################################################################################################################
        self.exciterCornersFrame_register = 0
        self.addExciterCorners()
        ########################################################################################################################
        ## ObjectParameters labels:
        ########################################################################################################################
        #label_ObjectParam_Title = Label(objectParamFrame, text="Object parameters" , font=("Arial", self.titleFontSize))
        label_ObjectParam_NumberWindings = Label(objectParamFrame, text="Number of windings: ", font=("Arial", self.paramFontSize))
        label_ObjectParam_Resistance = Label(objectParamFrame, text="Coil resistance (Ohm): ", font=("Arial", self.paramFontSize))
        label_ObjectParam_ObjectType = Label(objectParamFrame, text="Object type: ",font=("Arial", self.paramFontSize))
        ########################################################################################################################
        #objectParameters entries:
        ########################################################################################################################
        self.entry_ObjectParam_NumberWindings = Entry(objectParamFrame, width=int(5*self.scale_factor))
        self.entry_ObjectParam_NumberWindings.insert(0, str(self.objectParam_NumberOfWindings))
        self.entry_ObjectParam_Resistance = Entry(objectParamFrame, width=int(5*self.scale_factor))
        self.entry_ObjectParam_Resistance.insert(0, str(self.objectParam_CoilResistance))
        ########################################################################################################################
        ## ObjectParameters check boxes:
        ########################################################################################################################
        self.ObjectParamTypeFrame_register = 0
        self.objectTypePolygon = IntVar()
        self.objectTypeEllipse = IntVar()
        self.objectTypePuk = IntVar()
        self.objectTypePuk.set(1)
        self.objectTypeBall = IntVar()
        self.objectTypeWearable = IntVar()
        self.displayObjectTypePuk()


        checkBox_ObjectParam_Polygon = Checkbutton(objectParamFrame, text="Polygon", variable=self.objectTypePolygon,command=self.displayObjectTypePolygon)
        checkBox_ObjectParam_Ellipse = Checkbutton(objectParamFrame, text="Ellipse",variable=self.objectTypeEllipse,command=self.displayObjectTypeEllipse)
        checkBox_ObjectParam_Puk = Checkbutton(objectParamFrame, text="Puk",variable=self.objectTypePuk,command=self.displayObjectTypePuk)
        checkBox_ObjectParam_Ball = Checkbutton(objectParamFrame, text="Ball",variable=self.objectTypeBall,command=self.displayObjectTypeBall)
        checkBox_ObjectParam_Wearable = Checkbutton(objectParamFrame, text="Wearable", variable=self.objectTypeWearable,
                                                command=self.displayObjectTypeWearable)
        ########################################################################################################################
        ## Gridding ObjectParameters-labels, entries and check boxes:
        ########################################################################################################################
        #label_ObjectParam_Title.grid(row=0)
        label_ObjectParam_NumberWindings.grid(row=0, column=0)
        label_ObjectParam_Resistance.grid(row=1, column=0)
        self.entry_ObjectParam_NumberWindings.grid(row=0, column=1)
        self.entry_ObjectParam_Resistance.grid(row=1, column=1)
        label_ObjectParam_ObjectType.grid(row=2, column=0)
        checkBox_ObjectParam_Polygon.grid(row=2, column=1)
        checkBox_ObjectParam_Ellipse.grid(row=2, column=2)
        checkBox_ObjectParam_Puk.grid(row=2, column=3)
        checkBox_ObjectParam_Ball.grid(row=2, column=4)
        checkBox_ObjectParam_Wearable.grid(row=2, column=5)
        ########################################################################################################################
        #AntennasParameters labels:
        ########################################################################################################################
        #label_AntennasParam_Title = Label(antennaParamFrame, text="Antennas parameters" , font=("Arial", self.titleFontSize))
        ########################################################################################################################
        ## AntennaParameters check boxes variables:
        ########################################################################################################################
        self.antTypeMain = IntVar()
        self.antTypeFrame = IntVar()
        ########################################################################################################################
        #AntennasParameters check boxes:
        ########################################################################################################################
        checkBox_AntennasParam_MainAntennas = Checkbutton(antennaParamFrame, text="Main", variable=self.antTypeMain,command=self.displayAntTypeMain)
        checkBox_AntennasParam_FrameAntennas = Checkbutton(antennaParamFrame, text="Frame", variable=self.antTypeFrame,command=self.displayAntTypeFrame)
        ########################################################################################################################
        #Gridding AntennasParameters-labels and checkboxes:
        ########################################################################################################################
        #label_AntennasParam_Title.grid(row=0)
        checkBox_AntennasParam_MainAntennas.grid(row=0, column=0)
        checkBox_AntennasParam_FrameAntennas.grid(row=0, column=1)
        ########################################################################################################################
        ## Table parameters labels
        ########################################################################################################################
        #abel_TableParam_Title = Label(self.tableParamFrame, text="Excel table parameters",
                                        # font=("Arial", self.titleFontSize))
        label_TableParam_SavedAt = ttk.Label(self.tableParamFrame,
                                              text="Tables will be saved at \\tables",
                                              font=("Arial", int(8*self.scale_factor)), fg='blue')
        label_TableParam_NumberOfCores = Label(self.tableParamFrame, text="Number of cores: ",
                                      font=("Arial", self.paramFontSize))
        label_TableParam_XPos = Label(self.tableParamFrame, text="xpos (m): ",
                                           font=("Arial", self.paramFontSize))
        label_TableParam_YPos = Label(self.tableParamFrame, text="ypos (m): ",
                                      font=("Arial", self.paramFontSize))
        label_TableParam_ZPos = Label(self.tableParamFrame, text="zpos (m): ",
                                      font=("Arial", self.paramFontSize))
        label_TableParam_XAngels = Label(self.tableParamFrame, text=u'\u03b1'+' (deg)',
                                      font=("Arial", self.paramFontSize))
        label_TableParam_YAngels = Label(self.tableParamFrame, text=u'\u03b2'+' (deg)',
                                         font=("Arial", self.paramFontSize))
        label_TableParam_ZAngels = Label(self.tableParamFrame, text=u'\u03b3'+' (deg)',
                                         font=("Arial", self.paramFontSize))
        label_TableParam_TableName = Label(self.tableParamFrame, text="Table name: ",
                                                  font=("Arial", self.paramFontSize))
        #label_TableParam_SaveTo = Label(self.tableParamFrame, text="Save to (path): ", font=("Arial", self.paramFontSize))
        label_TableParam_Author = Label(self.tableParamFrame, text="Author name: ", font=("Arial", self.paramFontSize))
        ########################################################################################################################
        ## Gridding table parameters labels:
        ########################################################################################################################
        #label_TableParam_Title.grid(row=0, column=0)
        label_TableParam_SavedAt.grid(row=0, column=1)
        label_TableParam_NumberOfCores.grid(row=1, column=0)
        label_TableParam_XPos.grid(row=2, column=0)
        label_TableParam_YPos.grid(row=4, column=0)
        label_TableParam_ZPos.grid(row=6, column=0)
        label_TableParam_XAngels.grid(row=8, column=0)
        label_TableParam_YAngels.grid(row=10, column=0)
        label_TableParam_ZAngels.grid(row=12, column=0)
        label_TableParam_TableName.grid(row=14, column=0)
        #label_TableParam_SaveTo.grid(row=9, column=0)
        label_TableParam_Author.grid(row=15, column=0)
        ########################################################################################################################
        # Table Parameters check boxes:
        ########################################################################################################################
        self.xPosOption_register = []
        self.yPosOption_register = []
        self.zPosOption_register = []
        self.alphaOption_register = []
        self.betaOption_register = []
        self.gammaOption_register = []

        self.xpos_singleMultiplePoints = IntVar()
        self.xpos_singleMultiplePoints.set(1)
        self.xpos_sweep = IntVar()
        self.xpos_sweep.set(0)
        self.displayXPosSingleMultiplePoints()


        self.ypos_singleMultiplePoints = IntVar()
        self.ypos_singleMultiplePoints.set(1)
        self.ypos_sweep = IntVar()
        self.ypos_sweep.set(0)
        self.displayYPosSingleMultiplePoints()


        self.zpos_singleMultiplePoints = IntVar()
        self.zpos_singleMultiplePoints.set(1)
        self.zpos_sweep = IntVar()
        self.zpos_sweep.set(0)
        self.displayZPosSingleMultiplePoints()


        self.alpha_singleMultipleAngles = IntVar()
        self.alpha_singleMultipleAngles.set(1)
        self.alpha_sweep = IntVar()
        self.alpha_sweep.set(0)
        self.displayAlphaSingleMultipleAngles()

        self.beta_singleMultipleAngles = IntVar()
        self.beta_singleMultipleAngles.set(1)
        self.beta_sweep = IntVar()
        self.beta_sweep.set(0)
        self.displayBetaSingleMultipleAngles()


        self.gamma_singleMultipleAngles = IntVar()
        self.gamma_singleMultipleAngles.set(1)
        self.gamma_sweep = IntVar()
        self.gamma_sweep.set(0)
        self.displayGammaSingleMultipleAngles()


        checkBox_TableParam_XPosSingleMultiplePoints = Checkbutton(self.tableParamFrame, text="Single/Multiple points", variable=self.xpos_singleMultiplePoints,
                                                          command=self.displayXPosSingleMultiplePoints)
        checkBox_TableParam_XPosSweep = Checkbutton(self.tableParamFrame, text="Sweep",
                                                                   variable=self.xpos_sweep,
                                                                   command=self.displayXPosSweep)
        checkBox_TableParam_YPosSingleMultiplePoints = Checkbutton(self.tableParamFrame, text="Single/Multiple points",
                                                                   variable=self.ypos_singleMultiplePoints,
                                                                   command=self.displayYPosSingleMultiplePoints)
        checkBox_TableParam_YPosSweep = Checkbutton(self.tableParamFrame, text="Sweep",
                                                    variable=self.ypos_sweep,
                                                    command=self.displayYPosSweep)

        checkBox_TableParam_ZPosSingleMultiplePoints = Checkbutton(self.tableParamFrame, text="Single/Multiple points",
                                                                    variable=self.zpos_singleMultiplePoints,
                                                                    command=self.displayZPosSingleMultiplePoints)
        checkBox_TableParam_ZPosSweep = Checkbutton(self.tableParamFrame, text="Sweep",
                                                     variable=self.zpos_sweep,
                                                     command=self.displayZPosSweep)

        checkBox_TableParam_AlphaSingleMultiplePoints = Checkbutton(self.tableParamFrame, text="Single/Multiple angles",
                                                                    variable=self.alpha_singleMultipleAngles,
                                                                    command=self.displayAlphaSingleMultipleAngles)
        checkBox_TableParam_AlphaSweep = Checkbutton(self.tableParamFrame, text="Sweep",
                                                     variable=self.alpha_sweep,
                                                     command=self.displayAlphaSweep)
        checkBox_TableParam_BetaSingleMultiplePoints = Checkbutton(self.tableParamFrame, text="Single/Multiple angles",
                                                                    variable=self.beta_singleMultipleAngles,
                                                                    command=self.displayBetaSingleMultipleAngles)
        checkBox_TableParam_BetaSweep = Checkbutton(self.tableParamFrame, text="Sweep",
                                                     variable=self.beta_sweep,
                                                     command=self.displayBetaSweep)
        checkBox_TableParam_GammaSingleMultiplePoints = Checkbutton(self.tableParamFrame, text="Single/Multiple angles",
                                                                   variable=self.gamma_singleMultipleAngles,
                                                                   command=self.displayGammaSingleMultipleAngles)
        checkBox_TableParam_GammaSweep = Checkbutton(self.tableParamFrame, text="Sweep",
                                                    variable=self.gamma_sweep,
                                                    command=self.displayGammaSweep)

        ########################################################################################################################
        ##Gridding table parameters checkboxes
        ########################################################################################################################
        checkBox_TableParam_XPosSingleMultiplePoints.grid(row=2, column=1)
        checkBox_TableParam_XPosSweep.grid(row=2, column=2)
        checkBox_TableParam_YPosSingleMultiplePoints.grid(row=4, column=1)
        checkBox_TableParam_YPosSweep.grid(row=4, column=2)
        checkBox_TableParam_ZPosSingleMultiplePoints.grid(row=6, column=1)
        checkBox_TableParam_ZPosSweep.grid(row=6, column=2)
        checkBox_TableParam_AlphaSingleMultiplePoints.grid(row=8, column=1)
        checkBox_TableParam_AlphaSweep.grid(row=8, column=2)
        checkBox_TableParam_BetaSingleMultiplePoints.grid(row=10, column=1)
        checkBox_TableParam_BetaSweep.grid(row=10, column=2)
        checkBox_TableParam_GammaSingleMultiplePoints.grid(row=12, column=1)
        checkBox_TableParam_GammaSweep.grid(row=12, column=2)
        ########################################################################################################################
        ##Table parameters entries
        ########################################################################################################################

        self.entry_TableParam_NumberOfCores = Entry(self.tableParamFrame,  width=int(5*self.scale_factor))
        self.entry_TableParam_NumberOfCores.insert(0, str(self.excelTableParam_NumberOfCores))

        self.entry_TableParam_TableName = Entry(self.tableParamFrame, width=int(15*self.scale_factor))
        self.entry_TableParam_TableName.insert(0, str(self.excelTableParam_TableName))
        self.entry_TableParam_Author = Entry(self.tableParamFrame, width=int(15*self.scale_factor))
        self.entry_TableParam_Author.insert(0, str(self.excelTableParam_Author))
        ########################################################################################################################
        ## Gridding table parameters entries:
        ########################################################################################################################
        self.entry_TableParam_NumberOfCores.grid(row=1, column=1)
        self.entry_TableParam_TableName.grid(row=14, column=1)
        self.entry_TableParam_Author.grid(row=15, column=1)
        ########################################################################################################################
        ## Stop button
        ########################################################################################################################
        button_Stop = ttk.Button(stopFrame, text="Stop", fg="red", command=self.stop, height=int(2*self.scale_factor), width=int(6*self.scale_factor),font=("Arial", int(8*self.scale_factor),'bold'))
        button_Stop.grid()
        ########################################################################################################################
        ## Generate table button
        ########################################################################################################################
        self.button_GenerateTable = ttk.Button(generateTableFrame, text="Generate"+'\n'+"table",fg="dark green", command=self.generateTable, height=int(2*self.scale_factor), width=int(8*self.scale_factor))
        self.button_GenerateTable.grid()
        ########################################################################################################################
        ## Plot setup button
        ########################################################################################################################
        self.button_PlotSetup = ttk.Button(plotSetupFrame, text="Plot setup", fg="blue", command=self.plotSetup, height=int(2*self.scale_factor), width=int(8*self.scale_factor))
        self.button_PlotSetup.grid()
        ########################################################################################################################
        ## Plot/Generate/Stop text messages:
        ########################################################################################################################
        self.label_PlotGenerateStopMessages = ttk.Label(self.inducedVoltageTab, text="",
                                       font=("Arial", int(8*self.scale_factor)))
        self.label_PlotGenerateStopMessages.place(x=self.screenWidth-int(80*self.scale_factor), y=int(self.screenHeight/2))
        ########################################################################################################################
        ## Estimate time labels and entries
        ########################################################################################################################
        self.label_totalPoints = ttk.Label(estimateTimeFrame, text="Total number of points: ")
        self.label_maxTime = Label(estimateTimeFrame, text="Max. estimated time (min): ")
        self.label_minTime = Label(estimateTimeFrame, text="Min. estimated time (min): ")
        self.label_avgTime = Label(estimateTimeFrame, text="Avg. estimated time (min): ")
        self.entry_totalPoints = ttk.Entry(estimateTimeFrame, width=int(10*self.scale_factor))
        self.entry_maxTime = ttk.Entry(estimateTimeFrame, width = int(10*self.scale_factor))
        self.entry_minTime = ttk.Entry(estimateTimeFrame, width = int(10*self.scale_factor))
        self.entry_avgTime = ttk.Entry(estimateTimeFrame, width=int( 10*self.scale_factor))

        ########################################################################################################################
        ## Gridding estimate time labels and entries
        ########################################################################################################################
        self.label_totalPoints.grid(row=1, column=0)
        self.label_maxTime.grid(row=2, column = 0)
        self.label_minTime.grid(row=3, column = 0)
        self.label_avgTime.grid(row=4, column = 0)
        self.entry_totalPoints.grid(row=1, column = 1)
        self.entry_maxTime.grid(row=2, column = 1)
        self.entry_minTime.grid(row=3, column = 1)
        self.entry_avgTime.grid(row=4, column = 1)
        ########################################################################################################################
        ## Configure files save parameters labels and entries
        ########################################################################################################################
        label_configFilesSave_Title = Label(configureFileSaveFrame, text="Save configuration file",
                                       font=("Arial", self.paramFontSize))
        label_configFilesSave_SavedAt = ttk.Label(configureFileSaveFrame, text="Config. files will be saved at \\configFiles",
                                            font=("Arial", int(8*self.scale_factor)), fg='blue')
        label_configFilesSave_EnterConfigFileName = Label(configureFileSaveFrame, text="Enter name for configuration file:",
                                       font=("Arial", self.paramFontSize))
        self.label_SaveConfigTextMessage = ttk.Label(configureFileSaveTextFrame, text="",
                                       font=("Arial", int(8*self.scale_factor)))
        self.entry_configFilesSave_EnterConfigFileName = ttk.Entry(configureFileSaveFrame, width=int(40*self.scale_factor))

        ########################################################################################################################
        ## Gridding Configure files save parameters labels
        ########################################################################################################################
        label_configFilesSave_Title.grid(row=0, column=0)
        label_configFilesSave_SavedAt.grid(row=0, column=1)
        label_configFilesSave_EnterConfigFileName.grid(row=1, column=0)
        self.entry_configFilesSave_EnterConfigFileName.grid(row=1, column=1)
        self.entry_configFilesSave_EnterConfigFileName.insert(0, 'configFile_')
        self.label_SaveConfigTextMessage.grid()
        ########################################################################################################################
        ## Configure files save button
        ########################################################################################################################
        button_SaveConfigFile = ttk.Button(configureFileSaveFrame,text="Save config. file", command=self.saveConfigFile,
                                           height=int(1*self.scale_factor), width=int(10*self.scale_factor),font=("Arial", int(8*self.scale_factor)))
        ########################################################################################################################
        ## Gridding Configure files save button
        ########################################################################################################################
        button_SaveConfigFile.grid(row=1, column=3)
        ########################################################################################################################
        ## Configure files load parameters labels and entries
        ########################################################################################################################
        label_configFilesLoad_Title = Label(configureFileLoadFrame, text="Load configuration file",
                                            font=("Arial", self.paramFontSize))
        label_configFilesLoad_EnterConfigFileName = Label(configureFileLoadFrame,
                                                          text="Full path of configuration file:",
                                                          font=("Arial", self.paramFontSize))
        self.entry_configFilesLoad_BrowseConfigFilePath = ttk.Entry(configureFileLoadFrame, width=int(40*self.scale_factor))
        label_configFilesLoad_TextMessage = ttk.Label(configureFileLoadFrame,
                                                          text="Enter full path of config. file or browse for it.",
                                                          font=("Arial", int(8*self.scale_factor)), fg= 'blue')
        ########################################################################################################################
        ## Gridding Configure files load parameters labels and entries
        ########################################################################################################################
        label_configFilesLoad_Title.grid(row=0)
        label_configFilesLoad_EnterConfigFileName.grid(row=1, column=0)
        label_configFilesLoad_TextMessage.grid(row=0, column=1)
        self.entry_configFilesLoad_BrowseConfigFilePath.grid(row=1, column=1)

        ########################################################################################################################
        ## Configure files load buttons
        ########################################################################################################################
        button_BrowseConfigFile = ttk.Button(configureFileLoadFrame, text="Browse", command=self.browseForConfigFile,
                                             height=int(1 * self.scale_factor), width=int(8 * self.scale_factor),
                                             font=("Arial", int(8 * self.scale_factor)))
        button_LoadConfigFile = ttk.Button(configureFileLoadFrame, text="Load", command=self.loadConfigFile,
                                           height=int(1 * self.scale_factor), width=int(8 * self.scale_factor),
                                           font=("Arial", int(8 * self.scale_factor)))
        ########################################################################################################################
        ## Gridding Configure files browse button
        ########################################################################################################################
        button_BrowseConfigFile.grid(row=1, column=3)
        button_LoadConfigFile.grid(row=2, column=1)
        ########################################################################################################################
        ## PlotTab labels, entries
        ########################################################################################################################
        label_loadSimualtedTable = Label(self.loadSimulatedTableFrame, text="Load a simulated table: ",
                                                          font=("Arial", int(self.paramFontSize)))
        self.entry_simulatedTablePath = ttk.Entry(self.loadSimulatedTableFrame, width=int(40*self.scale_factor))
        button_browseForSimualtedTable = ttk.Button(self.loadSimulatedTableFrame, text="Browse", command=self.browseForTable,
                                                    height=int(1 * self.scale_factor), width=int(8 * self.scale_factor),
                                                    font=("Arial", int(8 * self.scale_factor))
                                                    )
        button_loadForSimualtedTable = ttk.Button(self.loadSimulatedTableFrame, text="Load", command=self.loadTable,
                                                  height=int(1 * self.scale_factor), width=int(8 * self.scale_factor),
                                                  font=("Arial", int(8 * self.scale_factor))
                                                  )
        label_enterFullPathOfSimulatedTable = ttk.Label(self.loadSimulatedTableFrame,
                                                          text="Enter full path of table.csv or browse for it.",
                                                          font=("Arial", int(8*self.scale_factor)), fg= 'blue')

        ########################################################################################################################
        ## Gridding PlotTab labels, entries
        ########################################################################################################################
        label_loadSimualtedTable.grid(row=0, column=0, sticky='W')
        self.entry_simulatedTablePath.grid(row=0, column=1, sticky='W')
        button_browseForSimualtedTable.grid(row=0, column=2, sticky='W')
        label_enterFullPathOfSimulatedTable.grid(row=1, column=1, sticky='W')
        button_loadForSimualtedTable.grid(row=1, column=2, sticky='W')
        ########################################################################################################################
        ## estimateTimeRefresher function initialization
        ########################################################################################################################
        self.estimateTimeRefresher()
        self.generateDifferentFrequencyTablesStart()
## class GUI_couplingFactorTab
class GUI_couplingFactorTab(GUI):
    def __init__(self, scale_factor, screenWidth, screenHeight, paramFontSize, titleFontSize, frame):
        self.scale_factor = scale_factor
        self.screenWidth = screenWidth
        self.screenHeight = screenHeight
        self.paramFontSize = paramFontSize
        self.titleFontSize = titleFontSize
        self.frame = frame
        self.couplingFactorTab = self.create_add_tab(self.frame, None, None, 'Coupling Factor')
        self.nb_couplingFactor = self.create_grid_notebook(self.couplingFactorTab, 0,1)
        self.objectParametersTab = self.create_add_tab(self.nb_couplingFactor, self.screenWidth - int(100*self.scale_factor),
                                                       self.screenHeight - int(150*self.scale_factor),'Objects parameters')
        self.createStartUpElments()
    def objectTypeFrameDestroy(self, frame, option, objectNumber):
        if objectNumber ==1:
            if option == 'all':
                if len(frame) !=0:
                    for element in frame:
                        element.destroy()
                    self.object1TypeFrame_register =[]

            elif option == 'last':
                if len(frame) >1:
                    frame[-1].destroy()
                    del frame[-1]
                    self.object1TypeFrame_register = frame
        elif objectNumber ==2:
            if option == 'all':
                if len(frame) != 0:
                    for element in frame:
                        element.destroy()
                    self.object2TypeFrame_register = []

            elif option == 'last':
                if len(frame) > 1:
                    frame[-1].destroy()
                    del frame[-1]
                    self.object2TypeFrame_register = frame
    def addPolygonCorners(self):
        if self.Object1numberOfPolygonCornersBuffer != self.Object1numberOfPolygonCorners.get():
            self.objectTypeFrameDestroy(self.object1TypeFrame_register, 'last',1)

            polygonCornersFrame = ttk.Frame(self.objectParametersTab)
            polygonCornersFrame.place(x=0, y=int(250*self.scale_factor))
            self.object1TypeFrame_register.append(polygonCornersFrame)

            self.Object1polygonCornersEntriesDict = {}
            self.Object1numberOfPolygonCornersBuffer = self.Object1numberOfPolygonCorners.get()
            self.registeredPolygonCorners = self.Object1numberOfPolygonCorners.get()
            for i in range(1,self.Object1numberOfPolygonCorners.get()+1):
                if i < 6:
                    self.Object1polygonCornersEntriesDict["Corner {0}".format(i)] = Label(polygonCornersFrame, text="Corner {0} [x(m),y(m),z(m)]: ".format(i),
                                                                      font=("Arial", self.paramFontSize-1),justify='left')
                    self.Object1polygonCornersEntriesDict["Corner {0}".format(i)].grid(row=i-1, column=0)

                    self.Object1polygonCornersEntriesDict["Corner {0} entry".format(i)] = Entry(polygonCornersFrame,width=12)
                    self.Object1polygonCornersEntriesDict["Corner {0} entry".format(i)].grid(row=i-1, column=1)
                    self.Object1polygonCornersEntriesDict["Corner {0} entry".format(i)].insert(0, str(self.polygonParam_PolygonPolygonCorners[i - 1]))
                else:
                    self.Object1polygonCornersEntriesDict["Corner {0}".format(i)] = Label(polygonCornersFrame,
                                                                                   text="Corner {0} [x(m),y(m),z(m)]: ".format(
                                                                                       i),
                                                                                   font=("Arial", self.paramFontSize-1),
                                                                                   justify='left')
                    self.Object1polygonCornersEntriesDict["Corner {0}".format(i)].grid(row=i-6, column=2)

                    self.Object1polygonCornersEntriesDict["Corner {0} entry".format(i)] = Entry(
                        polygonCornersFrame, width=12)
                    self.Object1polygonCornersEntriesDict["Corner {0} entry".format(i)].grid(row=i-6, column=3)
                    self.Object1polygonCornersEntriesDict["Corner {0} entry".format(i)].insert(0,
                                                                                        str(self.polygonParam_PolygonPolygonCorners[
                                                                                            i - 1]))
    def displayObject1TypeAntennaFrame(self):
        if self.object1TypeAntennaFrame.get() == 1:
            self.entry_object1_inductance.delete(0, END)
            self.entry_object1_inductance.insert(0, 495.9e-6)
            self.object1TypeAntennaMain.set(0)
            self.objectTypeFrameDestroy(self.object1TypeFrame_register, 'last',1)
            self.object1_AntennaTypeFrame_Frame = ttk.Frame(self.objectParametersTab)
            self.object1_AntennaTypeFrame_Frame.place(x=0, y=int(220*self.scale_factor))
            self.object1TypeFrame_register.append(self.object1_AntennaTypeFrame_Frame)

            label_object1_AntennaTypeFrame_Length= Label(self.object1_AntennaTypeFrame_Frame, text='Length (m): ', font=('Arial', self.paramFontSize))
            label_object1_AntennaTypeFrame_Height = Label(self.object1_AntennaTypeFrame_Frame, text='Height (m): ',
                                                         font=('Arial', self.paramFontSize))

            self.entry_object1_AntennaTypeFrame_Length = Entry(self.object1_AntennaTypeFrame_Frame, width=10)
            self.entry_object1_AntennaTypeFrame_Height = Entry(self.object1_AntennaTypeFrame_Frame, width=10)

            self.entry_object1_AntennaTypeFrame_Length.insert(0, str(self.object1_AntennaTypeFrame_Length) )
            self.entry_object1_AntennaTypeFrame_Height.insert(0, str(self.object1_AntennaTypeFrame_Height) )


            label_object1_AntennaTypeFrame_Length.grid(row=0, column=0)
            label_object1_AntennaTypeFrame_Height.grid(row=1, column=0)

            self.entry_object1_AntennaTypeFrame_Length.grid(row=0, column=1)
            self.entry_object1_AntennaTypeFrame_Height.grid(row=1, column=1)

        else:
            self.entry_object1_inductance.delete(0, END)
            self.entry_object1_inductance.insert(0, 'None')
            self.objectTypeFrameDestroy(self.object1TypeFrame_register, 'last',1)
            self.object1TypeAntennaMain.set(1)
            self.displayObject1TypeAntennaMain()
    def displayObject1TypeAntennaMain(self):
        if self.object1TypeAntennaMain.get() == 1:
            self.entry_object1_inductance.delete(0, END)
            self.entry_object1_inductance.insert(0, 527.3e-6)
            self.object1TypeAntennaFrame.set(0)
            self.objectTypeFrameDestroy(self.object1TypeFrame_register, 'last',1)
            self.object1_AntennaTypeMain_Frame = ttk.Frame(self.objectParametersTab)
            self.object1_AntennaTypeMain_Frame.place(x=0, y=int(220*self.scale_factor))
            self.object1TypeFrame_register.append(self.object1_AntennaTypeMain_Frame)

            label_object1_AntennaTypeMain_Length= Label(self.object1_AntennaTypeMain_Frame, text='Length (m): ', font=('Arial', self.paramFontSize))
            label_object1_AntennaTypeMain_Height = Label(self.object1_AntennaTypeMain_Frame, text='Height (m): ',
                                                         font=('Arial', self.paramFontSize))

            self.entry_object1_AntennaTypeMain_Length = Entry(self.object1_AntennaTypeMain_Frame, width=10)
            self.entry_object1_AntennaTypeMain_Height = Entry(self.object1_AntennaTypeMain_Frame, width=10)

            self.entry_object1_AntennaTypeMain_Length.insert(0, str(self.object1_AntennaTypeMain_Length) )
            self.entry_object1_AntennaTypeMain_Height.insert(0, str(self.object1_AntennaTypeMain_Height) )


            label_object1_AntennaTypeMain_Length.grid(row=0, column=0)
            label_object1_AntennaTypeMain_Height.grid(row=1, column=0)

            self.entry_object1_AntennaTypeMain_Length.grid(row=0, column=1)
            self.entry_object1_AntennaTypeMain_Height.grid(row=1, column=1)

        else:
            self.entry_object1_inductance.delete(0, END)
            self.entry_object1_inductance.insert(0, 'None')
            self.objectTypeFrameDestroy(self.object1TypeFrame_register, 'last',1)
            self.object1TypeAntennaFrame.set(1)
            self.displayObject1TypeAntennaFrame()
    def displayObject1TypePolygon(self):
        if self.object1TypePolygon.get() == 1:
            self.object1TypeAntenna.set(0)
            self.object1TypeEllipse.set(0)
            self.object1TypePuk.set(0)
            self.object1TypeBall.set(0)
            self.entry_object1_inductance.delete(0, END)
            self.entry_object1_inductance.insert(0, 7.20793033828e-6)
            ## Remove the previous object entries and labels from the GUI if any.
            self.objectTypeFrameDestroy(self.object1TypeFrame_register,'all',1)
            ## Create and place a frame that will contain the Polygon object labels and entries.
            self.object1FramePolygon = ttk.Frame(self.objectParametersTab)
            self.object1FramePolygon.place(x=0, y=int(60*self.scale_factor))
            ## This regesiter contains the current object type displayed frame.
            # This is important when the function objectParamFrameDestroy() is called.
            self.object1TypeFrame_register.append(self.object1FramePolygon)
            ######################################################################################################################
            ## create polygon parameters labels
            ######################################################################################################################

            label_Object1PolygonParam_NumberWindings = Label(self.object1FramePolygon, text="Number of windings: ",
                                                      font=("Arial", self.paramFontSize))
            label_Object1PolygonParam_Current = Label(self.object1FramePolygon, text="Current (A): ",
                                               font=("Arial", self.paramFontSize))
            label_Object1PolygonParam_Resistance = Label(self.object1FramePolygon, text="Resistance (Ohm): ",
                                               font=("Arial", self.paramFontSize))
            label_Object1PolygonParam_Frequency = Label(self.object1FramePolygon, text="Frequency (Hz): ",
                                                 font=("Arial", self.paramFontSize))
            label_Object1PolygonParam_PolygonOrigin = Label(self.object1FramePolygon, text="Position [x(m), y(m), z(m)]: ",
                                                     font=("Arial", self.paramFontSize))
            label_Object1PolygonParam_NumberOfPolygonCorners = Label(self.object1FramePolygon, text="Number of polygon corners: ",
                                                              font=("Arial", self.paramFontSize),
                                                              justify='left')
            label_Object1PolygonParam_Orientation = Label(self.object1FramePolygon,text="Polygon orientation (deg): ",font=("Arial", self.paramFontSize))
            label_Object1PolygonParam_OrientationAlpha = Label(self.object1FramePolygon, text=u'\u03b1',
                                                   font=("Arial", self.paramFontSize))
            label_Object1PolygonParam_OrientationBeta = Label(self.object1FramePolygon, text=u'\u03b2',
                                                   font=("Arial", self.paramFontSize))
            label_Object1PolygonParam_OrientationGamma = Label(self.object1FramePolygon, text=u'\u03b3',
                                                   font=("Arial", self.paramFontSize))
            self.object1_OpenExciter = IntVar()
            self.checkbox_object1_OpenExciter = ttk.Checkbutton(self.object1FramePolygon, text="Open last edge of the polygon", font=self.paramFontSize,variable=self.object1_OpenExciter)

            #######################################################################################################################################
            # PolygonParameters Dropdown list:
            #########################################################################################################################
            cornersChoices = {'', 2, 3, 4, 5, 6, 7, 8, 9, 10}
            self.Object1numberOfPolygonCorners = IntVar()
            self.Object1numberOfPolygonCorners.set(4)
            self.Object1numberOfPolygonCornersBuffer = 0

            Object1polygonParam_dropDownList = OptionMenu(self.object1FramePolygon, self.Object1numberOfPolygonCorners, *cornersChoices)
            Object1polygonParam_addPolygonCornersButton = ttk.Button(self.object1FramePolygon, text="Add", command=self.addPolygonCorners,
                                                        height = int(1 * self.scale_factor), width = int(6 * self.scale_factor))

            #######################################################################################################################################
            # PolygonParameters entries:
            #########################################################################################################################
            self.entry_Object1PolygonParam_NumberWindings = Entry(self.object1FramePolygon, width=5)
            self.entry_Object1PolygonParam_NumberWindings.insert(0, self.Object1_DefaultParameters_NumberOfWindings)
            self.entry_Object1PolygonParam_Current = Entry(self.object1FramePolygon, width=5)
            self.entry_Object1PolygonParam_Current.insert(0, str(self.Object1_DefaultParameters_Current))
            self.entry_Object1PolygonParam_Resistance = Entry(self.object1FramePolygon, width=5)
            self.entry_Object1PolygonParam_Resistance.insert(0, str(self.Object1_DefaultParameters_Resistance))
            self.entry_Object1PolygonParam_Frequency = Entry(self.object1FramePolygon, width=10)
            self.entry_Object1PolygonParam_Frequency.insert(0, str(self.Object1_DefaultParameters_Frequency))
            self.entry_Object1PolygonParam_PolygonOrigin = Entry(self.object1FramePolygon, width=10)
            self.entry_Object1PolygonParam_PolygonOrigin.insert(0, str(self.Object1_DefaultParameters_Origin))
            self.entry_Object1PolygonParam_OrientationAlpha= Entry(self.object1FramePolygon, width=5)
            self.entry_Object1PolygonParam_OrientationAlpha.insert(0, str(self.Object1_DefaultParameters_Alpha))
            self.entry_Object1PolygonParam_OrientationBeta= Entry(self.object1FramePolygon, width=5)
            self.entry_Object1PolygonParam_OrientationBeta.insert(0, str(self.Object1_DefaultParameters_Beta))
            self.entry_Object1PolygonParam_OrientationGamma= Entry(self.object1FramePolygon, width=5)
            self.entry_Object1PolygonParam_OrientationGamma.insert(0, str(self.Object1_DefaultParameters_Gamma))
            ########################################################################################################################
            ## Gridding PolygonParameters-labels and entries:
            ########################################################################################################################
            label_Object1PolygonParam_NumberWindings.grid(row=0, column=0,sticky='W')
            self.entry_Object1PolygonParam_NumberWindings.grid(row=0, column=2, sticky='W')
            label_Object1PolygonParam_Current.grid(row=1, column=0, sticky='W')
            self.entry_Object1PolygonParam_Current.grid(row=1, column=2, sticky='W')
            label_Object1PolygonParam_Resistance.grid(row=2, column=0, sticky='W')
            self.entry_Object1PolygonParam_Resistance.grid(row=2, column=2, sticky='W')
            label_Object1PolygonParam_Frequency.grid(row=3, column=0, sticky='W')
            self.entry_Object1PolygonParam_Frequency.grid(row=3, column=2, sticky='W')
            label_Object1PolygonParam_PolygonOrigin.grid(row=4, column=0, sticky='W')
            self.entry_Object1PolygonParam_PolygonOrigin.grid(row=4, column=2, sticky='W')
            label_Object1PolygonParam_Orientation.grid(row=5, column=0,sticky='W')
            label_Object1PolygonParam_OrientationAlpha.grid(row=5, column=1,sticky='E')
            self.entry_Object1PolygonParam_OrientationAlpha.grid(row=5, column=2,sticky='W')
            label_Object1PolygonParam_OrientationBeta.grid(row=5, column=3, sticky='W')
            self.entry_Object1PolygonParam_OrientationBeta.grid(row=5, column=4, sticky='W')
            label_Object1PolygonParam_OrientationGamma.grid(row=5, column=5, sticky='W')
            self.entry_Object1PolygonParam_OrientationGamma.grid(row=5, column=6, sticky='W')
            self.checkbox_object1_OpenExciter.grid(row=6, column=0, sticky='W')
            label_Object1PolygonParam_NumberOfPolygonCorners.grid(row=7, column=0,sticky='W')
            Object1polygonParam_dropDownList.grid(row=7, column=1,sticky='W')
            Object1polygonParam_addPolygonCornersButton.grid(row=7, column=2,sticky='W')



            ########################################################################################################################
            ## Initialize polygon corners:
            ########################################################################################################################

            self.addPolygonCorners()
        else:
            self.entry_object1_inductance.delete(0, END)
            self.entry_object1_inductance.insert(0, 'None')
            self.objectTypeFrameDestroy(self.object1TypeFrame_register,'all',1)
    def displayObject1TypeAntenna(self):
        if self.object1TypeAntenna.get() == 1:
            self.object1TypePolygon.set(0)
            self.object1TypeEllipse.set(0)
            self.object1TypePuk.set(0)
            self.object1TypeBall.set(0)
            ## Remove the previous object entries and labels from the GUI if any.
            self.objectTypeFrameDestroy(self.object1TypeFrame_register, 'all',1)
            ## Create and place a frame that will contain the Polygon object labels and entries.
            self.object1FrameAntenna = ttk.Frame(self.objectParametersTab)
            self.object1FrameAntenna.place(x=0, y=int(60*self.scale_factor))
            self.object1TypeFrame_register.append(self.object1FrameAntenna)
            ######################################################################################################################
            ## create antenna parameters labels
            ######################################################################################################################

            label_Object1AntennaParam_NumberWindings = Label(self.object1FrameAntenna, text="Number of windings: ",
                                                      font=("Arial", self.paramFontSize))
            label_Object1AntennaParam_Current = Label(self.object1FrameAntenna, text="Current (A): ",
                                               font=("Arial", self.paramFontSize))
            label_Object1AntennaParam_Resistance = Label(self.object1FrameAntenna, text="Resistance (Ohm): ",
                                                  font=("Arial", self.paramFontSize))
            label_Object1AntennaParam_Frequency = Label(self.object1FrameAntenna, text="Frequency (Hz): ",
                                                 font=("Arial", self.paramFontSize))
            label_Object1AntennaParam_AntennaPosition = Label(self.object1FrameAntenna, text="Position [x(m), y(m), z(m)]: ",
                                                     font=("Arial", self.paramFontSize))

            label_Object1AntennaParam_Orientation = Label(self.object1FrameAntenna, text="Antenna orientation (deg): ",
                                                   font=("Arial", self.paramFontSize))
            label_Object1AntennaParam_OrientationAlpha = Label(self.object1FrameAntenna, text=u'\u03b1',
                                                        font=("Arial", self.paramFontSize))
            label_Object1AntennaParam_OrientationBeta = Label(self.object1FrameAntenna, text=u'\u03b2',
                                                       font=("Arial", self.paramFontSize))
            label_Object1AntennaParam_OrientationGamma = Label(self.object1FrameAntenna, text=u'\u03b3',
                                                        font=("Arial", self.paramFontSize))
            label_Object1AntennaParam_Type = Label(self.object1FrameAntenna, text="Antenna type: ",
                                            font=("Arial", self.paramFontSize))
            self.object1TypeAntennaMain = IntVar()
            self.object1TypeAntennaMain.set(1)
            checkBox_Object1_TypeAntennaMain = Checkbutton(self.object1FrameAntenna, text="Main", variable=self.object1TypeAntennaMain,
                                                   command=self.displayObject1TypeAntennaMain)
            self.object1TypeAntennaFrame = IntVar()
            checkBox_Object1_TypeAntennaFrame = Checkbutton(self.object1FrameAntenna, text="Frame",
                                                           variable=self.object1TypeAntennaFrame,
                                                           command=self.displayObject1TypeAntennaFrame)
            self.displayObject1TypeAntennaMain()
            #######################################################################################################################################
            # Antenna Parameters entries:
            #########################################################################################################################
            self.entry_Object1AntennaParam_NumberWindings = Entry(self.object1FrameAntenna, width=5)
            self.entry_Object1AntennaParam_NumberWindings.insert(0, self.Object1_DefaultParameters_NumberOfWindings)
            self.entry_Object1AntennaParam_Current = Entry(self.object1FrameAntenna, width=5)
            self.entry_Object1AntennaParam_Current.insert(0, str(self.Object1_DefaultParameters_Current))
            self.entry_Object1AntennaParam_Resistance = Entry(self.object1FrameAntenna, width=5)
            self.entry_Object1AntennaParam_Resistance.insert(0, str(self.Object1_DefaultParameters_Resistance))
            self.entry_Object1AntennaParam_Frequency = Entry(self.object1FrameAntenna, width=10)
            self.entry_Object1AntennaParam_Frequency.insert(0, str(self.Object1_DefaultParameters_Frequency))
            self.entry_Object1AntennaParam_AntennaOrigin = Entry(self.object1FrameAntenna, width=10)
            self.entry_Object1AntennaParam_AntennaOrigin.insert(0, str(self.Object1_DefaultParameters_Origin))
            self.entry_Object1AntennaParam_OrientationAlpha = Entry(self.object1FrameAntenna, width=5)
            self.entry_Object1AntennaParam_OrientationAlpha.insert(0, str(self.Object1_DefaultParameters_Alpha))
            self.entry_Object1AntennaParam_OrientationBeta = Entry(self.object1FrameAntenna, width=5)
            self.entry_Object1AntennaParam_OrientationBeta.insert(0, str(self.Object1_DefaultParameters_Beta))
            self.entry_Object1AntennaParam_OrientationGamma = Entry(self.object1FrameAntenna, width=5)
            self.entry_Object1AntennaParam_OrientationGamma.insert(0, str(self.Object1_DefaultParameters_Gamma))
            ########################################################################################################################
            ## Gridding AntennaParameters-labels and entries:
            ########################################################################################################################
            label_Object1AntennaParam_NumberWindings.grid(row=0, column=0, sticky='W')
            label_Object1AntennaParam_Current.grid(row=1, column=0, sticky='W')
            label_Object1AntennaParam_Resistance.grid(row=2, column=0, sticky='W')
            label_Object1AntennaParam_Frequency.grid(row=3, column=0, sticky='W')
            label_Object1AntennaParam_AntennaPosition.grid(row=4, column=0, sticky='W')
            label_Object1AntennaParam_Orientation.grid(row=5, column=0, sticky='W')
            label_Object1AntennaParam_OrientationAlpha.grid(row=5, column=1, sticky='E')
            self.entry_Object1AntennaParam_OrientationAlpha.grid(row=5, column=2, sticky='W')
            label_Object1AntennaParam_OrientationBeta.grid(row=5, column=3, sticky='W')
            self.entry_Object1AntennaParam_OrientationBeta.grid(row=5, column=4, sticky='W')
            label_Object1AntennaParam_OrientationGamma.grid(row=5, column=5, sticky='W')
            self.entry_Object1AntennaParam_OrientationGamma.grid(row=5, column=6, sticky='W')


            self.entry_Object1AntennaParam_NumberWindings.grid(row=0, column=2, sticky='W')
            self.entry_Object1AntennaParam_Current.grid(row=1, column=2, sticky='W')
            self.entry_Object1AntennaParam_Resistance.grid(row=2, column=2, sticky='W')

            self.entry_Object1AntennaParam_Frequency.grid(row=3, column=2, sticky='W')
            self.entry_Object1AntennaParam_AntennaOrigin.grid(row=4, column=2, sticky='W')
            label_Object1AntennaParam_Type.grid(row=6, column=0, sticky='W')
            checkBox_Object1_TypeAntennaMain.grid(row=6, column=1, sticky='W')
            checkBox_Object1_TypeAntennaFrame.grid(row=6, column=2, sticky='W')

        else:
            self.entry_object1_inductance.delete(0, END)
            self.entry_object1_inductance.insert(0, 'None')
            self.objectTypeFrameDestroy(self.object1TypeFrame_register, 'all',1)
    def displayObject1TypeEllipse(self):
        if self.object1TypeEllipse.get() == 1:
            self.entry_object1_inductance.delete(0, END)
            self.entry_object1_inductance.insert(0, 'None')
            self.object1TypePolygon.set(0)
            self.object1TypeAntenna.set(0)
            self.object1TypePuk.set(0)
            self.object1TypeBall.set(0)
            ## Remove the previous object entries and labels from the GUI if any.
            self.objectTypeFrameDestroy(self.object1TypeFrame_register, 'all',1)
            ## Create and place a frame that will contain the Polygon object labels and entries.
            self.object1FrameEllipse = ttk.Frame(self.objectParametersTab)
            self.object1FrameEllipse.place(x=0, y=int(60*self.scale_factor))
            self.object1TypeFrame_register.append(self.object1FrameEllipse)
            ######################################################################################################################
            ## create antenna parameters labels
            ######################################################################################################################

            label_Object1EllipseParam_NumberWindings = Label(self.object1FrameEllipse, text="Number of windings: ",
                                                      font=("Arial", self.paramFontSize))
            label_Object1EllipseParam_Current = Label(self.object1FrameEllipse, text="Current (A): ",
                                               font=("Arial", self.paramFontSize))
            label_Object1EllipseParam_Resistance = Label(self.object1FrameEllipse, text="Resistance (Ohm): ",
                                                  font=("Arial", self.paramFontSize))
            label_Object1EllipseParam_Frequency = Label(self.object1FrameEllipse, text="Frequency (Hz): ",
                                                 font=("Arial", self.paramFontSize))
            label_Object1EllipseParam_EllipsePosition = Label(self.object1FrameEllipse, text="Position [x(m), y(m), z(m)]: ",
                                                       font=("Arial", self.paramFontSize))

            label_Object1EllipseParam_Orientation = Label(self.object1FrameEllipse, text="Ellipse orientation (deg): ",
                                                   font=("Arial", self.paramFontSize))
            label_Object1EllipseParam_OrientationAlpha = Label(self.object1FrameEllipse, text=u'\u03b1',
                                                        font=("Arial", self.paramFontSize))
            label_Object1EllipseParam_OrientationBeta = Label(self.object1FrameEllipse, text=u'\u03b2',
                                                       font=("Arial", self.paramFontSize))
            label_Object1EllipseParam_OrientationGamma = Label(self.object1FrameEllipse, text=u'\u03b3',
                                                        font=("Arial", self.paramFontSize))
            label_Object1EllipseParam_MajorAxis = Label(self.object1FrameEllipse, text="Major axis length (m): ",
                                                   font=("Arial", self.paramFontSize))
            label_Object1EllipseParam_MinorAxis = Label(self.object1FrameEllipse, text="Minor axis length (m): ",
                                                 font=("Arial", self.paramFontSize))


            #######################################################################################################################################
            # Ellipse Parameters entries:
            #########################################################################################################################
            self.entry_Object1EllipseParam_NumberWindings = Entry(self.object1FrameEllipse, width=5)
            self.entry_Object1EllipseParam_NumberWindings.insert(0, self.Object1_DefaultParameters_NumberOfWindings)
            self.entry_Object1EllipseParam_Current = Entry(self.object1FrameEllipse, width=5)
            self.entry_Object1EllipseParam_Current.insert(0, str(self.Object1_DefaultParameters_Current))
            self.entry_Object1EllipseParam_Resistance = Entry(self.object1FrameEllipse, width=5)
            self.entry_Object1EllipseParam_Resistance.insert(0, str(self.Object1_DefaultParameters_Resistance))
            self.entry_Object1EllipseParam_Frequency = Entry(self.object1FrameEllipse, width=10)
            self.entry_Object1EllipseParam_Frequency.insert(0, str(self.Object1_DefaultParameters_Frequency))
            self.entry_Object1EllipseParam_EllipseOrigin = Entry(self.object1FrameEllipse, width=10)
            self.entry_Object1EllipseParam_EllipseOrigin.insert(0, str(self.Object1_DefaultParameters_Origin))
            self.entry_Object1EllipseParam_OrientationAlpha = Entry(self.object1FrameEllipse, width=5)
            self.entry_Object1EllipseParam_OrientationAlpha.insert(0, str(self.Object1_DefaultParameters_Alpha))
            self.entry_Object1EllipseParam_OrientationBeta = Entry(self.object1FrameEllipse, width=5)
            self.entry_Object1EllipseParam_OrientationBeta.insert(0, str(self.Object1_DefaultParameters_Beta))
            self.entry_Object1EllipseParam_OrientationGamma = Entry(self.object1FrameEllipse, width=5)
            self.entry_Object1EllipseParam_OrientationGamma.insert(0, str(self.Object1_DefaultParameters_Gamma))
            self.entry_Object1EllipseParam_MajorAxis = Entry(self.object1FrameEllipse, width=5)
            self.entry_Object1EllipseParam_MajorAxis.insert(0, str(self.Object1_DefaultParameters_MajorAxis))
            self.entry_Object1EllipseParam_MinorAxis = Entry(self.object1FrameEllipse, width=5)
            self.entry_Object1EllipseParam_MinorAxis.insert(0, str(self.Object1_DefaultParameters_MinorAxis))
            ########################################################################################################################
            ## Gridding EllipseParameters-labels and entries:
            ########################################################################################################################
            label_Object1EllipseParam_NumberWindings.grid(row=0, column=0, sticky='W')
            label_Object1EllipseParam_Current.grid(row=1, column=0, sticky='W')
            label_Object1EllipseParam_Resistance.grid(row=2, column=0, sticky='W')
            label_Object1EllipseParam_Frequency.grid(row=3, column=0, sticky='W')
            label_Object1EllipseParam_EllipsePosition.grid(row=4, column=0, sticky='W')
            label_Object1EllipseParam_Orientation.grid(row=5, column=0, sticky='W')
            label_Object1EllipseParam_OrientationAlpha.grid(row=5, column=1, sticky='E')
            self.entry_Object1EllipseParam_OrientationAlpha.grid(row=5, column=2, sticky='W')
            label_Object1EllipseParam_OrientationBeta.grid(row=5, column=3, sticky='W')
            self.entry_Object1EllipseParam_OrientationBeta.grid(row=5, column=4, sticky='W')
            label_Object1EllipseParam_OrientationGamma.grid(row=5, column=5, sticky='W')
            self.entry_Object1EllipseParam_OrientationGamma.grid(row=5, column=6, sticky='W')

            self.entry_Object1EllipseParam_NumberWindings.grid(row=0, column=2, sticky='W')
            self.entry_Object1EllipseParam_Current.grid(row=1, column=2, sticky='W')
            self.entry_Object1EllipseParam_Resistance.grid(row=2, column=2, sticky='W')
            self.entry_Object1EllipseParam_Frequency.grid(row=3, column=2, sticky='W')
            self.entry_Object1EllipseParam_EllipseOrigin.grid(row=4, column=2, sticky='W')
            label_Object1EllipseParam_MajorAxis.grid(row=6, column=0, sticky='W')
            label_Object1EllipseParam_MinorAxis.grid(row=7, column=0, sticky='W')
            self.entry_Object1EllipseParam_MajorAxis.grid(row=6, column=1, sticky='W')
            self.entry_Object1EllipseParam_MinorAxis.grid(row=7, column=1, sticky='W')
        else:
            self.entry_object1_inductance.delete(0, END)
            self.entry_object1_inductance.insert(0, 'None')
            self.objectTypeFrameDestroy(self.object1TypeFrame_register, 'all',1)
    def displayObject1TypePuk(self):
        if self.object1TypePuk.get() == 1:
            self.entry_object1_inductance.delete(0, END)
            self.entry_object1_inductance.insert(0, 'None')
            self.object1TypePolygon.set(0)
            self.object1TypeAntenna.set(0)
            self.object1TypeEllipse.set(0)
            self.object1TypeBall.set(0)
            ## Remove the previous object entries and labels from the GUI if any.
            self.objectTypeFrameDestroy(self.object1TypeFrame_register, 'all',1)
            ## Create and place a frame that will contain the Polygon object labels and entries.
            self.object1FramePuk = ttk.Frame(self.objectParametersTab)
            self.object1FramePuk.place(x=0, y=int(60*self.scale_factor))
            self.object1TypeFrame_register.append(self.object1FramePuk)
            ######################################################################################################################
            ## create antenna parameters labels
            ######################################################################################################################

            label_Object1PukParam_NumberWindings = Label(self.object1FramePuk, text="Number of windings: ",
                                                      font=("Arial", self.paramFontSize))
            label_Object1PukParam_Current = Label(self.object1FramePuk, text="Current (A): ",
                                               font=("Arial", self.paramFontSize))
            label_Object1PukParam_Resistance = Label(self.object1FramePuk, text="Resistance (Ohm): ",
                                                  font=("Arial", self.paramFontSize))
            label_Object1PukParam_Frequency = Label(self.object1FramePuk, text="Frequency (Hz): ",
                                                 font=("Arial", self.paramFontSize))
            label_Object1PukParam_PukPosition = Label(self.object1FramePuk, text="Position [x(m), y(m), z(m)]: ",
                                                       font=("Arial", self.paramFontSize))

            label_Object1PukParam_Orientation = Label(self.object1FramePuk, text="Puk orientation (deg): ",
                                                   font=("Arial", self.paramFontSize))
            label_Object1PukParam_OrientationAlpha = Label(self.object1FramePuk, text=u'\u03b1',
                                                        font=("Arial", self.paramFontSize))
            label_Object1PukParam_OrientationBeta = Label(self.object1FramePuk, text=u'\u03b2',
                                                       font=("Arial", self.paramFontSize))
            label_Object1PukParam_OrientationGamma = Label(self.object1FramePuk, text=u'\u03b3',
                                                        font=("Arial", self.paramFontSize))
            label_Object1PukParam_PukHeight = Label(self.object1FramePuk, text="Puk height (m): ",
                                                   font=("Arial", self.paramFontSize))
            label_Object1PukParam_PukCircularCoilRadius = Label(self.object1FramePuk, text="Circular coil radius (m): ",
                                                 font=("Arial", self.paramFontSize))


            #######################################################################################################################################
            # Puk Parameters entries:
            #########################################################################################################################
            self.entry_Object1PukParam_NumberWindings = Entry(self.object1FramePuk, width=5)
            self.entry_Object1PukParam_NumberWindings.insert(0, self.Object1_DefaultParameters_NumberOfWindings)
            self.entry_Object1PukParam_Current = Entry(self.object1FramePuk, width=5)
            self.entry_Object1PukParam_Current.insert(0, str(self.Object1_DefaultParameters_Current))
            self.entry_Object1PukParam_Resistance = Entry(self.object1FramePuk, width=5)
            self.entry_Object1PukParam_Resistance.insert(0, str(self.Object1_DefaultParameters_Resistance))
            self.entry_Object1PukParam_Frequency = Entry(self.object1FramePuk, width=10)
            self.entry_Object1PukParam_Frequency.insert(0, str(self.Object1_DefaultParameters_Frequency))
            self.entry_Object1PukParam_PukOrigin = Entry(self.object1FramePuk, width=10)
            self.entry_Object1PukParam_PukOrigin.insert(0, str(self.Object1_DefaultParameters_Origin))
            self.entry_Object1PukParam_OrientationAlpha = Entry(self.object1FramePuk, width=5)
            self.entry_Object1PukParam_OrientationAlpha.insert(0, str(self.Object1_DefaultParameters_Alpha))
            self.entry_Object1PukParam_OrientationBeta = Entry(self.object1FramePuk, width=5)
            self.entry_Object1PukParam_OrientationBeta.insert(0, str(self.Object1_DefaultParameters_Beta))
            self.entry_Object1PukParam_OrientationGamma = Entry(self.object1FramePuk, width=5)
            self.entry_Object1PukParam_OrientationGamma.insert(0, str(self.Object1_DefaultParameters_Gamma))
            self.entry_Object1PukParam_PukHeight = Entry(self.object1FramePuk, width=5)
            self.entry_Object1PukParam_PukHeight.insert(0, str(self.Object1_DefaultParameters_PukHeight))
            self.entry_Object1PukParam_PukCircularCoilRadius = Entry(self.object1FramePuk, width=5)
            self.entry_Object1PukParam_PukCircularCoilRadius.insert(0, str(self.Object1_DefaultParameters_PukCircularCoilRadius))
            ########################################################################################################################
            ## Gridding PukParameters-labels and entries:
            ########################################################################################################################
            label_Object1PukParam_NumberWindings.grid(row=0, column=0, sticky='W')
            label_Object1PukParam_Current.grid(row=1, column=0, sticky='W')
            label_Object1PukParam_Resistance.grid(row=2, column=0, sticky='W')
            label_Object1PukParam_Frequency.grid(row=3, column=0, sticky='W')
            label_Object1PukParam_PukPosition.grid(row=4, column=0, sticky='W')
            label_Object1PukParam_Orientation.grid(row=5, column=0, sticky='W')
            label_Object1PukParam_OrientationAlpha.grid(row=5, column=1, sticky='E')
            self.entry_Object1PukParam_OrientationAlpha.grid(row=5, column=2, sticky='W')
            label_Object1PukParam_OrientationBeta.grid(row=5, column=3, sticky='W')
            self.entry_Object1PukParam_OrientationBeta.grid(row=5, column=4, sticky='W')
            label_Object1PukParam_OrientationGamma.grid(row=5, column=5, sticky='W')
            self.entry_Object1PukParam_OrientationGamma.grid(row=5, column=6, sticky='W')

            self.entry_Object1PukParam_NumberWindings.grid(row=0, column=2, sticky='W')
            self.entry_Object1PukParam_Current.grid(row=1, column=2, sticky='W')
            self.entry_Object1PukParam_Resistance.grid(row=2, column=2, sticky='W')
            self.entry_Object1PukParam_Frequency.grid(row=3, column=2, sticky='W')
            self.entry_Object1PukParam_PukOrigin.grid(row=4, column=2, sticky='W')
            label_Object1PukParam_PukHeight.grid(row=6, column=0, sticky='W')
            label_Object1PukParam_PukCircularCoilRadius.grid(row=7, column=0, sticky='W')
            self.entry_Object1PukParam_PukHeight.grid(row=6, column=1, sticky='W')
            self.entry_Object1PukParam_PukCircularCoilRadius.grid(row=7, column=1, sticky='W')
        else:
            self.entry_object1_inductance.delete(0, END)
            self.entry_object1_inductance.insert(0, 'None')
            self.objectTypeFrameDestroy(self.object1TypeFrame_register, 'all',1)
    def displayObject1TypeBall(self):
        if self.object1TypeBall.get() == 1:
            self.entry_object1_inductance.delete(0, END)
            self.entry_object1_inductance.insert(0, 'None')
            self.object1TypePolygon.set(0)
            self.object1TypeAntenna.set(0)
            self.object1TypeEllipse.set(0)
            self.object1TypePuk.set(0)
            ## Remove the previous object entries and labels from the GUI if any.
            self.objectTypeFrameDestroy(self.object1TypeFrame_register, 'all',1)
            ## Create and place a frame that will contain the Polygon object labels and entries.
            self.object1FrameBall = ttk.Frame(self.objectParametersTab)
            self.object1FrameBall.place(x=0, y=int(60*self.scale_factor))
            self.object1TypeFrame_register.append(self.object1FrameBall)
            ######################################################################################################################
            ## create antenna parameters labels
            ######################################################################################################################

            label_Object1BallParam_NumberWindings = Label(self.object1FrameBall, text="Number of windings: ",
                                                      font=("Arial", self.paramFontSize))
            label_Object1BallParam_Current = Label(self.object1FrameBall, text="Current (A): ",
                                               font=("Arial", self.paramFontSize))
            label_Object1BallParam_Resistance = Label(self.object1FrameBall, text="Resistance (Ohm): ",
                                                  font=("Arial", self.paramFontSize))
            label_Object1BallParam_Frequency = Label(self.object1FrameBall, text="Frequency (Hz): ",
                                                 font=("Arial", self.paramFontSize))
            label_Object1BallParam_BallPosition = Label(self.object1FrameBall, text="Position [x(m), y(m), z(m)]: ",
                                                       font=("Arial", self.paramFontSize))

            label_Object1BallParam_Orientation = Label(self.object1FrameBall, text="Ball orientation (deg): ",
                                                   font=("Arial", self.paramFontSize))
            label_Object1BallParam_OrientationAlpha = Label(self.object1FrameBall, text=u'\u03b1',
                                                        font=("Arial", self.paramFontSize))
            label_Object1BallParam_OrientationBeta = Label(self.object1FrameBall, text=u'\u03b2',
                                                       font=("Arial", self.paramFontSize))
            label_Object1BallParam_OrientationGamma = Label(self.object1FrameBall, text=u'\u03b3',
                                                        font=("Arial", self.paramFontSize))
            label_Object1BallParam_BallRadius = Label(self.object1FrameBall, text="Coil's radius (m): ",
                                                   font=("Arial", self.paramFontSize))



            #######################################################################################################################################
            # Ball Parameters entries:
            #########################################################################################################################
            self.entry_Object1BallParam_NumberWindings = Entry(self.object1FrameBall, width=5)
            self.entry_Object1BallParam_NumberWindings.insert(0, self.Object1_DefaultParameters_NumberOfWindings)
            self.entry_Object1BallParam_Current = Entry(self.object1FrameBall, width=5)
            self.entry_Object1BallParam_Current.insert(0, str(self.Object1_DefaultParameters_Current))
            self.entry_Object1BallParam_Resistance = Entry(self.object1FrameBall, width=5)
            self.entry_Object1BallParam_Resistance.insert(0, str(self.Object1_DefaultParameters_Resistance))
            self.entry_Object1BallParam_Frequency = Entry(self.object1FrameBall, width=10)
            self.entry_Object1BallParam_Frequency.insert(0, str(self.Object1_DefaultParameters_Frequency))
            self.entry_Object1BallParam_BallOrigin = Entry(self.object1FrameBall, width=10)
            self.entry_Object1BallParam_BallOrigin.insert(0, str(self.Object1_DefaultParameters_Origin))
            self.entry_Object1BallParam_OrientationAlpha = Entry(self.object1FrameBall, width=5)
            self.entry_Object1BallParam_OrientationAlpha.insert(0, str(self.Object1_DefaultParameters_Alpha))
            self.entry_Object1BallParam_OrientationBeta = Entry(self.object1FrameBall, width=5)
            self.entry_Object1BallParam_OrientationBeta.insert(0, str(self.Object1_DefaultParameters_Beta))
            self.entry_Object1BallParam_OrientationGamma = Entry(self.object1FrameBall, width=5)
            self.entry_Object1BallParam_OrientationGamma.insert(0, str(self.Object1_DefaultParameters_Gamma))
            self.entry_Object1BallParam_BallRadius = Entry(self.object1FrameBall, width=5)
            self.entry_Object1BallParam_BallRadius.insert(0, str(self.Object1_DefaultParameters_BallRadius))

            ########################################################################################################################
            ## Gridding BallParameters-labels and entries:
            ########################################################################################################################
            label_Object1BallParam_NumberWindings.grid(row=0, column=0, sticky='W')
            label_Object1BallParam_Current.grid(row=1, column=0, sticky='W')
            label_Object1BallParam_Resistance.grid(row=2, column=0, sticky='W')
            label_Object1BallParam_Frequency.grid(row=3, column=0, sticky='W')
            label_Object1BallParam_BallPosition.grid(row=4, column=0, sticky='W')
            label_Object1BallParam_Orientation.grid(row=5, column=0, sticky='W')
            label_Object1BallParam_OrientationAlpha.grid(row=5, column=1, sticky='E')
            self.entry_Object1BallParam_OrientationAlpha.grid(row=5, column=2, sticky='W')
            label_Object1BallParam_OrientationBeta.grid(row=5, column=3, sticky='W')
            self.entry_Object1BallParam_OrientationBeta.grid(row=5, column=4, sticky='W')
            label_Object1BallParam_OrientationGamma.grid(row=5, column=5, sticky='W')
            self.entry_Object1BallParam_OrientationGamma.grid(row=5, column=6, sticky='W')

            self.entry_Object1BallParam_NumberWindings.grid(row=0, column=2, sticky='W')
            self.entry_Object1BallParam_Current.grid(row=1, column=2, sticky='W')
            self.entry_Object1BallParam_Resistance.grid(row=2, column=2, sticky='W')
            self.entry_Object1BallParam_Frequency.grid(row=3, column=2, sticky='W')
            self.entry_Object1BallParam_BallOrigin.grid(row=4, column=2, sticky='W')
            label_Object1BallParam_BallRadius.grid(row=6, column=0, sticky='W')

            self.entry_Object1BallParam_BallRadius.grid(row=6, column=1, sticky='W')

        else:
            self.entry_object1_inductance.delete(0, END)
            self.entry_object1_inductance.insert(0, 'None')
            self.objectTypeFrameDestroy(self.object1TypeFrame_register, 'all',1)

    def addPolygonCorners2(self):
        if self.numberOfPolygonCornersBuffer2 != self.numberOfPolygonCorners2.get():
            self.objectTypeFrameDestroy(self.object2TypeFrame_register, 'last',2)

            polygonCornersFrame = ttk.Frame(self.objectParametersTab)
            polygonCornersFrame.place(x=int((self.screenWidth/2) -int(100*self.scale_factor)), y=int(250*self.scale_factor))
            self.object2TypeFrame_register.append(polygonCornersFrame)

            self.Object2polygonCornersEntriesDict = {}
            self.numberOfPolygonCornersBuffer2 = self.numberOfPolygonCorners2.get()
            self.registeredPolygonCorners2 = self.numberOfPolygonCorners2.get()
            for i in range(1,self.numberOfPolygonCorners2.get()+1):
                if i < 6:
                    self.Object2polygonCornersEntriesDict["Corner {0}".format(i)] = Label(polygonCornersFrame, text="Corner {0} [x(m),y(m),z(m)]: ".format(i),
                                                                      font=("Arial", self.paramFontSize-1),justify='left')
                    self.Object2polygonCornersEntriesDict["Corner {0}".format(i)].grid(row=i-1, column=0)

                    self.Object2polygonCornersEntriesDict["Corner {0} entry".format(i)] = Entry(polygonCornersFrame,width=12)
                    self.Object2polygonCornersEntriesDict["Corner {0} entry".format(i)].grid(row=i-1, column=1)
                    self.Object2polygonCornersEntriesDict["Corner {0} entry".format(i)].insert(0, str(self.polygonParam_PolygonPolygonCorners[i - 1]))
                else:
                    self.Object2polygonCornersEntriesDict["Corner {0}".format(i)] = Label(polygonCornersFrame,
                                                                                   text="Corner {0} [x(m),y(m),z(m)]: ".format(
                                                                                       i),
                                                                                   font=("Arial", self.paramFontSize-1),
                                                                                   justify='left')
                    self.Object2polygonCornersEntriesDict["Corner {0}".format(i)].grid(row=i-6, column=2)

                    self.Object2polygonCornersEntriesDict["Corner {0} entry".format(i)] = Entry(
                        polygonCornersFrame, width=12)
                    self.Object2polygonCornersEntriesDict["Corner {0} entry".format(i)].grid(row=i-6, column=3)
                    self.Object2polygonCornersEntriesDict["Corner {0} entry".format(i)].insert(0,
                                                                                        str(self.polygonParam_PolygonPolygonCorners[
                                                                                            i - 1]))
    def displayObject2TypeAntennaFrame(self):
        if self.object2TypeAntennaFrame.get() == 1:
            self.entry_object2_inductance.delete(0, END)
            self.entry_object2_inductance.insert(0, 495.9e-6)
            self.object2TypeAntennaMain.set(0)
            self.objectTypeFrameDestroy(self.object2TypeFrame_register, 'last',2)
            self.object2_AntennaTypeFrame_Frame = ttk.Frame(self.objectParametersTab)
            self.object2_AntennaTypeFrame_Frame.place(x=int((self.screenWidth/2) -int(100*self.scale_factor)), y=int(220*self.scale_factor))
            self.object2TypeFrame_register.append(self.object2_AntennaTypeFrame_Frame)

            label_object2_AntennaTypeFrame_Length= Label(self.object2_AntennaTypeFrame_Frame, text='Length (m): ', font=('Arial', self.paramFontSize))
            label_object2_AntennaTypeFrame_Height = Label(self.object2_AntennaTypeFrame_Frame, text='Height (m): ',
                                                         font=('Arial', self.paramFontSize))

            self.entry_object2_AntennaTypeFrame_Length = Entry(self.object2_AntennaTypeFrame_Frame, width=10)
            self.entry_object2_AntennaTypeFrame_Height = Entry(self.object2_AntennaTypeFrame_Frame, width=10)

            self.entry_object2_AntennaTypeFrame_Length.insert(0, str(self.object2_AntennaTypeFrame_Length) )
            self.entry_object2_AntennaTypeFrame_Height.insert(0, str(self.object2_AntennaTypeFrame_Height) )


            label_object2_AntennaTypeFrame_Length.grid(row=0, column=0)
            label_object2_AntennaTypeFrame_Height.grid(row=1, column=0)

            self.entry_object2_AntennaTypeFrame_Length.grid(row=0, column=1)
            self.entry_object2_AntennaTypeFrame_Height.grid(row=1, column=1)

        else:
            self.objectTypeFrameDestroy(self.object2TypeFrame_register, 'last',2)
            self.object2TypeAntennaMain.set(1)
            self.displayObject2TypeAntennaMain()
    def displayObject2TypeAntennaMain(self):
        if self.object2TypeAntennaMain.get() == 1:
            self.entry_object2_inductance.delete(0, END)
            self.entry_object2_inductance.insert(0, 527.3e-6)
            self.object2TypeAntennaFrame.set(0)
            self.objectTypeFrameDestroy(self.object2TypeFrame_register, 'last',2)
            self.object2_AntennaTypeMain_Frame = ttk.Frame(self.objectParametersTab)
            self.object2_AntennaTypeMain_Frame.place(x=int((self.screenWidth/2) -int(100*self.scale_factor)), y=int(220*self.scale_factor))
            self.object2TypeFrame_register.append(self.object2_AntennaTypeMain_Frame)

            label_object2_AntennaTypeMain_Length= Label(self.object2_AntennaTypeMain_Frame, text='Length (m): ', font=('Arial', self.paramFontSize))
            label_object2_AntennaTypeMain_Height = Label(self.object2_AntennaTypeMain_Frame, text='Height (m): ',
                                                         font=('Arial', self.paramFontSize))

            self.entry_object2_AntennaTypeMain_Length = Entry(self.object2_AntennaTypeMain_Frame, width=10)
            self.entry_object2_AntennaTypeMain_Height = Entry(self.object2_AntennaTypeMain_Frame, width=10)

            self.entry_object2_AntennaTypeMain_Length.insert(0, str(self.object2_AntennaTypeMain_Length) )
            self.entry_object2_AntennaTypeMain_Height.insert(0, str(self.object2_AntennaTypeMain_Height) )


            label_object2_AntennaTypeMain_Length.grid(row=0, column=0)
            label_object2_AntennaTypeMain_Height.grid(row=1, column=0)

            self.entry_object2_AntennaTypeMain_Length.grid(row=0, column=1)
            self.entry_object2_AntennaTypeMain_Height.grid(row=1, column=1)

        else:
            self.objectTypeFrameDestroy(self.object2TypeFrame_register, 'last',2)
            self.object2TypeAntennaFrame.set(1)
            self.displayObject2TypeAntennaFrame()
    def displayObject2TypePolygon(self):
        if self.object2TypePolygon.get() == 1:
            self.entry_object2_inductance.delete(0, END)
            self.entry_object2_inductance.insert(0, 7.20793033828e-6)
            self.object2TypeAntenna.set(0)
            self.object2TypeEllipse.set(0)
            self.object2TypePuk.set(0)
            self.object2TypeBall.set(0)
            ## Remove the previous object entries and labels from the GUI if any.
            self.objectTypeFrameDestroy(self.object2TypeFrame_register,'all',2)
            ## Create and place a frame that will contain the Polygon object labels and entries.
            self.object2FramePolygon = ttk.Frame(self.objectParametersTab)
            self.object2FramePolygon.place(x=int((self.screenWidth/2) -int(100*self.scale_factor)), y=int(60*self.scale_factor))
            ## This regesiter contains the current object type displayed frame.
            # This is important when the function objectParamFrameDestroy() is called.
            self.object2TypeFrame_register.append(self.object2FramePolygon)
            ######################################################################################################################
            ## create polygon parameters labels
            ######################################################################################################################

            label_PolygonParam_NumberWindings = Label(self.object2FramePolygon, text="Number of windings: ",
                                                      font=("Arial", self.paramFontSize))
            label_PolygonParam_Current = Label(self.object2FramePolygon, text="Current (A): ",
                                               font=("Arial", self.paramFontSize))
            label_PolygonParam_Resistance = Label(self.object2FramePolygon, text="Resistance (Ohm): ",
                                               font=("Arial", self.paramFontSize))
            label_PolygonParam_Frequency = Label(self.object2FramePolygon, text="Frequency (Hz): ",
                                                 font=("Arial", self.paramFontSize))
            label_PolygonParam_PolygonOrigin = Label(self.object2FramePolygon, text="Position [x(m), y(m), z(m)]: ",
                                                     font=("Arial", self.paramFontSize))
            label_PolygonParam_NumberOfPolygonCorners = Label(self.object2FramePolygon, text="Number of polygon corners: ",
                                                              font=("Arial", self.paramFontSize),
                                                              justify='left')
            label_PolygonParam_Orientation = Label(self.object2FramePolygon,text="Polygon orientation (deg): ",font=("Arial", self.paramFontSize))
            label_PolygonParam_OrientationAlpha = Label(self.object2FramePolygon, text=u'\u03b1',
                                                   font=("Arial", self.paramFontSize))
            label_PolygonParam_OrientationBeta = Label(self.object2FramePolygon, text=u'\u03b2',
                                                   font=("Arial", self.paramFontSize))
            label_PolygonParam_OrientationGamma = Label(self.object2FramePolygon, text=u'\u03b3',
                                                   font=("Arial", self.paramFontSize))

            self.object2_OpenExciter = IntVar()
            self.checkbox_object2_OpenExciter = ttk.Checkbutton(self.object2FramePolygon,
                                                                text="Open last edge of the polygon",
                                                                font=self.paramFontSize,
                                                                variable=self.object2_OpenExciter)
            #######################################################################################################################################
            # PolygonParameters Dropdown list:
            #########################################################################################################################
            cornersChoices = {'', 2, 3, 4, 5, 6, 7, 8, 9, 10}
            self.numberOfPolygonCorners2 = IntVar()
            self.numberOfPolygonCorners2.set(4)
            self.numberOfPolygonCornersBuffer2 = 0

            polygonParam_dropDownList = OptionMenu(self.object2FramePolygon, self.numberOfPolygonCorners2, *cornersChoices)
            polygonParam_addPolygonCornersButton = ttk.Button(self.object2FramePolygon, text="Add", command=self.addPolygonCorners2,
                                                         height = int(1 * self.scale_factor), width = int(6 * self.scale_factor))

            #######################################################################################################################################
            # PolygonParameters entries:
            #########################################################################################################################
            self.entry_Object2PolygonParam_NumberWindings = Entry(self.object2FramePolygon, width=5)
            self.entry_Object2PolygonParam_NumberWindings.insert(0, self.Object2_DefaultParameters_NumberOfWindings)
            self.entry_Object2PolygonParam_Current = Entry(self.object2FramePolygon, width=5)
            self.entry_Object2PolygonParam_Current.insert(0, str(self.Object2_DefaultParameters_Current))
            self.entry_Object2PolygonParam_Resistance = Entry(self.object2FramePolygon, width=5)
            self.entry_Object2PolygonParam_Resistance.insert(0, str(self.Object2_DefaultParameters_Resistance))
            self.entry_Object2PolygonParam_Frequency = Entry(self.object2FramePolygon, width=10)
            self.entry_Object2PolygonParam_Frequency.insert(0, str(self.Object2_DefaultParameters_Frequency))
            self.entry_Object2PolygonParam_PolygonOrigin = Entry(self.object2FramePolygon, width=10)
            self.entry_Object2PolygonParam_PolygonOrigin.insert(0, str(self.Object2_DefaultParameters_Origin))
            self.entry_Object2PolygonParam_OrientationAlpha= Entry(self.object2FramePolygon, width=5)
            self.entry_Object2PolygonParam_OrientationAlpha.insert(0, str(self.Object2_DefaultParameters_Alpha))
            self.entry_Object2PolygonParam_OrientationBeta= Entry(self.object2FramePolygon, width=5)
            self.entry_Object2PolygonParam_OrientationBeta.insert(0, str(self.Object2_DefaultParameters_Beta))
            self.entry_Object2PolygonParam_OrientationGamma= Entry(self.object2FramePolygon, width=5)
            self.entry_Object2PolygonParam_OrientationGamma.insert(0, str(self.Object2_DefaultParameters_Gamma))
            ########################################################################################################################
            ## Gridding PolygonParameters-labels and entries:
            ########################################################################################################################
            label_PolygonParam_NumberWindings.grid(row=0, column=0,sticky='W')
            label_PolygonParam_Current.grid(row=1, column=0,sticky='W')
            label_PolygonParam_Resistance.grid(row=2, column=0,sticky='W')
            label_PolygonParam_Frequency.grid(row=3, column=0,sticky='W')
            label_PolygonParam_PolygonOrigin.grid(row=4, column=0,sticky='W')
            label_PolygonParam_Orientation.grid(row=5, column=0,sticky='W')
            label_PolygonParam_OrientationAlpha.grid(row=5, column=1,sticky='E')
            self.entry_Object2PolygonParam_OrientationAlpha.grid(row=5, column=2,sticky='W')
            label_PolygonParam_OrientationBeta.grid(row=5, column=3, sticky='W')
            self.entry_Object2PolygonParam_OrientationBeta.grid(row=5, column=4, sticky='W')
            label_PolygonParam_OrientationGamma.grid(row=5, column=5, sticky='W')
            self.entry_Object2PolygonParam_OrientationGamma.grid(row=5, column=6, sticky='W')

            label_PolygonParam_NumberOfPolygonCorners.grid(row=7, column=0,sticky='W')
            polygonParam_dropDownList.grid(row=7, column=1,sticky='W')
            polygonParam_addPolygonCornersButton.grid(row=7, column=2,sticky='W')
            self.entry_Object2PolygonParam_NumberWindings.grid(row=0, column=2,sticky='W')
            self.entry_Object2PolygonParam_Current.grid(row=1, column=2,sticky='W')
            self.entry_Object2PolygonParam_Resistance.grid(row=2, column=2, sticky='W')

            self.entry_Object2PolygonParam_Frequency.grid(row=3, column=2,sticky='W')
            self.entry_Object2PolygonParam_PolygonOrigin.grid(row=4, column=2,sticky='W')
            self.checkbox_object2_OpenExciter.grid(row=6, column=0, sticky='W')
            ########################################################################################################################
            ## Initialize polygon corners:
            ########################################################################################################################

            self.addPolygonCorners2()
        else:
            self.entry_object2_inductance.delete(0, END)
            self.entry_object2_inductance.insert(0, 'None')
            self.objectTypeFrameDestroy(self.object2TypeFrame_register,'all',2)
    def displayObject2TypeAntenna(self):
        if self.object2TypeAntenna.get() == 1:
            self.object2TypePolygon.set(0)
            self.object2TypeEllipse.set(0)
            self.object2TypePuk.set(0)
            self.object2TypeBall.set(0)
            ## Remove the previous object entries and labels from the GUI if any.
            self.objectTypeFrameDestroy(self.object2TypeFrame_register, 'all',2)
            ## Create and place a frame that will contain the Polygon object labels and entries.
            self.object2FrameAntenna = ttk.Frame(self.objectParametersTab)
            self.object2FrameAntenna.place(x=int((self.screenWidth/2) -int(100*self.scale_factor)), y=int(60*self.scale_factor))
            self.object2TypeFrame_register.append(self.object2FrameAntenna)
            ######################################################################################################################
            ## create antenna parameters labels
            ######################################################################################################################

            label_AntennaParam_NumberWindings = Label(self.object2FrameAntenna, text="Number of windings: ",
                                                      font=("Arial", self.paramFontSize))
            label_AntennaParam_Current = Label(self.object2FrameAntenna, text="Current (A): ",
                                               font=("Arial", self.paramFontSize))
            label_AntennaParam_Resistance = Label(self.object2FrameAntenna, text="Resistance (Ohm): ",
                                                  font=("Arial", self.paramFontSize))
            label_AntennaParam_Frequency = Label(self.object2FrameAntenna, text="Frequency (Hz): ",
                                                 font=("Arial", self.paramFontSize))
            label_AntennaParam_AntennaPosition = Label(self.object2FrameAntenna, text="Position [x(m), y(m), z(m)]: ",
                                                     font=("Arial", self.paramFontSize))

            label_AntennaParam_Orientation = Label(self.object2FrameAntenna, text="Antenna orientation (deg): ",
                                                   font=("Arial", self.paramFontSize))
            label_AntennaParam_OrientationAlpha = Label(self.object2FrameAntenna, text=u'\u03b1',
                                                        font=("Arial", self.paramFontSize))
            label_AntennaParam_OrientationBeta = Label(self.object2FrameAntenna, text=u'\u03b2',
                                                       font=("Arial", self.paramFontSize))
            label_AntennaParam_OrientationGamma = Label(self.object2FrameAntenna, text=u'\u03b3',
                                                        font=("Arial", self.paramFontSize))
            label_AntennaParam_Type = Label(self.object2FrameAntenna, text="Antenna type: ",
                                            font=("Arial", self.paramFontSize))
            self.object2TypeAntennaMain = IntVar()
            self.object2TypeAntennaMain.set(1)
            checkBox_Object2_TypeAntennaMain = Checkbutton(self.object2FrameAntenna, text="Main", variable=self.object2TypeAntennaMain,
                                                   command=self.displayObject2TypeAntennaMain)

            self.object2TypeAntennaFrame = IntVar()
            checkBox_Object2_TypeAntennaFrame = Checkbutton(self.object2FrameAntenna, text="Frame",
                                                           variable=self.object2TypeAntennaFrame,
                                                           command=self.displayObject2TypeAntennaFrame)
            self.displayObject2TypeAntennaMain()
            #######################################################################################################################################
            # Antenna Parameters entries:
            #########################################################################################################################
            self.entry_Object2AntennaParam_NumberWindings = Entry(self.object2FrameAntenna, width=5)
            self.entry_Object2AntennaParam_NumberWindings.insert(0, self.Object2_DefaultParameters_NumberOfWindings)
            self.entry_Object2AntennaParam_Current = Entry(self.object2FrameAntenna, width=5)
            self.entry_Object2AntennaParam_Current.insert(0, str(self.Object2_DefaultParameters_Current))
            self.entry_Object2AntennaParam_Resistance = Entry(self.object2FrameAntenna, width=5)
            self.entry_Object2AntennaParam_Resistance.insert(0, str(self.Object2_DefaultParameters_Resistance))
            self.entry_Object2AntennaParam_Frequency = Entry(self.object2FrameAntenna, width=10)
            self.entry_Object2AntennaParam_Frequency.insert(0, str(self.Object2_DefaultParameters_Frequency))
            self.entry_Object2AntennaParam_AntennaOrigin = Entry(self.object2FrameAntenna, width=10)
            self.entry_Object2AntennaParam_AntennaOrigin.insert(0, str(self.Object2_DefaultParameters_Origin))
            self.entry_Object2AntennaParam_OrientationAlpha = Entry(self.object2FrameAntenna, width=5)
            self.entry_Object2AntennaParam_OrientationAlpha.insert(0, str(self.Object2_DefaultParameters_Alpha))
            self.entry_Object2AntennaParam_OrientationBeta = Entry(self.object2FrameAntenna, width=5)
            self.entry_Object2AntennaParam_OrientationBeta.insert(0, str(self.Object2_DefaultParameters_Beta))
            self.entry_Object2AntennaParam_OrientationGamma = Entry(self.object2FrameAntenna, width=5)
            self.entry_Object2AntennaParam_OrientationGamma.insert(0, str(self.Object2_DefaultParameters_Gamma))
            ########################################################################################################################
            ## Gridding AntennaParameters-labels and entries:
            ########################################################################################################################
            label_AntennaParam_NumberWindings.grid(row=0, column=0, sticky='W')
            label_AntennaParam_Current.grid(row=1, column=0, sticky='W')
            label_AntennaParam_Resistance.grid(row=2, column=0, sticky='W')
            label_AntennaParam_Frequency.grid(row=3, column=0, sticky='W')
            label_AntennaParam_AntennaPosition.grid(row=4, column=0, sticky='W')
            label_AntennaParam_Orientation.grid(row=5, column=0, sticky='W')
            label_AntennaParam_OrientationAlpha.grid(row=5, column=1, sticky='E')
            self.entry_Object2AntennaParam_OrientationAlpha.grid(row=5, column=2, sticky='W')
            label_AntennaParam_OrientationBeta.grid(row=5, column=3, sticky='W')
            self.entry_Object2AntennaParam_OrientationBeta.grid(row=5, column=4, sticky='W')
            label_AntennaParam_OrientationGamma.grid(row=5, column=5, sticky='W')
            self.entry_Object2AntennaParam_OrientationGamma.grid(row=5, column=6, sticky='W')


            self.entry_Object2AntennaParam_NumberWindings.grid(row=0, column=2, sticky='W')
            self.entry_Object2AntennaParam_Current.grid(row=1, column=2, sticky='W')
            self.entry_Object2AntennaParam_Resistance.grid(row=2, column=2, sticky='W')

            self.entry_Object2AntennaParam_Frequency.grid(row=3, column=2, sticky='W')
            self.entry_Object2AntennaParam_AntennaOrigin.grid(row=4, column=2, sticky='W')
            label_AntennaParam_Type.grid(row=6, column=0, sticky='W')
            checkBox_Object2_TypeAntennaMain.grid(row=6, column=1, sticky='W')
            checkBox_Object2_TypeAntennaFrame.grid(row=6, column=2, sticky='W')

        else:
            self.entry_object2_inductance.delete(0, END)
            self.entry_object2_inductance.insert(0, 'None')
            self.objectTypeFrameDestroy(self.object2TypeFrame_register, 'all',2)
    def displayObject2TypeEllipse(self):
        if self.object2TypeEllipse.get() == 1:
            self.entry_object2_inductance.delete(0, END)
            self.entry_object2_inductance.insert(0, 'None')
            self.object2TypePolygon.set(0)
            self.object2TypeAntenna.set(0)
            self.object2TypePuk.set(0)
            self.object2TypeBall.set(0)
            ## Remove the previous object entries and labels from the GUI if any.
            self.objectTypeFrameDestroy(self.object2TypeFrame_register, 'all',2)
            ## Create and place a frame that will contain the Polygon object labels and entries.
            self.object2FrameEllipse = ttk.Frame(self.objectParametersTab)
            self.object2FrameEllipse.place(x=int((self.screenWidth/2) -int(100*self.scale_factor)), y=int(60*self.scale_factor))
            self.object2TypeFrame_register.append(self.object2FrameEllipse)
            ######################################################################################################################
            ## create antenna parameters labels
            ######################################################################################################################

            label_EllipseParam_NumberWindings = Label(self.object2FrameEllipse, text="Number of windings: ",
                                                      font=("Arial", self.paramFontSize))
            label_EllipseParam_Current = Label(self.object2FrameEllipse, text="Current (A): ",
                                               font=("Arial", self.paramFontSize))
            label_EllipseParam_Resistance = Label(self.object2FrameEllipse, text="Resistance (Ohm): ",
                                                  font=("Arial", self.paramFontSize))
            label_EllipseParam_Frequency = Label(self.object2FrameEllipse, text="Frequency (Hz): ",
                                                 font=("Arial", self.paramFontSize))
            label_EllipseParam_EllipsePosition = Label(self.object2FrameEllipse, text="Position [x(m), y(m), z(m)]: ",
                                                       font=("Arial", self.paramFontSize))

            label_EllipseParam_Orientation = Label(self.object2FrameEllipse, text="Ellipse orientation (deg): ",
                                                   font=("Arial", self.paramFontSize))
            label_EllipseParam_OrientationAlpha = Label(self.object2FrameEllipse, text=u'\u03b1',
                                                        font=("Arial", self.paramFontSize))
            label_EllipseParam_OrientationBeta = Label(self.object2FrameEllipse, text=u'\u03b2',
                                                       font=("Arial", self.paramFontSize))
            label_EllipseParam_OrientationGamma = Label(self.object2FrameEllipse, text=u'\u03b3',
                                                        font=("Arial", self.paramFontSize))
            label_EllipseParam_MajorAxis = Label(self.object2FrameEllipse, text="Major axis length (m): ",
                                                   font=("Arial", self.paramFontSize))
            label_EllipseParam_MinorAxis = Label(self.object2FrameEllipse, text="Minor axis length (m): ",
                                                 font=("Arial", self.paramFontSize))


            #######################################################################################################################################
            # Ellipse Parameters entries:
            #########################################################################################################################
            self.entry_Object2EllipseParam_NumberWindings = Entry(self.object2FrameEllipse, width=5)
            self.entry_Object2EllipseParam_NumberWindings.insert(0, self.Object2_DefaultParameters_NumberOfWindings)
            self.entry_Object2EllipseParam_Current = Entry(self.object2FrameEllipse, width=5)
            self.entry_Object2EllipseParam_Current.insert(0, str(self.Object2_DefaultParameters_Current))
            self.entry_Object2EllipseParam_Resistance = Entry(self.object2FrameEllipse, width=5)
            self.entry_Object2EllipseParam_Resistance.insert(0, str(self.Object2_DefaultParameters_Resistance))
            self.entry_Object2EllipseParam_Frequency = Entry(self.object2FrameEllipse, width=10)
            self.entry_Object2EllipseParam_Frequency.insert(0, str(self.Object2_DefaultParameters_Frequency))
            self.entry_Object2EllipseParam_EllipseOrigin = Entry(self.object2FrameEllipse, width=10)
            self.entry_Object2EllipseParam_EllipseOrigin.insert(0, str(self.Object2_DefaultParameters_Origin))
            self.entry_Object2EllipseParam_OrientationAlpha = Entry(self.object2FrameEllipse, width=5)
            self.entry_Object2EllipseParam_OrientationAlpha.insert(0, str(self.Object2_DefaultParameters_Alpha))
            self.entry_Object2EllipseParam_OrientationBeta = Entry(self.object2FrameEllipse, width=5)
            self.entry_Object2EllipseParam_OrientationBeta.insert(0, str(self.Object2_DefaultParameters_Beta))
            self.entry_Object2EllipseParam_OrientationGamma = Entry(self.object2FrameEllipse, width=5)
            self.entry_Object2EllipseParam_OrientationGamma.insert(0, str(self.Object2_DefaultParameters_Gamma))
            self.entry_Object2EllipseParam_MajorAxis = Entry(self.object2FrameEllipse, width=5)
            self.entry_Object2EllipseParam_MajorAxis.insert(0, str(self.Object2_DefaultParameters_MajorAxis))
            self.entry_Object2EllipseParam_MinorAxis = Entry(self.object2FrameEllipse, width=5)
            self.entry_Object2EllipseParam_MinorAxis.insert(0, str(self.Object2_DefaultParameters_MinorAxis))
            ########################################################################################################################
            ## Gridding EllipseParameters-labels and entries:
            ########################################################################################################################
            label_EllipseParam_NumberWindings.grid(row=0, column=0, sticky='W')
            label_EllipseParam_Current.grid(row=1, column=0, sticky='W')
            label_EllipseParam_Resistance.grid(row=2, column=0, sticky='W')
            label_EllipseParam_Frequency.grid(row=3, column=0, sticky='W')
            label_EllipseParam_EllipsePosition.grid(row=4, column=0, sticky='W')
            label_EllipseParam_Orientation.grid(row=5, column=0, sticky='W')
            label_EllipseParam_OrientationAlpha.grid(row=5, column=1, sticky='E')
            self.entry_Object2EllipseParam_OrientationAlpha.grid(row=5, column=2, sticky='W')
            label_EllipseParam_OrientationBeta.grid(row=5, column=3, sticky='W')
            self.entry_Object2EllipseParam_OrientationBeta.grid(row=5, column=4, sticky='W')
            label_EllipseParam_OrientationGamma.grid(row=5, column=5, sticky='W')
            self.entry_Object2EllipseParam_OrientationGamma.grid(row=5, column=6, sticky='W')

            self.entry_Object2EllipseParam_NumberWindings.grid(row=0, column=2, sticky='W')
            self.entry_Object2EllipseParam_Current.grid(row=1, column=2, sticky='W')
            self.entry_Object2EllipseParam_Resistance.grid(row=2, column=2, sticky='W')
            self.entry_Object2EllipseParam_Frequency.grid(row=3, column=2, sticky='W')
            self.entry_Object2EllipseParam_EllipseOrigin.grid(row=4, column=2, sticky='W')
            label_EllipseParam_MajorAxis.grid(row=6, column=0, sticky='W')
            label_EllipseParam_MinorAxis.grid(row=7, column=0, sticky='W')
            self.entry_Object2EllipseParam_MajorAxis.grid(row=6, column=1, sticky='W')
            self.entry_Object2EllipseParam_MinorAxis.grid(row=7, column=1, sticky='W')
        else:
            self.entry_object2_inductance.delete(0, END)
            self.entry_object2_inductance.insert(0, 'None')
            self.objectTypeFrameDestroy(self.object2TypeFrame_register, 'all',2)
    def displayObject2TypePuk(self):
        if self.object2TypePuk.get() == 1:
            self.entry_object2_inductance.delete(0, END)
            self.entry_object2_inductance.insert(0, 'None')
            self.object2TypePolygon.set(0)
            self.object2TypeAntenna.set(0)
            self.object2TypeEllipse.set(0)
            self.object2TypeBall.set(0)
            ## Remove the previous object entries and labels from the GUI if any.
            self.objectTypeFrameDestroy(self.object2TypeFrame_register, 'all',2)
            ## Create and place a frame that will contain the Polygon object labels and entries.
            self.object2FramePuk = ttk.Frame(self.objectParametersTab)
            self.object2FramePuk.place(x=int((self.screenWidth/2) -int(100*self.scale_factor)), y=int(60*self.scale_factor))
            self.object2TypeFrame_register.append(self.object2FramePuk)
            ######################################################################################################################
            ## create antenna parameters labels
            ######################################################################################################################

            label_PukParam_NumberWindings = Label(self.object2FramePuk, text="Number of windings: ",
                                                      font=("Arial", self.paramFontSize))
            label_PukParam_Current = Label(self.object2FramePuk, text="Current (A): ",
                                               font=("Arial", self.paramFontSize))
            label_PukParam_Resistance = Label(self.object2FramePuk, text="Resistance (Ohm): ",
                                                  font=("Arial", self.paramFontSize))
            label_PukParam_Frequency = Label(self.object2FramePuk, text="Frequency (Hz): ",
                                                 font=("Arial", self.paramFontSize))
            label_PukParam_PukPosition = Label(self.object2FramePuk, text="Position [x(m), y(m), z(m)]: ",
                                                       font=("Arial", self.paramFontSize))

            label_PukParam_Orientation = Label(self.object2FramePuk, text="Puk orientation (deg): ",
                                                   font=("Arial", self.paramFontSize))
            label_PukParam_OrientationAlpha = Label(self.object2FramePuk, text=u'\u03b1',
                                                        font=("Arial", self.paramFontSize))
            label_PukParam_OrientationBeta = Label(self.object2FramePuk, text=u'\u03b2',
                                                       font=("Arial", self.paramFontSize))
            label_PukParam_OrientationGamma = Label(self.object2FramePuk, text=u'\u03b3',
                                                        font=("Arial", self.paramFontSize))
            label_PukParam_PukHeight = Label(self.object2FramePuk, text="Puk height (m): ",
                                                   font=("Arial", self.paramFontSize))
            label_PukParam_PukCircularCoilRadius = Label(self.object2FramePuk, text="Circular coil radius (m): ",
                                                 font=("Arial", self.paramFontSize))


            #######################################################################################################################################
            # Puk Parameters entries:
            #########################################################################################################################
            self.entry_Object2PukParam_NumberWindings = Entry(self.object2FramePuk, width=5)
            self.entry_Object2PukParam_NumberWindings.insert(0, self.Object2_DefaultParameters_NumberOfWindings)
            self.entry_Object2PukParam_Current = Entry(self.object2FramePuk, width=5)
            self.entry_Object2PukParam_Current.insert(0, str(self.Object2_DefaultParameters_Current))
            self.entry_Object2PukParam_Resistance = Entry(self.object2FramePuk, width=5)
            self.entry_Object2PukParam_Resistance.insert(0, str(self.Object2_DefaultParameters_Resistance))
            self.entry_Object2PukParam_Frequency = Entry(self.object2FramePuk, width=10)
            self.entry_Object2PukParam_Frequency.insert(0, str(self.Object2_DefaultParameters_Frequency))
            self.entry_Object2PukParam_PukOrigin = Entry(self.object2FramePuk, width=10)
            self.entry_Object2PukParam_PukOrigin.insert(0, str(self.Object2_DefaultParameters_Origin))
            self.entry_Object2PukParam_OrientationAlpha = Entry(self.object2FramePuk, width=5)
            self.entry_Object2PukParam_OrientationAlpha.insert(0, str(self.Object2_DefaultParameters_Alpha))
            self.entry_Object2PukParam_OrientationBeta = Entry(self.object2FramePuk, width=5)
            self.entry_Object2PukParam_OrientationBeta.insert(0, str(self.Object2_DefaultParameters_Beta))
            self.entry_Object2PukParam_OrientationGamma = Entry(self.object2FramePuk, width=5)
            self.entry_Object2PukParam_OrientationGamma.insert(0, str(self.Object2_DefaultParameters_Gamma))
            self.entry_Object2PukParam_PukHeight = Entry(self.object2FramePuk, width=5)
            self.entry_Object2PukParam_PukHeight.insert(0, str(self.Object2_DefaultParameters_PukHeight))
            self.entry_Object2PukParam_PukCircularCoilRadius = Entry(self.object2FramePuk, width=5)
            self.entry_Object2PukParam_PukCircularCoilRadius.insert(0, str(self.Object2_DefaultParameters_PukCircularCoilRadius))
            ########################################################################################################################
            ## Gridding PukParameters-labels and entries:
            ########################################################################################################################
            label_PukParam_NumberWindings.grid(row=0, column=0, sticky='W')
            label_PukParam_Current.grid(row=1, column=0, sticky='W')
            label_PukParam_Resistance.grid(row=2, column=0, sticky='W')
            label_PukParam_Frequency.grid(row=3, column=0, sticky='W')
            label_PukParam_PukPosition.grid(row=4, column=0, sticky='W')
            label_PukParam_Orientation.grid(row=5, column=0, sticky='W')
            label_PukParam_OrientationAlpha.grid(row=5, column=1, sticky='E')
            self.entry_Object2PukParam_OrientationAlpha.grid(row=5, column=2, sticky='W')
            label_PukParam_OrientationBeta.grid(row=5, column=3, sticky='W')
            self.entry_Object2PukParam_OrientationBeta.grid(row=5, column=4, sticky='W')
            label_PukParam_OrientationGamma.grid(row=5, column=5, sticky='W')
            self.entry_Object2PukParam_OrientationGamma.grid(row=5, column=6, sticky='W')

            self.entry_Object2PukParam_NumberWindings.grid(row=0, column=2, sticky='W')
            self.entry_Object2PukParam_Current.grid(row=1, column=2, sticky='W')
            self.entry_Object2PukParam_Resistance.grid(row=2, column=2, sticky='W')
            self.entry_Object2PukParam_Frequency.grid(row=3, column=2, sticky='W')
            self.entry_Object2PukParam_PukOrigin.grid(row=4, column=2, sticky='W')
            label_PukParam_PukHeight.grid(row=6, column=0, sticky='W')
            label_PukParam_PukCircularCoilRadius.grid(row=7, column=0, sticky='W')
            self.entry_Object2PukParam_PukHeight.grid(row=6, column=1, sticky='W')
            self.entry_Object2PukParam_PukCircularCoilRadius.grid(row=7, column=1, sticky='W')
        else:
            self.entry_object2_inductance.delete(0, END)
            self.entry_object2_inductance.insert(0, 'None')
            self.objectTypeFrameDestroy(self.object2TypeFrame_register, 'all',2)
    def displayObject2TypeBall(self):
        if self.object2TypeBall.get() == 1:
            self.entry_object2_inductance.delete(0, END)
            self.entry_object2_inductance.insert(0, 'None')
            self.object2TypePolygon.set(0)
            self.object2TypeAntenna.set(0)
            self.object2TypeEllipse.set(0)
            self.object2TypePuk.set(0)
            ## Remove the previous object entries and labels from the GUI if any.
            self.objectTypeFrameDestroy(self.object2TypeFrame_register, 'all',2)
            ## Create and place a frame that will contain the Polygon object labels and entries.
            self.object2FrameBall = ttk.Frame(self.objectParametersTab)
            self.object2FrameBall.place(x=int((self.screenWidth/2) -int(100*self.scale_factor)), y=int(60*self.scale_factor))
            self.object2TypeFrame_register.append(self.object2FrameBall)
            ######################################################################################################################
            ## create antenna parameters labels
            ######################################################################################################################

            label_BallParam_NumberWindings = Label(self.object2FrameBall, text="Number of windings: ",
                                                      font=("Arial", self.paramFontSize))
            label_BallParam_Current = Label(self.object2FrameBall, text="Current (A): ",
                                               font=("Arial", self.paramFontSize))
            label_BallParam_Resistance = Label(self.object2FrameBall, text="Resistance (Ohm): ",
                                                  font=("Arial", self.paramFontSize))
            label_BallParam_Frequency = Label(self.object2FrameBall, text="Frequency (Hz): ",
                                                 font=("Arial", self.paramFontSize))
            label_BallParam_BallPosition = Label(self.object2FrameBall, text="Position [x(m), y(m), z(m)]: ",
                                                       font=("Arial", self.paramFontSize))

            label_BallParam_Orientation = Label(self.object2FrameBall, text="Ball orientation (deg): ",
                                                   font=("Arial", self.paramFontSize))
            label_BallParam_OrientationAlpha = Label(self.object2FrameBall, text=u'\u03b1',
                                                        font=("Arial", self.paramFontSize))
            label_BallParam_OrientationBeta = Label(self.object2FrameBall, text=u'\u03b2',
                                                       font=("Arial", self.paramFontSize))
            label_BallParam_OrientationGamma = Label(self.object2FrameBall, text=u'\u03b3',
                                                        font=("Arial", self.paramFontSize))
            label_BallParam_BallRadius = Label(self.object2FrameBall, text="Coil's radius (m): ",
                                                   font=("Arial", self.paramFontSize))



            #######################################################################################################################################
            # Ball Parameters entries:
            #########################################################################################################################
            self.entry_Object2BallParam_NumberWindings = Entry(self.object2FrameBall, width=5)
            self.entry_Object2BallParam_NumberWindings.insert(0, self.Object2_DefaultParameters_NumberOfWindings)
            self.entry_Object2BallParam_Current = Entry(self.object2FrameBall, width=5)
            self.entry_Object2BallParam_Current.insert(0, str(self.Object2_DefaultParameters_Current))
            self.entry_Object2BallParam_Resistance = Entry(self.object2FrameBall, width=5)
            self.entry_Object2BallParam_Resistance.insert(0, str(self.Object2_DefaultParameters_Resistance))
            self.entry_Object2BallParam_Frequency = Entry(self.object2FrameBall, width=10)
            self.entry_Object2BallParam_Frequency.insert(0, str(self.Object2_DefaultParameters_Frequency))
            self.entry_Object2BallParam_BallOrigin = Entry(self.object2FrameBall, width=10)
            self.entry_Object2BallParam_BallOrigin.insert(0, str(self.Object2_DefaultParameters_Origin))
            self.entry_Object2BallParam_OrientationAlpha = Entry(self.object2FrameBall, width=5)
            self.entry_Object2BallParam_OrientationAlpha.insert(0, str(self.Object2_DefaultParameters_Alpha))
            self.entry_Object2BallParam_OrientationBeta = Entry(self.object2FrameBall, width=5)
            self.entry_Object2BallParam_OrientationBeta.insert(0, str(self.Object2_DefaultParameters_Beta))
            self.entry_Object2BallParam_OrientationGamma = Entry(self.object2FrameBall, width=5)
            self.entry_Object2BallParam_OrientationGamma.insert(0, str(self.Object2_DefaultParameters_Gamma))
            self.entry_Object2BallParam_BallRadius = Entry(self.object2FrameBall, width=5)
            self.entry_Object2BallParam_BallRadius.insert(0, str(self.Object2_DefaultParameters_BallRadius))

            ########################################################################################################################
            ## Gridding BallParameters-labels and entries:
            ########################################################################################################################
            label_BallParam_NumberWindings.grid(row=0, column=0, sticky='W')
            label_BallParam_Current.grid(row=1, column=0, sticky='W')
            label_BallParam_Resistance.grid(row=2, column=0, sticky='W')
            label_BallParam_Frequency.grid(row=3, column=0, sticky='W')
            label_BallParam_BallPosition.grid(row=4, column=0, sticky='W')
            label_BallParam_Orientation.grid(row=5, column=0, sticky='W')
            label_BallParam_OrientationAlpha.grid(row=5, column=1, sticky='E')
            self.entry_Object2BallParam_OrientationAlpha.grid(row=5, column=2, sticky='W')
            label_BallParam_OrientationBeta.grid(row=5, column=3, sticky='W')
            self.entry_Object2BallParam_OrientationBeta.grid(row=5, column=4, sticky='W')
            label_BallParam_OrientationGamma.grid(row=5, column=5, sticky='W')
            self.entry_Object2BallParam_OrientationGamma.grid(row=5, column=6, sticky='W')

            self.entry_Object2BallParam_NumberWindings.grid(row=0, column=2, sticky='W')
            self.entry_Object2BallParam_Current.grid(row=1, column=2, sticky='W')
            self.entry_Object2BallParam_Resistance.grid(row=2, column=2, sticky='W')
            self.entry_Object2BallParam_Frequency.grid(row=3, column=2, sticky='W')
            self.entry_Object2BallParam_BallOrigin.grid(row=4, column=2, sticky='W')
            label_BallParam_BallRadius.grid(row=6, column=0, sticky='W')

            self.entry_Object2BallParam_BallRadius.grid(row=6, column=1, sticky='W')

        else:
            self.entry_object2_inductance.delete(0, END)
            self.entry_object2_inductance.insert(0, 'None')
            self.objectTypeFrameDestroy(self.object2TypeFrame_register, 'all',2)

    def generateConfigFile(self, fileName):
        with open(fileName, 'w') as config:
            config.write('import numpy as np' +'\n')
            config.write('import math' + '\n')
            year, month, day, hour, minute = time.localtime()[:5]
            config.write('Date_Time = ' + "'"+'%s.%s.%s_%s:%s' % (day, month, year, hour, minute) +"'"'\n')
            #config.write('Author = ' + "'"+str(self.entry_TableParam_Author.get())+"'" + '\n')
            config.write('App = ' + "'" + 'Coupling factor' + "'" + '\n')

            if self.object1TypePolygon.get()==1:
                config.write('object1Type = ' + "'" + 'Polygon' + "'" + '\n')
                config.write('object1Windings = ' + str(self.entry_Object1PolygonParam_NumberWindings.get()) + '\n')
                config.write('object1Current = ' + str(self.entry_Object1PolygonParam_Current.get()) + '\n')
                config.write('object1Resistance= ' + str(self.entry_Object1PolygonParam_Resistance.get()) + '\n')
                config.write('object1Frequency = ' + str(self.entry_Object1PolygonParam_Frequency.get()) + '\n')
                config.write('object1Position = ' + str(self.entry_Object1PolygonParam_PolygonOrigin.get()) + '\n')
                config.write('object1OrientationAlpha = ' + str(self.entry_Object1PolygonParam_OrientationAlpha.get()) + '\n')
                config.write('object1OrientationBeta = ' + str(self.entry_Object1PolygonParam_OrientationBeta.get()) + '\n')
                config.write('object1OpenExciter = ' + str(self.object1_OpenExciter.get()) + '\n')
                config.write('object1OrientationGamma = ' + str(self.entry_Object1PolygonParam_OrientationGamma.get()) + '\n')
                config.write('object1RegisteredPolygonCorners = ' + str(self.registeredPolygonCorners) + '\n')
                for i in range(1, self.registeredPolygonCorners + 1):
                    config.write('object1E{0} = '.format(i) + self.Object1polygonCornersEntriesDict[
                        "Corner {0} entry".format(i)].get() + '\n')
            elif self.object1TypeAntenna.get()==1:
                config.write('object1Type = ' + "'" + 'Antenna' + "'" + '\n')
                config.write('object1Windings = ' + str(self.entry_Object1AntennaParam_NumberWindings.get()) + '\n')
                config.write('object1Current = ' + str(self.entry_Object1AntennaParam_Current.get()) + '\n')
                config.write('object1Resistance= ' + str(self.entry_Object1AntennaParam_Resistance.get()) + '\n')
                config.write('object1Frequency = ' + str(self.entry_Object1AntennaParam_Frequency.get()) + '\n')
                config.write('object1Position = ' + str(self.entry_Object1AntennaParam_AntennaOrigin.get()) + '\n')
                config.write('object1OrientationAlpha = ' + str(self.entry_Object1AntennaParam_OrientationAlpha.get()) + '\n')
                config.write('object1OrientationBeta = ' + str(self.entry_Object1AntennaParam_OrientationBeta.get()) + '\n')
                config.write('object1OrientationGamma = ' + str(self.entry_Object1AntennaParam_OrientationGamma.get()) + '\n')
                if self.object1TypeAntennaMain.get() == 1:
                    config.write('object1AntennaType = ' + "'" + 'Main' + "'" + '\n')
                    config.write('object1MainAntennaLength = ' + str(self.entry_object1_AntennaTypeMain_Length.get()) + '\n')
                    config.write('object1MainAntennaHeight = ' + str(self.entry_object1_AntennaTypeMain_Height.get()) + '\n')
                elif self.object1TypeAntennaFrame.get() == 1:
                    config.write('object1AntennaType = ' + "'" + 'Frame' + "'" + '\n')
                    config.write('object1FrameAntennaLength = ' + str(self.entry_object1_AntennaTypeFrame_Length.get()) + '\n')
                    config.write('object1FrameAntennaHeight = ' + str(self.entry_object1_AntennaTypeFrame_Height.get()) + '\n')
            elif self.object1TypeEllipse.get() == 1:
                config.write('object1Type = ' + "'" + 'Ellipse' + "'" + '\n')
                config.write('object1Windings = ' + str(self.entry_Object1EllipseParam_NumberWindings.get()) + '\n')
                config.write('object1Current = ' + str(self.entry_Object1EllipseParam_Current.get()) + '\n')
                config.write('object1Resistance= ' + str(self.entry_Object1EllipseParam_Resistance.get()) + '\n')
                config.write('object1Frequency = ' + str(self.entry_Object1EllipseParam_Frequency.get()) + '\n')
                config.write('object1Position = ' + str(self.entry_Object1EllipseParam_EllipseOrigin.get()) + '\n')
                config.write(
                    'object1OrientationAlpha = ' + str(self.entry_Object1EllipseParam_OrientationAlpha.get()) + '\n')
                config.write(
                    'object1OrientationBeta = ' + str(self.entry_Object1EllipseParam_OrientationBeta.get()) + '\n')
                config.write(
                    'object1OrientationGamma = ' + str(self.entry_Object1EllipseParam_OrientationGamma.get()) + '\n')
                config.write('object1MajorAxisLength = ' + str( self.entry_Object1EllipseParam_MajorAxis.get()) + '\n')
                config.write('object1MinorAxisLength =  ' + str( self.entry_Object1EllipseParam_MinorAxis.get()) + '\n')
            elif self.object1TypePuk.get()==1:
                config.write('object1Type = ' + "'" + 'Puk' + "'" + '\n')
                config.write('object1Windings = ' + str(self.entry_Object1PukParam_NumberWindings.get()) + '\n')
                config.write('object1Current = ' + str(self.entry_Object1PukParam_Current.get()) + '\n')
                config.write('object1Resistance= ' + str(self.entry_Object1PukParam_Resistance.get()) + '\n')
                config.write('object1Frequency = ' + str(self.entry_Object1PukParam_Frequency.get()) + '\n')
                config.write('object1Position = ' + str(self.entry_Object1PukParam_PukOrigin.get()) + '\n')
                config.write(
                    'object1OrientationAlpha = ' + str(self.entry_Object1PukParam_OrientationAlpha.get()) + '\n')
                config.write(
                    'object1OrientationBeta = ' + str(self.entry_Object1PukParam_OrientationBeta.get()) + '\n')
                config.write(
                    'object1OrientationGamma = ' + str(self.entry_Object1PukParam_OrientationGamma.get()) + '\n')
                config.write('object1PukHeight = ' + str(self.entry_Object1PukParam_PukHeight.get()) + '\n')
                config.write('object1CircularCoilRadius =  ' + str(self.entry_Object1PukParam_PukCircularCoilRadius.get()) + '\n')
            elif self.object1TypeBall.get() == 1:
                config.write('object1Type = ' + "'" + 'Ball' + "'" + '\n')
                config.write('object1Windings = ' + str(self.entry_Object1BallParam_NumberWindings.get()) + '\n')
                config.write('object1Current = ' + str(self.entry_Object1BallParam_Current.get()) + '\n')
                config.write('object1Resistance= ' + str(self.entry_Object1BallParam_Resistance.get()) + '\n')
                config.write('object1Frequency = ' + str(self.entry_Object1BallParam_Frequency.get()) + '\n')
                config.write('object1Position = ' + str(self.entry_Object1BallParam_BallOrigin.get()) + '\n')
                config.write(
                    'object1OrientationAlpha = ' + str(self.entry_Object1BallParam_OrientationAlpha.get()) + '\n')
                config.write(
                    'object1OrientationBeta = ' + str(self.entry_Object1BallParam_OrientationBeta.get()) + '\n')
                config.write(
                    'object1OrientationGamma = ' + str(self.entry_Object1BallParam_OrientationGamma.get()) + '\n')
                config.write(
                    'object1CoilsRadius =  ' + str(self.entry_Object1BallParam_BallRadius.get()) + '\n')
            else:config.write('object1Type = ' + str(0) + '\n')

            if self.object2TypePolygon.get()==1:
                config.write('object2Type = ' + "'" + 'Polygon' + "'" + '\n')
                config.write('object2Windings = ' + str(self.entry_Object2PolygonParam_NumberWindings.get()) + '\n')
                config.write('object2Current = ' + str(self.entry_Object2PolygonParam_Current.get()) + '\n')
                config.write('object2Resistance= ' + str(self.entry_Object2PolygonParam_Resistance.get()) + '\n')
                config.write('object2Frequency = ' + str(self.entry_Object2PolygonParam_Frequency.get()) + '\n')
                config.write('object2Position = ' + str(self.entry_Object2PolygonParam_PolygonOrigin.get()) + '\n')
                config.write('object2OrientationAlpha = ' + str(self.entry_Object2PolygonParam_OrientationAlpha.get()) + '\n')
                config.write('object2OrientationBeta = ' + str(self.entry_Object2PolygonParam_OrientationBeta.get()) + '\n')
                config.write('object2OrientationGamma = ' + str(self.entry_Object2PolygonParam_OrientationGamma.get()) + '\n')
                config.write('object2OpenExciter = ' + str(self.object2_OpenExciter.get()) + '\n')
                config.write('object2RegisteredPolygonCorners = ' + str(self.registeredPolygonCorners2) + '\n')
                for i in range(1, self.registeredPolygonCorners2 + 1):
                    config.write('object2E{0} = '.format(i) + self.Object2polygonCornersEntriesDict[
                        "Corner {0} entry".format(i)].get() + '\n')
            elif self.object2TypeAntenna.get()==1:
                config.write('object2Type = ' + "'" + 'Antenna' + "'" + '\n')
                config.write('object2Windings = ' + str(self.entry_Object2AntennaParam_NumberWindings.get()) + '\n')
                config.write('object2Current = ' + str(self.entry_Object2AntennaParam_Current.get()) + '\n')
                config.write('object2Resistance= ' + str(self.entry_Object2AntennaParam_Resistance.get()) + '\n')
                config.write('object2Frequency = ' + str(self.entry_Object2AntennaParam_Frequency.get()) + '\n')
                config.write('object2Position = ' + str(self.entry_Object2AntennaParam_AntennaOrigin.get()) + '\n')
                config.write('object2OrientationAlpha = ' + str(self.entry_Object2AntennaParam_OrientationAlpha.get()) + '\n')
                config.write('object2OrientationBeta = ' + str(self.entry_Object2AntennaParam_OrientationBeta.get()) + '\n')
                config.write('object2OrientationGamma = ' + str(self.entry_Object2AntennaParam_OrientationGamma.get()) + '\n')
                if self.object2TypeAntennaMain.get() == 1:
                    config.write('object2AntennaType = ' + "'" + 'Main' + "'" + '\n')
                    config.write('object2MainAntennaLength = ' + str(self.entry_object2_AntennaTypeMain_Length.get()) + '\n')
                    config.write('object2MainAntennaHeight = ' + str(self.entry_object2_AntennaTypeMain_Height.get()) + '\n')
                elif self.object2TypeAntennaFrame.get() == 1:
                    config.write('object2AntennaType = ' + "'" + 'Frame' + "'" + '\n')
                    config.write('object2FrameAntennaLength = ' + str(self.entry_object2_AntennaTypeFrame_Length.get()) + '\n')
                    config.write('object2FrameAntennaHeight = ' + str(self.entry_object2_AntennaTypeFrame_Height.get()) + '\n')
            elif self.object2TypeEllipse.get() == 1:
                config.write('object2Type = ' + "'" + 'Ellipse' + "'" + '\n')
                config.write('object2Windings = ' + str(self.entry_Object2EllipseParam_NumberWindings.get()) + '\n')
                config.write('object2Current = ' + str(self.entry_Object2EllipseParam_Current.get()) + '\n')
                config.write('object2Resistance= ' + str(self.entry_Object2EllipseParam_Resistance.get()) + '\n')
                config.write('object2Frequency = ' + str(self.entry_Object2EllipseParam_Frequency.get()) + '\n')
                config.write('object2Position = ' + str(self.entry_Object2EllipseParam_EllipseOrigin.get()) + '\n')
                config.write(
                    'object2OrientationAlpha = ' + str(self.entry_Object2EllipseParam_OrientationAlpha.get()) + '\n')
                config.write(
                    'object2OrientationBeta = ' + str(self.entry_Object2EllipseParam_OrientationBeta.get()) + '\n')
                config.write(
                    'object2OrientationGamma = ' + str(self.entry_Object2EllipseParam_OrientationGamma.get()) + '\n')
                config.write('object2MajorAxisLength = ' + str( self.entry_Object2EllipseParam_MajorAxis.get()) + '\n')
                config.write('object2MinorAxisLength =  ' + str( self.entry_Object2EllipseParam_MinorAxis.get()) + '\n')
            elif self.object2TypePuk.get()==1:
                config.write('object2Type = ' + "'" + 'Puk' + "'" + '\n')
                config.write('object2Windings = ' + str(self.entry_Object2PukParam_NumberWindings.get()) + '\n')
                config.write('object2Current = ' + str(self.entry_Object2PukParam_Current.get()) + '\n')
                config.write('object2Resistance= ' + str(self.entry_Object2PukParam_Resistance.get()) + '\n')
                config.write('object2Frequency = ' + str(self.entry_Object2PukParam_Frequency.get()) + '\n')
                config.write('object2Position = ' + str(self.entry_Object2PukParam_PukOrigin.get()) + '\n')
                config.write(
                    'object2OrientationAlpha = ' + str(self.entry_Object2PukParam_OrientationAlpha.get()) + '\n')
                config.write(
                    'object2OrientationBeta = ' + str(self.entry_Object2PukParam_OrientationBeta.get()) + '\n')
                config.write(
                    'object2OrientationGamma = ' + str(self.entry_Object2PukParam_OrientationGamma.get()) + '\n')
                config.write('object2PukHeight = ' + str(self.entry_Object2PukParam_PukHeight.get()) + '\n')
                config.write('object2CircularCoilRadius =  ' + str(self.entry_Object2PukParam_PukCircularCoilRadius.get()) + '\n')
            elif self.object2TypeBall.get() == 1:
                config.write('object2Type = ' + "'" + 'Ball' + "'" + '\n')
                config.write('object2Windings = ' + str(self.entry_Object2BallParam_NumberWindings.get()) + '\n')
                config.write('object2Current = ' + str(self.entry_Object2BallParam_Current.get()) + '\n')
                config.write('object2Resistance= ' + str(self.entry_Object2BallParam_Resistance.get()) + '\n')
                config.write('object2Frequency = ' + str(self.entry_Object2BallParam_Frequency.get()) + '\n')
                config.write('object2Position = ' + str(self.entry_Object2BallParam_BallOrigin.get()) + '\n')
                config.write(
                    'object2OrientationAlpha = ' + str(self.entry_Object2BallParam_OrientationAlpha.get()) + '\n')
                config.write(
                    'object2OrientationBeta = ' + str(self.entry_Object2BallParam_OrientationBeta.get()) + '\n')
                config.write(
                    'object2OrientationGamma = ' + str(self.entry_Object2BallParam_OrientationGamma.get()) + '\n')
                config.write(
                    'object2CoilsRadius =  ' + str(self.entry_Object2BallParam_BallRadius.get()) + '\n')
            else:config.write('object2Type = ' + str(0) + '\n')
    def plotSetup(self):
        try:
            self.label_PlotGenerateStopMessages.configure(text="Plotting" + '\n' + "setup.", fg='blue')
            self.generateConfigFile("GUI_configFile.py")
            self.proc_plotSetup = subprocess.Popen("python GUI_plot_setup.py")
            #self.proc.append(self.proc_plotSetup)
            self.label_PlotGenerateStopMessages.configure(text="Setup plotted.", fg='blue')
        except:
            self.label_PlotGenerateStopMessages.configure(
                    text="Error" + '\n' + "during " + '\n' + "plotting" + '\n' + "setup!", fg='red')
    def calculateCouplingFactor(self):
        try:
            self.generateConfigFile('GUI_configFile.py')
            self.label_PlotGenerateStopMessages.configure(text="Calculating"+'\n'+"coupling"+'\n'+"factor.", fg='dark green')
            object1, object2 = self.getTwoObjects()


            if object1 == 0 or object2 == 0:
                self.label_PlotGenerateStopMessages.configure(
                    text="Please" + '\n' + "define two" + '\n' + "objects for" + '\n' + "coupling factor" + '\n'+"calcualtions", fg='red')
            else:
                 calculateCouplingFactor = couplingFactor(object1, object2)
                 couplingFactor21 = calculateCouplingFactor.get_computedCouplingFactor()
                 calculateCouplingFactor = couplingFactor(object2, object1)
                 couplingFactor12 = calculateCouplingFactor.get_computedCouplingFactor()
                 absoluteCouplingFactor = couplingFactor.get_absoluteCouplingFactor(couplingFactor21, couplingFactor12)
                 self.entry_couplingFactor21.delete(0,END)
                 self.entry_couplingFactor12.delete(0,END)
                 self.entry_couplingFactorAbsolute.delete(0,END)
                 self.entry_couplingFactor21.insert(0, couplingFactor21)
                 self.entry_couplingFactor12.insert(0, couplingFactor12)
                 self.entry_couplingFactorAbsolute.insert(0, absoluteCouplingFactor)

        except:
           self.label_PlotGenerateStopMessages.configure(text="Error during"+'\n'+"calculating "+'\n'+"coupling"+'\n'+" factor !", fg='red')
    def getRotationMatrix(self,alpha, beta, gamma):
        c_x = np.array([[1, 0, 0],
                        [0, np.cos(math.radians(alpha)),
                         -np.sin(math.radians(alpha))],
                        [0, np.sin(math.radians(alpha)),
                         np.cos(math.radians(alpha))]])

        c_y = np.array([[np.cos(math.radians(beta)), 0,
                         np.sin(math.radians(beta))],
                        [0, 1, 0],
                        [-np.sin(math.radians(beta)), 0,
                         np.cos(math.radians(beta))]])

        c_z = np.array([[np.cos(math.radians(gamma)),
                         -np.sin(math.radians(gamma)), 0],
                        [np.sin(math.radians(gamma)),
                         np.cos(math.radians(gamma)), 0],
                        [0, 0, 1]])
        return np.dot(c_z, np.dot(c_y, c_x))
    def getTwoObjects(self):
        if self.object1TypePolygon.get() ==1 :
            shape = []
            closed_loop = True
            for i in range(1, self.registeredPolygonCorners + 1):
                e = eval(self.Object1polygonCornersEntriesDict["Corner {0} entry".format(i)].get())
                shape.append(np.array(e))
            if self.checkbox_object1_OpenExciter == 0:
                shape.append(shape[0])
            else:
                closed_loop = False

            rotationMatrix = self.getRotationMatrix(eval(self.entry_Object1PolygonParam_OrientationAlpha.get()),
                                                    eval(self.entry_Object1PolygonParam_OrientationBeta.get()),
                                                         eval(self.entry_Object1PolygonParam_OrientationGamma.get()))
            object1 = PolygonExtended(shape,
                                      eval(self.entry_object1_windingsWidth.get()),
                                      eval(self.entry_object1_windingsDistance.get()),
                                    eval(self.entry_object1_radius.get()),
                                    eval(self.entry_object1_inductance.get()),
                                      eval(self.entry_Object1PolygonParam_Resistance.get()),
                                    eval(self.entry_Object1PolygonParam_NumberWindings.get()),
                                    eval(self.entry_Object1PolygonParam_Current.get()),
                                    np.array(eval(self.entry_Object1PolygonParam_PolygonOrigin.get())),
                                      rotationMatrix,
                                      closed_loop)
        elif self.object1TypeAntenna.get() == 1:
            if self.object1TypeAntennaMain.get() == 1:
                length= eval(self.entry_object1_AntennaTypeMain_Length.get())
                height = eval(self.entry_object1_AntennaTypeMain_Height.get())
                aM0 = np.array([-length / 2, 0, -height / 2])
                aM1 = np.array([+length / 2, 0, -height / 2])
                aM2 = np.array([+length / 2, 0, +height / 2])
                aM3 = np.array([-length / 2, 0, +height / 2])
                shape = [aM0,aM1,aM2,aM3]
            elif self.object1TypeAntennaFrame.get() == 1:
                length = eval(self.entry_object1_AntennaTypeFrame_Length.get())
                height = eval(self.entry_object1_AntennaTypeFrame_Height.get())
                aF0 = np.array(
                    [-length / 2, -height / 2, 0])
                aF1 = np.array(
                    [+length / 2, -height / 2, 0])
                aF2 = np.array(
                    [+length / 2, +height / 2, 0])
                aF3 = np.array(
                    [-length / 2, +height / 2, 0])
                shape = [aF0, aF1, aF2, aF3]
            rotationMatrix = self.getRotationMatrix(eval(self.entry_Object1AntennaParam_OrientationAlpha.get()),
                                                         eval(self.entry_Object1AntennaParam_OrientationBeta.get()),
                                                              eval(self.entry_Object1AntennaParam_OrientationGamma.get()))
            object1 = PolygonExtended(shape,
                                      eval(self.entry_object1_windingsWidth.get()),
                                      eval(self.entry_object1_windingsDistance.get()),
                                      eval(self.entry_object1_radius.get()),
                                      eval(self.entry_object1_inductance.get()),
                                      eval(self.entry_Object1AntennaParam_Resistance.get()),
                                      eval(self.entry_Object1AntennaParam_NumberWindings.get()),
                                      eval(self.entry_Object1AntennaParam_Current.get()),
                                      np.array(eval(self.entry_Object1AntennaParam_AntennaOrigin.get())),
                                      rotationMatrix,
                                      closed_loop=True)
        elif self.object1TypeEllipse.get() == 1:
            rotationMatrix = self.getRotationMatrix(eval(self.entry_Object1EllipseParam_OrientationAlpha.get()),
                                                         eval(self.entry_Object1EllipseParam_OrientationBeta.get()),
                                                              eval(self.entry_Object1EllipseParam_OrientationGamma.get()))
            object1 = EllipseExtended(eval(self.entry_Object1EllipseParam_MajorAxis.get()),
                                        eval(self.entry_Object1EllipseParam_MinorAxis.get()),
                                        eval(self.entry_Object1EllipseParam_NumberWindings.get()),
                                        eval(self.entry_Object1EllipseParam_Resistance.get()),
                                        eval(self.entry_object1_inductance.get()),
                                        np.array(eval(self.entry_Object1EllipseParam_EllipseOrigin.get())),
                                        rotationMatrix,
                                             eval(self.entry_Object1EllipseParam_Current.get()))
        elif self.object1TypePuk.get() == 1:
            rotationMatrix = self.getRotationMatrix(eval(self.entry_Object1PukParam_OrientationAlpha.get()),
                                                         eval(self.entry_Object1PukParam_OrientationBeta.get()),
                                                              eval(self.entry_Object1PukParam_OrientationGamma.get()))
            object1 = PukExtended(eval(self.entry_Object1PukParam_PukCircularCoilRadius.get()),
                                       eval(self.entry_Object1PukParam_PukHeight.get()),
                                            eval(self.entry_Object1PukParam_NumberWindings.get()),
                                                 eval(self.entry_Object1PukParam_Resistance.get()),
                                                      eval(self.entry_object1_inductance.get()),
                                      np.array(eval(self.entry_Object1PukParam_PukOrigin.get())),
                                      rotationMatrix,
                                      np.array([eval(self.entry_Object1PukParam_Current.get()), eval(self.entry_Object1PukParam_Current.get()), eval(self.entry_Object1PukParam_Current.get())]))
        elif self.object1TypeBall.get() == 1:
            rotationMatrix = self.getRotationMatrix(eval(self.entry_Object1BallParam_OrientationAlpha.get()),
                                                         eval(self.entry_Object1BallParam_OrientationBeta.get()),
                                                              eval(self.entry_Object1BallParam_OrientationGamma.get()))
            object1 = BallExtended(eval(self.entry_Object1BallParam_BallRadius.get()),
                                        eval(self.entry_Object1BallParam_NumberWindings.get()),
                                             eval(self.entry_Object1BallParam_Resistance.get()),
                                                  eval(self.entry_object1_inductance.get()),
                                      np.array(eval(self.entry_Object1BallParam_BallOrigin.get())),
                                      rotationMatrix,
                                      np.array([eval(self.entry_Object1BallParam_Current.get()), eval(self.entry_Object1BallParam_Current.get()),
                                                                                                     eval(self.entry_Object1BallParam_Current.get())]))
        else:
            object1 = 0
        if self.object2TypePolygon.get() ==1 :
            shape = []
            closed_loop = True
            for i in range(1, self.registeredPolygonCorners2 + 1):
                e = eval(self.Object2polygonCornersEntriesDict["Corner {0} entry".format(i)].get())
                shape.append(np.array(e))
            if self.checkbox_object2_OpenExciter == 0:
                shape.append(shape[0])
            else:
                closed_loop = False

            rotationMatrix = self.getRotationMatrix(eval(self.entry_Object2PolygonParam_OrientationAlpha.get()),
                                                    eval(self.entry_Object2PolygonParam_OrientationBeta.get()),
                                                         eval(self.entry_Object2PolygonParam_OrientationGamma.get()))
            object2 = PolygonExtended(shape,
                                      eval(self.entry_object2_windingsWidth.get()),
                                      eval(self.entry_object2_windingsDistance.get()),
                                    eval(self.entry_object2_radius.get()),
                                    eval(self.entry_object2_inductance.get()),
                                      eval(self.entry_Object2PolygonParam_Resistance.get()),
                                    eval(self.entry_Object2PolygonParam_NumberWindings.get()),
                                    eval(self.entry_Object2PolygonParam_Current.get()),
                                    np.array(eval(self.entry_Object2PolygonParam_PolygonOrigin.get())),
                                      rotationMatrix,
                                      closed_loop)
        elif self.object2TypeAntenna.get() == 1:
            if self.object2TypeAntennaMain.get() == 1:
                length= eval(self.entry_object2_AntennaTypeMain_Length.get())
                height = eval(self.entry_object2_AntennaTypeMain_Height.get())
                aM0 = np.array([-length / 2, 0, -height / 2])
                aM1 = np.array([+length / 2, 0, -height / 2])
                aM2 = np.array([+length / 2, 0, +height / 2])
                aM3 = np.array([-length / 2, 0, +height / 2])
                shape = [aM0,aM1,aM2,aM3]
            elif self.object2TypeAntennaFrame.get() == 1:
                length = eval(self.entry_object2_AntennaTypeFrame_Length.get())
                height = eval(self.entry_object2_AntennaTypeFrame_Height.get())
                aF0 = np.array(
                    [-length / 2, -height / 2, 0])
                aF1 = np.array(
                    [+length / 2, -height / 2, 0])
                aF2 = np.array(
                    [+length / 2, +height / 2, 0])
                aF3 = np.array(
                    [-length / 2, +height / 2, 0])
                shape = [aF0, aF1, aF2, aF3]
            rotationMatrix = self.getRotationMatrix(eval(self.entry_Object2AntennaParam_OrientationAlpha.get()),
                                                         eval(self.entry_Object2AntennaParam_OrientationBeta.get()),
                                                              eval(self.entry_Object2AntennaParam_OrientationGamma.get()))
            object2 = PolygonExtended(shape,
                                      eval(self.entry_object2_windingsWidth.get()),
                                      eval(self.entry_object2_windingsDistance.get()),
                                      eval(self.entry_object2_radius.get()),
                                      eval(self.entry_object2_inductance.get()),
                                      eval(self.entry_Object2AntennaParam_Resistance.get()),
                                      eval(self.entry_Object2AntennaParam_NumberWindings.get()),
                                      eval(self.entry_Object2AntennaParam_Current.get()),
                                      np.array(eval(self.entry_Object2AntennaParam_AntennaOrigin.get())),
                                      rotationMatrix,
                                      closed_loop=True)
        elif self.object2TypeEllipse.get() == 1:
            rotationMatrix = self.getRotationMatrix(eval(self.entry_Object2EllipseParam_OrientationAlpha.get()),
                                                         eval(self.entry_Object2EllipseParam_OrientationBeta.get()),
                                                              eval(self.entry_Object2EllipseParam_OrientationGamma.get()))

            object2 = EllipseExtended(eval(self.entry_Object2EllipseParam_MajorAxis.get()),
                                        eval(self.entry_Object2EllipseParam_MinorAxis.get()),
                                        eval(self.entry_Object2EllipseParam_NumberWindings.get()),
                                        eval(self.entry_Object2EllipseParam_Resistance.get()),
                                        eval(self.entry_object2_inductance.get()),
                                        np.array(eval(self.entry_Object2EllipseParam_EllipseOrigin.get())),
                                        rotationMatrix,
                                             eval(self.entry_Object2EllipseParam_Current.get()))
        elif self.object2TypePuk.get() == 1:
            rotationMatrix = self.getRotationMatrix(eval(self.entry_Object2PukParam_OrientationAlpha.get()),
                                                         eval(self.entry_Object2PukParam_OrientationBeta.get()),
                                                              eval(self.entry_Object2PukParam_OrientationGamma.get()))
            object2 = PukExtended(eval(self.entry_Object2PukParam_PukCircularCoilRadius.get()),
                                       eval(self.entry_Object2PukParam_PukHeight.get()),
                                            eval(self.entry_Object2PukParam_NumberWindings.get()),
                                                 eval(self.entry_Object2PukParam_Resistance.get()),
                                                      eval(self.entry_object2_inductance.get()),
                                      np.array(eval(self.entry_Object2PukParam_PukOrigin.get())),
                                      rotationMatrix,
                                      np.array([eval(self.entry_Object2PukParam_Current.get()), eval(self.entry_Object2PukParam_Current.get()), eval(self.entry_Object2PukParam_Current.get())]))
        elif self.object2TypeBall.get() == 1:
            rotationMatrix = self.getRotationMatrix(eval(self.entry_Object2BallParam_OrientationAlpha.get()),
                                                         eval(self.entry_Object2BallParam_OrientationBeta.get()),
                                                              eval(self.entry_Object2BallParam_OrientationGamma.get()))
            object2 = BallExtended(eval(self.entry_Object2BallParam_BallRadius.get()),
                                        eval(self.entry_Object2BallParam_NumberWindings.get()),
                                             eval(self.entry_Object2BallParam_Resistance.get()),
                                                  eval(self.entry_object2_inductance.get()),
                                      np.array(eval(self.entry_Object2BallParam_BallOrigin.get())),
                                      rotationMatrix,
                                      np.array([eval(self.entry_Object2BallParam_Current.get()), eval(self.entry_Object2BallParam_Current.get()),
                                                                                                     eval(self.entry_Object2BallParam_Current.get())]))
        else:
            object2 = 0


        return object1, object2
    def createStartUpElments(self):
        self.object1Frame = ttk.Frame(self.objectParametersTab)
        self.object1Frame.place(x=0, y=int(10*self.scale_factor))
        self.object2Frame = ttk.Frame(self.objectParametersTab)
        self.object2Frame.place(x=int((self.screenWidth/2) -int(100*self.scale_factor)), y=int(10*self.scale_factor))
        plotSetupFrame = ttk.Frame(self.couplingFactorTab)
        plotSetupFrame.place(x=self.screenWidth - int(80*self.scale_factor), y=int((self.screenHeight / 2) - int(200*self.scale_factor)))
        calculateCouplingFactorFrame = ttk.Frame(self.couplingFactorTab)
        calculateCouplingFactorFrame.place(x=self.screenWidth - int(80*self.scale_factor), y=int((self.screenHeight / 2) - int(150*self.scale_factor)))
        couplingFactorResultsFrame = ttk.Frame(self.objectParametersTab)
        couplingFactorResultsFrame.place(x=int(10*self.scale_factor), y=int(450*self.scale_factor))
        ######################################################################################################################
        ## Initial values for GUI entries:
        ######################################################################################################################
        ## Number of windings of the polygon loop, default = 1
        self.Object1_DefaultParameters_NumberOfWindings = 1
        ## AC current amplitude in the polygon loop, default = 1 A
        self.Object1_DefaultParameters_Current = 1
        self.Object1_DefaultParameters_Resistance=10
        ## Frequency of the polygon's current in hz, default = 119000 hz
        self.Object1_DefaultParameters_Frequency = 119000
        ## Origin of the polygon coordinate system relative
        # to the intertial system, default = [0,0,0]
        self.Object1_DefaultParameters_Origin = [0, 0, 0]
        ## Default Positions of the  corners of the polygon wire.
        self.polygonParam_PolygonPolygonCorners = [[0, 1.256, 0], [0, 0, 0], [1.255,0, 0],
                                                   [1.255, 1.256, 0], [0, 1, 0], [0, 0.75, 0],
                                                   [0, 0.5, 0], [0, 0.25, 0], [0, 0.1, 0]
            , [0, 0.05, 0]]
        self.Object1_DefaultParameters_Alpha = 0
        self.Object1_DefaultParameters_Beta = 0
        self.Object1_DefaultParameters_Gamma = 0




        ##
        self.object1_AntennaTypeFrame_Length = 0.319
        self.object1_AntennaTypeFrame_Height = 0.06

        self.object1_AntennaTypeMain_Length = 0.3423
        self.object1_AntennaTypeMain_Height = 0.0597

        self.Object1_DefaultParameters_MajorAxis = 0.05
        self.Object1_DefaultParameters_MinorAxis = 0.02

        self.Object1_DefaultParameters_PukHeight = 0.015
        self.Object1_DefaultParameters_PukCircularCoilRadius = 0.0381

        self.Object1_DefaultParameters_BallRadius = 0.0381

        ## Number of windings of the polygon loop, default = 1
        self.Object2_DefaultParameters_NumberOfWindings = 1
        ## AC current amplitude in the polygon loop, default = 1 A
        self.Object2_DefaultParameters_Current = 1
        self.Object2_DefaultParameters_Resistance = 10
        ## Frequency of the polygon's current in hz, default = 119000 hz
        self.Object2_DefaultParameters_Frequency = 119000
        ## Origin of the polygon coordinate system relative
        # to the intertial system, default = [0,0,0]
        self.Object2_DefaultParameters_Origin = [0, 0, 0]

        self.Object2_DefaultParameters_Alpha = 0
        self.Object2_DefaultParameters_Beta = 0
        self.Object2_DefaultParameters_Gamma = 0

        ##
        self.object2_AntennaTypeFrame_Length = 0.5
        self.object2_AntennaTypeFrame_Height = 0.02

        self.object2_AntennaTypeMain_Length = 0.5
        self.object2_AntennaTypeMain_Height = 0.04

        self.Object2_DefaultParameters_MajorAxis = 0.05
        self.Object2_DefaultParameters_MinorAxis = 0.02

        self.Object2_DefaultParameters_PukHeight = 0.015
        self.Object2_DefaultParameters_PukCircularCoilRadius = 0.0381

        self.Object2_DefaultParameters_BallRadius = 0.0381
        ################################################################################################################
        ## Object1 labels, checkboxes
        ################################################################################################################
        label_object1_title = Label(self.object1Frame, text='Object number 1:',font=("Arial", self.paramFontSize))
        label_object1_shape = Label(self.object1Frame, text='Shape/Type:', font=("Arial", self.paramFontSize))
        self.object1TypeFrame_register = []
        self.object1TypePolygon = IntVar()
        self.object1TypeAntenna = IntVar()
        self.object1TypeEllipse = IntVar()
        self.object1TypePuk = IntVar()
        self.object1TypeBall = IntVar()

        checkBox_Object1_Polygon = Checkbutton(self.object1Frame, text="Polygon", variable=self.object1TypePolygon,
                                                   command=self.displayObject1TypePolygon)
        checkBox_Object1_Antenna = Checkbutton(self.object1Frame, text="Rect. Antenna", variable=self.object1TypeAntenna,
                                               command=self.displayObject1TypeAntenna)
        checkBox_Object1_Ellipse = Checkbutton(self.object1Frame, text="Ellipse", variable=self.object1TypeEllipse,
                                                   command=self.displayObject1TypeEllipse)
        checkBox_Object1_Puk = Checkbutton(self.object1Frame, text="Puk", variable=self.object1TypePuk,
                                               command=self.displayObject1TypePuk)
        checkBox_Object1_Ball = Checkbutton(self.object1Frame, text="Ball", variable=self.object1TypeBall,
                                                command=self.displayObject1TypeBall)

        label_object1_title.grid(row=0, column=0, sticky='W')
        label_object1_shape.grid(row=1, column=0, sticky='W')
        checkBox_Object1_Polygon.grid(row=1, column=1, sticky='W')
        checkBox_Object1_Antenna.grid(row=1, column=2, sticky='W')
        checkBox_Object1_Ellipse.grid(row=1, column=3, sticky='W')
        checkBox_Object1_Puk.grid(row=1, column=4, sticky='W')
        checkBox_Object1_Ball.grid(row=1, column=5, sticky='W')
        ################################################################################################################
        ## Object2 labels, checkboxes
        ################################################################################################################
        label_object2_title = Label(self.object2Frame, text='Object number 2:', font=("Arial", self.paramFontSize))
        label_object2_shape = Label(self.object2Frame, text='Shape/Type:', font=("Arial", self.paramFontSize))
        self.object2TypeFrame_register = []
        self.object2TypePolygon = IntVar()
        self.object2TypeAntenna = IntVar()
        self.object2TypeEllipse = IntVar()
        self.object2TypePuk = IntVar()
        self.object2TypeBall = IntVar()

        checkBox_Object2_Polygon = Checkbutton(self.object2Frame, text="Polygon", variable=self.object2TypePolygon,
                                               command=self.displayObject2TypePolygon)
        checkBox_Object2_Antenna = Checkbutton(self.object2Frame, text="Rect. Antenna",
                                               variable=self.object2TypeAntenna,
                                               command=self.displayObject2TypeAntenna)
        checkBox_Object2_Ellipse = Checkbutton(self.object2Frame, text="Ellipse", variable=self.object2TypeEllipse,
                                               command=self.displayObject2TypeEllipse)
        checkBox_Object2_Puk = Checkbutton(self.object2Frame, text="Puk", variable=self.object2TypePuk,
                                           command=self.displayObject2TypePuk)
        checkBox_Object2_Ball = Checkbutton(self.object2Frame, text="Ball", variable=self.object2TypeBall,
                                            command=self.displayObject2TypeBall)

        label_object2_title.grid(row=0, column=0, sticky='W')
        label_object2_shape.grid(row=1, column=0, sticky='W')
        checkBox_Object2_Polygon.grid(row=1, column=1, sticky='W')
        checkBox_Object2_Antenna.grid(row=1, column=2, sticky='W')
        checkBox_Object2_Ellipse.grid(row=1, column=3, sticky='W')
        checkBox_Object2_Puk.grid(row=1, column=4, sticky='W')
        checkBox_Object2_Ball.grid(row=1, column=5, sticky='W')

        ########################################################################################################################
        ## Plot setup button
        ########################################################################################################################
        self.button_PlotSetup = ttk.Button(plotSetupFrame, text="Plot setup", fg="blue", command=self.plotSetup,
                                           height=int(2*self.scale_factor), width=int(8*self.scale_factor))
        self.button_PlotSetup.grid()
        ########################################################################################################################
        ## Plot/Generate/Stop text messages:
        ########################################################################################################################
        self.label_PlotGenerateStopMessages = ttk.Label(self.couplingFactorTab, text="",
                                                        font=("Arial", int(7*self.scale_factor)))
        self.label_PlotGenerateStopMessages.place(x=self.screenWidth - int(80*self.scale_factor), y=int(self.screenHeight / 2))
        ########################################################################################################################
        ## Calculate coupling factor button
        ########################################################################################################################
        self.button_CalculateCouplingFactor = ttk.Button(calculateCouplingFactorFrame, text="Calculate " + '\n' + "coupling" +'\n' + "factor", fg="dark green",
                                               command=self.calculateCouplingFactor, height=int(3*self.scale_factor), width=int(8*self.scale_factor))
        self.button_CalculateCouplingFactor.grid()
        ########################################################################################################################
        ## Coupling factor results labels and entries
        ########################################################################################################################
        label_object1_radius = Label(couplingFactorResultsFrame, text='Object 1 wire\'s radius (m): ')
        label_object1_windingsWidth = Label(couplingFactorResultsFrame, text='Object 1 windings\' width (m): ')
        label_object1_windingsDistance = Label(couplingFactorResultsFrame, text='Object 1 windings\' distance (m): ')

        label_object2_radius = Label(couplingFactorResultsFrame, text='Object 2 wire\'s radius (m): ')
        label_object2_windingsWidth = Label(couplingFactorResultsFrame, text='Object 2 windings\' width (m): ')
        label_object2_windingsDistance = Label(couplingFactorResultsFrame, text='Object 2 windings\' distance (m): ')


        label_object1_inductance = Label(couplingFactorResultsFrame, text='Object 1 inductance (H): ')
        label_object2_inductance = Label(couplingFactorResultsFrame, text='Object 2 inductance (H): ')
        label_couplingFactor12 = ttk.Label(couplingFactorResultsFrame, text='Coupling factor from object 1 to object 2: ', fg='blue')
        label_couplingFactor21 = ttk.Label(couplingFactorResultsFrame, text='Coupling factor from object 2 to object 1: ', fg='blue')
        label_couplingFactorAbsolute = ttk.Label(couplingFactorResultsFrame, text='Absolute coupling factor: ', fg='blue')
        self.entry_object1_radius = Entry(couplingFactorResultsFrame, width=10)
        self.entry_object1_windingsWidth = Entry(couplingFactorResultsFrame, width=10)
        self.entry_object1_windingsDistance = Entry(couplingFactorResultsFrame, width=10)

        self.entry_object2_radius = Entry(couplingFactorResultsFrame, width=10)
        self.entry_object2_windingsWidth = Entry(couplingFactorResultsFrame, width=10)
        self.entry_object2_windingsDistance = Entry(couplingFactorResultsFrame, width=10)


        self.entry_object1_inductance = Entry(couplingFactorResultsFrame,  width=10)
        self.entry_object2_inductance = Entry(couplingFactorResultsFrame,  width=10)
        self.entry_couplingFactor12 = Entry(couplingFactorResultsFrame, width=10)
        self.entry_couplingFactor21 = Entry(couplingFactorResultsFrame, width=10)
        self.entry_couplingFactorAbsolute = Entry(couplingFactorResultsFrame, width=10)
        label_object1_radius.grid(row=0,column=0, sticky='W')
        self.entry_object1_radius.grid(row=0,column=1, sticky='W')
        label_object1_windingsWidth.grid(row=1,column=0, sticky='W')
        self.entry_object1_windingsWidth.grid(row=1,column=1, sticky='W')
        label_object1_windingsDistance.grid(row=2,column=0, sticky='W')
        self.entry_object1_windingsDistance.grid(row=2,column=1, sticky='W')
        label_object1_inductance.grid(row=3, column=0, sticky='W')
        self.entry_object1_inductance.grid(row=3, column=1, sticky='W')

        self.entry_object1_radius.insert(0, 'None')
        self.entry_object1_windingsWidth.insert(0, 'None')
        self.entry_object1_windingsDistance.insert(0, 'None')
        self.entry_object1_inductance.insert(0, 'None')
        label_object2_radius.grid(row=0,column=2, sticky='W')
        self.entry_object2_radius.grid(row=0,column=3, sticky='W')
        label_object2_windingsWidth.grid(row=1,column=2, sticky='W')
        self.entry_object2_windingsWidth.grid(row=1,column=3, sticky='W')
        label_object2_windingsDistance.grid(row=2,column=2, sticky='W')
        self.entry_object2_windingsDistance.grid(row=2,column=3, sticky='W')
        label_object2_inductance.grid(row=3,column=2, sticky='W')
        self.entry_object2_inductance.grid(row=3,column=3, sticky='W')
        label_couplingFactor12.grid(row=4,column=0, sticky='W')
        self.entry_couplingFactor12.grid(row=4,column=1, sticky='W')
        label_couplingFactor21.grid(row=4,column=2, sticky='W')
        self.entry_couplingFactor21.grid(row=4,column=3, sticky='W')
        label_couplingFactorAbsolute.grid(row=4,column=4, sticky='W')
        self.entry_couplingFactorAbsolute.grid(row=4,column=5, sticky='W')

        self.entry_object2_radius.insert(0, 'None')
        self.entry_object2_windingsWidth.insert(0, 'None')
        self.entry_object2_windingsDistance.insert(0, 'None')
        self.entry_object2_inductance.insert(0, 'None')

def calculate_magnetic_field(thread_queue=None, stop_queue = None,exciterObject=None, xyzPositionsList=None):
    mu0 = 4 * np.pi * 1e-7
    for i in range(len(xyzPositionsList)):
        if i < len(xyzPositionsList)-1:
            flag = False
        else:
            flag = True
        if stop_queue.empty() == True:
            result=[]
            result.append(xyzPositionsList[i][0])
            result.append(xyzPositionsList[i][1])
            result.append(xyzPositionsList[i][2])
            try:
                mF = exciterObject.fluxDensity(xyzPositionsList[i])
                mF = mF / mu0
            except:
                flag = 'Near Exciter'
                result.append(None)
                result.append(None)
                result.append(None)
                result.append(i)
                result.append(flag)
                thread_queue.put(result)
                return

            for ii in range(3):
                result.append(mF[ii])
            result.append(i)
            result.append(flag)
            thread_queue.put(result)
        elif stop_queue.empty() == False:
            return

class GUI_magneticFieldSimulation(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        self.create_widgets()

    def create_widgets(self):
        self.tabs = Notebook(self)
        self.tabs.pack(side="top", expand=True, fill="both")

        self.tab_simulation = GUI_magneticFieldSimulationTab(self.tabs)
        self.tabs.add(self.tab_simulation, text="Simulation")

        self.tab_plot = GUI_magneticFieldPlotTab(self.tabs)
        self.tabs.add(self.tab_plot, text="Plot")

        self.tabs.bind('<<NotebookTabChanged>>', self.tab_changed)

    def tab_changed(self, evt):
        tab_idx = self.tabs.index(self.tabs.select())
        tabs = self.tabs.winfo_children()

        for i in range(len(tabs)):
            if i == tab_idx:
                tabs[i].activate()
            else:
                tabs[i].deactivate()

class GUI_magneticFieldSimulationTab(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        self.visible = True
        self.numberOfCoulmns = 6
        self.period = 5 #time to recheck the queue in the after method call
        self.create_widgets()
        self.default_show_exciter_corners()
        self.finishedFlag = False
        self.button_stopSimulation.state(["disabled"])

    def create_widgets(self):
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=0)
        self.rowconfigure(2, weight=0)
        # self.rowconfigure(3, weight=1)
        # self.rowconfigure(4, weight=2)
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=0)
        self.columnconfigure(2, weight=0)
        #self.columnconfigure(3, weight=0)
        # self.columnconfigure(4, weight=1)
        #self.columnconfigure(2, weight=1)

        self.default_numberOfWindings = 1
        self.default_current = 1
        self.default_frequency = 119000
        self.default_origin = str([0,0,0])
        self.default_numberOfExciterCorners = 4
        self.default_exciterCornersPositions = [[0, 0, 0], [1.9, 0, 0],[1.9, 1.26, 0],
                                                   [0, 1.26, 0],[0, 1, 0],[0, 0.75, 0],
                                                   [0, 0.5, 0], [0, 0.25, 0],[0, 0.1, 0]
                                                   ,[0, 0.05, 0]]
        self.default_tablePath = str(os.path.dirname(os.path.abspath(__file__))) + '\\tables'
        self.default_tableName = 'MagnFieldSim'
        self.default_tableAuthor = str(getpass.getuser())
        self.default_configFilePath = str(os.path.dirname(os.path.abspath(__file__))) + '\\configFiles'
        self.default_configFileName = 'config_' + self.default_tableName
        #####Exciter parameters Labelframe##########################################
        self.labelFrame_exciterParameters = LabelFrame(self, text="Exciter parameters: ")
        self.labelFrame_exciterParameters.grid(row=0, column=0, columnspan=2, sticky=W + E + N + S, padx=10, pady=(10, 0))
        self.labelFrame_exciterParameters.rowconfigure(0,weight=0)
        self.labelFrame_exciterParameters.rowconfigure(1, weight=0)
        self.labelFrame_exciterParameters.rowconfigure(2, weight=0)
        self.labelFrame_exciterParameters.rowconfigure(3, weight=0)
        self.labelFrame_exciterParameters.rowconfigure(4, weight=0)
        self.labelFrame_exciterParameters.rowconfigure(5, weight=0)
        self.labelFrame_exciterParameters.columnconfigure(0, weight=0)
        self.labelFrame_exciterParameters.columnconfigure(1, weight=0)
        self.label_numberOfWindings = Label(self.labelFrame_exciterParameters, text='Number of windings: ')
        self.label_numberOfWindings.grid(row=0, column=0, sticky=W + E + N + S, padx=10, pady=(10, 0))
        self.label_current = Label(self.labelFrame_exciterParameters, text='Current (A): ')
        self.label_current.grid(row=1, column=0, sticky=W + E + N + S, padx=10, pady=(10, 0))
        self.label_frequency = Label(self.labelFrame_exciterParameters, text='Frequency (Hz): ')
        self.label_frequency.grid(row=2, column=0, sticky=W + E + N + S, padx=10, pady=(10, 0))
        self.label_origin = Label(self.labelFrame_exciterParameters, text='Origin [x(m),y(m),z(m)]: ')
        self.label_origin.grid(row=3, column=0, sticky=W + E + N + S, padx=10, pady=(10, 0))
        self.label_numberOfExciterCorners = Label(self.labelFrame_exciterParameters, text='Number of exciter corners: ')
        self.label_numberOfExciterCorners.grid(row=4, column=0, sticky=W + E + N + S, padx=10, pady=(10, 0))

        self.entry_numberOfWindings = Entry(self.labelFrame_exciterParameters, width=10)
        self.entry_numberOfWindings.insert(0, self.default_numberOfWindings)
        self.entry_numberOfWindings.grid(row=0, column=1, sticky=W + E + N + S, padx=10, pady=(10, 0))
        self.entry_current = Entry(self.labelFrame_exciterParameters, width=10)
        self.entry_current.insert(0, self.default_current)
        self.entry_current.grid(row=1, column=1, sticky=W + E + N + S, padx=10, pady=(10, 0))
        self.entry_frequency = Entry(self.labelFrame_exciterParameters, width=10)
        self.entry_frequency.insert(0, self.default_frequency)
        self.entry_frequency.grid(row=2, column=1, sticky=W + E + N + S, padx=10, pady=(10, 0))
        self.entry_origin = Entry(self.labelFrame_exciterParameters, width=10)
        self.entry_origin.insert(0, self.default_origin)
        self.entry_origin.grid(row=3, column=1, sticky=W + E + N + S, padx=10, pady=(10, 0))
        cornersChoices = {'', 2, 3, 4, 5, 6, 7, 8, 9, 10}
        self.numberOfExciterCorners = IntVar()
        self.numberOfExciterCorners.set(self.default_numberOfExciterCorners)
        self.dropdown_numberOfExciterCorners = OptionMenu(self.labelFrame_exciterParameters, self.numberOfExciterCorners, *cornersChoices, command=self.event_show_exciter_corners)
        self.dropdown_numberOfExciterCorners.grid(row=4, column=1, sticky=W + E + N + S, padx=10, pady=(10, 0))
        self.labelFrame_exciterCorners = LabelFrame(self.labelFrame_exciterParameters, text="Corners positions: ")

        #####xyz positions Labelframe##########################################
        self.default_xPos = [0.1, 1]
        self.default_yPos = [0.1, 1]
        self.default_zPos = [0.1, 1]

        self.labelFrame_xyzPositions = LabelFrame(self, text="Simulation points: ")
        self.labelFrame_xyzPositions.grid(row=1, column=0, columnspan=2, sticky=W + E + N + S, padx=10,
                                               pady=(10, 0))
        self.labelFrame_xyzPositions.rowconfigure(0, weight=0)
        self.labelFrame_xyzPositions.rowconfigure(1, weight=0)
        self.labelFrame_xyzPositions.rowconfigure(2, weight=0)
        self.labelFrame_xyzPositions.rowconfigure(3, weight=0)
        self.labelFrame_xyzPositions.rowconfigure(4, weight=0)
        self.labelFrame_xyzPositions.rowconfigure(5, weight=0)
        self.labelFrame_xyzPositions.rowconfigure(6, weight=0)
        # self.labelFrame_xyzPositions.rowconfigure(4, weight=0)
        # self.labelFrame_xyzPositions.rowconfigure(5, weight=0)
        self.labelFrame_xyzPositions.columnconfigure(0, weight=0)
        self.labelFrame_xyzPositions.columnconfigure(1, weight=0)
        self.labelFrame_xyzPositions.columnconfigure(2, weight=0)

        self.label_x = Label(self.labelFrame_xyzPositions, text='X (m): ')
        self.label_x.grid(row=0, column=0, sticky=W + E + N + S, padx=10, pady=(10, 0))
        self.label_xListSweep = Label(self.labelFrame_xyzPositions, text='List [x1,x2,..]: ')
        self.label_xListSweep.grid(row=1, column=0, sticky=W + E + N + S, padx=10, pady=(10, 0))
        self.entry_x = Entry(self.labelFrame_xyzPositions, width=20)
        self.entry_x.insert(0, str(self.default_xPos))
        self.entry_x.grid(row=1, column=1, sticky=W + E + N + S, padx=10, pady=(10, 0))
        self.label_y = Label(self.labelFrame_xyzPositions, text='Y (m): ')
        self.label_y.grid(row=2, column=0, sticky=W + E + N + S, padx=10, pady=(10, 0))
        self.label_yListSweep = Label(self.labelFrame_xyzPositions, text='List [y1,y2,..]: ')
        self.label_yListSweep.grid(row=3, column=0, sticky=W + E + N + S, padx=10, pady=(10, 0))
        self.entry_y = Entry(self.labelFrame_xyzPositions, width=20)
        self.entry_y.insert(0, str(self.default_yPos))
        self.entry_y.grid(row=3, column=1, sticky=W + E + N + S, padx=10, pady=(10, 0))
        self.label_z = Label(self.labelFrame_xyzPositions, text='Z (m): ')
        self.label_z.grid(row=4, column=0, sticky=W + E + N + S, padx=10, pady=(10, 0))
        self.label_zListSweep = Label(self.labelFrame_xyzPositions, text='List [z1,z2,..]: ')
        self.label_zListSweep.grid(row=5, column=0, sticky=W + E + N + S, padx=10, pady=(10, 0))
        self.entry_z = Entry(self.labelFrame_xyzPositions, width=20)
        self.entry_z.insert(0, str(self.default_zPos))
        self.entry_z.grid(row=5, column=1, sticky=W + E + N + S, padx=10, pady=(10, 0))


        self.var_checkbox_sweepX = IntVar()
        self.var_checkbox_sweepX.set(0)
        self.checkbox_sweepX = Checkbutton(self.labelFrame_xyzPositions, text="Sweep", variable=self.var_checkbox_sweepX,
                                                command=self.show_sweep_x)
        self.checkbox_sweepX.grid(row=0, column=1, sticky=W + E + N + S, padx=10, pady=(10, 0))

        self.var_checkbox_sweepY = IntVar()
        self.var_checkbox_sweepY.set(0)
        self.checkbox_sweepY = Checkbutton(self.labelFrame_xyzPositions, text="Sweep",
                                           variable=self.var_checkbox_sweepY,
                                           command=self.show_sweep_y)
        self.checkbox_sweepY.grid(row=2, column=1, sticky=W + E + N + S, padx=10, pady=(10, 0))

        self.var_checkbox_sweepZ = IntVar()
        self.var_checkbox_sweepZ.set(0)
        self.checkbox_sweepZ = Checkbutton(self.labelFrame_xyzPositions, text="Sweep",
                                           variable=self.var_checkbox_sweepZ,
                                           command=self.show_sweep_z)
        self.checkbox_sweepZ.grid(row=4, column=1, sticky=W + E + N + S, padx=10, pady=(10, 0))

        self.var_checkbox_combinationOrXYZPoints = IntVar()
        self.var_checkbox_combinationOrXYZPoints.set(1)
        self.checkbox_combinationOrXYZPoints = Checkbutton(self.labelFrame_xyzPositions, text="All x,y,z combinations",
                                           variable=self.var_checkbox_combinationOrXYZPoints,
                                           command=self.combinationOrXYZPoints)
        self.checkbox_combinationOrXYZPoints.grid(row=6, column=0, sticky=W + E + N + S, padx=10, pady=(10, 0))

        self.label_infoAboutXYZ = Label(self.labelFrame_xyzPositions, text='If ticked, all the possible'+
                                                                           '\nx,y,z points combinations will be'
                                                                           '\nsimulated. The x,y,z lists do\n'
                                                                           'not have to be the same length.\n If not ticked,'
                                                                           'only the given x,y,z\nwill be simulated.'
                                                                           ' The x,y,z lists\nmust be'
                                                                           ' the same length otherwise\n you will get an error.')
        self.label_infoAboutXYZ.grid(row=7, column=0, sticky=W + E + N + S, padx=10, pady=(10, 0))

        #####Save to Labelframe##########################################
        self.labelFrame_saveLoad = LabelFrame(self, text="Save/Load: ")
        self.labelFrame_saveLoad.grid(row=0, column=2, columnspan=3, sticky=W + E + N + S, padx=10,
                                          pady=(10, 0))
        self.labelFrame_saveLoad.columnconfigure(0, weight=0)
        self.labelFrame_saveLoad.columnconfigure(1, weight=0)
        self.labelFrame_saveLoad.columnconfigure(2, weight=0)
        self.labelFrame_saveLoad.columnconfigure(3, weight=0)


        self.label_tablePath = Label(self.labelFrame_saveLoad, text="Table path: ")
        self.label_tablePath.grid(row=0, column=0, sticky=W + E + N + S, padx=10,
                                               pady=(10, 0))
        self.entry_tablePath = Entry(self.labelFrame_saveLoad, width=60)
        self.entry_tablePath.grid(row=0, column=1, sticky=W + E + N + S, padx=10,
              pady=(10, 0))
        self.entry_tablePath.insert(0, self.default_tablePath)
        self.button_browseForTablePath = Button(self.labelFrame_saveLoad, text="Browse", command=self.browse)
        self.button_browseForTablePath.grid(row=0, column=2, sticky=W + E + N + S, padx=10,
                                               pady=(10, 0))

        self.label_tableName = Label(self.labelFrame_saveLoad, text="Table name: ")
        self.label_tableName.grid(row=1, column=0, sticky=W + E + N + S, padx=10,
                                  pady=(10, 0))
        self.entry_tableName = Entry(self.labelFrame_saveLoad, width=40)
        self.entry_tableName.grid(row=1, column=1, sticky=W + E + N + S, padx=10,
                                  pady=(10, 0))
        self.entry_tableName.insert(0, self.default_tableName)

        self.label_tableAuthor = Label(self.labelFrame_saveLoad, text="Author: ")
        self.label_tableAuthor.grid(row=2, column=0, sticky=W + E + N + S, padx=10,
                                  pady=(10, 0))
        self.entry_tableAuthor = Entry(self.labelFrame_saveLoad, width=40)
        self.entry_tableAuthor.grid(row=2, column=1, sticky=W + E + N + S, padx=10,
                                  pady=(10, 0))
        self.entry_tableAuthor.insert(0, self.default_tableAuthor)

        self.label_configFilePath = Label(self.labelFrame_saveLoad, text="Congfig. file path: ")
        self.label_configFilePath.grid(row=3, column=0, sticky=W + E + N + S, padx=10,
                                    pady=(10, 0))
        self.entry_configFilePath = Entry(self.labelFrame_saveLoad, width=40)
        self.entry_configFilePath.grid(row=3, column=1, sticky=W + E + N + S, padx=10,
                                    pady=(10, 0))
        self.entry_configFilePath.insert(0, self.default_configFilePath)
        self.button_browseConfigFile = Button(self.labelFrame_saveLoad, text="Browse", command=self.browse_for_config)
        self.button_browseConfigFile.grid(row=3, column=2, sticky=W + E + N + S, padx=10,
                                        pady=(10, 0))

        self.button_saveConfigFile = Button(self.labelFrame_saveLoad, text="Save config.", command=self.save_config_file)
        self.button_saveConfigFile.grid(row=4, column=2, sticky=W + E + N + S, padx=10,
                                            pady=(10, 0))

        self.label_configFileName = Label(self.labelFrame_saveLoad, text="Congfig. file name: ")
        self.label_configFileName.grid(row=4, column=0, sticky=W + E + N + S, padx=10,
                                       pady=(10, 0))

        self.entry_configFileName = Entry(self.labelFrame_saveLoad, width=40)
        self.entry_configFileName.grid(row=4, column=1, sticky=W + E + N + S, padx=10,
                                  pady=(10, 0))
        self.entry_configFileName.insert(0, self.default_configFileName)


        self.label_saveConfigTextMessage = Label(self.labelFrame_saveLoad)
        self.label_saveConfigTextMessage.grid(row=4, column=3, sticky=W + E + N + S, padx=10,
                                  pady=(10, 0))

        self.label_loadConfigFile = Label(self.labelFrame_saveLoad, text="Load config. file: ")
        self.label_loadConfigFile.grid(row=5, column=0, sticky=W + E + N + S, padx=10,
                                       pady=(10, 0))

        self.entry_loadConfigFile = Entry(self.labelFrame_saveLoad, width=40)
        self.entry_loadConfigFile.grid(row=5, column=1, sticky=W + E + N + S, padx=10,
                                       pady=(10, 0))

        self.button_browseForLoadConfig = Button(self.labelFrame_saveLoad, text="Browse",
                                            command=self.browse_for_load_config)
        self.button_browseForLoadConfig.grid(row=5, column=2, sticky=W + E + N + S, padx=10,
                                        pady=(10, 0))

        self.button_loadConfigFile = Button(self.labelFrame_saveLoad, text="Load config.", command=self.load_config_file)
        self.button_loadConfigFile.grid(row=5, column=3, sticky=W + E + N + S, padx=10,
                                            pady=(10, 0))

        style_warningLabel = Style()
        style_warningLabel.configure("Red.TLabel", foreground="red", fontsize=40)
        self.label_warningMessage = Label(self.labelFrame_saveLoad, text='WARNING: THIS SIMULATION DOES NOT WORK WITH THE C COMIPLED CLASS_CONDUCTOER'
                                                                         '\nPLEASE RENAME IT OR ONLY USE THE PYTHON VERSION OF CLASS_CONDUCTOR')
        self.label_warningMessage["style"] = "Red.TLabel"
        self.label_warningMessage.config(font=("Courier", 14))
        self.label_warningMessage.grid(row=5, column=0, columnspan=4, sticky=W + E + N + S, padx=10,
                                              pady=(10, 0))
        #####Simulate Labelframe##########################################
        self.labelFrame_startStop = LabelFrame(self, text="Start/Stop: ")
        self.labelFrame_startStop.grid(row=1, column=2, columnspan=3, sticky=W + E + N + S, padx=10,
                                      pady=(10, 0))
        self.labelFrame_startStop.rowconfigure(0, weight=0)
        self.labelFrame_startStop.rowconfigure(1, weight=0)
        self.labelFrame_startStop.rowconfigure(2, weight=0)
        self.labelFrame_startStop.rowconfigure(3, weight=1)

        self.labelFrame_startStop.columnconfigure(0, weight=0)
        self.labelFrame_startStop.columnconfigure(1, weight=0)
        self.labelFrame_startStop.columnconfigure(2, weight=1)

        self.label_numberOfCores = Label(self.labelFrame_startStop, text='Number of cores: ')
        self.label_numberOfCores.grid(row=0, column=0, sticky=W + E + N + S, padx=10,
                                              pady=(10, 0))
        self.entry_numberOfCores = Entry(self.labelFrame_startStop, width=5)
        self.entry_numberOfCores.grid(row=0, column=1, sticky=W + E + N + S, padx=10,
                                       pady=(10, 0))
        self.entry_numberOfCores.insert(0, str(1))

        style_startButton = Style()
        style_startButton.configure("Green.TButton", foreground="green")

        self.button_startSimulation = Button(self.labelFrame_startStop, text="Start Simulation",
                                            command=self.start_simulation_thread)
        self.button_startSimulation["style"] = "Green.TButton"
        self.button_startSimulation.grid(row=1, column=0, columnspan=2, sticky=W + E + N + S, padx=10,
                                        pady=(10, 0))

        style_stopButton = Style()
        style_stopButton.configure("Red.TButton", foreground="red")

        self.button_stopSimulation = Button(self.labelFrame_startStop, text="Stop",
                                             command=self.stop_simulation)
        self.button_stopSimulation["style"] = "Red.TButton"
        self.button_stopSimulation.grid(row=2, column=0, columnspan=2, sticky=W + E + N + S, padx=10,
                                         pady=(10, 0))

        #####Output messages Labelframe##########################################
        self.labelFrame_outputMessages = LabelFrame(self.labelFrame_startStop, text="Output messages: ")
        self.labelFrame_outputMessages.grid(row=3, column=0, columnspan=3, sticky=W + E + N + S, padx=10,
                                       pady=(10, 0))
        self.labelFrame_outputMessages.rowconfigure(0, weight=0)
        self.labelFrame_outputMessages.rowconfigure(1, weight=0)
        self.labelFrame_outputMessages.columnconfigure(0, weight=1)

        self.label_outputMessages = Label(self.labelFrame_outputMessages)
        self.label_outputMessages.grid(row=0, column=0, columnspan=3, sticky=W + E + N + S, padx=10,
                                       pady=(10, 0))

    def event_show_exciter_corners(self, event):
        self.default_show_exciter_corners()

    def default_show_exciter_corners(self):
        self.labelFrame_exciterCorners.destroy()
        self.labelFrame_exciterCorners = LabelFrame(self.labelFrame_exciterParameters, text="Corners positions: ")
        self.labelFrame_exciterCorners.grid(row=5, column=0, columnspan=2, sticky=W + E + N + S, padx=10, pady=(10, 0))
        self.dict_labels_exciterCorners = {}
        self.dict_entries_exciterCorners = {}
        if self.numberOfExciterCorners.get() <= 5:
            self.labelFrame_exciterCorners.grid(row=5, column=0, columnspan=2, sticky=W + E + N + S, padx=10,
                                                pady=(10, 0))
            for i in range(self.numberOfExciterCorners.get()):
                self.labelFrame_exciterCorners.rowconfigure(i, weight=0)
                self.dict_labels_exciterCorners['corner' + str(i)] = Label(self.labelFrame_exciterCorners,
                                                                           text='Corner' + str(
                                                                               i + 1) + ' [x(m),y(m),z(m)]: ')
                self.dict_labels_exciterCorners['corner' + str(i)].grid(row=i, column=0, sticky=W + E + N + S, padx=10,
                                                                        pady=(10, 0))
                self.dict_entries_exciterCorners['corner' + str(i)] = Entry(self.labelFrame_exciterCorners, width=10)
                self.dict_entries_exciterCorners['corner' + str(i)].grid(row=i, column=1, sticky=W + E + N + S, padx=10,
                                                                         pady=(10, 0))
                self.dict_entries_exciterCorners['corner' + str(i)].insert(0,
                                                                           str(self.default_exciterCornersPositions[i]))
            self.labelFrame_exciterCorners.rowconfigure(0, weight=0)
            self.labelFrame_exciterCorners.rowconfigure(1, weight=0)

        else:
            self.labelFrame_exciterCorners.grid(row=5, column=0, columnspan=4, sticky=W + E + N + S, padx=10,
                                                pady=(10, 0))
            for i in range(5):
                self.labelFrame_exciterCorners.rowconfigure(i, weight=0)
                self.dict_labels_exciterCorners['corner' + str(i)] = Label(self.labelFrame_exciterCorners,
                                                                           text='Corner' + str(
                                                                               i + 1) + ' [x(m),y(m),z(m)]: ')
                self.dict_labels_exciterCorners['corner' + str(i)].grid(row=i, column=0, sticky=W + E + N + S, padx=10,
                                                                        pady=(10, 0))
                self.dict_entries_exciterCorners['corner' + str(i)] = Entry(self.labelFrame_exciterCorners, width=12)
                self.dict_entries_exciterCorners['corner' + str(i)].grid(row=i, column=1, sticky=W + E + N + S, padx=10,
                                                                         pady=(10, 0))
                self.dict_entries_exciterCorners['corner' + str(i)].insert(0,
                                                                           str(self.default_exciterCornersPositions[i]))
            for i in range(5, 10):
                self.dict_labels_exciterCorners['corner' + str(i)] = Label(self.labelFrame_exciterCorners,
                                                                           text='Corner' + str(
                                                                               i + 1) + ' [x(m),y(m),z(m)]: ')
                self.dict_labels_exciterCorners['corner' + str(i)].grid(row=i - 5, column=2, sticky=W + E + N + S,
                                                                        padx=10,
                                                                        pady=(10, 0))
                self.dict_entries_exciterCorners['corner' + str(i)] = Entry(self.labelFrame_exciterCorners, width=12)
                self.dict_entries_exciterCorners['corner' + str(i)].grid(row=i - 5, column=3, sticky=W + E + N + S,
                                                                         padx=10,
                                                                         pady=(10, 0))
                self.dict_entries_exciterCorners['corner' + str(i)].insert(0,
                                                                           str(self.default_exciterCornersPositions[i]))

            self.labelFrame_exciterCorners.rowconfigure(0, weight=1)
            self.labelFrame_exciterCorners.rowconfigure(1, weight=1)
            self.labelFrame_exciterCorners.rowconfigure(2, weight=1)
            self.labelFrame_exciterCorners.rowconfigure(3, weight=1)

    def show_exciter_corners(self, cornersList):
        self.labelFrame_exciterCorners.destroy()
        self.labelFrame_exciterCorners = LabelFrame(self.labelFrame_exciterParameters,text="Corners positions: ")
        self.labelFrame_exciterCorners.grid(row=5, column=0, columnspan=2,sticky=W + E + N + S, padx=10, pady=(10, 0))
        self.dict_labels_exciterCorners = {}
        self.dict_entries_exciterCorners = {}
        if self.numberOfExciterCorners.get() <=5:
            self.labelFrame_exciterCorners.grid(row=5, column=0, columnspan=2, sticky=W + E + N + S, padx=10,
                                                pady=(10, 0))
            for i in range(self.numberOfExciterCorners.get()):
                self.labelFrame_exciterCorners.rowconfigure(i, weight=0)
                self.dict_labels_exciterCorners['corner'+str(i)] = Label(self.labelFrame_exciterCorners, text='Corner' +str(i+1)+' [x(m),y(m),z(m)]: ')
                self.dict_labels_exciterCorners['corner' + str(i)].grid(row=i, column=0,sticky=W + E + N + S, padx=10, pady=(10, 0))
                self.dict_entries_exciterCorners['corner' + str(i)] = Entry(self.labelFrame_exciterCorners,width =10)
                self.dict_entries_exciterCorners['corner' + str(i)].grid(row=i, column=1, sticky=W + E + N + S, padx=10,
                                                                        pady=(10, 0))
                self.dict_entries_exciterCorners['corner' + str(i)].insert(0, str(cornersList[i]))
            self.labelFrame_exciterCorners.rowconfigure(0, weight=0)
            self.labelFrame_exciterCorners.rowconfigure(1, weight=0)

        else:
            self.labelFrame_exciterCorners.grid(row=5, column=0, columnspan=4, sticky=W + E + N + S, padx=10,
                                                pady=(10, 0))
            for i in range(5):
                self.labelFrame_exciterCorners.rowconfigure(i, weight=0)
                self.dict_labels_exciterCorners['corner' + str(i)] = Label(self.labelFrame_exciterCorners,
                                                                           text='Corner' + str(
                                                                               i + 1) + ' [x(m),y(m),z(m)]: ')
                self.dict_labels_exciterCorners['corner' + str(i)].grid(row=i, column=0, sticky=W + E + N + S, padx=10,
                                                                        pady=(10, 0))
                self.dict_entries_exciterCorners['corner' + str(i)] = Entry(self.labelFrame_exciterCorners, width=12)
                self.dict_entries_exciterCorners['corner' + str(i)].grid(row=i, column=1, sticky=W + E + N + S, padx=10,
                                                                         pady=(10, 0))
                self.dict_entries_exciterCorners['corner' + str(i)].insert(0,
                                                                           str(cornersList[i]))
            for i in range(5,10):
                self.dict_labels_exciterCorners['corner' + str(i)] = Label(self.labelFrame_exciterCorners,
                                                                           text='Corner' + str(
                                                                               i + 1) + ' [x(m),y(m),z(m)]: ')
                self.dict_labels_exciterCorners['corner' + str(i)].grid(row=i-5, column=2, sticky=W + E + N + S, padx=10,
                                                                        pady=(10, 0))
                self.dict_entries_exciterCorners['corner' + str(i)] = Entry(self.labelFrame_exciterCorners, width=12)
                self.dict_entries_exciterCorners['corner' + str(i)].grid(row=i-5, column=3, sticky=W + E + N + S, padx=10,
                                                                         pady=(10, 0))
                self.dict_entries_exciterCorners['corner' + str(i)].insert(0,
                                                                           str(cornersList[i]))

            self.labelFrame_exciterCorners.rowconfigure(0, weight=1)
            self.labelFrame_exciterCorners.rowconfigure(1, weight=1)
            self.labelFrame_exciterCorners.rowconfigure(2, weight=1)
            self.labelFrame_exciterCorners.rowconfigure(3, weight=1)




        # self.defaultNthSample = 33
        # self.defaultRecordPeriod = 10
        # self.defaultCountdown = 2
        # self.defaultTable1Path = str(os.path.dirname(os.path.abspath(__file__))) + '\\recorded_freq1.csv'
        # self.defaultTable2Path = str(os.path.dirname(os.path.abspath(__file__))) + '\\recorded_freq2.csv'
        #
        # labelFrame_Configure = LabelFrame(self, text="Configure recording parameters: ")
        # labelFrame_Configure.grid(row=0, column=0, columnspan=2, sticky=W + E + N + S, padx=10, pady=(10, 0))
        # label_nthSample = Label(labelFrame_Configure, text="Record every nth sample (integer between 1 and 2499): ")
        # label_nthSample.grid(row=0, column=0, sticky=W + E + N + S, padx=10, pady=(10, 0))
        # self.entry_nthSample = Entry(labelFrame_Configure, width=5)
        # self.entry_nthSample.grid(row=0, column=1, sticky=W + E + N + S, padx=10, pady=(10, 0))
        # self.entry_nthSample.insert(0, self.defaultNthSample)
        # label_recordPeriod = Label(labelFrame_Configure, text="Recording period in seconds: ")
        # label_recordPeriod.grid(row=1, column=0, sticky=W + E + N + S, padx=10, pady=(10, 0))
        # self.entry_recordPeriod = Entry(labelFrame_Configure, width=5)
        # self.entry_recordPeriod.grid(row=1, column=1, sticky=W + E + N + S, padx=10, pady=(10, 0))
        # self.entry_recordPeriod.insert(0, self.defaultRecordPeriod)
        #
        # label_countdown = Label(labelFrame_Configure, text="Countdown in seconds: ")
        # label_countdown.grid(row=2, column=0, sticky=W + E + N + S, padx=10, pady=(10, 0))
        # self.entry_countdown = Entry(labelFrame_Configure, width=5)
        # self.entry_countdown.grid(row=2, column=1, sticky=W + E + N + S, padx=10, pady=(10, 0))
        # self.entry_countdown.insert(0, self.defaultCountdown)

    def show_sweep_x(self):
        if self.var_checkbox_sweepX.get() == 1:
            self.label_xListSweep.configure(text='Sweep [start, end, step]: ')
            self.entry_x.delete( 0,'end')
            self.entry_x.insert(0, '[0.1,1,0.1]')
        else:
            self.label_xListSweep.configure(text='List [x1,x2,..]: ')
            self.entry_x.delete(0, 'end')
            self.entry_x.insert(0, str(self.default_xPos))

    def show_sweep_y(self):
        if self.var_checkbox_sweepY.get() == 1:
            self.label_yListSweep.configure(text='Sweep [start, end, step]: ')
            self.entry_y.delete( 0,'end')
            self.entry_y.insert(0, '[0.1,1,0.1]')
        else:
            self.label_yListSweep.configure(text='List [y1,y2,..]: ')
            self.entry_y.delete(0, 'end')
            self.entry_y.insert(0, str(self.default_yPos))

    def show_sweep_z(self):
        if self.var_checkbox_sweepZ.get() == 1:
            self.label_zListSweep.configure(text='Sweep [start, end, step]: ')
            self.entry_z.delete( 0,'end')
            self.entry_z.insert(0, '[0.1,1,0.1]')
        else:
            self.label_zListSweep.configure(text='List [z1,z2,..]: ')
            self.entry_z.delete(0, 'end')
            self.entry_z.insert(0, str(self.default_zPos))

    def combinationOrXYZPoints(self):
        pass

    def get_dir_path(self):
        toplevel = ttk.Tk()
        toplevel.withdraw()
        dirPath = fileDialog.askdirectory()
        if os.path.isdir(dirPath):
            return os.path.abspath(dirPath)
        else:
            return 0

    def get_file_path(self):
        toplevel = ttk.Tk()
        toplevel.withdraw()
        filePath = fileDialog.askopenfilename()
        if os.path.isfile(filePath):
            return filePath
        else:
            return 0

    def browse(self):
        filePath = self.get_dir_path()
        if filePath != 0:
            self.entry_tablePath.delete(0, 'end')
            style = Style()
            style.configure("Black.TEntry", foreground="black")
            self.entry_tablePath["style"] = "Black.TEntry"
            self.entry_tablePath.insert(0, filePath)
        else:
            self.entry_tablePath.delete(0, 'end')
            style = Style()
            style.configure("Red.TEntry", foreground="red")
            self.entry_tablePath["style"] = "Red.TEntry"
            self.entry_tablePath.insert(0, 'Table path was not chosen or invalid path')

    def browse_for_config(self):
        filePath = self.get_dir_path()
        if filePath != 0:
            self.entry_configFilePath.delete(0, 'end')
            style = Style()
            style.configure("Black.TEntry", foreground="black")
            self.entry_configFilePath["style"] = "Black.TEntry"
            self.entry_configFilePath.insert(0, filePath)
        else:
            self.entry_configFilePath.delete(0, 'end')
            style = Style()
            style.configure("Red.TEntry", foreground="red")
            self.entry_configFilePath["style"] = "Red.TEntry"
            self.entry_configFilePath.insert(0, 'Config. file path was not chosen or invalid path')

    def browse_for_load_config(self):
        filePath = self.get_file_path()
        if filePath != 0:
            self.entry_loadConfigFile.delete(0, 'end')
            style = Style()
            style.configure("Black.TEntry", foreground="black")
            self.entry_loadConfigFile["style"] = "Black.TEntry"
            self.entry_loadConfigFile.insert(0, filePath)
        else:
            self.entry_loadConfigFile.delete(0, 'end')
            style = Style()
            style.configure("Red.TEntry", foreground="red")
            self.entry_loadConfigFile["style"] = "Red.TEntry"
            self.entry_loadConfigFile.insert(0, 'Config. file path was not chosen or invalid path')

    def create_config_file(self, fileName):
        with open(fileName, 'w') as config:
            config.write('import numpy as np' +'\n')
            config.write('import math' + '\n')
            year, month, day, hour, minute = time.localtime()[:5]
            config.write('Date_Time = ' + "'"+'%s.%s.%s_%s:%s' % (day, month, year, hour, minute) +"'"'\n')
            config.write('Author = ' + "'"+str(self.entry_tableAuthor.get())+"'" + '\n')
            config.write('App = ' + "'" + 'Magnetic Field Simulation' + "'" + '\n')
            config.write('exciterWindings = ' + str(self.entry_numberOfWindings.get()) + '\n')
            config.write('exciterCurrent = '+str(self.entry_current.get())+'\n')
            config.write('frequency = ' + str(self.entry_frequency.get()) + '\n')
            config.write('exciterOrigin = ' + str(self.entry_origin.get()) + '\n')
            config.write('exciterCorners = ' + str(self.numberOfExciterCorners.get()) + '\n')
            for i in range(self.numberOfExciterCorners.get()):
                config.write('e{0} = '.format(i) + self.dict_entries_exciterCorners['corner'+str(i)].get() + '\n')

            if self.var_checkbox_sweepX.get() == 1:
                config.write('xOption = ' + str(1) + '\n')
                config.write('x = ' + str(self.entry_x.get()) + '\n')
            elif self.var_checkbox_sweepX.get() == 0:
                config.write('xOption = ' + str(0) + '\n')
                config.write('x = ' + str(self.entry_x.get()) + '\n')

            if self.var_checkbox_sweepY.get() == 1:
                config.write('yOption = ' + str(1) + '\n')
                config.write('y = ' + str(self.entry_y.get()) + '\n')
            elif self.var_checkbox_sweepY.get() == 0:
                config.write('yOption = ' + str(0) + '\n')
                config.write('y = ' + str(self.entry_y.get()) + '\n')

            if self.var_checkbox_sweepZ.get() == 1:
                config.write('zOption = ' + str(1) + '\n')
                config.write('z = ' + str(self.entry_z.get()) + '\n')
            elif self.var_checkbox_sweepZ.get() == 0:
                config.write('zOption = ' + str(0) + '\n')
                config.write('z = ' + str(self.entry_z.get()) + '\n')

            if self.var_checkbox_combinationOrXYZPoints.get() == 1:
                config.write('combinationOrXYZPoints = ' + str(1) + '\n')
            else:
                config.write('combinationOrXYZPoints = ' + str(0) + '\n')

    def save_config_file(self):

        filename_full = os.path.join(self.entry_configFilePath.get(), self.entry_configFileName.get()+ '.py')
        # Extend file name, if a file with the same name already exists:
        count = 0
        if (os.path.isfile(filename_full)):
            while (os.path.isfile(filename_full)):
                 count += 1
                 filename_full = os.path.join(self.entry_configFilePath.get(), self.entry_configFileName.get()+ '(%s)'% count+ '.py')
                 self.label_outputMessages.configure(text="File already exists. It is renamed to:\n"+
                                                                 self.entry_configFileName.get()+ '(%s)'% count+ '.py' + ' and saved.')
        else:
            self.label_outputMessages.configure(text="File: " +
                                                            self.entry_configFileName.get() + '.py' + ' saved.')
        self.create_config_file(filename_full)

    def load_config_file(self):
        filePath = self.entry_loadConfigFile.get()
        try:
            if os.path.isfile(filePath):
                # load the parameters values
                self.label_outputMessages.configure(text='Loading parameters values')
                #print('Loading parameters values')
                file = open(filePath, 'r')
                contents = file.readlines()
                paramDict = {}
                for elment in contents:
                    if '=' in elment:
                        key = ''
                        for character in elment:
                            if character != '=':
                                key += character
                            elif character == '=':
                                paramDict[key[0:-1]] = eval(elment[elment.index('=') + 1:-1])
                                break
                self.entry_numberOfWindings.delete(0, END)
                self.entry_numberOfWindings.insert(0, str(paramDict['exciterWindings']))
                self.entry_current.delete(0, END)
                self.entry_current.insert(0, str(paramDict['exciterCurrent']))
                self.entry_frequency.delete(0, END)
                self.entry_frequency.insert(0, str(paramDict['frequency']))
                self.entry_origin.delete(0, END)
                self.entry_origin.insert(0, str(paramDict['exciterOrigin']))
                self.numberOfExciterCorners.set(paramDict['exciterCorners'])
                cornersList = []
                for i in range(self.numberOfExciterCorners.get()):
                    cornersList.append(str(paramDict['e{0}'.format(i)]))
                self.show_exciter_corners(cornersList)
                self.var_checkbox_sweepX.set(int(paramDict['xOption']))
                self.show_sweep_x()
                self.entry_x.delete(0, 'end')
                self.entry_x.insert(0, str(paramDict['x']))
                self.var_checkbox_sweepY.set(int(paramDict['yOption']))
                self.show_sweep_y()
                self.entry_y.delete(0, 'end')
                self.entry_y.insert(0, str(paramDict['y']))
                self.var_checkbox_sweepZ.set(int(paramDict['zOption']))
                self.show_sweep_z()
                self.entry_z.delete(0, 'end')
                self.entry_z.insert(0, str(paramDict['z']))
                self.var_checkbox_combinationOrXYZPoints.set(int(paramDict['combinationOrXYZPoints']))
                self.entry_tableAuthor.delete(0, 'end')
                self.entry_tableAuthor.insert(0, str(paramDict['Author']))

        except:
            self.label_outputMessages.configure(text='Error while loading parameters values')
            #print('Error while loading parameters values')

    def start_simulation_thread(self):
        self.xyzPositionsList = self.process_positions()
        self.exciterObject = self.create_exciter_object()
        if self.xyzPositionsList != None and self.exciterObject != None:
            self.button_startSimulation.state(["disabled"])
            self.create_results_array()
            self.resultsQueue = queue.Queue()
            self.stopQueue = queue.Queue()
            self.calculateThread = Thread(
                target=calculate_magnetic_field, daemon=True,
                kwargs={'thread_queue': self.resultsQueue, 'stop_queue': self.stopQueue, 'exciterObject': self.exciterObject,
                        'xyzPositionsList': self.xyzPositionsList})
            self.calculateThread.start()
            self.button_stopSimulation.state(["!disabled"])
            self.after(self.period, self.listen_for_result)

    def stop_simulation(self):
        self.button_stopSimulation.state(["disabled"])
        self.finishedFlag = True
        self.stopQueue.put('stop')

    def listen_for_result(self):
        if self.finishedFlag == False:
            if self.resultsQueue.empty() == False:
                self.result = self.resultsQueue.get()
                self.finishedFlag = self.result[-1]
                if self.finishedFlag != 'Near Exciter':
                    self.resultsArray[self.result[self.numberOfCoulmns]] = self.result[0:self.numberOfCoulmns]
                    self.label_outputMessages.configure(text='Calculating {0}% ({1} out of total {2})'.format(int((self.result[self.numberOfCoulmns]/len(self.xyzPositionsList))*100),self.result[self.numberOfCoulmns], len(self.xyzPositionsList)))
                elif self.finishedFlag == 'Near Exciter':
                    self.label_outputMessages.configure(
                        text='Simulation inturrepted.\nError point [{0}, {1}, {2}]is less than 1 mm near to the exciter'.format(self.result[0], self.result[1], self.result[2]))
                    self.button_startSimulation.state(["!disabled"])
                    self.button_stopSimulation.state(["disabled"])
                    self.reset_flags_lists()
                    return

            self.after(self.period, self.listen_for_result)
        else:
            if self.result[self.numberOfCoulmns] == len(self.xyzPositionsList)-1:
                self.label_outputMessages.configure(
                    text='Simulation finished.\nCalculated {0} out of total {1}'.format(len(self.resultsArray), len(self.xyzPositionsList)))
                self.save_table(self.resultsArray)
                self.button_startSimulation.state(["!disabled"])
                self.button_stopSimulation.state(["disabled"])
            else:
                self.label_outputMessages.configure(
                    text='Simulation stopped.\nLast calculated position index {0} with x,y,z =[{1}, {2}, {3}] '.format(self.result[self.numberOfCoulmns], self.result[0], self.result[1], self.result[2]))
                self.button_startSimulation.state(["!disabled"])
            self.reset_flags_lists()

    def reset_flags_lists(self):
        self.finishedFlag = False
        self.resultsQueue = None
        self.stopQueue = None

    def process_positions(self):
        self.listsLength = 0
        xyz = []
        try:
            evalX = eval(self.entry_x.get())
        except:
            self.entry_x.delete(0, 'end')
            self.entry_x.insert(0, 'ERROR! It must be [Xstart, Xend, Xstep] for sweep or [x1,...] for list')
            return None
        try:
            evalY = eval(self.entry_y.get())
        except:
            self.entry_y.delete(0, 'end')
            self.entry_y.insert(0, 'ERROR! It must be [Ystart, Yend, Ystep] for sweep or [y1,...] for list')
            return None
        try:
            evalZ = eval(self.entry_z.get())
        except:
            self.entry_z.delete(0, 'end')
            self.entry_z.insert(0, 'ERROR! It must be [Zstart, Zend, Zstep] for sweep or [z1,...] for list')
            return None
        if self.var_checkbox_sweepX.get() == 1:
            if len(evalX) ==3:
                x = np.arange(evalX[0], evalX[1], evalX[2])
                self.xStart = evalX[0]
                self.xEnd = evalX[1]
                self.xStep = evalX[2]
            else:
                self.entry_x.delete(0, 'end')
                self.entry_x.insert(0, 'ERROR! It must be [Xstart, Xend, Xstep]')
                return None
        else:
            x = evalX
            self.xStart = x[0]
            self.xEnd = x[-1]

        if self.var_checkbox_sweepY.get() == 1:
            if len(evalY) ==3:
                y = np.arange(evalY[0], evalY[1], evalY[2])
                self.yStart = evalY[0]
                self.yEnd = evalY[1]
                self.yStep = evalY[2]
            else:
                self.entry_y.delete(0, 'end')
                self.entry_y.insert(0, 'ERROR! It must be [Ystart, Yend, Ystep]')
                return None
        else:
            y = evalY
            self.yStart = y[0]
            self.yEnd = y[-1]

        if self.var_checkbox_sweepZ.get() == 1:
            if len(evalZ) ==3:
                z = np.arange(evalZ[0], evalZ[1], evalZ[2])
                self.zStart = evalZ[0]
                self.zEnd = evalZ[1]
                self.zStep = evalZ[2]
            else:
                self.entry_z.delete(0, 'end')
                self.entry_z.insert(0, 'ERROR! It must be [Zstart, Zend, Zstep]')
                return  None
        else:
            z = evalZ
            self.zStart = z[0]
            self.zEnd = z[-1]

        if self.var_checkbox_combinationOrXYZPoints.get() == 1:
            self.allCombFlag = 1
            for xPoint in x:
                for yPoint in y:
                    for zPoint in z:
                        xyz.append([xPoint, yPoint, zPoint])
        else:
            self.allCombFlag = 0
            if len(x) == len(y) and len(y) == len(z):
                for i in range(len(x)):
                    xyz.append([x[i], y[i], z[i]])
            else:
                self.label_outputMessages.configure(text='ERROR!!! X, Y, Z lists must have the same length.')
                return None

        return xyz

    def create_results_array(self):
        self.resultsArray = np.empty([len(self.xyzPositionsList), self.numberOfCoulmns])

    def create_exciter_object(self):
        exciterShape = []
        for i in range(len(self.dict_entries_exciterCorners)):
            try:
                e = eval(self.dict_entries_exciterCorners['corner' + str(i)].get())
                exciterShape.append(np.array([e[0], e[1], e[2]]))
            except:
                self.dict_entries_exciterCorners['corner' + str(i)].delete(0, 'end')
                self.dict_entries_exciterCorners['corner' + str(i)].insert(0, 'ERROR! it must be [x,y,z]')
                return None

        try:
            o = eval(self.entry_origin.get())
            exciterOrigin = np.array([o[0], o[1], o[2]])
        except:
            self.entry_origin.delete(0, 'end')
            self.entry_origin.insert(0, 'ERROR! it must be [x,y,z]')
            return None

        try:
            w = eval(self.entry_numberOfWindings.get())
            exciterWindings = int(w)
        except:
            self.entry_numberOfWindings.delete(0, 'end')
            self.entry_numberOfWindings.insert(0, 'ERROR! it must be positive integer')
            return None

        try:
            current = eval(self.entry_current.get())
            exciterCurrent = float(current)
        except:
            self.entry_current.delete(0, 'end')
            self.entry_current.insert(0, 'ERROR! it must be positive integer')
            return None

        return c.Polygon(exciterShape, exciterWindings, exciterOrigin, current=exciterCurrent)

    def save_table(self, resultsArray):
        header= self.create_header()
        filename_full = os.path.join(self.entry_tablePath.get(), self.entry_tableName.get() + '.csv')
        # Extend file name, if a file with the same name already exists:
        count = 0
        if (os.path.isfile(filename_full)):
            while (os.path.isfile(filename_full)):
                count += 1
                filename_full = os.path.join(self.entry_tablePath.get(),
                                             self.entry_tableName.get() + '(%s)' % count + '.csv')
                tableExistFlag = 1
        else:
            tableExistFlag = 0

        tempResult = np.empty([np.shape(resultsArray)[0], np.shape(resultsArray)[1] + 1])
        for i in range(len(resultsArray)):
            hSum = np.sqrt(sum([np.square(resultsArray[i][3]), np.square(resultsArray[i][4]), np.square(resultsArray[i][5])]))
            for ii in range(7):
                if ii < 6:
                    tempResult[i][ii] = resultsArray[i][ii]
                else:
                    tempResult[i][ii] = hSum
            #self.of.write(';'.join(str(tempResult[i])))
            #self.of.flush()
        np.savetxt(filename_full , tempResult, fmt='%10.5E', delimiter=';', header=header)
        if tableExistFlag == 1:
            self.label_outputMessages.configure(
                text=self.label_outputMessages.cget("text") + "\n" + "Table already exists. It is renamed and saved in:\n" +filename_full)
        else:
            self.label_outputMessages.configure(
                text=self.label_outputMessages.cget(
                    "text") + "\n" + "Table saved in:\n" + filename_full)

    def create_header(self):

        # Configure column heading:
        column_heading = 'X-position;Y-position;Z-position;Hx;Hy;Hz;' \
                         'Hsum (sqrt(Hx^2 + Hy^2 + Hz^2))'

        if self.var_checkbox_sweepX.get() == 1:
            xheader = 'X: {0} ... {1} ({2}) [m]'.format(self.xStart, self.xEnd, self.xStep)
        else:
            xheader = 'X: {0} ... {1}  [m]'.format(self.xStart, self.xEnd)

        if self.var_checkbox_sweepY.get() == 1:
            yheader = 'Y: {0} ... {1} ({2}) [m]'.format(self.yStart, self.yEnd, self.yStep)
        else:
            yheader = 'Y: {0} ... {1}  [m]'.format(self.yStart, self.yEnd)

        if self.var_checkbox_sweepZ.get() == 1:
            zheader = 'Z: {0} ... {1} ({2}) [m]'.format(self.zStart, self.zEnd, self.zStep)
        else:
            zheader = 'Z: {0} ... {1}  [m]'.format(self.zStart, self.zEnd)

        if self.allCombFlag == 1:
            combintationHeader = 'All'
        else:
            combintationHeader = 'Unique'

        # Get current date:
        year, month, day, hour, minute = time.localtime()[:5]


        # Put all parts together:
        header = '--- Magnetic Field Simulation table ---\n   ' \
                 '%s.%s.%s\n   ' % (day, month, year) \
                 + self.entry_tableAuthor.get() + '\n' \
                 + 'Positions:'  + '\n' \
                 + xheader + '\n' \
                 + yheader + '\n' \
                 + zheader + '\n' \
                 + combintationHeader + '\n' \
                 + column_heading
        return header

    def activate(self):
        self.visible = True

    def deactivate(self):
        self.visible = False

class GUI_magneticFieldPlotTab(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        self.create_widgets()

    def create_widgets(self):
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=0)
        self.rowconfigure(2, weight=0)
        self.rowconfigure(3, weight=0)
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=0)
        self.columnconfigure(2, weight=0)
        self.columnconfigure(3, weight=0)
        #########Default values########################################

        #####Load table Labelframe##########################################
        self.labelFrame_load = LabelFrame(self, text="Load table: ")
        self.labelFrame_load.grid(row=0, column=0, columnspan=3, sticky=W + E + N + S, padx=10,
                                      pady=(10, 0))
        self.labelFrame_load.columnconfigure(0, weight=0)
        self.labelFrame_load.columnconfigure(1, weight=0)
        self.labelFrame_load.columnconfigure(2, weight=0)

        self.label_tablePath = Label(self.labelFrame_load, text="Table* path: ")
        self.label_tablePath.grid(row=0, column=0, sticky=W + E + N + S, padx=10,
                                  pady=(10, 0))
        self.entry_tablePath = Entry(self.labelFrame_load, width=80)
        self.entry_tablePath.grid(row=0, column=1, sticky=W + E + N + S, padx=10,
                                  pady=(10, 0))
        #self.entry_tablePath.insert(0, self.default_tablePath)
        self.button_browseForTablePath = Button(self.labelFrame_load, text="Browse", command=self.browse)
        self.button_browseForTablePath.grid(row=0, column=2, sticky=W + E + N + S, padx=10,
                                            pady=(10, 0))

        self.label_tableInstructions = Label(self.labelFrame_load, text='*The chosen table must be generated from the Simulation tab with correct header information.')
        self.label_tableInstructions.grid(row=1, column=1, sticky=W + E + N + S, padx=10,
              pady=(10, 0))

        self.button_loadTable = Button(self.labelFrame_load, text="Load", command=self.loadTable)
        self.button_loadTable.grid(row=1, column=2, sticky=W + E + N + S, padx=10,
                                            pady=(10, 0))
        #####Table data Labelframe##########################################
        self.labelFrame_tableData = LabelFrame(self, text="Table data: ")
        self.labelFrame_tableData.grid(row=0, column=3, columnspan=4, sticky=W + E + N + S, padx=10,
                                  pady=(10, 0))
        self.labelFrame_tableData.columnconfigure(0, weight=0)
        self.labelFrame_tableData.columnconfigure(1, weight=0)
        self.labelFrame_tableData.columnconfigure(2, weight=0)
        self.labelFrame_tableData.columnconfigure(3, weight=0)
        self.labelFrame_tableData.columnconfigure(4, weight=0)
        self.labelFrame_tableData.columnconfigure(5, weight=0)

        self.label_dateCreated = Label(self.labelFrame_tableData, text="Date created: ")
        self.label_dateCreated.grid(row=0, column=0, sticky=W + E + N + S, padx=10,
                                  pady=(10, 0))
        self.label_dateCreatedValue = Label(self.labelFrame_tableData)
        self.label_dateCreatedValue.grid(row=0, column=1, sticky=W + E + N + S, padx=10,
                                    pady=(10, 0))

        self.label_author = Label(self.labelFrame_tableData, text="Author: ")
        self.label_author.grid(row=0, column=2, sticky=W + E + N + S, padx=10,
                                    pady=(10, 0))
        self.label_authorValue = Label(self.labelFrame_tableData)
        self.label_authorValue.grid(row=0, column=3, sticky=W + E + N + S, padx=10,
                                         pady=(10, 0))

        self.label_xyzComb = Label(self.labelFrame_tableData, text="Positions combination: ")
        self.label_xyzComb.grid(row=0, column=4, sticky=W + E + N + S, padx=10,
                               pady=(10, 0))
        self.label_xyzCombValue = Label(self.labelFrame_tableData)
        self.label_xyzCombValue.grid(row=0, column=5, sticky=W + E + N + S, padx=10,
                                    pady=(10, 0))



        self.label_simulatedX = Label(self.labelFrame_tableData, text="Simulated X(m): ")
        self.label_simulatedX.grid(row=1, column=0, sticky=W + E + N + S, padx=10,
                               pady=(10, 0))
        self.label_simulatedXValue = Label(self.labelFrame_tableData)
        self.label_simulatedXValue.grid(row=1, column=1, sticky=W + E + N + S, padx=10,
                                    pady=(10, 0))

        self.label_simulatedY = Label(self.labelFrame_tableData, text="Simulated Y(m): ")
        self.label_simulatedY.grid(row=1, column=2, sticky=W + E + N + S, padx=10,
                                   pady=(10, 0))
        self.label_simulatedYValue = Label(self.labelFrame_tableData)
        self.label_simulatedYValue.grid(row=1, column=3, sticky=W + E + N + S, padx=10,
                                        pady=(10, 0))
        self.label_simulatedZ = Label(self.labelFrame_tableData, text="Simulated Z(m): ")
        self.label_simulatedZ.grid(row=1, column=4, sticky=W + E + N + S, padx=10,
                                   pady=(10, 0))
        self.label_simulatedZValue = Label(self.labelFrame_tableData)
        self.label_simulatedZValue.grid(row=1, column=5, sticky=W + E + N + S, padx=10,
                                        pady=(10, 0))



    def browse(self):
        filePath = self.get_file_path()
        if filePath != 0:
            self.entry_tablePath.delete(0, 'end')
            style = Style()
            style.configure("Black.TEntry", foreground="black")
            self.entry_tablePath["style"] = "Black.TEntry"
            self.entry_tablePath.insert(0, filePath)
        else:
            self.entry_tablePath.delete(0, 'end')
            style = Style()
            style.configure("Red.TEntry", foreground="red")
            self.entry_tablePath["style"] = "Red.TEntry"
            self.entry_tablePath.insert(0, 'Table path was not chosen or invalid path')

    def get_file_path(self):
        toplevel = ttk.Tk()
        toplevel.withdraw()
        filePath = fileDialog.askopenfilename()
        if os.path.isfile(filePath):
            return filePath
        else:
            return 0

    def loadTable(self):
        filename = self.entry_tablePath.get()
        self.update_table_data(filename)
        tableContent = np.loadtxt(filename, comments='#',delimiter=';')

    def update_table_data(self, filename):
        with open(filename) as table:
            reader = csv.reader(table, delimiter=';')
            for i, row in enumerate(reader):
                if i == 1:
                    self.label_dateCreatedValue.configure(text= str(row[0][4:]))
                elif i == 2:
                    self.label_authorValue.configure(text=str(row[0][4:]))
                elif i == 4:
                    self.label_simulatedXValue.configure(text=str(row[0][3:]))
                elif i == 5:
                    self.label_simulatedYValue.configure(text=str(row[0][3:]))
                elif i == 6:
                    self.label_simulatedZValue.configure(text=str(row[0][3:]))
                elif i == 7:
                    self.label_xyzCombValue.configure(text=str(row[0][3:]))


    def activate(self):
        self.visible = True

    def deactivate(self):
        self.visible = False


## class GUI_aboutTab
class GUI_aboutTab(GUI):
    def __init__(self,scale_factor, screenWidth, screenHeight, paramFontSize, titleFontSize, frame):
        self.scale_factor = scale_factor
        self.screenWidth = screenWidth
        self.screenHeight = screenHeight
        self.paramFontSize = paramFontSize
        self.titleFontSize = titleFontSize
        self.frame = frame
        self.tab3 = self.create_add_tab(self.frame, None, None, 'About')
        self.logo_img = Image.open('iis_logo.png')
        self.lblLogo = Label(self.tab3)
        self.lblLogo.grid(row=0, column=0, sticky=W)#, padx=100, pady=(10, 0))
        img = ImageTk.PhotoImage(self.logo_img.resize((int(float(self.logo_img.size[0])/15),int(float(self.logo_img.size[1])/15)),Image.ANTIALIAS))
        #img = ImageTk.PhotoImage(self.logo_img)
        self.lblLogo.configure(image=img)
        self.lblLogo.image = img
        label_legalInfo = Label(self.tab3, text='\n'+'\n'+'The'+'\n'+
                                            'Fraunhofer Institute for Integrated Circuits IIS' +'\n'+
                                                'Am Wolfsmantel 33' +'\n'+
                                                    '91058 Erlangen'+'\n'+
                                                    'Germany'+'\n'+'\n'+
                                                    'is a constituent entity of the Fraunhofer-Gesellschaft,'+'\n'+
                                                    'and as such has no separate legal status.'+'\n'+'\n'+

                                                    'Fraunhofer-Gesellschaft'+'\n'+
                                                    'zur Foerderung der angewandten Forschung e.V.'+'\n'+
                                                    'Hansastrasse 27 c'+'\n'+
                                                    '80686 Muenchen'+'\n'+
                                                    'Internet: www.fraunhofer.de'+'\n'+
                                                    'E-Mail: info(at)zv.fraunhofer.de' +'\n'+'\n'+
                                                'Contact:'+'\n'+
                                                'Ibrahim Ibrahim' + '\n' +
                                                'ibrahim.ibrahim@iis.fraunhofer.de' + '\n' +'\n'+
                                                'Usage rights:'+'\n'+
                                                'Copyright by Fraunhofer - Gesellschaft 2018.' +'\n'+
                                                '\n' +
                                                'All rights reserved.'+'\n'+
                                                'All copyright are owned in full by the Fraunhofer-Gesellschaft.'+'\n'+
                                                '\n' +
                                                'Disclaimer:'+'\n'+
                                                'Registered trademarks and proprietary names, and copyrighted text and images,' + '\n' +
                                                'are not generally indicated as such. But the absence of such indications in' + '\n' +
                                                'no way implies that these names, images or text belong to the public domain' + '\n' +
                                                'in the context of trademark or copyright law.' + '\n'
        ,font=("Arial", self.paramFontSize-1))
        label_legalInfo.grid(row=1,column=0)

    def logo_resize(self,event):
        aspect = float(self.logo_img.size[1]) / self.logo_img.size[0]
        width = self.lblLogo.winfo_width()
        height = int(width * aspect)
        img = ImageTk.PhotoImage(self.logo_img.resize((width, height)))
        self.lblLogo.configure(image=img)
        self.lblLogo.image = img

class couplingFactor:
    """!
    Restore the absolute coupling factor between two coils:
        1. Compute the coupling factor from the first coil to the second:
            - Compute the flux through the primary coil
            - Compute the flux through the secondary coil
            - Divide the flux of the second coil through the first.
        2. Interconvert the two coils.
        3. Do the same as in step 1 again.
        4. Compute the absolute coupling factor.
    """
    def __init__(self, primaryCoil, secondaryCoil):
        """!
        @param primaryCoil:         coil one, with all it's attributes

        @param secondaryCoil:       coil two, with all it's attributes
        """


        self.primaryCoil = primaryCoil
        self.secondaryCoil = secondaryCoil
        self.primaryCoil_flux = 0
        self.secondaryCoil_flux = 0
        self.absoluteCouplingFactor = 0
        self.frequency = 1 / (2 * np.pi)
        self.primaryCoil_flux = self.get_primaryCoil_flux(self.primaryCoil.inductance, self.primaryCoil.current)
        self.secondaryCoil_flux = self.secondaryCoil.inducedVoltage(self.primaryCoil, self.frequency)
        self.couplingFactor = self.get_couplingFactor(self.secondaryCoil_flux, self.primaryCoil_flux)
    def get_computedCouplingFactor(self):
        return self.couplingFactor
    def get_couplingFactor(self, flux2, flux1):
        """!
        This function computes the coupling factor.

        @param  flux1:              flux of the secondary coil
        @param  flux2:              flux of the primary coil

        @return                     coupling factor between these two coils
        """
        self.flux1 = flux1
        self.flux2 = flux2

        return np.abs(self.flux2 / self.flux1)
    def get_primaryCoil_flux(self, inductance, current):
        """!
        This function computes the coupling factor.

        @param  inductance:         inductance of the primary coil
        @param  current:            current flux of the primary coil

        @return                     computed flux of the primary coil
        """
        self.inductance = inductance
        self.current = current


        return self.inductance * self.current
    def get_absoluteCouplingFactor(couplingFactor21, couplingFactor12):
        return  np.sqrt(couplingFactor21 * couplingFactor12)

class PolygonExtended(c.Polygon):
    """!
       Subclass of Polygon with more atrributes.
       The shape is defined by a list of support vectors, which define the corners of the polygon.

       Attributes inherited from Conductor:
           - windings
           - position               > default:  np.array([0, 0, 0])
           - orientation            > default:  np.eye(3)
           - current                > default:  0

       Attributes inherited from Polygon:
           - vector_list            = list of vectors, which specify the corners of the polygon
           - closed_loop            = if <B>True</B>, last point of support_vectors will be connected to first point, to obtain a closed current loop
                                    if <B>False</B>, the loop will not be closed

       Further Attributes:
           - winding_width          = width of one winding
           - winding_distance       = distance between the several windings
           - radius                 = radius of the Polygon wire
           - inductance             = inductance of the Polygon
    """
    def __init__(self, vector_list, winding_width, winding_distance, radius, inductance, resistance,
                 windings, current=0, position=np.array([0, 0, 0]), rotationMatrix=np.eye(3), closed_loop=True):

        """!
        @param vector_list          list of vectors, which specify the corners of the polygon

        @winding_width              width of one winding

        @winding_distance           distance between the several windings

        @param radius:              radius of the exciter wire

        @param inductance:          inductance of the exciter wire
        @param inductance:          resistance of the exciter wire

        @param windings:            number of windings

        @param current:             current, running through the object <BR>
                                    <I> > default value: </I>  0

        @param position:            origin of the object's coordinate system <BR>
                                    <I> > default value: </I>  <B> np.array([0, 0, 0]) </B>

        @param rotationMatrix:      states the orientation of the object in relation to the inertial frame; <BR>
                                    see documentation of Conductor for detailed information <BR>
                                    <I> > default value: </I>  <B> np.eye(3) </B>

        @param closed_loop:         if <B>True</B>, last point of support_vectors will be connected to first point, to obtain a closed current loop
                                    if <B>False</B>, the loop will not be closed

        """
        self.vector_list = vector_list
        self.winding_width = winding_width
        self.winding_distance = winding_distance
        self.radius = radius
        self.inductance = inductance
        self.resistance = resistance

        ## Width:
        w = 0
        ## Length:
        l = 0
        ## Height:
        h = 0

        ## Compute the circumference of the Polygon:
        for corner in self.vector_list:
            w += abs(corner[0])
            l += abs(corner[1])
            h += abs(corner[2])
        self.circumference = (w + l + h ) * 100    #[cm]

        ## For type exciter compute the inductance if not given in config:
        if self.radius != None:
            if self.inductance == None:
                self.inductance = 2 * self.circumference * (np.log(self.circumference / self.radius) + 4.0 *
                                                       self.radius / self.circumference - 1.91) * 1e-9
            else:
                self.inductance = inductance

        ## For type antenna with no radius compute the inductance if not given in config:
        elif self.radius == None:
            if self.inductance == None:
                self.inductance = 0
            else:
                self.inductance = inductance


        c.Polygon.__init__(self, vector_list=self.vector_list, windings=windings, position=position,
                         rotationMatrix=rotationMatrix, current=current, closed_loop=closed_loop)

class EllipseExtended(c.Ellipse):
    """!
    Subclass of Ellipse with more attributes.
    The shape is given implicit by the parameters 'majorAxis' and 'minorAxis', which characterise an ellipse.
    The elliptic coil is located at the origin of the x-y-plane of it's coordinate system with it's major axis
    coinciding with the x-axis and the minor axis coinciding with the y-axis of the plane.

    Attributes inherited from conductor:
        - windings
        - position                  > default:  np.array([0, 0, 0])
        - orientation               > default:  np.eye(3)
        - current                   > default:  0

    Attributes inherited from Ellipse:
        - majorAxis                 = major Axis of the ellipse (coincides with the X-axis of it's coordinate system)
        - minorAxis                 = minor Axis of the ellipse (coincides with the Y-axis of it's coordinate system)

    Further Attributes:
        - resistance                = resistance of the ellipse
        - inductance                = inductance of the ellipse, without induced inductance
    """
    def __init__(self, majorAxis, minorAxis, windings, resistance, inductance, position=np.array([0, 0, 0]), rotationMatrix=np.eye(3), current=0):
        """!
        @param  majorAxis:          major Axis of the ball (coincides with the X-axis of it's coordinate system)

        @param  minorAxis:          minor Axis of the ball (coincides with the Y-axis of it's coordinate system)

        @param  windings:           number of windings

        @param  resistance:         resistance of the ball

        @param  inductance:         inductance of the ball

        @param  position:           origin of the object's coordinate system <BR>
                                    <I> > default value: </I>  <B> np.array([0, 0, 0]) </B>

        @param  rotationMatrix:     states the orientation of the object in relation to the inertial frame; <BR>
                                    see documentation of Conductor for detailed information <BR>
                                    <I> > default value: </I>  <B> np.eye(3) </B>

        @param  current:            current, running through the object <BR>
                                    <I> > default value: </I>  0
        """
        self.resistance = resistance
        self.inductance = inductance

        ## Compute the inductance if not given in config:
        if self.inductance == None:
            self.inductance = 0

        c.Ellipse.__init__(self, majorAxis, minorAxis, windings, position, rotationMatrix, current)

class BallExtended(c.Ball):
    """!
    Subclass of Ball with more attributes.
    An object of class Ball consists of 3 orthogonal round coils, which are each instances of class Conductor.Ellipse. <BR>
    Every coil is named after the orientation of it's normal vector (-> coilX, coilY, coilZ).

    Attributes inherited from conductor:
            - windings
            - position              > default:  np.array([0, 0, 0])
            - orientation           > default:  np.eye(3)
            - current               > default:  0

    Attributes inherited from Ball:
            - coilX                 = coil with normal vector in X-direction of the ball frame
            - coilY                 = coil with normal vector in Y-direction of the ball frame
            - coilZ                 = coil with normal vector in Z-direction of the ball frame

            - c_X                   = transformation matrix from coilX- to ball-frame
            - c_Y                   = transformation matrix from coilY- to ball-frame
            - radius                = radius of the three balls

    Further Attributes:
            - resistance            = resistance of the ball
            - inductance            = inductance of the ball

    """
    def __init__(self, radius,  windings, resistance, inductance, position=np.array([0, 0, 0]), rotationMatrix=np.eye(3), current=np.array([0,0,0])):
        """!
        @param radius:              ball radius = radius of the three balls

        @param windings:            number of windings per coil

        @param resistance:          resistance of the ball

        @param inductance:          inductance of the ball

        @param position:            ball position = origin of the three coil coordinate systems

        @param rotationMatrix:      orientation of coilZ <BR>
                                    see documentation of Conductor for detailed information <BR>
                                    <I> > default value: </I>  <B> np.eye(3) </B>

        @param current:             np.array([self.coilX.current, self.coilY.current, self.coilZ.current]) <BR>
                                    <I> > default value: </I> <B> np.array([0, 0, 0]) </B>
        """
        self.resistance = resistance
        self.inductance = inductance

        ## Compute the inductance if not given in config:
        if self.inductance == None:
            self.inductance = 0

        c.Ball.__init__(self, radius, windings, position, rotationMatrix, current)

class PukExtended(c.Puk):
    """!
    Subclass of Puk with more attributes.

    An object of class ball consists of 3 orthogonal coils:
     - 1 round coil of class Conductor.Ellipse (orientation: Y)
     - 2 rectangular coils of class Conductor.Polygon (orientation: X and Z)
     Every coil is named after the orientation of it's normal vector (-> coilX, coilY, coilZ).

     Attributes inherited from conductor:
            - windings
            - position              > default:  np.array([0, 0, 0])
            - orientation           > default:  np.eye(3)
            - current               > default:  0

    Attributes inherited from Puk:
            - coilX                 = coil with normal vector in X-direction of the ball frame
            - coilY                 = coil with normal vector in Y-direction of the ball frame
            - coilZ                 = coil with normal vector in Z-direction of the ball frame

            - c_X                   = transformation matrix from coilX- to ball-frame
            - c_Y                   = transformation matrix from coilY- to ball-frame

            - radius                = radius of the circular Puk's Y-coil
            - height                = height of the Puk's rectangular coils

    Further Attributes:
            - resistance            = resistance of the ball
            - inductance            = inductance of the ball
    """
    def __init__(self, radius, height, windings, resistance, inductance, position=np.array([0, 0, 0]), rotationMatrix=np.eye(3), current=np.array([0,0,0])):
        """!
        @param radius:              radius of the circular Puk's Y-coil

        @param height:              height of the Puk's rectangular coils

        @param windings:            number of windings per coil

        @param resistance:          resistance of the Puk

        @param inductance:          inductance of the Puk

        @param position:            Puk position = origin of the three coil coordinate systems

        @param rotationMatrix:      orientation of coilZ <BR>
                                    see documentation of Conductor for detailed information <BR>
                                    <I> > default value: </I>  <B> np.eye(3) </B>

        @param current:             np.array([self.coilX.current, self.coilY.current, self.coilZ.current]) <BR>
                                    <I> > default value: </I> <B> np.array([0, 0, 0]) </B>
        """
        self.resistance = resistance
        self.inductance = inductance

        ## Compute the inductance if not given in config:
        if self.inductance == None:
            self.inductance = 0

        c.Puk.__init__(self, radius, height, windings, position, rotationMatrix, current)


c = GUI()


















