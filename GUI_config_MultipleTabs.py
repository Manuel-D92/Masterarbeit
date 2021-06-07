## @package config_GUI_MultipleTabs
# @brief A python logic to get the configured parameters values from the GUI genrated
# configFile.py and to create class_conductor objects which will be used
# for further calculations.
#@author ihm
#@version 2.0
#@date Created on Fri Dec 18 11:37:12 2015, Modified on Tue Dec 19 10:26:50 2017

import numpy as np
import class_conductor as c
import math
import config_DWRegal as configFile
from config_DWRegal import *
__author__ = 'ihm'



exciterWindings = configFile.exciterWindings
exciterCurrent = configFile.exciterCurrent
frequency = configFile.frequency
exciterCorners = configFile.registeredExciterCorners
## exciter shape (relative to the exciter coordinate system):
exciterShape = []
for i in range(1, exciterCorners+1):
    e = np.array(eval("e%s" % str(i)))
    exciterShape.append(e)



## Origin of the exciter coordinate system (relative to the inertial coordinate system):
exciterPosition = np.array(configFile.exciterPosition)

## Generate a Polygonal exciter object
exciter = c.Polygon(exciterShape, exciterWindings, exciterPosition, current=exciterCurrent)



## Coil data

coilWindings = configFile.coilWindings
coilResistance = configFile.coilResistance
objectType = configFile.objectType

if configFile.objectType != 0:
    if configFile.objectType == 'Polygon':
        coilLength = configFile.coilLength
        coilWidth = configFile.coilWidth
        c0 = np.array([-coilLength/2, -coilWidth/2, 0])
        c1 = np.array([coilLength/2, -coilWidth/2, 0])
        c2 = np.array([coilLength/2, coilWidth/2, 0])
        c3 = np.array([-coilLength/2, coilWidth/2, 0])
        coilShape = [c0, c1, c2, c3]
        ## Generate a Polygonal object
        coil = c.Polygon(coilShape, coilWindings)
    elif configFile.objectType == 'Ellipse':
        majorAxisLength = configFile.majorAxisLength
        minorAxisLength = configFile.minorAxisLength
        ## Generate an Elliptical object
        coil = c.Ellipse(majorAxisLength, minorAxisLength, coilWindings)
    elif configFile.objectType == 'Puk':
        pukHeight = configFile.pukHeight
        circularCoilRadius = configFile.circularCoilRadius
        ## Generate a Puk object
        coil = c.Puk(circularCoilRadius, pukHeight, coilWindings)
    elif configFile.objectType == 'Ball':
        coilsRadius = configFile.coilsRadius
        ## Generate a Ball object
        coil = c.Ball(coilsRadius, coilWindings)
    elif configFile.objectType == 'Wearable':
        coil1Width = configFile.coil1Width
        coil2Width = configFile.coil2Width
        coil3Width = configFile.coil3Width
        coil1Length = configFile.coil1Length
        coil2Length = configFile.coil2Length
        coil3Length = configFile.coil3Length
        ## Generate a Ball object
        coil = c.Wearable(coil1Width, coil1Length,
                           coil2Width, coil2Length,
                           coil3Width, coil3Length,
                           coilWindings)
else:
    print('No coil object was selected. Please select a Polygon or an Ellipse or a Puk or a Ball.')
    exit()

### Antennas data
#antennaList = []
#if configFile.frameAntennaCheckBox == 1:
#    if configFile.registeredFrameAntennas != 0:
#        frameAntennaWindings = configFile.frameAntennaWindings
#        frameAntennaLength = configFile.frameAntennaLength
#        frameAntennaHeight = configFile.frameAntennaHeight
#        aF0 = np.array([-frameAntennaLength/2, -frameAntennaHeight/2, 0])
#        aF1 = np.array([+frameAntennaLength/2, -frameAntennaHeight/2, 0])
#        aF2 = np.array([+frameAntennaLength/2, +frameAntennaHeight/2, 0])
#        aF3 = np.array([-frameAntennaLength/2, +frameAntennaHeight/2, 0])
#        ## Frame antennas
#        #  shape
#        frameShape = [aF0, aF1, aF2, aF3]
#
#
#        for i in range(1, configFile.registeredFrameAntennas + 1):
#            frameAntennaOrientation = eval("frameAntennaOrientation%s" %str(i))
#            c_x = np.array([[1, 0, 0],
#                            [0, np.cos(math.radians(frameAntennaOrientation[0])), -np.sin(math.radians(frameAntennaOrientation[0]))],
#                            [0, np.sin(math.radians(frameAntennaOrientation[0])), np.cos(math.radians(frameAntennaOrientation[0]))]])
#
#            c_y = np.array([[np.cos(math.radians(frameAntennaOrientation[1])), 0, np.sin(math.radians(frameAntennaOrientation[1]))],
#                            [0, 1, 0],
#                            [-np.sin(math.radians(frameAntennaOrientation[1])), 0, np.cos(math.radians(frameAntennaOrientation[1]))]])
#
#            c_z = np.array([[np.cos(math.radians(frameAntennaOrientation[2])), -np.sin(math.radians(frameAntennaOrientation[2])), 0],
#                            [np.sin(math.radians(frameAntennaOrientation[2])), np.cos(math.radians(frameAntennaOrientation[2])), 0],
#                            [0, 0, 1]])
#            rotationMatrix = np.dot(c_z, np.dot(c_y, c_x))
#            antennaList.append(c.Polygon(frameShape, frameAntennaWindings,
#                                    np.array(eval("frameAntennaPosition%s" %str(i))),
#                                         rotationMatrix))
#
antennaList = configFile.antennaList
#
#if configFile.mainAntennaCheckBox == 1:
#    if configFile.registeredMainAntennas:
#        mainAntennaWindings = configFile.mainAntennaWindings
#        mainAntennaLength = configFile.mainAntennaLength
#        mainAntennaHeight = configFile.mainAntennaHeight
#
#        aM0 = np.array([-mainAntennaLength/2, 0, -mainAntennaHeight/2])
#        aM1 = np.array([+mainAntennaLength/2, 0, -mainAntennaHeight/2])
#        aM2 = np.array([+mainAntennaLength/2, 0, +mainAntennaHeight/2])
#        aM3 = np.array([-mainAntennaLength/2, 0, +mainAntennaHeight/2])
#        ## Main antennas
#        #  shape
#        mainShape = [aM0, aM1, aM2, aM3]
#        mainAntennasDict = {}
#        for i in range(1, configFile.registeredMainAntennas + 1):
#            mainAntennaOrientation = eval("mainAntennaOrientation%s" % str(i))
#
#            c_x = np.array([[1, 0, 0],
#                            [0, np.cos(math.radians(mainAntennaOrientation[0])),
#                             -np.sin(math.radians(mainAntennaOrientation[0]))],
#                            [0, np.sin(math.radians(mainAntennaOrientation[0])),
#                             np.cos(math.radians(mainAntennaOrientation[0]))]])
#
#            c_y = np.array([[np.cos(math.radians(mainAntennaOrientation[1])), 0,
#                             np.sin(math.radians(mainAntennaOrientation[1]))],
#                            [0, 1, 0],
#                            [-np.sin(math.radians(mainAntennaOrientation[1])), 0,
#                             np.cos(math.radians(mainAntennaOrientation[1]))]])
#
#            c_z = np.array([[np.cos(math.radians(mainAntennaOrientation[2])),
#                             -np.sin(math.radians(mainAntennaOrientation[2])), 0],
#                            [np.sin(math.radians(mainAntennaOrientation[2])),
#                             np.cos(math.radians(mainAntennaOrientation[2])), 0],
#                            [0, 0, 1]])
#            rotationMatrix = np.dot(c_z, np.dot(c_y, c_x))
#            antennaList.append(c.Polygon(mainShape, mainAntennaWindings, np.array(eval("mainAntennaPosition%s" %str(i))),
#                                         rotationMatrix))
#
#
#

