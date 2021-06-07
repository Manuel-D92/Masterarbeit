# -*- coding: utf-8 -*-
"""
Created on Tue Sep 08 2020

@author: mler
"""

import numpy as np
import class_conductor as c

########################################################################################################################
# Exciter data:
########################################################################################################################
Date_Time = '20.8.2018_11:37'
Author = 'ihm'
App = 'Induced voltage'
exciterWindings = 1
exciterCurrent = 1
frequency = 119000
registeredExciterCorners = 4
objectType = 'Ball'

# rectangular exciter in x-y-plane:
e1 = np.array([0,       0,      0])
e2 = np.array([1.365,   0,      0])
e3 = np.array([1.365,   1.01,   0])
e4 = np.array([0,       1.01,   0])

# --> exciter shape (relative to the exciter coordinate system):
exciterShape = [e1, e2, e3, e4]

# Origin of the exciter coordinate system (relative to the inertial coordinate system):
exciterPosition = np.array([0, 0, 0])

# Generate object which contains exciter data:
exciter = c.Polygon(exciterShape, exciterWindings, exciterPosition, current=exciterCurrent)

########################################################################################################################
# Coil data:
########################################################################################################################
coilWindings = 52
coilResistance = 0.9
coilsRadius = a = b = 1.8e-2

# Generate object which contains coil data:
coil = c.Ellipse(a, b, coilWindings)

########################################################################################################################
# Antenna data:
########################################################################################################################
antennaWindings = 40

# ----------------------------------------------------------------------------------------------------------------------
# Main antenna (with amplifier):
# x-y-plane (O = origin, roughly coincides with center of plug when in case)
# Order of points matches direction of winding
#
#     y--> 
#   -------aM3---------------------------------aM2
# x |       |                                   |
# | O       | -------------(frame)------------- |
# V |       | [          cal. loop            ] |
#   -------aM0---------------------------------aM1
#
aM0 = 1e-3*np.array([22.5, 62, 0]) #y = 150+62
aM1 = 1e-3*np.array([22.5, 397, 0])#y = 470+150+62 ?
aM2 = 1e-3*np.array([-22.5, 397, 0])
aM3 = 1e-3*np.array([-22.5, 62, 0])
mainShape = np.array([aM0, aM1, aM2, aM3])

# ----------------------------------------------------------------------------------------------------------------------
# Frame antenna:
# z-y-plane (O = origin, roughly coincides with center of plug when in case)
# Order of points matches direction of winding
#
#     y--> 
#          aF0---------------------------------aF1
# ^         | [          cal. loop            ] |
# | O       |                                   |
# z         |                                   |
#          aF3---------------------------------aF2
#
aF0 = 1e-3*np.array([0, 69, 22])
aF1 = 1e-3*np.array([0, 390, 22])
aF2 = 1e-3*np.array([0, 390, -22])
aF3 = 1e-3*np.array([0, 69, -22])
frameShape = np.array([aF0, aF1, aF2, aF3])

# Antenna positions:
antennaPosition1 = 1e-3*np.array([0, 1010-25-25-30-465, 0])
antennaPosition2 = 1e-3*np.array([0, 1010-25-25-30, 0])
antennaPosition3 = 1e-3*np.array([1365-25-150-30-465, 1010, 0])
antennaPosition4 = 1e-3*np.array([1365-25-150-30, 1010, 0])
antennaPosition5 = 1e-3*np.array([1365, 25+25+30+465, 0])
antennaPosition6 = 1e-3*np.array([1365, 25+25+30, 0])
antennaPosition7 = 1e-3*np.array([25+155+30+465, 0, 0])
antennaPosition8 = 1e-3*np.array([25+155+30, 0, 0])

# Rotation matrices:
# Base rotation is first 180 degrees around z, then -45 degrees around y
c_base = (np.array([[np.cos(np.deg2rad(-45)), 0, np.sin(np.deg2rad(-45))],
                   [0, 1, 0],
                   [-np.sin(np.deg2rad(-45)), 0, np.cos(np.deg2rad(-45))]]) @ 
          np.array([[-1, 0, 0],
                   [0, -1, 0],
                   [0, 0, 1]]))
# Individual antenna pairs are rotated around z
c_z_90 = np.array([[0, 1, 0],
                   [-1,  0, 0],
                   [0,  0, 1]])
c_z_left = np.eye(3)
c_z_top = c_z_90
c_z_right = c_z_90 @ c_z_90
c_z_bottom = c_z_90 @ c_z_90 @ c_z_90

main1 = c.Polygon(mainShape, antennaWindings, antennaPosition1, c_base)
frame1 = c.Polygon(frameShape, antennaWindings, antennaPosition1, c_base)
main2 = c.Polygon(mainShape, antennaWindings, antennaPosition2, c_base)
frame2 = c.Polygon(frameShape, antennaWindings, antennaPosition2, c_base)
main3 = c.Polygon(mainShape, antennaWindings, antennaPosition3, c_z_top @ c_base)
frame3 = c.Polygon(frameShape, antennaWindings, antennaPosition3, c_z_top @ c_base)
main4 = c.Polygon(mainShape, antennaWindings, antennaPosition4, c_z_top @ c_base)
frame4 = c.Polygon(frameShape, antennaWindings, antennaPosition4, c_z_top @ c_base)
main5 = c.Polygon(mainShape, antennaWindings, antennaPosition5, c_z_right @ c_base)
frame5 = c.Polygon(frameShape, antennaWindings, antennaPosition5, c_z_right @ c_base)
main6 = c.Polygon(mainShape, antennaWindings, antennaPosition6, c_z_right @ c_base)
frame6 = c.Polygon(frameShape, antennaWindings, antennaPosition6, c_z_right @ c_base)
main7 = c.Polygon(mainShape, antennaWindings, antennaPosition7, c_z_bottom @ c_base)
frame7 = c.Polygon(frameShape, antennaWindings, antennaPosition7, c_z_bottom @ c_base)
main8 = c.Polygon(mainShape, antennaWindings, antennaPosition8, c_z_bottom @ c_base)
frame8 = c.Polygon(frameShape, antennaWindings, antennaPosition8, c_z_bottom @ c_base)

# ----------------------------------------------------------------------------------------------------------------------
# Select desired antennas and put them in a list:
antennaList = [frame1,  frame2, frame3, frame4, frame5, frame6, frame7, frame8,
               main1,   main2,  main3,  main4,  main5,  main6,  main7,  main8]


mainAntennaCheckBox = 0

frameAntennaCheckBox = 0

numberOfCores = 6
tableName = 'DW_Shelf_45DegAntennas'
