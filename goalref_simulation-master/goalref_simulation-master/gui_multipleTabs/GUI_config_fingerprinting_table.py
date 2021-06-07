import numpy as np
import GUI_configFile as configFile
__author__ = 'ihm'
## @package config_fingerprinting_table
# @brief A python file to configure the objects positions, orientation, the number of
# processing cores, the table name...
#@author ihm
#@version 2.0
#@date Created on Fri Dec 18 11:37:12 2015, Modified on Tue Dec 19 10:26:50 2017
## Select application:


app = configFile.App
numberOfCores = configFile.numberOfCores
## Header information:
author = configFile.Author
additional_info = 'Origin of the inertial frame: x=0, y=0, z=0'
## Table format: (table entries will be depicted in given format)
# exponential representation with 5 decimal places
tableFormat = '%10.5E'
## Select a set of Positions and rotations or add a new one:


tableName = configFile.tableName
## 1. define positions of the coil [m]

if configFile.xPosOption == 'SingleMultiple':
    xpos = configFile.xpos
elif configFile.xPosOption == 'Sweep':
    xpos = np.arange(configFile.xposSweepStart, configFile.xposSweepEnd, configFile.xposSweepStep)

if configFile.yPosOption == 'SingleMultiple':
    ypos = configFile.ypos
elif configFile.yPosOption == 'Sweep':
    ypos = np.arange(configFile.yposSweepStart, configFile.yposSweepEnd, configFile.yposSweepStep)

if configFile.zPosOption == 'SingleMultiple':
    zpos = configFile.zpos
elif configFile.zPosOption == 'Sweep':
    zpos = np.arange(configFile.zposSweepStart, configFile.zposSweepEnd, configFile.zposSweepStep)
## positions of the coil [m]
positions = [xpos, ypos, zpos]

## 2. define rotation angles
unit = 'deg'
if configFile.alphaOption == 'SingleMultiple':
    xangles = configFile.alpha
elif configFile.alphaOption == 'Sweep':
    xangles = np.arange(configFile.alphaSweepStart, configFile.alphaSweepEnd, configFile.alphaSweepStep)

if configFile.betaOption == 'SingleMultiple':
    yangles = configFile.beta
elif configFile.betaOption == 'Sweep':
    yangles = np.arange(configFile.betaSweepStart, configFile.betaSweepEnd, configFile.betaSweepStep)

if configFile.gammaOption == 'SingleMultiple':
    zangles = configFile.gamma
elif configFile.gammaOption == 'Sweep':
    zangles = np.arange(configFile.gammaSweepStart, configFile.gammaSweepEnd, configFile.gammaSweepStep)

## rotation angles
angles = [xangles, yangles, zangles, unit]
#print(angles)


