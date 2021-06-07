import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import math
import numpy as np
import matplotlib.pyplot as plt
import GUI_configFile as configFile
from GUI_configFile import *
## @package plot_setup
# @brief A python logic to plot the cofigured setup in configFile.py for exciter, antennas and objects coils
#@author ibrahiim
#@version 0.6
#@date Created on Wed Dec 6 09:00:00 2017, Updated on Tue Dec 19 10:26:50 2017

## This function rotates a circle in 3d
# it takes the radius of the circle, rotation matrix
# and the shift in the position of the center of the circle
def rotateCircle(radius, rotationMatrix, Position):
    r = radius
    MTX = rotationMatrix
    # Set of all angles:
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, 2 * np.pi, 100)
    x = r * np.cos(u)
    y = r * np.sin(v)
    z = 0 * np.cos(v)
    x_rotated = []
    y_rotated = []
    z_rotated = []
    for i in range(0, len(x)):
        #xyzRotTemp = np.dot(np.transpose(MTX), np.array([x[i], y[i], z[i]])) +Position
        xyzRotTemp = np.dot(MTX, np.array([x[i], y[i], z[i]])) + Position
        x_rotated.append(xyzRotTemp[0])
        y_rotated.append(xyzRotTemp[1])
        z_rotated.append(xyzRotTemp[2])
    return x_rotated, y_rotated, z_rotated
## This function rotates an ellipse in 3d
# it takes the minorAxis, majorAxis of the ellipse, rotation matrix
# and the shift in the position of the center of the ellipse.
def rotateEllipse(minorAxis, majorAxis, rotationMatirx,Position):
    MTX = rotationMatirx
    ytemp = np.linspace(-np.abs(minorAxis), np.abs(minorAxis), 100, endpoint=True)
    xtemp = np.sqrt((1 - ((np.square(ytemp)) / (minorAxis * minorAxis))) * (majorAxis * majorAxis))
    x = []
    y = []
    z = []
    for i in range(0, len(xtemp)):
        x.append(xtemp[i])
        y.append(ytemp[i])
        z.append(0)
    yreversed = list(reversed(ytemp))
    for i in range(0, len(xtemp)):
        x.append(-xtemp[i])
        y.append(yreversed[i])
        z.append(0)

    x_rotated = []
    y_rotated = []
    z_rotated = []

    for i in range(0, len(x)):
        #xyzRotTemp = np.dot(np.transpose(MTX), [x[i], y[i], z[i]]) + Position
        xyzRotTemp = np.dot(MTX, [x[i], y[i], z[i]]) + Position
        x_rotated.append(xyzRotTemp[0])
        y_rotated.append(xyzRotTemp[1])
        z_rotated.append(xyzRotTemp[2])
    return x_rotated, y_rotated, z_rotated
if configFile.App == 'Induced voltage':
    ## Open 3d-figure:
    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    exciterCorners = configFile.registeredExciterCorners
    exciterShape = []
    for i in range(1, exciterCorners+1):
        e = np.array(eval("e%s" % str(i)))
        exciterShape.append(e)
    ## exciter shape (relative to the exciter coordinate system):
    exciterShape.append(exciterShape[0])




    x, y, z = [], [], []
    for e in exciterShape:
        x.append(e[0])
        y.append(e[1])
        z.append(e[2])

    ## Plot exciter:
    ax.plot(x, z, y, label='exciter')

    minExciterShape_x = min(x)
    maxExciterShape_x = max(x)
    minExciterShape_y = min(y)
    maxExciterShape_y = max(y)
    minExciterShape_z = min(z)
    maxExciterShape_z = max(z)


    if configFile.frameAntennaCheckBox ==1 :
        if configFile.registeredFrameAntennas != 0:
            frameAntennaLength = configFile.frameAntennaLength
            frameAntennaHeight = configFile.frameAntennaHeight
            for i in range(1, configFile.registeredFrameAntennas +1):
                frameAntennaOrientation = eval("frameAntennaOrientation%s" % str(i))
                c_x = np.array([[1, 0, 0],
                                [0, np.cos(math.radians(frameAntennaOrientation[0])),
                                 -np.sin(math.radians(frameAntennaOrientation[0]))],
                                [0, np.sin(math.radians(frameAntennaOrientation[0])),
                                 np.cos(math.radians(frameAntennaOrientation[0]))]])

                c_y = np.array([[np.cos(math.radians(frameAntennaOrientation[1])), 0,
                                 np.sin(math.radians(frameAntennaOrientation[1]))],
                                [0, 1, 0],
                                [-np.sin(math.radians(frameAntennaOrientation[1])), 0,
                                 np.cos(math.radians(frameAntennaOrientation[1]))]])

                c_z = np.array([[np.cos(math.radians(frameAntennaOrientation[2])),
                                 -np.sin(math.radians(frameAntennaOrientation[2])), 0],
                                [np.sin(math.radians(frameAntennaOrientation[2])),
                                 np.cos(math.radians(frameAntennaOrientation[2])), 0],
                                [0, 0, 1]])
                rotationMatrix = np.dot(c_z, np.dot(c_y, c_x))
                aF0 = np.array([-frameAntennaLength / 2, -frameAntennaHeight / 2, 0])
                aF1 = np.array([+frameAntennaLength / 2, -frameAntennaHeight / 2, 0])
                aF2 = np.array([+frameAntennaLength / 2, +frameAntennaHeight / 2, 0])
                aF3 = np.array([-frameAntennaLength / 2, +frameAntennaHeight / 2, 0])
                # aF0 = np.dot(np.transpose(rotationMatrix),aF0)+ np.array(eval("frameAntennaPosition%s" %str(i)))
                # aF1 = np.dot(np.transpose(rotationMatrix),aF1)+np.array(eval("frameAntennaPosition%s" %str(i)))
                # aF2 = np.dot(np.transpose(rotationMatrix),aF2)+np.array(eval("frameAntennaPosition%s" %str(i)))
                # aF3 = np.dot(np.transpose(rotationMatrix),aF3)+np.array(eval("frameAntennaPosition%s" %str(i)))
                aF0 = np.dot(rotationMatrix, aF0) + np.array(eval("frameAntennaPosition%s" % str(i)))
                aF1 = np.dot(rotationMatrix, aF1) + np.array(eval("frameAntennaPosition%s" % str(i)))
                aF2 = np.dot(rotationMatrix, aF2) + np.array(eval("frameAntennaPosition%s" % str(i)))
                aF3 = np.dot(rotationMatrix, aF3) + np.array(eval("frameAntennaPosition%s" % str(i)))
                frameShape = [aF0, aF1, aF2, aF3, aF0]
                x, y, z = [], [], []
                for aF in frameShape:
                    x.append(aF[0])
                    y.append(aF[1])
                    z.append(aF[2])
                ## Plot Frame Antennas, if any:
                ax.plot(x, z, y, label='frame antenna %s' %i)

    if configFile.mainAntennaCheckBox ==1 :
        if configFile.registeredMainAntennas != 0:
            mainAntennaLength = configFile.mainAntennaLength
            mainAntennaHeight = configFile.mainAntennaHeight
            for i in range(1, configFile.registeredMainAntennas + 1):
                mainAntennaOrientation = eval("mainAntennaOrientation%s" % str(i))
                c_x = np.array([[1, 0, 0],
                                [0, np.cos(math.radians(mainAntennaOrientation[0])),
                                 -np.sin(math.radians(mainAntennaOrientation[0]))],
                                [0, np.sin(math.radians(mainAntennaOrientation[0])),
                                 np.cos(math.radians(mainAntennaOrientation[0]))]])

                c_y = np.array([[np.cos(math.radians(mainAntennaOrientation[1])), 0,
                                 np.sin(math.radians(mainAntennaOrientation[1]))],
                                [0, 1, 0],
                                [-np.sin(math.radians(mainAntennaOrientation[1])), 0,
                                 np.cos(math.radians(mainAntennaOrientation[1]))]])

                c_z = np.array([[np.cos(math.radians(mainAntennaOrientation[2])),
                                 -np.sin(math.radians(mainAntennaOrientation[2])), 0],
                                [np.sin(math.radians(mainAntennaOrientation[2])),
                                 np.cos(math.radians(mainAntennaOrientation[2])), 0],
                                [0, 0, 1]])
                rotationMatrix = np.dot(c_z, np.dot(c_y, c_x))
                aM0 = np.array([-mainAntennaLength/2, 0, -mainAntennaHeight/2])
                aM1 = np.array([+mainAntennaLength/2, 0, -mainAntennaHeight/2])
                aM2 = np.array([+mainAntennaLength/2, 0, +mainAntennaHeight/2])
                aM3 = np.array([-mainAntennaLength/2, 0, +mainAntennaHeight/2])
                # aM0 = np.dot(np.transpose(rotationMatrix), aM0)+np.array(eval("mainAntennaPosition%s" %str(i)))
                # aM1 = np.dot(np.transpose(rotationMatrix), aM1)+np.array(eval("mainAntennaPosition%s" %str(i)))
                # aM2 = np.dot(np.transpose(rotationMatrix), aM2)+np.array(eval("mainAntennaPosition%s" %str(i)))
                # aM3 = np.dot(np.transpose(rotationMatrix), aM3)+np.array(eval("mainAntennaPosition%s" %str(i)))
                aM0 = np.dot(rotationMatrix, aM0) + np.array(eval("mainAntennaPosition%s" % str(i)))
                aM1 = np.dot(rotationMatrix, aM1) + np.array(eval("mainAntennaPosition%s" % str(i)))
                aM2 = np.dot(rotationMatrix, aM2) + np.array(eval("mainAntennaPosition%s" % str(i)))
                aM3 = np.dot(rotationMatrix, aM3) + np.array(eval("mainAntennaPosition%s" % str(i)))
                mainShape = [aM0, aM1, aM2, aM3, aM0]
                x, y, z = [], [], []
                for aM in mainShape:
                    x.append(aM[0])
                    y.append(aM[1])
                    z.append(aM[2])
                ## Plot Main Antennas, if any:
                ax.plot(x, z, y, label='main antenna %s' % i)

    if configFile.objectType != 0:
        try:
            if configFile.coilPositionX != '':

                coilDefault3dPosition = np.array([configFile.coilPositionX, configFile.coilPositionY, configFile.coilPositionZ])
            else:
                coilDefault3dPosition = np.array([(maxExciterShape_x - minExciterShape_x) / 2, (maxExciterShape_y - minExciterShape_y) / 2,
                                                  (maxExciterShape_z - minExciterShape_z) / 2])
        except:
            coilDefault3dPosition = np.array([(maxExciterShape_x - minExciterShape_x) / 2, (maxExciterShape_y - minExciterShape_y) / 2,
                                                  (maxExciterShape_z - minExciterShape_z) / 2])
        ## Preparing and plotting a Polygon object
        if configFile.objectType == 'Polygon':
            coilLength = configFile.coilLength
            coilWidth = configFile.coilWidth
            ## 360- because the replaced y and z axis in 3d plot
            coilOrientationAlpha = configFile.coilOrientationAlpha
            ## 360- because the replaced y and z axis in 3d plot
            coilOrientationBeta = configFile.coilOrientationBeta
            ## 360- because the replaced y and z axis in 3d plot
            coilOrientationGamma = configFile.coilOrientationGamma
            c0 = np.array([-coilLength / 2, -coilWidth / 2, 0])
            c1 = np.array([coilLength / 2, -coilWidth / 2, 0])
            c2 = np.array([coilLength / 2, coilWidth / 2, 0])
            c3 = np.array([-coilLength / 2, coilWidth / 2, 0])
            c_x = np.array([[1, 0, 0],
                            [0, np.cos(math.radians(coilOrientationAlpha)), -np.sin(math.radians(coilOrientationAlpha))],
                            [0, np.sin(math.radians(coilOrientationAlpha)), np.cos(math.radians(coilOrientationAlpha))]])

            c_y = np.array([[np.cos(math.radians(coilOrientationBeta)), 0, np.sin(math.radians(coilOrientationBeta))],
                            [0, 1, 0],
                            [-np.sin(math.radians(coilOrientationBeta)), 0, np.cos(math.radians(coilOrientationBeta))]])

            c_z = np.array([[np.cos(math.radians(coilOrientationGamma)), -np.sin(math.radians(coilOrientationGamma)), 0],
                            [np.sin(math.radians(coilOrientationGamma)), np.cos(math.radians(coilOrientationGamma)), 0],
                            [0, 0, 1]])

            # First rotation around x-Axis, then y-Axis, then z-Axis
            coilOrientation = np.dot(c_z, np.dot(c_y, c_x))
            # c0 = np.dot(np.transpose(coilOrientation), c0)+coilDefault3dPosition
            # c1 = np.dot(np.transpose(coilOrientation), c1)+coilDefault3dPosition
            # c2 = np.dot(np.transpose(coilOrientation), c2)+coilDefault3dPosition
            # c3 = np.dot(np.transpose(coilOrientation), c3)+coilDefault3dPosition
            c0 = np.dot(coilOrientation, c0) + coilDefault3dPosition
            c1 = np.dot(coilOrientation, c1) + coilDefault3dPosition
            c2 = np.dot(coilOrientation, c2) + coilDefault3dPosition
            c3 = np.dot(coilOrientation, c3) + coilDefault3dPosition
            coilShape = [c0, c1, c2, c3, c0]
            x, y, z = [], [], []
            for c in coilShape:
                x.append(c[0])
                y.append(c[1])
                z.append(c[2])
            ax.plot(x, z, y, label='Polygon coil')
        ## Preparing and plotting an elliptical object
        elif configFile.objectType == 'Ellipse':
            majorAxisLength = configFile.majorAxisLength
            minorAxisLength = configFile.minorAxisLength
            coilOrientationAlpha = configFile.coilOrientationAlpha
            coilOrientationBeta = configFile.coilOrientationBeta
            coilOrientationGamma = configFile.coilOrientationGamma
            c_x = np.array([[1, 0, 0],
                            [0, np.cos(math.radians(coilOrientationAlpha)), -np.sin(math.radians(coilOrientationAlpha))],
                            [0, np.sin(math.radians(coilOrientationAlpha)), np.cos(math.radians(coilOrientationAlpha))]])

            c_y = np.array([[np.cos(math.radians(coilOrientationBeta)), 0, np.sin(math.radians(coilOrientationBeta))],
                            [0, 1, 0],
                            [-np.sin(math.radians(coilOrientationBeta)), 0, np.cos(math.radians(coilOrientationBeta))]])

            c_z = np.array([[np.cos(math.radians(coilOrientationGamma)), -np.sin(math.radians(coilOrientationGamma)), 0],
                            [np.sin(math.radians(coilOrientationGamma)), np.cos(math.radians(coilOrientationGamma)), 0],
                            [0, 0, 1]])

            # First rotation around x-Axis, then y-Axis, then z-Axis
            coilOrientation = np.dot(c_z, np.dot(c_y, c_x))
            x,y,z = rotateEllipse(minorAxisLength, majorAxisLength, coilOrientation, coilDefault3dPosition)
            ax.plot(x, z, y, label='Elliptical coil')
        ## Preparing and plotting a Puk object
        elif configFile.objectType == 'Puk':
            puckHeight = configFile.pukHeight
            circularCoilRadius = configFile.circularCoilRadius
            coilOrientationAlpha = configFile.coilOrientationAlpha
            coilOrientationBeta = configFile.coilOrientationBeta
            coilOrientationGamma = configFile.coilOrientationGamma
            c_x = np.array([[1, 0, 0],
                            [0, np.cos(math.radians(coilOrientationAlpha)), -np.sin(math.radians(coilOrientationAlpha))],
                            [0, np.sin(math.radians(coilOrientationAlpha)), np.cos(math.radians(coilOrientationAlpha))]])

            c_y = np.array([[np.cos(math.radians(coilOrientationBeta)), 0, np.sin(math.radians(coilOrientationBeta))],
                            [0, 1, 0],
                            [-np.sin(math.radians(coilOrientationBeta)), 0, np.cos(math.radians(coilOrientationBeta))]])

            c_z = np.array([[np.cos(math.radians(coilOrientationGamma)), -np.sin(math.radians(coilOrientationGamma)), 0],
                            [np.sin(math.radians(coilOrientationGamma)), np.cos(math.radians(coilOrientationGamma)), 0],
                            [0, 0, 1]])

            #coilXOrientation = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
            coilXOrientation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
            coilOrientation = np.dot(c_z, np.dot(c_y, c_x))
            c0 = np.array([-circularCoilRadius, -puckHeight/2, 0])
            c1 = np.array([circularCoilRadius, -puckHeight/2, 0])
            c2 = np.array([circularCoilRadius, puckHeight/2, 0])
            c3 = np.array([-circularCoilRadius, puckHeight/2, 0])
            # c0 = np.dot(np.transpose(coilXOrientation), c0) #+ coilDefault3dPosition
            # c1 = np.dot(np.transpose(coilXOrientation), c1) #+ coilDefault3dPosition
            # c2 = np.dot(np.transpose(coilXOrientation), c2) #+ coilDefault3dPosition
            # c3 = np.dot(np.transpose(coilXOrientation), c3) #+ coilDefault3dPosition
            c0 = np.dot(coilXOrientation, c0) #+ coilDefault3dPosition
            c1 = np.dot(coilXOrientation, c1) #+ coilDefault3dPosition
            c2 = np.dot(coilXOrientation, c2) #+ coilDefault3dPosition
            c3 = np.dot(coilXOrientation, c3) #+ coilDefault3dPosition
            c0 = np.dot(coilOrientation, c0)  + coilDefault3dPosition
            c1 = np.dot(coilOrientation, c1)  + coilDefault3dPosition
            c2 = np.dot(coilOrientation, c2)  + coilDefault3dPosition
            c3 = np.dot(coilOrientation, c3)  + coilDefault3dPosition
            coilShape = [c0, c1, c2, c3, c0]
            x1, y1, z1 = [], [], []
            for c in coilShape:
                x1.append(c[0])
                y1.append(c[1])
                z1.append(c[2])
            ax.plot(x1, z1, y1, label='Puk x-coil')

            x2, y2, z2 = rotateCircle(circularCoilRadius, np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]), [0,0,0])
            xyz_temp = []
            x2_rotated = []
            y2_rotated = []
            z2_rotated = []
            for i in range(0,len(x2)):
                #xyz_temp.append(np.dot(np.transpose(coilOrientation), np.array([x2[i], y2[i], z2[i]]))+coilDefault3dPosition)
                xyz_temp.append(
                    np.dot(coilOrientation, np.array([x2[i], y2[i], z2[i]])) + coilDefault3dPosition)
            for i in range(0, len(xyz_temp)):
                x2_rotated.append(xyz_temp[i][0])
                y2_rotated.append(xyz_temp[i][1])
                z2_rotated.append(xyz_temp[i][2])
            ax.plot(x2_rotated, z2_rotated, y2_rotated, label='Puk y-coil')

            c0 = np.array([-circularCoilRadius, -puckHeight/2, 0])
            c1 = np.array([circularCoilRadius, -puckHeight/2, 0])
            c2 = np.array([circularCoilRadius, puckHeight/2, 0])
            c3 = np.array([-circularCoilRadius, puckHeight/2, 0])
            # c0 = np.dot(np.transpose(coilOrientation), c0) + coilDefault3dPosition
            # c1 = np.dot(np.transpose(coilOrientation), c1) + coilDefault3dPosition
            # c2 = np.dot(np.transpose(coilOrientation), c2) + coilDefault3dPosition
            # c3 = np.dot(np.transpose(coilOrientation), c3) + coilDefault3dPosition
            c0 = np.dot(coilOrientation, c0) + coilDefault3dPosition
            c1 = np.dot(coilOrientation, c1) + coilDefault3dPosition
            c2 = np.dot(coilOrientation, c2) + coilDefault3dPosition
            c3 = np.dot(coilOrientation, c3) + coilDefault3dPosition

            coilShape = [c0, c1, c2, c3, c0]
            x3, y3, z3 = [], [], []
            for c in coilShape:
                x3.append(c[0])
                y3.append(c[1])
                z3.append(c[2])
            ax.plot(x3, z3, y3, label='Puk z-coil')
        ## Preparing and plotting a Ball object with three orthognal coils
        elif configFile.objectType == 'Ball':
            coilsRadius = configFile.coilsRadius
            coilOrientationAlpha = configFile.coilOrientationAlpha
            coilOrientationBeta = configFile.coilOrientationBeta
            coilOrientationGamma = configFile.coilOrientationGamma
            c_x = np.array([[1, 0, 0],
                            [0, np.cos(math.radians(coilOrientationAlpha)), -np.sin(math.radians(coilOrientationAlpha))],
                            [0, np.sin(math.radians(coilOrientationAlpha)), np.cos(math.radians(coilOrientationAlpha))]])

            c_y = np.array([[np.cos(math.radians(coilOrientationBeta)), 0, np.sin(math.radians(coilOrientationBeta))],
                            [0, 1, 0],
                            [-np.sin(math.radians(coilOrientationBeta)), 0, np.cos(math.radians(coilOrientationBeta))]])

            c_z = np.array([[np.cos(math.radians(coilOrientationGamma)), -np.sin(math.radians(coilOrientationGamma)), 0],
                            [np.sin(math.radians(coilOrientationGamma)), np.cos(math.radians(coilOrientationGamma)), 0],
                            [0, 0, 1]])

            coilOrientation = np.dot(c_z, np.dot(c_y, c_x))

            #x1, y1, z1 = rotateCircle(coilsRadius, np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]), [0,0,0])
            x1, y1, z1 = rotateCircle(coilsRadius,np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]), [0,0,0])
            xyz_temp = []
            x1_rotated = []
            y1_rotated = []
            z1_rotated = []
            for i in range(0, len(x1)):
                # xyz_temp.append(
                #     np.dot(np.transpose(coilOrientation), np.array([x1[i], y1[i], z1[i]])) + coilDefault3dPosition)
                xyz_temp.append(
                    np.dot(coilOrientation, np.array([x1[i], y1[i], z1[i]])) + coilDefault3dPosition)
            for i in range(0, len(xyz_temp)):
                x1_rotated.append(xyz_temp[i][0])
                y1_rotated.append(xyz_temp[i][1])
                z1_rotated.append(xyz_temp[i][2])
            ax.plot(x1_rotated, z1_rotated, y1_rotated, label='Ball x-coil')


            x2, y2, z2 = rotateCircle(coilsRadius,np.array([[1, 0, 0],[0,  0, 1],[0,  -1, 0]]), [0,0,0])
            xyz_temp = []
            x2_rotated = []
            y2_rotated = []
            z2_rotated = []
            for i in range(0, len(x2)):
                # xyz_temp.append(
                #     np.dot(np.transpose(coilOrientation), np.array([x2[i], y2[i], z2[i]])) + coilDefault3dPosition)
                xyz_temp.append(
                    np.dot(coilOrientation, np.array([x2[i], y2[i], z2[i]])) + coilDefault3dPosition)
            for i in range(0, len(xyz_temp)):
                x2_rotated.append(xyz_temp[i][0])
                y2_rotated.append(xyz_temp[i][1])
                z2_rotated.append(xyz_temp[i][2])
            ax.plot(x2_rotated, z2_rotated, y2_rotated, label='Ball y-coil')

            x3, y3, z3 = rotateCircle(coilsRadius, np.eye(3), [0,0,0])
            xyz_temp = []
            x3_rotated = []
            y3_rotated = []
            z3_rotated = []
            for i in range(0, len(x3)):
                # xyz_temp.append(
                #     np.dot(np.transpose(coilOrientation), np.array([x3[i], y3[i], z3[i]])) + coilDefault3dPosition)
                xyz_temp.append(
                    np.dot(coilOrientation, np.array([x3[i], y3[i], z3[i]])) + coilDefault3dPosition)
            for i in range(0, len(xyz_temp)):
                x3_rotated.append(xyz_temp[i][0])
                y3_rotated.append(xyz_temp[i][1])
                z3_rotated.append(xyz_temp[i][2])
            ax.plot(x3_rotated, z3_rotated, y3_rotated, label='Ball z-coil')

        elif configFile.objectType == 'Wearable':
            coil1Width = configFile.coil1Width
            coil2Width = configFile.coil2Width
            coil3Width = configFile.coil3Width
            coil1Length = configFile.coil1Length
            coil2Length = configFile.coil2Length
            coil3Length = configFile.coil3Length

            #puckHeight = configFile.pukHeight
            #circularCoilRadius = configFile.circularCoilRadius
            coilOrientationAlpha = configFile.coilOrientationAlpha
            coilOrientationBeta = configFile.coilOrientationBeta
            coilOrientationGamma = configFile.coilOrientationGamma
            c_x = np.array([[1, 0, 0],
                            [0, np.cos(math.radians(coilOrientationAlpha)), -np.sin(math.radians(coilOrientationAlpha))],
                            [0, np.sin(math.radians(coilOrientationAlpha)), np.cos(math.radians(coilOrientationAlpha))]])

            c_y = np.array([[np.cos(math.radians(coilOrientationBeta)), 0, np.sin(math.radians(coilOrientationBeta))],
                            [0, 1, 0],
                            [-np.sin(math.radians(coilOrientationBeta)), 0, np.cos(math.radians(coilOrientationBeta))]])

            c_z = np.array([[np.cos(math.radians(coilOrientationGamma)), -np.sin(math.radians(coilOrientationGamma)), 0],
                            [np.sin(math.radians(coilOrientationGamma)), np.cos(math.radians(coilOrientationGamma)), 0],
                            [0, 0, 1]])

            #coilXOrientation = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
            coilXOrientation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
            coilOrientation = np.dot(c_z, np.dot(c_y, c_x))
            c0 = np.array([-coil1Width/2, -coil1Length/2, 0])
            c1 = np.array([coil1Width/2, -coil1Length/2, 0])
            c2 = np.array([coil1Width/2, coil1Length/2, 0])
            c3 = np.array([-coil1Width/2, coil1Length/2, 0])
            # c0 = np.dot(np.transpose(coilXOrientation), c0) #+ coilDefault3dPosition
            # c1 = np.dot(np.transpose(coilXOrientation), c1) #+ coilDefault3dPosition
            # c2 = np.dot(np.transpose(coilXOrientation), c2) #+ coilDefault3dPosition
            # c3 = np.dot(np.transpose(coilXOrientation), c3) #+ coilDefault3dPosition
            c0 = np.dot(coilXOrientation, c0) #+ coilDefault3dPosition
            c1 = np.dot(coilXOrientation, c1) #+ coilDefault3dPosition
            c2 = np.dot(coilXOrientation, c2) #+ coilDefault3dPosition
            c3 = np.dot(coilXOrientation, c3) #+ coilDefault3dPosition
            c0 = np.dot(coilOrientation, c0)  + coilDefault3dPosition
            c1 = np.dot(coilOrientation, c1)  + coilDefault3dPosition
            c2 = np.dot(coilOrientation, c2)  + coilDefault3dPosition
            c3 = np.dot(coilOrientation, c3)  + coilDefault3dPosition
            coilShape = [c0, c1, c2, c3, c0]
            x1, y1, z1 = [], [], []
            for c in coilShape:
                x1.append(c[0])
                y1.append(c[1])
                z1.append(c[2])
            ax.plot(x1, z1, y1, label='Wearable x-coil')

            coilYOrientation = np.array([[1, 0, 0],[0,  0, 1],[0,  -1, 0]])
            c0 = np.array([-coil2Width / 2, -coil2Length / 2 , 0])
            c1 = np.array([coil2Width / 2, -coil2Length / 2 , 0])
            c2 = np.array([coil2Width / 2, coil2Length / 2 , 0])
            c3 = np.array([-coil2Width / 2, coil2Length / 2 , 0])
            # c0 = np.dot(np.transpose(coilXOrientation), c0) #+ coilDefault3dPosition
            # c1 = np.dot(np.transpose(coilXOrientation), c1) #+ coilDefault3dPosition
            # c2 = np.dot(np.transpose(coilXOrientation), c2) #+ coilDefault3dPosition
            # c3 = np.dot(np.transpose(coilXOrientation), c3) #+ coilDefault3dPosition
            c0 = np.dot(coilYOrientation, c0)  # + coilDefault3dPosition
            c1 = np.dot(coilYOrientation, c1)  # + coilDefault3dPosition
            c2 = np.dot(coilYOrientation, c2)  # + coilDefault3dPosition
            c3 = np.dot(coilYOrientation, c3)  # + coilDefault3dPosition
            c0 = np.dot(coilOrientation, c0) + coilDefault3dPosition
            c1 = np.dot(coilOrientation, c1) + coilDefault3dPosition
            c2 = np.dot(coilOrientation, c2) + coilDefault3dPosition
            c3 = np.dot(coilOrientation, c3) + coilDefault3dPosition
            coilShape = [c0, c1, c2, c3, c0]
            x2, y2, z2 = [], [], []
            for c in coilShape:
                x2.append(c[0])
                y2.append(c[1])
                z2.append(c[2])
            ax.plot(x2, z2, y2, label='Wearable y-coil')

            coilZOrientation = np.eye(3)
            c0 = np.array([-coil3Width / 2, -coil3Length / 2 , 0])
            c1 = np.array([coil3Width / 2, -coil3Length / 2 , 0])
            c2 = np.array([coil3Width / 2, coil3Length / 2 , 0])
            c3 = np.array([-coil3Width / 2, coil3Length / 2 , 0])
            # c0 = np.dot(np.transpose(coilXOrientation), c0) #+ coilDefault3dPosition
            # c1 = np.dot(np.transpose(coilXOrientation), c1) #+ coilDefault3dPosition
            # c2 = np.dot(np.transpose(coilXOrientation), c2) #+ coilDefault3dPosition
            # c3 = np.dot(np.transpose(coilXOrientation), c3) #+ coilDefault3dPosition
            c0 = np.dot(coilZOrientation, c0)  # + coilDefault3dPosition
            c1 = np.dot(coilZOrientation, c1)  # + coilDefault3dPosition
            c2 = np.dot(coilZOrientation, c2)  # + coilDefault3dPosition
            c3 = np.dot(coilZOrientation, c3)  # + coilDefault3dPosition
            c0 = np.dot(coilOrientation, c0) + coilDefault3dPosition
            c1 = np.dot(coilOrientation, c1) + coilDefault3dPosition
            c2 = np.dot(coilOrientation, c2) + coilDefault3dPosition
            c3 = np.dot(coilOrientation, c3) + coilDefault3dPosition
            coilShape = [c0, c1, c2, c3, c0]
            x3, y3, z3 = [], [], []
            for c in coilShape:
                x3.append(c[0])
                y3.append(c[1])
                z3.append(c[2])
            ax.plot(x3, z3, y3, label='Wearable z-coil')




    ## Labeling Axes according to right hand rule and showing the final plot:
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.invert_yaxis()
    ax.legend()
    plt.axis('equal')
    plt.show()
elif configFile.App == 'Coupling factor':
    ## Open 3d-figure:
    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    if configFile.object1Type !=0:
        if configFile.object1Type == 'Polygon':
            object1RegisteredPolygonCorners = configFile.object1RegisteredPolygonCorners
            object1PolygonShape = []
            for i in range(1, object1RegisteredPolygonCorners + 1):
                e = np.array(eval("object1E%s" % str(i)))
                object1PolygonShape.append(e)
            if configFile.object1OpenExciter == 0:
                ## exciter shape with closed loop (relative to the exciter coordinate system):
                object1PolygonShape.append(object1PolygonShape[0])
            object1PolygonPosition = configFile.object1Position
            object1PolygonOrientationAlpha = configFile.object1OrientationAlpha
            object1PolygonOrientationBeta = configFile.object1OrientationBeta
            object1PolygonOrientationGamma =configFile.object1OrientationGamma
            c_x = np.array([[1, 0, 0],
                            [0, np.cos(math.radians(object1PolygonOrientationAlpha)),
                             -np.sin(math.radians(object1PolygonOrientationAlpha))],
                            [0, np.sin(math.radians(object1PolygonOrientationAlpha)),
                             np.cos(math.radians(object1PolygonOrientationAlpha))]])

            c_y = np.array([[np.cos(math.radians(object1PolygonOrientationBeta)), 0,
                             np.sin(math.radians(object1PolygonOrientationBeta))],
                            [0, 1, 0],
                            [-np.sin(math.radians(object1PolygonOrientationBeta)), 0,
                             np.cos(math.radians(object1PolygonOrientationBeta))]])

            c_z = np.array([[np.cos(math.radians(object1PolygonOrientationGamma)),
                             -np.sin(math.radians(object1PolygonOrientationGamma)), 0],
                            [np.sin(math.radians(object1PolygonOrientationGamma)),
                             np.cos(math.radians(object1PolygonOrientationGamma)), 0],
                            [0, 0, 1]])
            rotationMatrix = np.dot(c_z, np.dot(c_y, c_x))
            object1PolygonShape_rotated = []
            for e in object1PolygonShape:
                E = np.array(e)
                #E_rotated = np.dot(np.transpose(rotationMatrix), E) + np.array(object1PolygonPosition)
                E_rotated = np.dot(rotationMatrix, E) + np.array(object1PolygonPosition)
                object1PolygonShape_rotated.append(E_rotated)


            x, y, z = [], [], []
            for e in object1PolygonShape_rotated:
                x.append(e[0])
                y.append(e[1])
                z.append(e[2])

            ## Plot exciter:
            ax.plot(x, z, y, label='object1_Polygon')

            # minObject1PolygonShape_x = min(x)
            # maxObject1PolygonShape_x = max(x)
            # minObject1PolygonShape_y = min(y)
            # maxObject1PolygonShape_y = max(y)
            # minObject1PolygonShape_z = min(z)
            # maxObject1PolygonShape_z = max(z)
        elif configFile.object1Type == 'Antenna':
            object1AntennaPosition = configFile.object1Position
            object1AntennaOrientationAlpha = configFile.object1OrientationAlpha
            object1AntennaOrientationBeta = configFile.object1OrientationBeta
            object1AntennaOrientationGamma = configFile.object1OrientationGamma
            c_x = np.array([[1, 0, 0],
                            [0, np.cos(math.radians(object1AntennaOrientationAlpha)),
                             -np.sin(math.radians(object1AntennaOrientationAlpha))],
                            [0, np.sin(math.radians(object1AntennaOrientationAlpha)),
                             np.cos(math.radians(object1AntennaOrientationAlpha))]])

            c_y = np.array([[np.cos(math.radians(object1AntennaOrientationBeta)), 0,
                             np.sin(math.radians(object1AntennaOrientationBeta))],
                            [0, 1, 0],
                            [-np.sin(math.radians(object1AntennaOrientationBeta)), 0,
                             np.cos(math.radians(object1AntennaOrientationBeta))]])

            c_z = np.array([[np.cos(math.radians(object1AntennaOrientationGamma)),
                             -np.sin(math.radians(object1AntennaOrientationGamma)), 0],
                            [np.sin(math.radians(object1AntennaOrientationGamma)),
                             np.cos(math.radians(object1AntennaOrientationGamma)), 0],
                            [0, 0, 1]])
            rotationMatrix = np.dot(c_z, np.dot(c_y, c_x))
            if configFile.object1AntennaType == 'Main':
                aM0 = np.array([-configFile.object1MainAntennaLength / 2, 0, -configFile.object1MainAntennaHeight / 2])
                aM1 = np.array([+configFile.object1MainAntennaLength / 2, 0, -configFile.object1MainAntennaHeight / 2])
                aM2 = np.array([+configFile.object1MainAntennaLength / 2, 0, +configFile.object1MainAntennaHeight / 2])
                aM3 = np.array([-configFile.object1MainAntennaLength / 2, 0, +configFile.object1MainAntennaHeight / 2])
                # aM0 = np.dot(np.transpose(rotationMatrix), aM0) + np.array(object1AntennaPosition)
                # aM1 = np.dot(np.transpose(rotationMatrix), aM1) + np.array(object1AntennaPosition)
                # aM2 = np.dot(np.transpose(rotationMatrix), aM2) + np.array(object1AntennaPosition)
                # aM3 = np.dot(np.transpose(rotationMatrix), aM3) + np.array(object1AntennaPosition)
                aM0 = np.dot(rotationMatrix, aM0) + np.array(object1AntennaPosition)
                aM1 = np.dot(rotationMatrix, aM1) + np.array(object1AntennaPosition)
                aM2 = np.dot(rotationMatrix, aM2) + np.array(object1AntennaPosition)
                aM3 = np.dot(rotationMatrix, aM3) + np.array(object1AntennaPosition)
                mainShape = [aM0, aM1, aM2, aM3, aM0]
                x, y, z = [], [], []
                for aM in mainShape:
                    x.append(aM[0])
                    y.append(aM[1])
                    z.append(aM[2])
                ## Plot Main Antennas, if any:
                ax.plot(x, z, y, label='object1 Main antenna')
            elif configFile.object1AntennaType == 'Frame':
                aF0 = np.array([-configFile.object1FrameAntennaLength / 2, -configFile.object1FrameAntennaHeight / 2, 0])
                aF1 = np.array([+configFile.object1FrameAntennaLength / 2, -configFile.object1FrameAntennaHeight / 2, 0])
                aF2 = np.array([+configFile.object1FrameAntennaLength / 2, +configFile.object1FrameAntennaHeight / 2, 0])
                aF3 = np.array([-configFile.object1FrameAntennaLength / 2, +configFile.object1FrameAntennaHeight / 2, 0])
                # aF0 = np.dot(np.transpose(rotationMatrix), aF0) + np.array(object1AntennaPosition)
                # aF1 = np.dot(np.transpose(rotationMatrix), aF1) + np.array(object1AntennaPosition)
                # aF2 = np.dot(np.transpose(rotationMatrix), aF2) + np.array(object1AntennaPosition)
                # aF3 = np.dot(np.transpose(rotationMatrix), aF3) + np.array(object1AntennaPosition)
                aF0 = np.dot(rotationMatrix, aF0) + np.array(object1AntennaPosition)
                aF1 = np.dot(rotationMatrix, aF1) + np.array(object1AntennaPosition)
                aF2 = np.dot(rotationMatrix, aF2) + np.array(object1AntennaPosition)
                aF3 = np.dot(rotationMatrix, aF3) + np.array(object1AntennaPosition)
                frameShape = [aF0, aF1, aF2, aF3, aF0]
                x, y, z = [], [], []
                for aF in frameShape:
                    x.append(aF[0])
                    y.append(aF[1])
                    z.append(aF[2])
                ## Plot Frame Antennas, if any:
                ax.plot(x, z, y, label='object1 Frame antenna')
        elif configFile.object1Type == 'Ellipse':
            object1EllipsePosition = np.array(configFile.object1Position)
            majorAxisLength = configFile.object1MajorAxisLength
            minorAxisLength = configFile.object1MinorAxisLength
            object1EllipseOrientationAlpha = configFile.object1OrientationAlpha
            object1EllipseOrientationBeta = configFile.object1OrientationBeta
            object1EllipseOrientationGamma = configFile.object1OrientationGamma

            c_x = np.array([[1, 0, 0],
                            [0, np.cos(math.radians(object1EllipseOrientationAlpha)),
                             -np.sin(math.radians(object1EllipseOrientationAlpha))],
                            [0, np.sin(math.radians(object1EllipseOrientationAlpha)),
                             np.cos(math.radians(object1EllipseOrientationAlpha))]])


            c_y = np.array([[np.cos(math.radians(object1EllipseOrientationBeta)), 0, np.sin(math.radians(object1EllipseOrientationBeta))],
                            [0, 1, 0],
                            [-np.sin(math.radians(object1EllipseOrientationBeta)), 0, np.cos(math.radians(object1EllipseOrientationBeta))]])

            c_z = np.array([[np.cos(math.radians(object1EllipseOrientationGamma)), -np.sin(math.radians(object1EllipseOrientationGamma)), 0],
                 [np.sin(math.radians(object1EllipseOrientationGamma)), np.cos(math.radians(object1EllipseOrientationGamma)), 0],
                 [0, 0, 1]])

            rotationMatrix = np.dot(c_z, np.dot(c_y, c_x))
            x, y, z = rotateEllipse(minorAxisLength, majorAxisLength, rotationMatrix, object1EllipsePosition)
            ax.plot(x, z, y, label='object1 Elliptical coil')
        elif configFile.object1Type == 'Puk':
            object1PukPosition = np.array(configFile.object1Position)
            object1PukHeight = configFile.object1PukHeight
            object1CircularCoilRadius= configFile.object1CircularCoilRadius
            object1PukOrientationAlpha = configFile.object1OrientationAlpha
            object1PukOrientationBeta = configFile.object1OrientationBeta
            object1PukOrientationGamma = configFile.object1OrientationGamma

            c_x = np.array([[1, 0, 0],
                            [0, np.cos(math.radians(object1PukOrientationAlpha)),
                             -np.sin(math.radians(object1PukOrientationAlpha))],
                            [0, np.sin(math.radians(object1PukOrientationAlpha)),
                             np.cos(math.radians(object1PukOrientationAlpha))]])

            c_y = np.array([[np.cos(math.radians(object1PukOrientationBeta)), 0,
                             np.sin(math.radians(object1PukOrientationBeta))],
                            [0, 1, 0],
                            [-np.sin(math.radians(object1PukOrientationBeta)), 0,
                             np.cos(math.radians(object1PukOrientationBeta))]])

            c_z = np.array([[np.cos(math.radians(object1PukOrientationGamma)),
                             -np.sin(math.radians(object1PukOrientationGamma)), 0],
                            [np.sin(math.radians(object1PukOrientationGamma)),
                             np.cos(math.radians(object1PukOrientationGamma)), 0],
                            [0, 0, 1]])


            #coilXOrientation = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
            coilXOrientation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
            coilOrientation = np.dot(c_z, np.dot(c_y, c_x))
            c0 = np.array([-object1CircularCoilRadius, -object1PukHeight / 2, 0])
            c1 = np.array([object1CircularCoilRadius, -object1PukHeight / 2, 0])
            c2 = np.array([object1CircularCoilRadius, object1PukHeight / 2, 0])
            c3 = np.array([-object1CircularCoilRadius, object1PukHeight / 2, 0])
            # c0 = np.dot(np.transpose(coilXOrientation), c0)  # + coilDefault3dPosition
            # c1 = np.dot(np.transpose(coilXOrientation), c1)  # + coilDefault3dPosition
            # c2 = np.dot(np.transpose(coilXOrientation), c2)  # + coilDefault3dPosition
            # c3 = np.dot(np.transpose(coilXOrientation), c3)  # + coilDefault3dPosition
            c0 = np.dot(coilXOrientation, c0)  # + coilDefault3dPosition
            c1 = np.dot(coilXOrientation, c1)  # + coilDefault3dPosition
            c2 = np.dot(coilXOrientation, c2)  # + coilDefault3dPosition
            c3 = np.dot(coilXOrientation, c3)  # + coilDefault3dPosition
            c0 = np.dot(coilOrientation, c0) + object1PukPosition
            c1 = np.dot(coilOrientation, c1) + object1PukPosition
            c2 = np.dot(coilOrientation, c2) + object1PukPosition
            c3 = np.dot(coilOrientation, c3) + object1PukPosition
            coilShape = [c0, c1, c2, c3, c0]
            x1, y1, z1 = [], [], []
            for c in coilShape:
                x1.append(c[0])
                y1.append(c[1])
                z1.append(c[2])
            ax.plot(x1, z1, y1, label='object 1 Puk x-coil')

            x2, y2, z2 = rotateCircle(object1CircularCoilRadius, np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]), [0, 0, 0])
            xyz_temp = []
            x2_rotated = []
            y2_rotated = []
            z2_rotated = []
            for i in range(0, len(x2)):
                # xyz_temp.append(
                #     np.dot(np.transpose(coilOrientation), np.array([x2[i], y2[i], z2[i]])) + object1PukPosition)
                xyz_temp.append(
                    np.dot(coilOrientation, np.array([x2[i], y2[i], z2[i]])) + object1PukPosition)
            for i in range(0, len(xyz_temp)):
                x2_rotated.append(xyz_temp[i][0])
                y2_rotated.append(xyz_temp[i][1])
                z2_rotated.append(xyz_temp[i][2])
            ax.plot(x2_rotated, z2_rotated, y2_rotated, label='object 1 Puk y-coil')

            c0 = np.array([-object1CircularCoilRadius, -object1PukHeight / 2, 0])
            c1 = np.array([object1CircularCoilRadius, -object1PukHeight / 2, 0])
            c2 = np.array([object1CircularCoilRadius, object1PukHeight / 2, 0])
            c3 = np.array([-object1CircularCoilRadius, object1PukHeight / 2, 0])
            # c0 = np.dot(np.transpose(coilOrientation), c0) + object1PukPosition
            # c1 = np.dot(np.transpose(coilOrientation), c1) + object1PukPosition
            # c2 = np.dot(np.transpose(coilOrientation), c2) + object1PukPosition
            # c3 = np.dot(np.transpose(coilOrientation), c3) + object1PukPosition
            c0 = np.dot(coilOrientation, c0) + object1PukPosition
            c1 = np.dot(coilOrientation, c1) + object1PukPosition
            c2 = np.dot(coilOrientation, c2) + object1PukPosition
            c3 = np.dot(coilOrientation, c3) + object1PukPosition

            coilShape = [c0, c1, c2, c3, c0]
            x3, y3, z3 = [], [], []
            for c in coilShape:
                x3.append(c[0])
                y3.append(c[1])
                z3.append(c[2])
            ax.plot(x3, z3, y3, label='object 1 Puk z-coil')
        elif configFile.object1Type == 'Ball':
            object1BallPosition = np.array(configFile.object1Position)
            object1BallCoilsRadius = configFile.object1CoilsRadius
            object1BallOrientationAlpha = configFile.object1OrientationAlpha
            object1BallOrientationBeta = configFile.object1OrientationBeta
            object1BallOrientationGamma = configFile.object1OrientationGamma
            c_x = np.array([[1, 0, 0],
                            [0, np.cos(math.radians(object1BallOrientationAlpha)), -np.sin(math.radians(object1BallOrientationAlpha))],
                            [0, np.sin(math.radians(object1BallOrientationAlpha)), np.cos(math.radians(object1BallOrientationAlpha))]])

            c_y = np.array([[np.cos(math.radians(object1BallOrientationBeta)), 0, np.sin(math.radians(object1BallOrientationBeta))],
                            [0, 1, 0],
                            [-np.sin(math.radians(object1BallOrientationBeta)), 0, np.cos(math.radians(object1BallOrientationBeta))]])

            c_z = np.array([[np.cos(math.radians(object1BallOrientationGamma)), -np.sin(math.radians(object1BallOrientationGamma)), 0],
                            [np.sin(math.radians(object1BallOrientationGamma)), np.cos(math.radians(object1BallOrientationGamma)), 0],
                            [0, 0, 1]])

            coilOrientation = np.dot(c_z, np.dot(c_y, c_x))
            #x1, y1, z1 = rotateCircle(object1BallCoilsRadius, np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]), [0, 0, 0])
            x1, y1, z1 = rotateCircle(object1BallCoilsRadius,np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]), [0, 0, 0])
            xyz_temp = []
            x1_rotated = []
            y1_rotated = []
            z1_rotated = []
            for i in range(0, len(x1)):
                # xyz_temp.append(
                #     np.dot(np.transpose(coilOrientation), np.array([x1[i], y1[i], z1[i]])) + object1BallPosition)
                xyz_temp.append(
                    np.dot(coilOrientation, np.array([x1[i], y1[i], z1[i]])) + object1BallPosition)
            for i in range(0, len(xyz_temp)):
                x1_rotated.append(xyz_temp[i][0])
                y1_rotated.append(xyz_temp[i][1])
                z1_rotated.append(xyz_temp[i][2])
            ax.plot(x1_rotated, z1_rotated, y1_rotated, label='object 1 Ball x-coil')

            x2, y2, z2 = rotateCircle(object1BallCoilsRadius, np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]), [0, 0, 0])
            xyz_temp = []
            x2_rotated = []
            y2_rotated = []
            z2_rotated = []
            for i in range(0, len(x2)):
                # xyz_temp.append(
                #     np.dot(np.transpose(coilOrientation), np.array([x2[i], y2[i], z2[i]])) + object1BallPosition)
                xyz_temp.append(
                    np.dot(coilOrientation, np.array([x2[i], y2[i], z2[i]])) + object1BallPosition)
            for i in range(0, len(xyz_temp)):
                x2_rotated.append(xyz_temp[i][0])
                y2_rotated.append(xyz_temp[i][1])
                z2_rotated.append(xyz_temp[i][2])
            ax.plot(x2_rotated, z2_rotated, y2_rotated, label='object 1 Ball y-coil')

            x3, y3, z3 = rotateCircle(object1BallCoilsRadius, np.eye(3), [0, 0, 0])
            xyz_temp = []
            x3_rotated = []
            y3_rotated = []
            z3_rotated = []
            for i in range(0, len(x3)):
                # xyz_temp.append(
                #     np.dot(np.transpose(coilOrientation), np.array([x3[i], y3[i], z3[i]])) + object1BallPosition)
                xyz_temp.append(
                    np.dot(coilOrientation, np.array([x3[i], y3[i], z3[i]])) + object1BallPosition)
            for i in range(0, len(xyz_temp)):
                x3_rotated.append(xyz_temp[i][0])
                y3_rotated.append(xyz_temp[i][1])
                z3_rotated.append(xyz_temp[i][2])
            ax.plot(x3_rotated, z3_rotated, y3_rotated, label='object 1 Ball z-coil')

    if configFile.object2Type != 0:
        if configFile.object2Type == 'Polygon':
            object2RegisteredPolygonCorners = configFile.object2RegisteredPolygonCorners
            object2PolygonShape = []
            for i in range(1, object2RegisteredPolygonCorners + 1):
                e = np.array(eval("object2E%s" % str(i)))
                object2PolygonShape.append(e)
            if configFile.object2OpenExciter == 0:
                ## exciter shape with closed loop (relative to the exciter coordinate system):
                object2PolygonShape.append(object2PolygonShape[0])
            object2PolygonPosition = configFile.object2Position
            object2PolygonOrientationAlpha = configFile.object2OrientationAlpha
            object2PolygonOrientationBeta = configFile.object2OrientationBeta
            object2PolygonOrientationGamma = configFile.object2OrientationGamma
            c_x = np.array([[1, 0, 0],
                            [0, np.cos(math.radians(object2PolygonOrientationAlpha)),
                             -np.sin(math.radians(object2PolygonOrientationAlpha))],
                            [0, np.sin(math.radians(object2PolygonOrientationAlpha)),
                             np.cos(math.radians(object2PolygonOrientationAlpha))]])

            c_y = np.array([[np.cos(math.radians(object2PolygonOrientationBeta)), 0,
                             np.sin(math.radians(object2PolygonOrientationBeta))],
                            [0, 1, 0],
                            [-np.sin(math.radians(object2PolygonOrientationBeta)), 0,
                             np.cos(math.radians(object2PolygonOrientationBeta))]])

            c_z = np.array([[np.cos(math.radians(object2PolygonOrientationGamma)),
                             -np.sin(math.radians(object2PolygonOrientationGamma)), 0],
                            [np.sin(math.radians(object2PolygonOrientationGamma)),
                             np.cos(math.radians(object2PolygonOrientationGamma)), 0],
                            [0, 0, 1]])
            rotationMatrix = np.dot(c_z, np.dot(c_y, c_x))
            object2PolygonShape_rotated = []
            for e in object2PolygonShape:
                E = np.array(e)
                #E_rotated = np.dot(np.transpose(rotationMatrix), E) + np.array(object2PolygonPosition)
                E_rotated = np.dot(rotationMatrix, E) + np.array(object2PolygonPosition)
                object2PolygonShape_rotated.append(E_rotated)

            x, y, z = [], [], []
            for e in object2PolygonShape_rotated:
                x.append(e[0])
                y.append(e[1])
                z.append(e[2])

            ## Plot exciter:
            ax.plot(x, z, y, label='object2_Polygon')

            # minObject1PolygonShape_x = min(x)
            # maxObject1PolygonShape_x = max(x)
            # minObject1PolygonShape_y = min(y)
            # maxObject1PolygonShape_y = max(y)
            # minObject1PolygonShape_z = min(z)
            # maxObject1PolygonShape_z = max(z)
        elif configFile.object2Type == 'Antenna':
            object2AntennaPosition = configFile.object2Position
            object2AntennaOrientationAlpha = configFile.object2OrientationAlpha
            object2AntennaOrientationBeta = configFile.object2OrientationBeta
            object2AntennaOrientationGamma = configFile.object2OrientationGamma
            c_x = np.array([[1, 0, 0],
                            [0, np.cos(math.radians(object2AntennaOrientationAlpha)),
                             -np.sin(math.radians(object2AntennaOrientationAlpha))],
                            [0, np.sin(math.radians(object2AntennaOrientationAlpha)),
                             np.cos(math.radians(object2AntennaOrientationAlpha))]])

            c_y = np.array([[np.cos(math.radians(object2AntennaOrientationBeta)), 0,
                             np.sin(math.radians(object2AntennaOrientationBeta))],
                            [0, 1, 0],
                            [-np.sin(math.radians(object2AntennaOrientationBeta)), 0,
                             np.cos(math.radians(object2AntennaOrientationBeta))]])

            c_z = np.array([[np.cos(math.radians(object2AntennaOrientationGamma)),
                             -np.sin(math.radians(object2AntennaOrientationGamma)), 0],
                            [np.sin(math.radians(object2AntennaOrientationGamma)),
                             np.cos(math.radians(object2AntennaOrientationGamma)), 0],
                            [0, 0, 1]])
            rotationMatrix = np.dot(c_z, np.dot(c_y, c_x))
            if configFile.object2AntennaType == 'Main':
                aM0 = np.array(
                    [-configFile.object2MainAntennaLength / 2, 0, -configFile.object2MainAntennaHeight / 2])
                aM1 = np.array(
                    [+configFile.object2MainAntennaLength / 2, 0, -configFile.object2MainAntennaHeight / 2])
                aM2 = np.array(
                    [+configFile.object2MainAntennaLength / 2, 0, +configFile.object2MainAntennaHeight / 2])
                aM3 = np.array(
                    [-configFile.object2MainAntennaLength / 2, 0, +configFile.object2MainAntennaHeight / 2])
                # aM0 = np.dot(np.transpose(rotationMatrix), aM0) + np.array(object2AntennaPosition)
                # aM1 = np.dot(np.transpose(rotationMatrix), aM1) + np.array(object2AntennaPosition)
                # aM2 = np.dot(np.transpose(rotationMatrix), aM2) + np.array(object2AntennaPosition)
                # aM3 = np.dot(np.transpose(rotationMatrix), aM3) + np.array(object2AntennaPosition)
                aM0 = np.dot(rotationMatrix, aM0) + np.array(object2AntennaPosition)
                aM1 = np.dot(rotationMatrix, aM1) + np.array(object2AntennaPosition)
                aM2 = np.dot(rotationMatrix, aM2) + np.array(object2AntennaPosition)
                aM3 = np.dot(rotationMatrix, aM3) + np.array(object2AntennaPosition)
                mainShape = [aM0, aM1, aM2, aM3, aM0]
                x, y, z = [], [], []
                for aM in mainShape:
                    x.append(aM[0])
                    y.append(aM[1])
                    z.append(aM[2])
                ## Plot Main Antennas, if any:
                ax.plot(x, z, y, label='object2 Main antenna')
            elif configFile.object2AntennaType == 'Frame':
                aF0 = np.array(
                    [-configFile.object2FrameAntennaLength / 2, -configFile.object2FrameAntennaHeight / 2, 0])
                aF1 = np.array(
                    [+configFile.object2FrameAntennaLength / 2, -configFile.object2FrameAntennaHeight / 2, 0])
                aF2 = np.array(
                    [+configFile.object2FrameAntennaLength / 2, +configFile.object2FrameAntennaHeight / 2, 0])
                aF3 = np.array(
                    [-configFile.object2FrameAntennaLength / 2, +configFile.object2FrameAntennaHeight / 2, 0])
                # aF0 = np.dot(np.transpose(rotationMatrix), aF0) + np.array(object2AntennaPosition)
                # aF1 = np.dot(np.transpose(rotationMatrix), aF1) + np.array(object2AntennaPosition)
                # aF2 = np.dot(np.transpose(rotationMatrix), aF2) + np.array(object2AntennaPosition)
                # aF3 = np.dot(np.transpose(rotationMatrix), aF3) + np.array(object2AntennaPosition)
                aF0 = np.dot(rotationMatrix, aF0) + np.array(object2AntennaPosition)
                aF1 = np.dot(rotationMatrix, aF1) + np.array(object2AntennaPosition)
                aF2 = np.dot(rotationMatrix, aF2) + np.array(object2AntennaPosition)
                aF3 = np.dot(rotationMatrix, aF3) + np.array(object2AntennaPosition)
                frameShape = [aF0, aF1, aF2, aF3, aF0]
                x, y, z = [], [], []
                for aF in frameShape:
                    x.append(aF[0])
                    y.append(aF[1])
                    z.append(aF[2])
                ## Plot Frame Antennas, if any:
                ax.plot(x, z, y, label='object2 Frame antenna')
        elif configFile.object2Type == 'Ellipse':
            object2EllipsePosition = np.array(configFile.object2Position)
            majorAxisLength = configFile.object2MajorAxisLength
            minorAxisLength = configFile.object2MinorAxisLength
            object2EllipseOrientationAlpha = configFile.object2OrientationAlpha
            object2EllipseOrientationBeta = configFile.object2OrientationBeta
            object2EllipseOrientationGamma = configFile.object2OrientationGamma

            c_x = np.array([[1, 0, 0],
                            [0, np.cos(math.radians(object2EllipseOrientationAlpha)),
                             -np.sin(math.radians(object2EllipseOrientationAlpha))],
                            [0, np.sin(math.radians(object2EllipseOrientationAlpha)),
                             np.cos(math.radians(object2EllipseOrientationAlpha))]])

            c_y = np.array([[np.cos(math.radians(object2EllipseOrientationBeta)), 0,
                             np.sin(math.radians(object2EllipseOrientationBeta))],
                            [0, 1, 0],
                            [-np.sin(math.radians(object2EllipseOrientationBeta)), 0,
                             np.cos(math.radians(object2EllipseOrientationBeta))]])

            c_z = np.array([[np.cos(math.radians(object2EllipseOrientationGamma)),
                             -np.sin(math.radians(object2EllipseOrientationGamma)), 0],
                            [np.sin(math.radians(object2EllipseOrientationGamma)),
                             np.cos(math.radians(object2EllipseOrientationGamma)), 0],
                            [0, 0, 1]])

            rotationMatrix = np.dot(c_z, np.dot(c_y, c_x))
            x, y, z = rotateEllipse(minorAxisLength, majorAxisLength, rotationMatrix, object2EllipsePosition)
            ax.plot(x, z, y, label='object2 Elliptical coil')
        elif configFile.object2Type == 'Puk':
            object2PukPosition = np.array(configFile.object2Position)
            object2PukHeight = configFile.object2PukHeight
            object2CircularCoilRadius = configFile.object2CircularCoilRadius
            object2PukOrientationAlpha = configFile.object2OrientationAlpha
            object2PukOrientationBeta = configFile.object2OrientationBeta
            object2PukOrientationGamma = configFile.object2OrientationGamma

            c_x = np.array([[1, 0, 0],
                            [0, np.cos(math.radians(object2PukOrientationAlpha)),
                             -np.sin(math.radians(object2PukOrientationAlpha))],
                            [0, np.sin(math.radians(object2PukOrientationAlpha)),
                             np.cos(math.radians(object2PukOrientationAlpha))]])

            c_y = np.array([[np.cos(math.radians(object2PukOrientationBeta)), 0,
                             np.sin(math.radians(object2PukOrientationBeta))],
                            [0, 1, 0],
                            [-np.sin(math.radians(object2PukOrientationBeta)), 0,
                             np.cos(math.radians(object2PukOrientationBeta))]])

            c_z = np.array([[np.cos(math.radians(object2PukOrientationGamma)),
                             -np.sin(math.radians(object2PukOrientationGamma)), 0],
                            [np.sin(math.radians(object2PukOrientationGamma)),
                             np.cos(math.radians(object2PukOrientationGamma)), 0],
                            [0, 0, 1]])

            # coilXOrientation = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
            coilXOrientation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
            coilOrientation = np.dot(c_z, np.dot(c_y, c_x))
            c0 = np.array([-object2CircularCoilRadius, -object2PukHeight / 2, 0])
            c1 = np.array([object2CircularCoilRadius, -object2PukHeight / 2, 0])
            c2 = np.array([object2CircularCoilRadius, object2PukHeight / 2, 0])
            c3 = np.array([-object2CircularCoilRadius, object2PukHeight / 2, 0])
            # c0 = np.dot(coilXOrientation), c0)  # + coilDefault3dPosition
            # c1 = np.dot(np.transpose(coilXOrientation), c1)  # + coilDefault3dPosition
            # c2 = np.dot(np.transpose(coilXOrientation), c2)  # + coilDefault3dPosition
            # c3 = np.dot(np.transpose(coilXOrientation), c3)  # + coilDefault3dPosition
            c0 = np.dot(coilXOrientation, c0)  # + coilDefault3dPosition
            c1 = np.dot(coilXOrientation, c1)  # + coilDefault3dPosition
            c2 = np.dot(coilXOrientation, c2)  # + coilDefault3dPosition
            c3 = np.dot(coilXOrientation, c3)  # + coilDefault3dPosition
            c0 = np.dot(coilOrientation, c0) + object2PukPosition
            c1 = np.dot(coilOrientation, c1) + object2PukPosition
            c2 = np.dot(coilOrientation, c2) + object2PukPosition
            c3 = np.dot(coilOrientation, c3) + object2PukPosition
            coilShape = [c0, c1, c2, c3, c0]
            x1, y1, z1 = [], [], []
            for c in coilShape:
                x1.append(c[0])
                y1.append(c[1])
                z1.append(c[2])
            ax.plot(x1, z1, y1, label='object 2 Puk x-coil')

            x2, y2, z2 = rotateCircle(object2CircularCoilRadius, np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
                                      [0, 0, 0])
            xyz_temp = []
            x2_rotated = []
            y2_rotated = []
            z2_rotated = []
            for i in range(0, len(x2)):
                # xyz_temp.append(
                #     np.dot(np.transpose(coilOrientation), np.array([x2[i], y2[i], z2[i]])) + object2PukPosition)
                xyz_temp.append(
                    np.dot(coilOrientation, np.array([x2[i], y2[i], z2[i]])) + object2PukPosition)
            for i in range(0, len(xyz_temp)):
                x2_rotated.append(xyz_temp[i][0])
                y2_rotated.append(xyz_temp[i][1])
                z2_rotated.append(xyz_temp[i][2])
            ax.plot(x2_rotated, z2_rotated, y2_rotated, label='object 2 Puk y-coil')

            c0 = np.array([-object2CircularCoilRadius, -object2PukHeight / 2, 0])
            c1 = np.array([object2CircularCoilRadius, -object2PukHeight / 2, 0])
            c2 = np.array([object2CircularCoilRadius, object2PukHeight / 2, 0])
            c3 = np.array([-object2CircularCoilRadius, object2PukHeight / 2, 0])
            # c0 = np.dot(np.transpose(coilOrientation), c0) + object2PukPosition
            # c1 = np.dot(np.transpose(coilOrientation), c1) + object2PukPosition
            # c2 = np.dot(np.transpose(coilOrientation), c2) + object2PukPosition
            # c3 = np.dot(np.transpose(coilOrientation), c3) + object2PukPosition
            c0 = np.dot(coilOrientation, c0) + object2PukPosition
            c1 = np.dot(coilOrientation, c1) + object2PukPosition
            c2 = np.dot(coilOrientation, c2) + object2PukPosition
            c3 = np.dot(coilOrientation, c3) + object2PukPosition

            coilShape = [c0, c1, c2, c3, c0]
            x3, y3, z3 = [], [], []
            for c in coilShape:
                x3.append(c[0])
                y3.append(c[1])
                z3.append(c[2])
            ax.plot(x3, z3, y3, label='object 2 Puk z-coil')
        elif configFile.object2Type == 'Ball':
            object2BallPosition = np.array(configFile.object2Position)
            object2BallCoilsRadius = configFile.object2CoilsRadius
            object2BallOrientationAlpha = configFile.object2OrientationAlpha
            object2BallOrientationBeta = configFile.object2OrientationBeta
            object2BallOrientationGamma = configFile.object2OrientationGamma
            c_x = np.array([[1, 0, 0],
                            [0, np.cos(math.radians(object2BallOrientationAlpha)),
                             -np.sin(math.radians(object2BallOrientationAlpha))],
                            [0, np.sin(math.radians(object2BallOrientationAlpha)),
                             np.cos(math.radians(object2BallOrientationAlpha))]])

            c_y = np.array([[np.cos(math.radians(object2BallOrientationBeta)), 0,
                             np.sin(math.radians(object2BallOrientationBeta))],
                            [0, 1, 0],
                            [-np.sin(math.radians(object2BallOrientationBeta)), 0,
                             np.cos(math.radians(object2BallOrientationBeta))]])

            c_z = np.array([[np.cos(math.radians(object2BallOrientationGamma)),
                             -np.sin(math.radians(object2BallOrientationGamma)), 0],
                            [np.sin(math.radians(object2BallOrientationGamma)),
                             np.cos(math.radians(object2BallOrientationGamma)), 0],
                            [0, 0, 1]])

            coilOrientation = np.dot(c_z, np.dot(c_y, c_x))
            # x1, y1, z1 = rotateCircle(object2BallCoilsRadius, np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),
            #                           [0, 0, 0])
            x1, y1, z1 = rotateCircle(object2BallCoilsRadius, np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
                                      [0, 0, 0])
            xyz_temp = []
            x1_rotated = []
            y1_rotated = []
            z1_rotated = []
            for i in range(0, len(x1)):
                # xyz_temp.append(
                #     np.dot(np.transpose(coilOrientation), np.array([x1[i], y1[i], z1[i]])) + object2BallPosition)
                xyz_temp.append(
                    np.dot(coilOrientation, np.array([x1[i], y1[i], z1[i]])) + object2BallPosition)
            for i in range(0, len(xyz_temp)):
                x1_rotated.append(xyz_temp[i][0])
                y1_rotated.append(xyz_temp[i][1])
                z1_rotated.append(xyz_temp[i][2])
            ax.plot(x1_rotated, z1_rotated, y1_rotated, label='object 2 Ball x-coil')

            x2, y2, z2 = rotateCircle(object2BallCoilsRadius, np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
                                      [0, 0, 0])
            xyz_temp = []
            x2_rotated = []
            y2_rotated = []
            z2_rotated = []
            for i in range(0, len(x2)):
                # xyz_temp.append(
                #     np.dot(np.transpose(coilOrientation), np.array([x2[i], y2[i], z2[i]])) + object2BallPosition)
                xyz_temp.append(
                    np.dot(coilOrientation, np.array([x2[i], y2[i], z2[i]])) + object2BallPosition)
            for i in range(0, len(xyz_temp)):
                x2_rotated.append(xyz_temp[i][0])
                y2_rotated.append(xyz_temp[i][1])
                z2_rotated.append(xyz_temp[i][2])
            ax.plot(x2_rotated, z2_rotated, y2_rotated, label='object 2 Ball y-coil')

            x3, y3, z3 = rotateCircle(object2BallCoilsRadius, np.eye(3), [0, 0, 0])
            xyz_temp = []
            x3_rotated = []
            y3_rotated = []
            z3_rotated = []
            for i in range(0, len(x3)):
                # xyz_temp.append(
                #     np.dot(np.transpose(coilOrientation), np.array([x3[i], y3[i], z3[i]])) + object2BallPosition)
                xyz_temp.append(
                    np.dot(coilOrientation, np.array([x3[i], y3[i], z3[i]])) + object2BallPosition)
            for i in range(0, len(xyz_temp)):
                x3_rotated.append(xyz_temp[i][0])
                y3_rotated.append(xyz_temp[i][1])
                z3_rotated.append(xyz_temp[i][2])
            ax.plot(x3_rotated, z3_rotated, y3_rotated, label='object 2 Ball z-coil')









    ## Labeling Axes according to right hand rule and showing the final plot:
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.invert_yaxis()
    ax.legend()
    plt.axis('equal')
    plt.show()

