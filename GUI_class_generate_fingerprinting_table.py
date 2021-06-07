from __future__ import print_function
import numpy as np
import math as m
import multiprocessing as mp
import time
import timeit
import config_DWRegal as configFile
import os

class GenerateFingerprintingTable:

    def __init__(self, exciter, coil, antennaList, frequency, coilResistance, positions, angles, objectType, numberOfCores=4,
                 app=' ', author=' ', additionalInfo=' ', headerFlag=True):
        self.exciter = exciter
        self.coil = coil
        self.antennaList = antennaList
        self.frequency = frequency
        self.coilResistance = coilResistance
        self.positions = positions
        self.angles = angles
        self.objectType = objectType
        self.numberOfCores = numberOfCores
        self.app = app
        self.author =  author
        self.additionalInfo = additionalInfo
        self.headerFlag = headerFlag
        self.header = None
        if self.headerFlag:
            self.header = self.create_header()
        self.ignoredPositionsOrientationsIndices = []
        self.dummy = np.empty([5,2])

    def preprocess_positions(self,positions, printOut=True):
        """
        :param      positions: [xsteps, ysteps, zsteps]
        :param      printOut: if True, it prints out the positions
        :return:    positions: list of 3D-position-vectors

        Example: positions = [np.array([1,2,3,4,5]), np.array([5,7,9]), np.array([20,21])
                 positions = [np.array([1,5,20]), np.array([1,5,21]), np.array([1,7,20]), np.array([1,7,21]),
                              np.array([1,9,20]), np.array([1,9,21]), np.array([2, 5, 20]), np.array([2, 5, 21]), ... ]
        """
        if printOut == True:
            print('\nConfiguring positions...')
            # Print position-steps:
            #for d, pos in zip(['X', 'Y', 'Z'], positions):
            #    if len(pos) > 1:
            #        print('     {0}:   {1} ... {2}   ({3})   [m]'.format(d, pos[0], pos[-1], (pos[1] - pos[0])))
            #    else:
            #        print('     {0}:   {1}   [m]'.format(d, pos[0]))
            for d in range(0,len(positions)):
                print(d)
        # Convert position-steps to 3D-positions:
        processedPositions = []

        #for x in positions[0]:
        #    for y in positions[1]:
        #        for z in positions[2]:
        #            processedPositions.append(np.array([x, y, z]))
        processedPositions = positions

        # for x in positions[0]:
        #     for y in positions[1]:
        #         for z in positions[2]:
        #             processedPositions.append(np.array([x, y, z]))

        return processedPositions

    def preprocess_angles(self,angles, printOut=True):
        """
        :param      angles: [xangles, yangles, zangles, unit]
        :param      printOut: if True, it prints out the angles
        :return:    rotationMatrices: list of rotation-matrices
                    eulerAngles: list of euler-angles
        """
        unit = angles[3]
        if printOut ==True:
            print('\nConfiguring rotations...')
            # Print angle-steps:
            for d, ang in zip(['X', 'Y', 'Z'], angles):
                if len(ang) > 1:
                    print('     {0}:   {1} ... {2}   ({3})   [{4}]'.format(d, ang[0], ang[-1], (ang[1] - ang[0]), unit))
                else:
                    print('     {0}:   {1}   [{2}]'.format(d, ang[0], angles[3]))

        # Create Rotation-Matrices and Euler-Angles from angle-steps:
        orientations = []
        rotationMatrices = []
        eulerAngles = []  # needed for output

        for x_i in angles[0]:
            for y_i in angles[1]:
                for z_i in angles[2]:
                    eulerAngle = np.array([x_i, y_i, z_i])
                    eulerAngles.append(np.array([x_i, y_i, z_i]))

                    if unit == 'deg':
                        phi_x, phi_y, phi_z = m.radians(x_i), m.radians(y_i), m.radians(z_i)
                    else:
                        phi_x, phi_y, phi_z = x_i, y_i, z_i

                    cos_x, cos_y, cos_z = np.cos(phi_x), np.cos(phi_y), np.cos(phi_z)
                    sin_x, sin_y, sin_z = np.sin(phi_x), np.sin(phi_y), np.sin(phi_z)

                    c_x = np.array([[1, 0, 0],
                                    [0, cos_x, -sin_x],
                                    [0, sin_x, cos_x]])

                    c_y = np.array([[cos_y, 0, sin_y],
                                    [0, 1, 0],
                                    [-sin_y, 0, cos_y]])

                    c_z = np.array([[cos_z, -sin_z, 0],
                                    [sin_z, cos_z, 0],
                                    [0, 0, 1]])

                    # First rotation around x-Axis, then y-Axis, then z-Axis
                    rotationMatrix = np.dot(c_z, np.dot(c_y, c_x))
                    rotationMatrices.append(np.dot(c_z, np.dot(c_y, c_x)))
                    orientations.append([rotationMatrix, eulerAngle])
        return orientations

    def create_header(self):
        ## Determine limits and step size of the grid of positions and angles:
        limits = '\n\n Positions:'
        for d, pos in zip(['X', 'Y', 'Z'], self.positions):
            if len(pos) > 1:
                limits += '\n     {0}:   {1} ... {2}   ({3})   [m]'.format(d, pos[0], pos[-1], (pos[1] - pos[0]))
            else:
                limits += '\n     {0}:   {1}   [m]'.format(d, pos[0])

        limits += '\n\n angle_steps:'
        for d, ang in zip(['X', 'Y', 'Z'], self.angles):
            if len(ang) > 1:
                limits += '\n     {0}:   {1} ... {2}   ({3})   [{4}]'.format(d, ang[0], ang[-1], (ang[1] - ang[0]),
                                                                             self.angles[3])
            else:
                limits += '\n     {0}:   {1}   [{2}]'.format(d, ang[0], self.angles[3])

        # Configure column heading:
        column_heading = 'X-position;Y-position;Z-position;X-angle;Y-angle;Z-angle;' \
                         'coil X;coil Y;coil Z;'

        frame_heading = main_heading = ''
        if 1 == 1:
            if 8 != 0:
                for i in range(8):
                    main_heading += 'main%s;' % (i + 1)
        if 1 == 1:
            if 8 != 0:
                for i in range(8):
                    frame_heading += 'frame%s;' % (i + 1)

        column_heading += frame_heading + main_heading

        # Get current date:
        year, month, day, hour, minute = time.localtime()[:5]

        # Get positions and angles lengths:
        positions = self.preprocess_positions(self.positions, printOut=False)
        orientations = self.preprocess_angles(self.angles, printOut=False)
        # Check if the table is arranged positions-wise or orientations-wise:
        #if len(positions) >= len(orientations):
        #if len(orientations) == 1:
        tableArrangment = 'positionsWise'
        # else:
        #     tableArrangment = 'orientationsWise'
        # Put all parts together:
        header = '--- Fingerprinting-Table ---\n   ' \
                 '%s.%s.%s\n   ' % (day, month, year) \
                 + self.author + '\n' \
                 + self.objectType + '\n' \
                 + self.app + '\n' \
                 + self.additionalInfo \
                 + limits + '\n' \
                 + tableArrangment + '\n' \
                 + column_heading
        return header

    def calculate_voltages_multipos(self,arg):
        exciter, f, coil, coilResistance, antennaList, position, orientations, queue = arg

        coil.set_position(position)
        subtable = np.empty([len(orientations), 9+len(antennaList)], dtype=float)
        for i, orientation in enumerate(orientations):
            rotationMatrix, eulerAngles = orientation
            coil.set_orientation(rotationMatrix)
            start = timeit.default_timer()

            # Voltage in coils:
            try:
                coil_voltage = coil.inducedVoltage(exciter, f)
            except AssertionError as e:
                print('    --> Position: {0}\n'.format(position), e,
                      '\n\x1b[1;31m         -> Position will be removed from table.\x1b[0m',
                      '\n\x1b[34mProblem occured while calculating COIL voltage!\x1b[0m\n')

                result = np.empty([1, 9+len(antennaList)])
                result[:] = np.nan
                subtable[i] = result
                continue
            else:
                coil.set_current(coil_voltage / coilResistance)
                if type(coil_voltage) is not np.ndarray:
                    coil_voltage = np.array([0, 0, coil_voltage])


            # Voltage in antennas:
            antenna_voltages = np.empty([1, len(antennaList)], dtype=float)
            try:
                for ii, antenna in enumerate(antennaList):
                    antenna_voltages[0][ii] = antenna.inducedVoltage(coil, f)

            except AssertionError as e:
                print('    --> Position: {0}\n'.format(position), e,
                      '\n\x1b[1;31m         -> Position will be removed from table.\x1b[0m',
                      '\n\x1b[34mProblem occured while calculating ANTENNA voltage!\x1b[0m\n')
                result = np.empty([1, 9 + len(antennaList)])
                result[:] = np.nan
                subtable[i] = result
                continue

            # Add line to table:
            subtable[i][0:3] = position
            subtable[i][3:6] = eulerAngles
            subtable[i][6:9] = coil_voltage
            subtable[i][9: ] = antenna_voltages
            stop = timeit.default_timer()
            print('    --> Position: {0},   Angle: {1}    /    Runtime: {2} s'.format(position[0:3], eulerAngles, (stop - start)))


        # Notify parent of completed work unit
        queue.put(position)


        return subtable

    def calculate_voltages_multipos_config_DWRegal(self,arg):
        exciter, f, coil, coilResistance, antennaList, position, orientations, queue = arg

        coil.set_position(position)
        subtable = np.empty([len(orientations), 9+len(antennaList)], dtype=float)
        for i, orientation in enumerate(orientations):
            rotationMatrix, eulerAngles = orientation
            coil.set_orientation(rotationMatrix)
            start = timeit.default_timer()

            # Voltage in coils:
            try:
                coil_voltage = coil.inducedVoltage(exciter, f)
            except AssertionError as e:
                print('    --> Position: {0}\n'.format(position), e,
                      '\n\x1b[1;31m         -> Position will be removed from table.\x1b[0m',
                      '\n\x1b[34mProblem occured while calculating COIL voltage!\x1b[0m\n')

                result = np.empty([1, 9+len(antennaList)])
                result[:] = np.nan
                subtable[i] = result
                continue
            else:
                coil.set_current(coil_voltage / coilResistance)
                if type(coil_voltage) is not np.ndarray:
                    coil_voltage = np.array([0, 0, coil_voltage])


            # Voltage in antennas:
            antenna_voltages = np.empty([1, len(antennaList)], dtype=float)
            try:
                for ii, antenna in enumerate(antennaList):
                    antenna_voltages[0][ii] = antenna.inducedVoltage(coil, f)

            except AssertionError as e:
                print('    --> Position: {0}\n'.format(position), e,
                      '\n\x1b[1;31m         -> Position will be removed from table.\x1b[0m',
                      '\n\x1b[34mProblem occured while calculating ANTENNA voltage!\x1b[0m\n')
                result = np.empty([1, 9 + len(antennaList)])
                result[:] = np.nan
                subtable[i] = result
                continue

            # Add line to table:
            subtable[i][0:3] = position
            subtable[i][3:6] = eulerAngles
            subtable[i][6:9] = coil_voltage
            subtable[i][9: ] = antenna_voltages
            stop = timeit.default_timer()
            print('    --> Position: {0},   Angle: {1}    /    Runtime: {2} s'.format(position[0:3], eulerAngles, (stop - start)))


        # Notify parent of completed work unit
        queue.put(position)


        return subtable

    def calculate_voltages_multirot(self,arg):
        exciter, f, coil, coilResistance, antennaList, positions, orientation, queue = arg
        rotationMatrix, eulerAngles = orientation
        coil.set_orientation(rotationMatrix)
        subtable = np.empty([len(positions), 9+len(antennaList)], dtype=float)
        for i, position in enumerate(positions):
            coil.set_position(position)
            start = timeit.default_timer()

            # Voltage in coils:
            try:
                coil_voltage = coil.inducedVoltage(exciter, f)
            except AssertionError as e:
                print('    --> Position: {0}\n'.format(position), e,
                      '\n\x1b[1;31m         -> Position will be removed from table.\x1b[0m',
                      '\n\x1b[34mProblem occured while calculating COIL voltage!\x1b[0m\n')
                result = np.empty([1, 9 + len(antennaList)])
                result[:] = np.nan
                subtable[i] = result
                continue
            coil.set_current(coil_voltage / coilResistance)
            if type(coil_voltage) is not np.ndarray:
                coil_voltage = np.array([0, 0, coil_voltage])

            # Voltage in antennas:
            antenna_voltages = np.empty([1, len(antennaList)], dtype=float)
            try:
                for ii, antenna in enumerate(antennaList):
                    antenna_voltages[0][ii] = antenna.inducedVoltage(coil, f)
            except AssertionError as e:
                print('    --> Position: {0}\n'.format(position), e,
                      '\n\x1b[1;31m         -> Position will be removed from table.\x1b[0m',
                      '\n\x1b[34mProblem occured while calculating ANTENNA voltage!\x1b[0m\n')
                result = np.empty([1, 9 + len(antennaList)])
                result[:] = np.nan
                subtable[i] = result
                continue
            # Add line to table:
            subtable[i][0:3] = position
            subtable[i][3:6] = eulerAngles
            subtable[i][6:9] = coil_voltage
            subtable[i][9:] = antenna_voltages
            stop = timeit.default_timer()
            print('    --> Position: {0},   Angle: {1}    /    Runtime: {2} s'.format(position[0:3], eulerAngles,
                                                                                      (stop - start)))
        # Notify parent of completed work unit
        queue.put(orientation)

        #return self.resultsArray
        return subtable

    def calculate_table(self):
        # Get positions, rotation matrices and euler angles from position- and angle-steps:
        positions = self.preprocess_positions(self.positions)
        orientations = self.preprocess_angles(self.angles)

        # Calculate antenna voltages for every coil position and rotation:
        print('\n\nCalculating voltages...')

        # Start a worker pool and let it process all the points
        pool = mp.Pool(self.numberOfCores)
        manager = mp.Manager()
        queue = manager.Queue()

        start = timeit.default_timer()


        length = len(positions)
        #result = pool.map_async(self.calculate_voltages_multipos, [(self.exciter, self.frequency, self.coil, self.coilResistance, self.antennaList, position,
        #                                                       orientations, queue) for position in positions])
        result = pool.map_async(self.calculate_voltages_multipos_config_DWRegal, [(self.exciter, self.frequency, self.coil, self.coilResistance, configFile.antennaList, position,
                                                              orientations, queue) for position in positions])


        # Show status
        while not result.ready():
            time.sleep(1)
            size = queue.qsize()
            stop = timeit.default_timer()
            print('\n    Simulation %.1f%% completed (%d of %d points)    runtime: %d s\n' \
                  % (size * 100.0 / length, size, length, (stop - start)))

        self.resultsArray = self.reshape_results(np.array(result.get()))
        self.resultsArray = self.clean_results_array(self.resultsArray)

    def save_results(self, path='.\\tables', tableName='table', tableFormat='%10.5E'):
        print('\n\nSaving data...')
        # Check, if directory exists and create new folder, if necessary:
        try:
            if os.path.isdir(path) == False:
                os.mkdir(path)

            filename_full = os.path.join(path, tableName)

        except IOError as exception:
            filename_full = tableName
            print(exception, '\n--> Table is saved to source-directory')

        # Extend file name, if a file with the same name already exists:
        count = 0
        while (os.path.isfile(filename_full + '.csv')):
            count += 1
            filename_full = os.path.join(path, tableName + '(%s)' % count)
        np.savetxt(filename_full + '.csv', self.resultsArray, fmt=tableFormat, delimiter=';', header=self.header)
        print('    Table is saved to:   ' + filename_full + '.csv')
    ## This method removes all the rows that contains
    # NaN (if any) from the resultsArray of calculate_voltages_multipos()
    # or calculate_voltages_multirot() methods.
    # It returns a new clean array without any NaN values.
    # If there are rows that includes NaN values, the returned
    # array length will be less than the original one.
    def clean_results_array(self, resultsArray):
        return resultsArray[~np.isnan(resultsArray).any(axis=1)]
    ## This method converts the given 3D numpy array from (m,1,n)
    # to (m,n) 2D numpy array
    def reshape_results(self, resultsArray):
        resultsArray = resultsArray.reshape(np.shape(resultsArray)[0]*np.shape(resultsArray)[1], -1)
        return resultsArray
