"""!
@brief      Send a test signal over the calibration path.
@details    The signal has 119.000kHz. The test use a FFT. If the test fails it ist not possible to tell exactly
            what is the problem. The test prove if it is possible to send a signal over the calibration path and
            and receive that signal with the FPGA.
            A successful test means not that a channel is working well. It just means that a signal come through
            the calibration path and back to the FPGA.

@date       01.09.2017
@author     rge
@copyright  Fraunhofer IIS
@see
"""


from goalref.reader_interface import ReaderInterface
from numpy.fft import fft, fftshift
from scipy.signal import get_window
import numpy as np
import config
import logging
import time

# Factor between raw reader output values and voltages
GOALREF_VOLTAGE_FACTOR = 4.47 / 200300000

class TestCalRef():
    """!
    @brief  A test to show if a signal comes through the calibration path and back
    """

    def __init__(self):
        """!
        @brief Setup of class TestCalSignal
        @param frequency Choose between 0-3.
        @param exciterFrequncy Is the frequency of the calibration signal.
        @param channel Channel of the reader.
        @param fft_length Length of the FFT.
        @param fft_window Set the window which is used in the FFT.
        """
        self._frequency = 0
        self._exciterFrequency = 119500
        self._channel = 0
        self._fft_length = 1024
        self._fft_window = get_window('blackman', 1024)

    def callback(self, data):
        """!
        @brief FFT callback function.
        @param data
        """
        tmp = np.array([sample.getSampleVal(self._channel, self._frequency) for sample in data])
        self._values = 20 * np.log10(np.abs(fftshift(fft(tmp * self._fft_window, self._fft_length)))/self._fft_length)


    def launchTestCalRef(self):
        """!
        @brief
        @param
        @param
        """
        self._reader = ReaderInterface(resistance=config.resistance,  # Resistance of the load seen by the amplifier.
                                       inductance=config.inductance,  # Inductance of the load seen by the amplifier.
                                       load_factor=config.loadFactor,  #
                                       num_antennas=config.numAntennas
                                       # Number of antenna pairs connected to the reader
                                       )
        #Disable the exciter signal and enable the calibration signal
        self._reader.setExciterGating(False, True)

        #Set allexciter channels to zero gain
        for c in range(4):
            self._reader.setExciterGain(c, 0)


        # Set all the first exciter signal setting
        self._reader.setExciterEnabled(self._frequency, True)
        self._reader.setFrequency(self._frequency, self._exciterFrequency)
        self._reader.setExciterGain(self._frequency, 10000)
        #Send the configuration and than wait for 5 seconds
        self._reader.enableConfiguration()
        time.sleep(5)

        # Switch the main relay to the calibration signal
        self._reader.setMainRelay(True)
        self._reader.setFrameRelay(True)
        time.sleep(2)
        self._reader.setMainRelay(False)
        self._reader.setFrameRelay(False)
        time.sleep(2)
        # Try it a second time, because sometimes the relay switch sometimes not at the first time
        self._reader.setMainRelay(True)
        self._reader.setFrameRelay(True)

        print('Requesting data...')
        self._reader.requestData(self.callback, len(self._fft_window), -1)
        print('Wait for FFT data')
        time.sleep(5)
        #test loop
        for i in range(20):
            self._channel = i
            time.sleep(1)
            self._voltageAmplitude =  np.round(10**(self._values[512] /20), 2)
            if self._values[512] > -13:
                print('Channel: %s, FFT: %s, Amplitude: %s V' % (self._channel, self._values[512], self._voltageAmplitude))
            else:
                print('Channel: %s, FFT: %s, Amplitude: %s V Not OK' % (self._channel, self._values[512], self._voltageAmplitude))


# Independent tests
if __name__ == "__main__":
    test = TestCalRef()
    test.launchTestCalRef()