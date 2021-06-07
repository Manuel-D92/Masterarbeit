"""!
@brief      Test the main relay.
@details

@date       01.09.2017
@author     rge
@copyright  Fraunhofer IIS
@see
"""

from goalref.reader_interface import ReaderInterface
import config
import time


class TestMainRelay():
    """!
    @brief  Switch the main relay six times.
    """

    def __init__(self):
        """!
        @brief Setup of class TestMainRelay
        """

    reader = ReaderInterface(resistance=config.resistance,  # Resistance of the load seen by the amplifier.
                             inductance=config.inductance,  # Inductance of the load seen by the amplifier.
                             load_factor=config.loadFactor,  #
                             num_antennas=config.numAntennas,  # Number of antenna pairs connected to the reader
                             channel_permutation=config.channelPermutation)

    for idx, c in enumerate(config.channels):
        reader.setFrequency(idx, c[0])
        if c[1] > 0:
            reader.setExciterCurrent(idx, c[1])
            reader.setExciterEnabled(idx, True)

    reader.setChannel20Switch(3)  # Set channel 20 to current feedback
    reader.enableConfiguration()
    time.sleep(5)  # wait for the termination of the config

    def testMainRelay(self):
        """!
        @brief Switches the relais of the main antennas six times
        """

    print('The relais of main antenna shall switch 6 times')
    for i in range(1, 7):
        print('Switch main relay: %d. try' % i)
        time.sleep(1)
        reader.setMainRelay(not reader.getMainRelay())
        print('Status main relay: %s' % reader.getMainRelay())
        time.sleep(2)




# Independent tests
if __name__ == "__main__":
    test = TestMainRelay()
    test.testMainRelay()