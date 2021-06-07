"""!
@brief

@data 14. Nov. 2016

@author amueller
"""

from threading import Thread, RLock
from struct import pack
from copy import deepcopy
import socket
import time
import numpy as np
from queue import Queue
from .goalref_sample import GoalrefSample

# Constants
# Factor between raw reader output values and voltages
#GOALREF_VOLTAGE_FACTOR = 4.47 / 200300000
GOALREF_VOLTAGE_FACTOR = 5. / 2**27
# Guard time to wait between sending commands to the reader
GOALREF_COMMAND_GUARD_DELAY = .3
# Number of bits in the sine table index of the DCO
GOALREF_TABLE_IDX_LEN = 16
GOALREF_TABLE_IDX_LEN_PRECISE = 24
# Clock frequency of the DCO
GOALREF_MAIN_CLOCK = 1e6

NUM_FREQUENCIES = 5
NUM_CHANNELS = 20

# Addresses in FPGA configuration memory
GOALREF_MEMORY = {
    'leds': {
        'ledSingleRgb': 0x00000000 + (0 << 5),
        'ledRange': 0x00000000 + (1 << 5),
        'ledRangeStep': 0x00000000 + (2 << 5),
    },
    'exciter': {
        'phaseInc': 0x00000100 + (0 << 5),
        'phaseOff': 0x00000100 + (1 << 5),
        'gain': 0x00000100 + (2 << 5),
        'enable': 0x00000100 + (3 << 5),
        'phaseIncPrecise': 0x00000100 + (4 << 5),
    },
    'mixer': {
        'phaseInc': 0x00000200 + (0 << 5)
    },
    'lc': {
        'phaseInc': 0x00000300 + (0 << 5),
        'phaseOff': 0x00000300 + (1 << 5),
        'gain': 0x00000300 + (2 << 5),
        #'enable': lcBase + (3 << 5), #unused
        'auto': 0x00000300 + (4 << 5)
    },
    'testSig': {
        'phaseInc': 0x00000400 + (0 << 5),
        'enable': 0x00000400 + (1 << 5)
    },
    'switchCtrlBase': 0x00000500,
    'switchCtrl': {
        'extRelay0': 0x00000500 + 0,
        'extRelay1': 0x00000500 + 1,
        'extRelay2': 0x00000500 + 2,
        'extRelay3': 0x00000500 + 3,
        'extRelay4': 0x00000500 + 4,
        'mainRelay': 0x00000500 + 5,
        'frameRelay':  0x00000500 + 6,
        'swA0': 0x00000500 + 7,
        'swA1': 0x00000500 + 8,
        'swA2': 0x00000500 + 9,
        'gateExcDisable': 0x00000500 + 10,
        'gateCalEnable': 0x00000500 + 11
    },
    'advanced': {
        'reset': 0x00000600 + 0,
        'channelConfig': 0x00000600 + 1,
        'dataRate': 0x00000600 + 2,
        'askRate': 0x00000600 + 3,
        'askSend': 0x00000600 + 4,
        'udpStream': 0x00000600 + 5,
        'antSetting': 0x00000600 + 6,
        'clockMode': 0x00000600 + 7,
        'clockEnable': 0x00000600 + 8,
        'clockCtrl': 0x00000600 + 9,
        'syncStatus': 0x00000600 + 10
    },
    'microblaze': {
        'lcAuto': 0xffffff00 + (0 << 5)
    }
}

class ReaderInterface(Thread):
    '''!
    @brief Interfaces with a GoalRef reader unit via UDP for data collection and
    configuration of the reader.
    '''

    def __init__(self, reader_ip="192.168.1.10", reader_port=4321, bind_address="0.0.0.0", bind_port=1234,
                 num_antennas=10, resistance=0, inductance=0, load_factor=1, channel_permutation=None, sync_status=False):
        '''!
        @brief Initializes a ReaderInterface to communicate with a GoalRef reader at the
        given reader_ip and reader_port, listening for reader data at the given bind_address, bind_port.
        
        A thread is spawned to listen for incoming data packets from the reader.
        '''
        Thread.__init__(self)
        self._reader_ip = reader_ip
        self._reader_port = reader_port
        self._num_antennas = num_antennas
        self._channel_permutation = channel_permutation
        
        # Open UDP socket
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.bind((bind_address, bind_port))
        self._socket.settimeout(3.0)
        
        # Initialize status
        self._readerAvailable = False
        self._dataRequests = []
        self._listLock = RLock()
        self._lastCommand = time.time()-GOALREF_COMMAND_GUARD_DELAY
        self._initialConfig = False
        self._commandCnt = 0
        self._expectedPacketLen = 2*4*NUM_FREQUENCIES*NUM_CHANNELS+8

        # Raw command queue
        self._rawQueue = Queue()

        # Status data from hardware
        self._lastPacketId = -1
        self._lastLanIdentifier = 0
        self._expectLanIdentifier = 0
        
        # Initialize configuration
        self._channels = [
            ChannelConfig(0, 70000, resistance, inductance, load_factor),
            ChannelConfig(1, 119000, resistance, inductance, load_factor),
            ChannelConfig(2, 128000, resistance, inductance, load_factor),
            ChannelConfig(3, 134000, resistance, inductance, load_factor)
        ]
        while len(self._channels) < NUM_FREQUENCIES:
            self._channels.append(ChannelConfig(len(self._channels), 0, resistance, inductance, load_factor))
        if len(self._channels) > NUM_FREQUENCIES:
            self._channels = self._channels[:NUM_FREQUENCIES]
        
        self._switchCtrlMask = [0, 0, 0, 0, 0, -1, -1, 0, 0, 0, -1, -1]
        self._oldSwitchCtrlMask = deepcopy(self._switchCtrlMask)
        self._lcGain = -np.ones(16, dtype=np.uint32)
        self._oldLcGain = deepcopy(self._lcGain)
        self._channelConfig = (NUM_CHANNELS << 8) | NUM_FREQUENCIES
        self._oldChannelConfig = 0
        self._askRate = 1
        self._oldAskRate = self._askRate
        self._askDataQueue = Queue()
        self._clockCtrl = 0
        self._oldClockCtrl = 0
        self._clockMode = 0
        self._oldClockMode = 0
        self._syncStatus = sync_status
        self._syncStatusRequested = False
        
        # Start thread
        self._stopped = False
        self.setDaemon(True)
        self.start()
        
    def run(self):
        # Loop while not stopped
        while not self._stopped:
            try:
                # Receive a packet
                data, addr = self._socket.recvfrom(4096)
                # Check if packet is from reader
                if addr[0] != self._reader_ip:
                    print("Packet from unknown source: %s" % addr[0])
                    continue
                if not self._readerAvailable:
                    print("Reader available.")
                    self._readerAvailable = True
                
                # Parse the packet and forward it
                self._processPacket(data)
            except socket.timeout:
                self._readerAvailable = False

            # If the reader is available and we may send a packet (guard delay expired)...
            # Caution: Separate from above condition because values may have changed.
            if self._readerAvailable and self._lastCommand+GOALREF_COMMAND_GUARD_DELAY < time.time() and self._initialConfig:
                cmd = self._getNextGlobalPacket()
                if cmd is not None:
                    # Add command counter to packet
                    self._commandCnt += 1
                    addr = cmd[0] | (self._commandCnt << 16)
                    self._expectLanIdentifier = (self._commandCnt << 16) | (cmd[1] & 0xFFFF)
                    # Build payload for datagram
                    payload = pack('!I', addr) + pack('!I', cmd[1])
                    # Send payload to output
                    self._socket.sendto(payload, (self._reader_ip, self._reader_port))
                    self._lastCommand = time.time()
                
            # If the reader is available and we may send a packet (guard delay expired)...
            if self._readerAvailable and self._lastCommand+GOALREF_COMMAND_GUARD_DELAY < time.time() and self._initialConfig:
                # ... check for changes in reader config
                for cc in self._channels:
                    cmd = cc.getNextUpdatePacket()
                    if cmd is not None:
                        # Add command counter to packet
                        self._commandCnt += 1
                        addr = cmd[0] | (self._commandCnt << 16)
                        self._expectLanIdentifier = (self._commandCnt << 16) | (cmd[1] & 0xFFFF)
                        # Build payload for datagram
                        payload = pack('!I', addr) + pack('!I', cmd[1])
                        # Send payload to output
                        self._socket.sendto(payload, (self._reader_ip, self._reader_port))
                        self._lastCommand = time.time()
                        break
                
    def _processPacket(self, packet):
        # Check packet length
        if (len(packet) != self._expectedPacketLen):
            print('Ignoring wrong length GoalRef message: %d - expected %d' % (len(packet), self._expectedPacketLen))
            return
        
        # Convert data packet to array of integers
        header = np.ndarray((2,), ">i4", packet, offset=0)
        data = np.ndarray((2, NUM_FREQUENCIES, NUM_CHANNELS), ">i4", packet, offset=8)
        # First two bytes are status/header infos
        if self._syncStatus and self._syncStatusRequested:
            if header[0] != self._lastPacketId:
                print('## Clock config change ##')
                print('ESEL EAVA / WCE  OCE ')
                print('%s %s / %s %s' % ('....' if (header[0]&0x80000000 == 0) else '*ON*', '....' if (header[0]&0x40000000 == 0) else '*ON*',
                                         '....' if (header[0]&0x00000200 == 0) else '*ON*', '....' if (header[0]&0x00000100 == 0) else '*ON*'))
                print('LF Sync:    %s' % ('-' if (header[0]&0x00800000 == 0) else 'ON'))
                print('SATA Sync:  %s' % ('-' if (header[0]&0x00400000 == 0) else 'ON'))
                print('Fiber TX:   %s' % ('-' if (header[0]&0x00200000 == 0) else 'ON'))
                print('Fiber RX:   %s' % ('-' if (header[0]&0x00100000 == 0) else 'ON'))
                print('Fiber TX from: %s' % ('Bypass' if (header[0]&0x00080000 == 0) else 'SoC'))
                print('SATA TX from:  %s' % ('Bypass' if (header[0]&0x00040000 == 0) else 'SoC'))
                print('Bypass from:   %s' % ('SATA RX' if (header[0]&0x00020000 == 0) else 'Fiber RX'))
                print('SoC Clk from:  %s' % ('SATA RX' if (header[0]&0x00010000 == 0) else 'Fiber RX'))
                print()
                self._lastPacketId = header[0]
        else:
            self._lastPacketId = header[0]
        if header[1] != self._lastLanIdentifier:
            if header[1] != self._expectLanIdentifier:
                print('INVALID IDENTIFIER RECEIVED: %08x (expected %08x).' % (header[1], self._expectLanIdentifier))
            else:
                print('Command accepted: %08x' % header[1])
                self._lastCommand = time.time()-GOALREF_COMMAND_GUARD_DELAY
            self._lastLanIdentifier = header[1]
            
        # Combine real and imaginary parts into complex numbers (ignore first two header bytes)
        sample = (data[0]+1j*data[1])
        # Scale input value to volts
        sample *= GOALREF_VOLTAGE_FACTOR
        # Wrap sample in GoalrefSample object
        sample = GoalrefSample(sample, self._num_antennas)
        # Rearrange channels as configured (if configured)
        if self._channel_permutation is not None:
            sample.applyPermutation(self._channel_permutation)
        
        # Forward data to requesting entities
        # At the same time remove requests which should not repeat
        with self._listLock:
            self._dataRequests = [x for x in self._dataRequests if x.addSample(sample)]
            
    def _getNextGlobalPacket(self):
        if self._channelConfig != self._oldChannelConfig:
            self._oldChannelConfig = self._channelConfig
            print('Setting channel config to %04x' % self._channelConfig)
            return (GOALREF_MEMORY['advanced']['channelConfig'], self._channelConfig)
        for i in range(len(self._switchCtrlMask)):
            if self._switchCtrlMask[i] != self._oldSwitchCtrlMask[i]:
                self._oldSwitchCtrlMask[i] = self._switchCtrlMask[i]
                print('Setting switch control for switch %d to %d' % (i, self._switchCtrlMask[i]))
                return (GOALREF_MEMORY['switchCtrlBase'] + i, self._switchCtrlMask[i])
        for i in range(len(self._lcGain)):
            if self._lcGain[i] != self._oldLcGain[i]:
                self._oldLcGain[i] = self._lcGain[i]
                print('Setting leakage cancellation gain for channel %d to %d' % (i, self._lcGain[i]))
                return (GOALREF_MEMORY['lc']['gain'] + i, self._lcGain[i])
        if self._syncStatus and not self._syncStatusRequested:
            self._syncStatusRequested = True
            print('Requesting sync status')
            return (GOALREF_MEMORY['advanced']['syncStatus'], 1)
        if self._clockCtrl != self._oldClockCtrl:
            self._oldClockCtrl = self._clockCtrl
            print('Sending clock control command')
            return (GOALREF_MEMORY['advanced']['clockCtrl'], self._clockCtrl)
        if self._clockMode != self._oldClockMode:
            self._oldClockMode = self._clockMode
            print('Sending clock mode command')
            return (GOALREF_MEMORY['advanced']['clockMode'], self._clockMode)
        if self._askRate != self._oldAskRate:
            self._oldAskRate = self._askRate
            print('Setting basic ASK rate to %d Hz' % self._askRate)
            return (GOALREF_MEMORY['advanced']['askRate'], self._askRate)
        if not self._askDataQueue.empty():
            data = self._askDataQueue.get()
            print('Sending %08x through basic ASK' % data)
            return (GOALREF_MEMORY['advanced']['askSend'], data)
        if not self._rawQueue.empty():
            data = self._rawQueue.get()
            print("Sending raw (antenna/LED) configuration command to address %08x with data %08x" %(data[0], data[1]))
            return data
            
        return None
                
    def requestData(self, callback, blockSize=1, blocks=1):
        """!
        @brief To get datas from the reader.
        @param callback Function to be called when a block of data is available
        @param blockSize Number of samples to collect per block
        @param blocks Number of blocks to return. When -1 for unlimited
        @return request Object identifier for this request. Can be used for cancelRequest()
        """
        # Limit parameters to integers, negative blocks indicate infinite
        blockSize, blocks = int(blockSize), int(blocks)
        if blockSize < 1: blockSize = 1
        # Store data request as specified
        request = ReaderDataRequest(callback, blockSize, blocks)
        with self._listLock:
            self._dataRequests.append(request)
        return request
    
    def setClockControl(self, lfsy, sata, fitx, firx, clo1, clo0, cli1, cli0):
        clockCtrl = 0
        if lfsy: clockCtrl |= 0x80
        if sata: clockCtrl |= 0x40
        if fitx: clockCtrl |= 0x20
        if firx: clockCtrl |= 0x10
        if clo1: clockCtrl |= 0x08
        if clo0: clockCtrl |= 0x04
        if cli1: clockCtrl |= 0x02
        if cli0: clockCtrl |= 0x01
        self._clockCtrl = clockCtrl
        
    def setClockMode(self, mode):
        self._clockMode = mode
    
    def cancelRequest(self, request):
        '''!
        @brief To terminate the callback
        @param request Object identifier from requestData().
        '''
        with self._listLock:
            self._dataRequests.remove(request)
        
    def setFrequency(self, channel, frequency, precise=False):
        '''!
        @brief Set the frequency of a channel of the exciter
        @param channel Exciter channel. Choose between 0-3
        @param frequency Set the frequency
        @param precise If True, use high resolution setting, otherwise compatible low resolution
        '''
        self._channels[channel].setFrequency(frequency, precise)
        
    def getFrequency(self, channel):
        '''!
        @brief Get the last software frequency value of a channel
        @param channel  0-3
        @param frequency Set the frequency 0-3
        '''
        return self._channels[channel].getFrequency()
        
    def setMixerFrequency(self, channel, frequency):
        '''!
        @brief
        @param channel  0-3
        @param frequency Set the frequency 0-3
        '''
        self._channels[channel].setMixerFrequency(frequency)
        
    def getMixerFrequency(self, channel):
        '''!
        @brief
        @param frequency Set the frequency
        '''
        return self._channels[channel].getMixerFrequency()
        
    def setExciterCurrent(self, channel, current):
        '''!
        @brief Set the current of the exciter
        @param channel
        @param current Exciter current which is set through the user.
        '''
        self._channels[channel].setExciterCurrent(current)
        
    def getExciterCurrent(self, channel):
        '''!
         @brief Get the last software current value of the exciter
         @param channel
         '''
        return self._channels[channel].getExciterCurrent()
        
    def setExciterGain(self, channel, gain):
        '''!
        @brief Set the amplitude of the exciter
        @param channel
        @param gain 0-65535
        '''
        self._channels[channel].setExciterGain(gain)
        
    def getExciterGain(self, channel):
        '''!
        @brief Get the last software amplitude value of the exciter
        @param channel
        '''
        return self._channels[channel].getExciterGain()

    def setMainGain(self, channel, gain):
        '''!
        @brief Set the main antenna potentiometer of a given channel to the given gain value 
        @param channel 0-16
        @param gain    0-64
        '''
        #option value -> setMainGain: 0
        data = ((gain << 24)|(channel << 16)|(0x00))
        self._rawQueue.put((GOALREF_MEMORY['advanced']['antSetting'], data))

    def setFrameGain(self, channel, gain):
        '''!
        @brief Set the frame antenna potentiometer of a given channel to the given gain value
        @param channel 0-16
        @parma gain    0-64
        '''
        #option value -> setFrameGain: 1
        data = ((gain << 24)|(channel << 16)|(0x01))
        self._rawQueue.put((GOALREF_MEMORY['advanced']['antSetting'], data))

    def setMainCalib(self, channel, calib):
        '''!
        @brief activate or deactivate the main antenna calibration for a given channel
        @param channel  0-16
        @param calib    0 or 1
        '''
        #option value -> setMainCalib: 2
        data = ((calib << 24)|(channel<<16)|(0x02))
        self._rawQueue.put((GOALREF_MEMORY['advanced']['antSetting'], data))

    def setFrameCalib(self, channel, calib):
        '''!
        @brief activate or deactivate the frame antenna calibration for a given channel
        @param channel  0-16
        @param calib    0 or 1
        '''
        #option value -> setFrameCalib: 3
        data = ((calib<<24)|(channel<<16)|(0x03))
        self._rawQueue.put((GOALREF_MEMORY['advanced']['antSetting'], data))
    
    def setLedSingle(self, chain, led, rgb):
        data = (led << 24) | rgb
        self._rawQueue.put((GOALREF_MEMORY['leds']['ledSingleRgb'] | chain, data))
        
    def setLedRange(self, chain, base_color, begin=0, end=255, step=1):
        data = (begin << 24) | (end << 16) | (step << 8) | base_color
        self._rawQueue.put((GOALREF_MEMORY['leds']['ledRangeStep'] | chain, data))

    def setExciterEnabled(self, channel, enable):
        '''!
        @brief
        @param channel Which
        @param enable True or False
        '''
        self._channels[channel].setExciterEnabled(enable)
        
    def isExciterEnabled(self, channel):
        '''!
         @brief
         @param channel Which
         '''
        return self._channels[channel].isExciterEnabled()
    
    def setMainRelay(self, value):
        """!
        @brief Set the main reader relay at the input of the reader to the calibration signal or antenna signal
        @param value Calibration signal = False, Antenna signal = True
        """
        self._switchCtrlMask[5] = 1 if value else 0
        
    def setFrameRelay(self, value):
        """!
        @brief Set the frame reader relay at the input of the reader to the calibration signal or antenna signal
        @param value Calibration signal = False, Antenna signal = True
        """
        self._switchCtrlMask[6] = 1 if value else 0

    def setExciterGating(self, excEnable, calEnable):
        """!
        @brief Turn the exciter and calibration signal on or off
        @param excEnable True - Enable exciter signal, False - Disable exciter signal
        @param calEnable True - Enable calibration signal, False - Disable calibration signal
        """
        self._switchCtrlMask[10] = 0 if excEnable else 1
        self._switchCtrlMask[11] = 1 if calEnable else 0
    
    def setChannel20Switch(self, value):
        """!
        @brief To switch the muliplexer of channal 20
        @return
        """
        self._switchCtrlMask[7] = 1 if value & 0x1 > 0 else 0
        self._switchCtrlMask[8] = 1 if value & 0x2 > 0 else 0
        self._switchCtrlMask[9] = 1 if value & 0x4 > 0 else 0
        
    def setLCGain(self, path, gain):
        if path > 15:
            raise Exception('Not implemented!')
        self._lcGain[path] = gain
        
    def getLCGain(self, path, gain):
        if path > 15:
            return 0
        return self._lcGain[path]
        
    def getMainRelay(self):
        """!
        @brief
        @return
        """
        return (self._switchCtrlMask[5] == 1)
    
    def getFrameRelay(self):
        """!
        @brief
        @return
        """
        return (self._switchCtrlMask[6] == 1)
    
    def getExciterGating(self):
        return (self._switchCtrlMask[10] == 0, self._switchCtrlMask[11] == 1)
        
    def getChannel20Switch(self):
        """!
        @brief
        @return
        """
        value = self._switchCtrlMask[7]
        value |= self._switchCtrlMask[8]<<1
        value |= self._switchCtrlMask[9]<<2
        return value
        
    def enableConfiguration(self):
        """!
        @brief
        """
        self._initialConfig = True
        
    def getNumAntennas(self):
        """!
        @brief
        @return
        """
        return self._num_antennas
        
    def setAskRate(self, rate):
        self._askRate = rate
        
    def getAskRate(self):
        return self._askRate
        
    def askSend(self, data):
        self._askDataQueue.put(data)
    

class ReaderDataRequest(object):
    def __init__(self, callback, blockSize, blocks):
        # Store parameters and initialize buffer
        self._callback = callback
        self._blockSize = blockSize
        self._blocks = blocks
        self._collectedSamples = []
        
    def addSample(self, sample):
        # Store sample in buffer
        self._collectedSamples.append(deepcopy(sample))
        # If buffer contains the requested number of samples, forward it
        if len(self._collectedSamples) == self._blockSize:
            # Trigger callback
            self._callback(self._collectedSamples)
            
            # Check if more blocks have been requested
            self._blocks -= 1
            if self._blocks == 0:
                return False
            self._collectedSamples = []
        return True
    

class FastReaderInterface(ReaderInterface):
    def __init__(self, reader_ip="192.168.1.10", reader_port=4321, bind_address="0.0.0.0", bind_port=1234,
                 num_antennas=10, resistance=0, inductance=0, load_factor=1, channel_permutation=None):
        self._callback = None
        self._block_cnt = 0
        self._block_size = 1000
        self._block = np.zeros((1000, NUM_FREQUENCIES, NUM_CHANNELS), dtype=np.complex128)
        self._block_idx = 0
        
        super().__init__(reader_ip, reader_port, bind_address, bind_port, num_antennas, resistance, inductance, load_factor, channel_permutation)
        
    def _processPacket(self, packet):
        # Convert data packet to array of integers
        header = np.ndarray((2,), ">i4", packet, offset=0)
        data = np.ndarray((2, NUM_FREQUENCIES, NUM_CHANNELS), ">i4", packet, offset=8)
        
        # Check packet ID and reset block if there are missing samples
        if header[0] != self._lastPacketId+1:
            self._block_idx = 0
        
        # First two bytes are status/header infos
        self._lastPacketId = header[0]
        if header[1] != self._lastLanIdentifier:
            if header[1] != self._expectLanIdentifier:
                print('INVALID IDENTIFIER RECEIVED: %08x (expected %08x).' % (header[1], self._expectLanIdentifier))
            else:
                print('Command accepted: %08x' % header[1])
                self._lastCommand = time.time()-GOALREF_COMMAND_GUARD_DELAY
            self._lastLanIdentifier = header[1]
            
        if self._callback is not None and self._block_cnt != 0:
            # Combine real and imaginary parts into complex numbers (ignore first two header bytes)
            # Save result into output block
            self._block[self._block_idx] = data[0]+1j*data[1]
            self._block_idx += 1
            
            # If the requested block site is reached, forward a copy to the callback function
            if self._block_idx == self._block_size:
                self._callback(deepcopy(self._block)*GOALREF_VOLTAGE_FACTOR)
                self._block_idx = 0
                self._block_cnt -= 1
            
    def requestData(self, callback, blockSize=1000, blocks=-1):
        self._block_cnt = 0
        self._callback = callback
        if self._block_size < blockSize:
            self._block = np.zeros((blockSize, NUM_FREQUENCIES, NUM_CHANNELS), dtype=np.complex128)
            self._block_size = blockSize
        else:
            self._block_size = blockSize
            self._block = np.zeros((blockSize, NUM_FREQUENCIES, NUM_CHANNELS), dtype=np.complex128)
        self._block_idx = 0
        self._block_cnt = blocks
        
class ChannelConfig(object):
    """!
    @brief
    @param object
    """
    def __init__(self, index, defaultFreq, resistance, inductance, load_factor):
        """!
        @param index
        @param defaultFreq
        @param resistance Resistance of the exciter.
        @param inductance Inductance of the exciter.
        @param load_factor Value of the load seen by the amplifier.
        """
        self._index = index
        self._resistance = resistance
        self._inductance = inductance
        self._load_factor = load_factor
        
        self._exciterFreq = defaultFreq
        self._exciterGain = 0
        self._exciterCurrent = None
        self._exciterEnable = False
        self._mixerFreq = defaultFreq
        
        self._oldExciterFreq = None
        self._oldExciterGain = None
        self._oldExciterEnable = None
        self._oldMixerFreq = None
        
    def setFrequency(self, frequency, precise):
        self._exciterFreq = self._phaseIncToFreq(self._freqToPhaseInc(frequency, precise=precise), precise=precise)
        self._mixerFreq = self._exciterFreq
        if self._exciterCurrent is not None:
            self._exciterGain = self._calculateGain(self._exciterCurrent, frequency)
            
    def getFrequency(self):
        return self._exciterFreq
        
    def setMixerFrequency(self, frequency):
        self._mixerFreq = self._phaseIncToFreq(self._freqToPhaseInc(frequency))
        
    def getMixerFrequency(self):
        return self._mixerFreq
        
    def setExciterCurrent(self, current):
        self._exciterCurrent = current
        self._exciterGain = self._calculateGain(current, self._exciterFreq)
        
    def getExciterCurrent(self):
        return self._exciterCurrent
        
    def setExciterGain(self, gain):
        self._exciterCurrent = None
        self._exciterGain = gain
        
    def getExciterGain(self):
        return self._exciterGain
        
    def setExciterEnabled(self, enable):
        self._exciterEnable = enable
        
    def isExciterEnabled(self):
        return self._exciterEnable
        
    def getNextUpdatePacket(self):
        if self._exciterFreq != self._oldExciterFreq:
            data = self._freqToPhaseInc(self._exciterFreq, precise=True)
            self._oldExciterFreq = self._exciterFreq
            self._oldMixerFreq = self._exciterFreq
            print('Setting exciter and mixer channel %d frequency to %f Hz' % (self._index, self._exciterFreq))
            return (GOALREF_MEMORY['exciter']['phaseIncPrecise'] + self._index, data)
        if self._mixerFreq != self._oldMixerFreq:
            data = self._freqToPhaseInc(self._mixerFreq)
            self._oldMixerFreq = self._mixerFreq
            print('Setting mixer channel %d frequency to %f Hz' % (self._index, self._mixerFreq))
            return (GOALREF_MEMORY['mixer']['phaseInc'] + self._index, data)
        if self._exciterGain != self._oldExciterGain:
            data = self._exciterGain
            self._oldExciterGain = self._exciterGain
            print('Setting exciter channel %d gain to %d' % (self._index, data))
            return (GOALREF_MEMORY['exciter']['gain'] + self._index, data)
        if self._exciterEnable != self._oldExciterEnable:
            data = 1 if self._exciterEnable else 0
            self._oldExciterEnable = self._exciterEnable
            print('Setting exciter channel %d to %s' % (self._index, 'ENABLED' if data == 1 else 'DISABLED'))
            return (GOALREF_MEMORY['exciter']['enable'] + self._index, data)
        return None

    def _freqToPhaseInc(self, freq, precise=False):
        """"!
        @brief Helper function to convert from frequency to phase increment used in FPGA
        """
        if precise:
            return int(round(freq * 2**GOALREF_TABLE_IDX_LEN_PRECISE / GOALREF_MAIN_CLOCK))
        else:
            return int(round(freq * 2**GOALREF_TABLE_IDX_LEN / GOALREF_MAIN_CLOCK))

    def _phaseIncToFreq(self, phaseInc, precise=False):
        """"!
        @brief Helper function to convert from phase increment used in FPGA to frequency
        """
        if precise:
            return GOALREF_MAIN_CLOCK * phaseInc / 2**GOALREF_TABLE_IDX_LEN_PRECISE
        else:
            return GOALREF_MAIN_CLOCK * phaseInc / 2**GOALREF_TABLE_IDX_LEN

    def _calculateGain(self, current, frequency):
        """!
        @brief Helper function to calculate exciter gain from target current and frequency.
        Intern calculation:
        *       Impedance: \f$ Z = \sqrt{R^2 + (2 \cdot L \cdot \pi \cdot f)^2} \f$
        *       Filter compensation:  \f$ filterCompensation = 0.0419 (\frac{f}{1000})^2 - 7.3564 (\frac {f}{1000}) + 586.38 \f$
        *       Gain: \f$ gain = int(floor(filterCompensation \cdot Z \cdot I \cdot loadfactor))\f$
        @param current Exciter current which is set through the user.
        @param frequency
        @return gain 0-65535 (0-5V)
        """

        if False:
            filterfactor = -23
            voltageSupply = 54

            gm = voltageSupply * 0.4 * 10**(filterfactor / 20)     #  gm= I/U  #Transkonduktanz
            voltageFPGAOutput = current/gm
            gain= int(65535/5 *voltageFPGAOutput)
            return gain
        else:
            # Absolute value of impedance
            impedance = np.sqrt(self._resistance**2 + (self._inductance*2*np.pi*frequency)**2)
            # Compensation function for amplifier input filter empirically fitted from measurement data 
            filterCompensation = .0419*(frequency/1000.0)**2 - 7.3564*(frequency/1000.0) + 586.38
            gain = int(np.floor(filterCompensation * impedance * current * self._load_factor))
            return gain
