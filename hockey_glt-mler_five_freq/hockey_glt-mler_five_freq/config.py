'''
Created on Nov 17, 2016

@author: muellead
'''

# System characteristics
# Set resistance and inductance the values of the load seen by the amplifier.
# Enable just a single frequency and adjust load factor till flowing current matches
# configured one. Current at other frequencies should then match as well.
resistance = .4
inductance = 14e-6
loadFactor = 2.3

# Channel configuration
# Four channels indexed 0..3.
# Each has (<frequency>, <current>).
# Set current to 0 to disable a channel.
channels = [
    ( 96500.014, 1120), 
    (104000.007, 11475), # 1A = 11k5
    (111500.000, 1175),
    (118999.993, 12000), # 1A = 12k
    (126499.986, 1220)
]

# Default calibration file
# This calibration file will be loaded at startup.
# Set to None to disable.
defaultCalibration = None

# Number of antenna pairs connected to the reader
numAntennas = 10

# Channel permutation
# Raw channels will be arranged in the samples as in this list.
# Channels correspond to connectors in consecutive pairs (0/1 = A, 2/3 = B, ...).
# In each pair the lower (even) channel is frame and the higher (odd) channel is main.
# If you switch antenna connections between ports or have main/frame switched on one
# port, you will have to reverse that change here to keep the same antenna names in software.
# THIS PERMUTATION MUST BE 21 ELEMENTS LONG (21 channels in each sample)
channelPermutation = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 
                      10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

# Noise cancellation settings
# The channel from which the current reference signal is taken. Caution: This is
# applied after the channel permutation specified above.
currentChannel = 2
# Definition of frequency combinations for noise cancellation.
# This is a list of tuples of the form (<targetFreq>, <refFreq0>, [<refFreq1>, [...]]).
# The signal at the frequency <targetFreq> will be compensated by subtracting a 
# weighted copy of the current channel signal. The weight is predicted through the
# use of the specified reference frequencies <refFreq0>, ...
noiseCancellation = [(1,0,2), (3,2,4)]
# Buffer length for averaging the weight factors determined from the reference frequencies
# Use buffers from 10 to 100 samples. This has a large performance impact.
bufferLen = 100

# DDFT frequency pairs and weights (<positiveFreq>, <negativeFreq>, <weight>)
# Defines which frequencies should be used to calculate DDFTs as a basis for goal detection.
# The overall sum signals will be a weighted sum of the result of these pairs.
# Frequencies are indexed from 0 to 3
# Set <negativeFreq> to -1 to directly use the <positiveFreq> values with taking differences
# Example: [(1,3,1)] -> classic DDFT between frequencies 1 and 3: total = (f1-f3)*1
# Example: [(0,-1,1),(1,-1,1)] -> sum of frequencies 0 and 1: total = (f0)*1 + (f1)*1
# Example: [(0,3,2),(1,3,1),(2,3,0.5)] -> weighted sum: total = (f0-f3)*2 + (f1-f3)*1 + (f2-f3)*0.5
ddftPairs = [(1,3,1)]

# Length of moving average done over input signals
# This is only applied to goal detection, not to other views
movingAverageLen = 100

# Initial main live threshold
#mainLiveThreshold = .02 # ball 
mainLiveThreshold = .005 # puck 
# Initial frame threshold
#frameInsideThreshold = .015 # ball
frameInsideThreshold = .0015 # puck
# Main offset scaling factor
mainOffsetScalingFactor = 0

# Invert main antenna signals (for goal detection only)
invertMain = False
# Invert frame antenna signals (for goal detection only)
invertFrame = False

# Window lengths for noise power calculation in debug and noise views
debugNoiseWindowLen = 2500
noiseNoiseWindowLen = 1250
