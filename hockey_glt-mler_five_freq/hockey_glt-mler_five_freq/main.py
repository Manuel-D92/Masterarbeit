'''
Created on Nov 13, 2016

@author: muellead
'''

from tkinter import Tk, N, S, E, W
from gui.application import Application
from goalref.reader_interface import ReaderInterface
from goalref.goalref_calibration import GoalrefCalibration

# Initialize config namespace with defaults
config = {
    "resistance": .55,
    "inductance": 39e-6,
    "loadFactor": 1.0,
    
    "channels": [
        (70000, 0),
        (119000, 0),
        (128000, 0),
        (134000, 0)
    ],
    "channelPermutation": list(range(21)),
    "numAntennas": 7,
    
    "defaultCalibration": None,
    "currentChannel": None,
    "noiseCancellation": [],
    "calBufferLen": 100,
    
    "ddftPairs": [(1,3,1)],
    "movingAverageLen": 100,
    
    "mainLiveThreshold": .005,
    "frameInsideThreshold": .0015,
    "mainOffsetScalingFactor": 0,
    "invertMain": False,
    "invertFrame": False
}

# Try to read the configuration file and execute its contents
try:
    config_file = open('config.py', 'r')
    config_contents = config_file.read()
    config_file.close()
    
    exec(config_contents, config)
except FileNotFoundError:
    # Ignore missing config file -> defaults will be used
    pass

# Connect to reader and configure it
reader = ReaderInterface(resistance=config["resistance"], 
                         inductance=config["inductance"], 
                         load_factor=config["loadFactor"],
                         num_antennas=config["numAntennas"],
                         channel_permutation=config["channelPermutation"],
                         reader_ip="192.168.50.10")
for idx, c in enumerate(config["channels"]):
    reader.setFrequency(idx, c[0], precise=True)
    if c[1] > 0:
        reader.setExciterGain(idx, c[1])
        reader.setExciterEnabled(idx, True)
reader.setExciterGating(True, False)
reader.enableConfiguration()

# Load calibration if specified
calibration = GoalrefCalibration(currentChannel=config["currentChannel"],
                                 noiseCancellation=config["noiseCancellation"],
                                 bufferLen=config["calBufferLen"])
if config["defaultCalibration"] is not None:
    calibration.load(config["defaultCalibration"])

# test examples for the i2c access
# reader.setMainGain(5, 64)
# reader.setFrameCalib(5, 1)
# reader.setMainCalib(5,1)
# reader.setFrameGain(5, 0)
# reader.setMainGain(5, 0)

# Leave uncommented:
reader.setFrameRelay(True)
reader.setMainRelay(True)
# To use cal signal uncomment:
#reader.setExciterGating(False, True)
# To use amplifier signal uncomment:
reader.setFrameRelay(False)
reader.setMainRelay(False)

# reader.setLedRange(0, 0, 0, 37)
# reader.setLedRange(1, 0, 0, 37)
# 
# reader.setLedRange(0, 1, 1, 11)
# reader.setLedRange(0, 2, 13, 24)
# reader.setLedRange(0, 3, 26, 36)
# 
# reader.setLedRange(1, 4, 1, 11)
# reader.setLedRange(1, 5, 13, 24)
# reader.setLedRange(1, 6, 26, 36)

# Convert config values to dict for passing into GoalDetector
detectorConfig = {
    'ddft_pairs': config["ddftPairs"],
    'moving_average_len': config["movingAverageLen"],
    'invert_main': config["invertMain"],
    'invert_frame': config["invertFrame"],
    'main_live_threshold': config["mainLiveThreshold"],
    'frame_inside_threshold': config["frameInsideThreshold"],
    'main_offset_factor' : config["mainOffsetScalingFactor"]
}

# Start up Tkinter application
root = Tk()
root.wm_title("Hockey GLT")
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
app = Application(master=root, reader=reader, calibration=calibration, detectorConfig=detectorConfig, 
                  debugNoiseWindowLen=config["debugNoiseWindowLen"], noiseNoiseWindowLen=config["noiseNoiseWindowLen"])
app.grid(sticky=N+S+E+W)
root.mainloop()
try:
    root.destroy()
except:
    pass
