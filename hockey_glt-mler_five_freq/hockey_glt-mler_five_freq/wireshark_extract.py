'''
Created on 25.08.2020

@author: mler
'''

import sys
import os
import numpy as np
from pcapng import FileScanner # pip install python-pcapng
from pcapng.blocks import EnhancedPacket
from goalref.goalref_calibration import GoalrefCalibration

#FILENAME = r"C:\Users\dauserml\Desktop\dauserml_Messungen_2020_07_22\meas1.pcapng"
FOLDERNAME = r"C:\Users\dauserml\Documents\2020_09_25\Messung_1"


def load_File_List_from_folder(path):
    File_list_cal=[]
    for folder in os.listdir(path):
        if (folder.endswith(".pcapng")):
            File_list_cal.append(path+"\\"+folder)
    return File_list_cal


FILENAME_folder = load_File_List_from_folder(FOLDERNAME)
# Must match reader interface constants
for FILENAME in FILENAME_folder:
    NUM_FREQUENCIES = 5
    NUM_CHANNELS = 20

    # Should match the configuration used for realtime processing
    CURRENT_CHANNEL = 2
    NOISE_CANCELLATION = [(1,0,2), (3,2,4)]
    BUFFER_LEN = 100

    if len(sys.argv) > 1:
        FILENAME = sys.argv[1]

    print("Loading file %s..." % FILENAME)
    expected_len = 2*4*NUM_FREQUENCIES*NUM_CHANNELS+8
    raw = []
    timestamp_wiresharks = []
    with open(FILENAME, 'rb') as fp:
        scanner = FileScanner(fp)
        for block in scanner:
            if isinstance(block, EnhancedPacket):
                #tes = block.timestamp_wireshark
                data = block.packet_data[0x2a:]
                if len(data) == expected_len:
                    raw.append(data)
                    timestamp_wiresharks.append(block.timestamp)

    time=[]
    time.append(np.float(0))
    for i in range(0,len(timestamp_wiresharks)-1):
        time.append(np.float(timestamp_wiresharks[i+1])-np.float(timestamp_wiresharks[0]))

    print("Processing %d packets..." % len(raw))
    out = np.zeros((len(raw), NUM_FREQUENCIES, NUM_CHANNELS), dtype=np.complex128)
    headers = np.zeros(len(raw), dtype=np.uint32)
    i = 0
    for packet in raw:
        if (len(packet) != expected_len):
            continue
        header = np.ndarray((2,), ">i4", packet, offset=0)
        data = np.ndarray((2, NUM_FREQUENCIES, NUM_CHANNELS), ">i4", packet, offset=8)
        sample = (data[0]+1j*data[1]) * (5. / 2**27)
        out[i] = sample
        headers[i] = header[0]
        i += 1

    print("Positions with lost samples:")
    print(np.nonzero(np.diff(headers) > 1))

    print("Writing raw output file...")
    #outpath = os.path.join(os.path.dirname(FILENAME), os.path.basename(FILENAME) + ".raw.npy")
    #outpath_timestamp_wireshark = os.path.join(os.path.dirname(FILENAME), os.path.basename(FILENAME) + ".timestamp_wireshark.npy")
    #np.save(outpath, out)
    #np.save(outpath_timestamp_wireshark,[time,timestamp_wiresharks])

    print("Performing noise-cancellation...")
    cal = GoalrefCalibration(currentChannel=CURRENT_CHANNEL, noiseCancellation=NOISE_CANCELLATION, bufferLen=BUFFER_LEN)
    cal.updateCurrentWeights(out[:BUFFER_LEN])
    for i in range(len(out)):
        cal.apply(out[i], offset=True, reference=False, current=True)

    print("Writing calibrated output file...")
    outpath = os.path.join(os.path.dirname(FILENAME), os.path.basename(FILENAME) + ".cal.npy")
    outpath_timestamp_wireshark = os.path.join(os.path.dirname(FILENAME), os.path.basename(FILENAME) + ".timestamp_wireshark.npy")
    np.save(outpath, out)
    np.save(outpath_timestamp_wireshark,[time,timestamp_wiresharks])

    print("Completed.")
