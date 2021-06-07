import numpy as np
import scipy.signal as signal_freqenz
import matplotlib.pyplot as plt

class Signal_resampling():
    @staticmethod
    def resampl(tm_timestamp,timestamp_qualisys,tm_wireshark):
        f = len(tm_timestamp)
        ff_wireshark_shape = []
        tt = []
        for i in range(0, len(tm_timestamp)):
            if (int(round(tm_timestamp[i], 3) * 1000) >= int(round(timestamp_qualisys[0],3)*1000) and int(round(tm_timestamp[i], 3) * 1000) <= int(round(timestamp_qualisys[-1],3)*1000)):
                ff_wireshark_shape.append(tm_wireshark[i, :])
                tt.append(tm_timestamp[i])
        ff_wireshark_shape = np.array(ff_wireshark_shape)
        tt = np.array(tt)
        wireshark_signal = signal_freqenz.resample(ff_wireshark_shape, len(timestamp_qualisys), axis=0)
        #plt.plot(tm_timestamp,tm_wireshark)
        #plt.figure()
        #plt.plot(timestamp_qualisys,wireshark_signal)
        #plt.show()
        #plt.plot(tm_timestamp,tm_wireshark,timestamp_qualisys,wireshark_signal)
        #plt.show()
        return wireshark_signal

    @staticmethod
    def resampl_self_made(timestamp_wireshark_in,Signal_in,decimal_digits): # timestamp_wireshark_in = input timestamp_wiresharks; Signal_in = input Signale; decimal_digits = welche Nachkommastelle (z.b. 2 -> 10^-2 -> 0.01)
        #print(timestamp_wireshark_in[263345:263350])
        e = np.round(np.transpose(timestamp_wireshark_in[ :],axes=0), decimal_digits)
        first_time = int(np.round(timestamp_wireshark_in[ 0], 2) * (10 ** decimal_digits)) # first time in INT
        last_time = int(np.round(timestamp_wireshark_in[ -1], 2) * (10 ** decimal_digits)) # last time in INT
        # last_time = last_time - first_time
        # first_time = first_time -first_time
        count_predecessor = 0
        timestamp_wireshark_neu_controlling = []
        timestamp_wireshark_neu = []
        ff_wireshark_neu = []
        #print(e[263345:263350])
        for i in range(first_time, last_time):
            float_i = (i / (10 ** decimal_digits))
            count = np.count_nonzero(e == float_i)
            timestamp_wireshark_tm = []
            ff_wireshark_tm = []
            for j in range(count_predecessor, count + count_predecessor):
                timestamp_wireshark_tm.append(np.transpose(timestamp_wireshark_in[ j]))
                ff_wireshark_tm.append(Signal_in[j, :])
            timestamp_wireshark_tm = np.array(timestamp_wireshark_tm)
            ff_wireshark_tm = np.array(ff_wireshark_tm)
            if (count == 0):
                nan_array=np.zeros((1, len(ff_wireshark_neu[0,:])),dtype=int)
                nan_array[:,:]=np.nan
                ff_wireshark_neu = np.vstack((ff_wireshark_neu, nan_array))
                timestamp_wireshark_neu = np.vstack((timestamp_wireshark_neu,float_i))

            if (len(ff_wireshark_neu) > 0 and len(ff_wireshark_tm) > 0):
                timestamp_wireshark_neu_controlling = np.vstack(((timestamp_wireshark_neu_controlling, np.mean(timestamp_wireshark_tm, axis=0))))
                ff_wireshark_neu = np.vstack((ff_wireshark_neu, np.mean(ff_wireshark_tm, axis=0)))
                timestamp_wireshark_neu = np.vstack((timestamp_wireshark_neu, e[count_predecessor]))

            elif (len(ff_wireshark_neu) == 0 and len(ff_wireshark_tm) > 0):
                timestamp_wireshark_neu_controlling = (np.mean(timestamp_wireshark_tm, axis=0))
                ff_wireshark_neu = np.mean(ff_wireshark_tm, axis=0)
                timestamp_wireshark_neu = (e[count_predecessor])
            count_predecessor = count_predecessor + count
            print("Wert:", i, " bis:", last_time)
        return timestamp_wireshark_neu,ff_wireshark_neu