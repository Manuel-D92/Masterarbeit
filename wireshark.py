class wireshark():
    @staticmethod
    def convert_Signal_wireshark(Signal_1):
        XX = [];
        YY = []
        # sample_rate =  1/((np.shape(Signal_1)[0]-1)/np.real(Signal_1[-1,0,0]))
        # lengh_Singal = len(Signal_1)
        # time_s = np.linspace(0,lengh_Singal*sample_rate,lengh_Singal)
        for i in range(0, len(Signal_1)):  # Signal_1.shape[0]): len(Signal_1)
            X = []
            for j in range(4, Signal_1.shape[2]):
                X.append(Signal_1.real[i, 3, j])
                X.append(Signal_1.imag[i, 3, j])
            XX.append(X)
        for i in range(0, len(Signal_1)):  # len(Signal_1)
            YY.append(0)
        return XX  # ,YY,time_s