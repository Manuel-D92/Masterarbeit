import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import math
from sklearn.metrics import mean_squared_error

class config_lutwqMSE():
    path = r"C:\Users\Dauser\Desktop\dauserml_Messungen_2020_07_22\Referenz_alfred\Used_Data"
    path_look_up_table = r"C:\Users\Dauser\Desktop\Skripte\tables\DW_Shelf_45DegAntennas(17).csv"

class look_up_table_with_qualisys_MSE():
    def __init__(self,path,path_look_up_table):
        self.path = path
        self.postions_look_up_table = pd.read_csv(path_look_up_table, skiprows=17,sep=';')

    def compare_look_up_table_with_real_Data(self):
        postions_look_up_table = self.postions_look_up_table.drop(self.postions_look_up_table.columns[[3,4,5,6,7,8,25]],axis='columns')

        antennen_signale = torch.load(self.path+'\\orginale_antennen_signale_look_up_table_compare.npy')#'\\antennen_signale_imag_compre.npy')
        #antennen_signale = torch.load('C:\\Users\\dauserml\\Desktop\\dauserml_Messungen_2020_07_22\\cali_2020_09_01\\orginale_antennen_signale_look_up_table_compare.npy')
        antennen_signale =pd.DataFrame(antennen_signale,columns=["X-position"," Y-position", "Z-position","frame1","frame2","frame3","frame4","frame5","frame6","frame7","frame8","main1","main2","main3","main4","main5","main6","main7","main8"])
        timestamp = torch.load(self.path+'\\timestamp_look_up_table_compare.npy')

        antennen_signale = np.array(antennen_signale)
        postions_look_up_table = np.array(postions_look_up_table)

        antennen_signale_tm = []
        postions_look_up_table_tm = []
        timestamp_tm=[]

        i = 0
        while i < len(antennen_signale): # X = X for the timestamp
            if ((int(round(antennen_signale[0, 0]*1000, 3) * 1000)  == int(round(postions_look_up_table[0, 0] * 1000, 3) * 1000)) ):#and (int(round(antennen_signale[0, 1], 3) * 1000) == int(round(postions_look_up_table_tm[0, 1] * 1000, 3) * 1000)) and (int(round(antennen_signale[0, 2], 3) * 1000) == int(round(postions_look_up_table_tm[0, 2] * 1000, 3) * 1000))):
                antennen_signale_tm.append(antennen_signale[0, :])
                postions_look_up_table_tm.append(postions_look_up_table[0, :])
                postions_look_up_table = np.delete(postions_look_up_table,[0],0)
                antennen_signale = np.delete(antennen_signale, [0], 0)
                timestamp_tm.append(timestamp[0])
                timestamp = np.delete(timestamp,[0],0)
            else:
                antennen_signale = np.delete(antennen_signale,[0],0)

        antennen_signale = np.array((antennen_signale_tm))
        postions_look_up_table_tm = np.array(postions_look_up_table_tm)
        #postions_look_up_table = np.abs(postions_look_up_table_tm)
        postions_look_up_table = postions_look_up_table_tm

        timestamp_tm = np.array(timestamp_tm)

        print("MSE_GESAMT = ", mean_squared_error(antennen_signale[:,3:19],(postions_look_up_table[:,3:19])))
        mse = mean_squared_error(antennen_signale[:,3:19],(postions_look_up_table[:,3:19]), multioutput='raw_values')
        print("\nmse = \n",mse)
        error_x_y_z = abs(antennen_signale[:,[0,1,2]]-postions_look_up_table[:,[0,1,2]])
        error = antennen_signale[:,3:19]-(postions_look_up_table[:,3:19])
        error = np.array(error)
        print("\n Mittelwert\n",np.mean(error,0))
        plt.plot(timestamp_tm,error)
        plt.legend(["frame1","frame2","frame3","frame4","frame5","frame6","frame7","frame8","main1","main2","main3","main4","main5","main6","main7","main8"])
        print("\n xyz-error\n", np.mean(error_x_y_z,0))
        plt.figure(2)
        plt.title("x_y_z_error")
        plt.plot(timestamp_tm,error_x_y_z)



        plt.figure(3)
        plt.subplot(4,1,1)
        plt.plot(timestamp_tm,error[:,[0,8]])
        plt.legend(['Frame 1','Main 1'])
        plt.subplot(4,1,2)
        plt.plot(timestamp_tm,error[:,[1,9]])
        plt.legend(['Frame 2','Main 2'])
        plt.subplot(4,1,3)
        plt.plot(timestamp_tm,error[:,[2,10]])
        plt.legend(['Frame 3','Main 3'])
        plt.subplot(4,1,4)
        plt.plot(timestamp_tm,error[:,[3,11]])
        plt.legend(['Frame 4','Main 4'])

        plt.figure(4)
        plt.subplot(4,1,1)
        plt.plot(timestamp_tm,error[:,[4,12]])
        plt.legend(['Frame 5','Main 5'])
        plt.subplot(4,1,2)
        plt.plot(timestamp_tm,error[:,[5,13]])
        plt.legend(['Frame 6','Main 6'])
        plt.subplot(4,1,3)
        plt.plot(timestamp_tm,error[:,[6,14]])
        plt.legend(['Frame 7','Main 7'])
        plt.subplot(4,1,4)
        plt.plot(timestamp_tm,error[:,[7,15]])
        plt.legend(['Frame 8','Main 8'])


if __name__ == '__main__':
    lutwqMSE = look_up_table_with_qualisys_MSE(path=config_lutwqMSE.path,path_look_up_table=config_lutwqMSE.path_look_up_table)
    lutwqMSE.compare_look_up_table_with_real_Data()
    plt.show()