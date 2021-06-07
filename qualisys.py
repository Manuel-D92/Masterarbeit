import numpy as np
import pandas as pd
import torch
import datetime

class qualisys():
    @staticmethod
    def load_convert_qulisys_Data(path,delet_Null):
        #ea = pd.read_csv(path,sep='\t',skiprows=10)
        Signal_1 = pd.read_table(path,sep='\n');
        Signal = []
        zwischen_speicher_ar = []
        #Kopf = []
        #for i in range(10,len(Signal_1)):
        #    zwischen_speicher = Signal_1.values[i]
        #    zwischen_speicher = str(zwischen_speicher)
        #    zwischen_speicher_ar = re.findall(r'[+-]?\d+[,.]?\d', zwischen_speicher)
        #
        #    vere = []
        #    if not zwischen_speicher_ar:
        #        zwischen_speicher = str(zwischen_speicher)
        #        zwischen_speicher = zwischen_speicher.replace("\\t", ";")
        #        zwischen_speicher = zwischen_speicher.replace("", "")
        #        zwischen_speicher = zwischen_speicher.replace("'", "")
        #        zwischen_speicher = zwischen_speicher.replace("[", "")
        #        zwischen_speicher = zwischen_speicher.replace("]", "")
        #        zwischen_speicher = zwischen_speicher.split(";")
        #        Kopf.append(zwischen_speicher)
        #
        #    for i in range(0,len(zwischen_speicher_ar)):
        #
        #        if(zwischen_speicher_ar[i]==''):
        #            zwischen_speicher_ar[i] = 0
        #        ver = np.array(zwischen_speicher_ar)
        #        vere.append(float(zwischen_speicher_ar[i]))
        #    #if(a==-1):
        #    if zwischen_speicher_ar:
        #        tt.append(vere)

        first_timestamp_qualisys = Signal_1.values[6]
        first_timestamp_qualisys = str(first_timestamp_qualisys)
        first_timestamp_qualisys = first_timestamp_qualisys.replace("\\t", ";")
        first_timestamp_qualisys = first_timestamp_qualisys.replace("", "")
        first_timestamp_qualisys = first_timestamp_qualisys.replace("'", "")
        first_timestamp_qualisys = first_timestamp_qualisys.replace("[", "")
        first_timestamp_qualisys = first_timestamp_qualisys.replace("]", "")
        first_timestamp_qualisys = first_timestamp_qualisys.split(";")

        first_timestamp_qualisys=first_timestamp_qualisys[1]
        first_timestamp_qualisys = (datetime.datetime.strptime(first_timestamp_qualisys,'%Y-%m-%d, %H:%M:%S.%f')).timestamp()
        print("first_timestamp from Qualisys system: ",first_timestamp_qualisys)

        Kopf = Signal_1.values[10] #10 bei 6D
        Kopf = str(Kopf)
        Kopf = Kopf.replace("\\t",";")
        Kopf = Kopf.replace("", "")
        Kopf = Kopf.replace("'","")
        Kopf = Kopf.replace("[","")
        Kopf = Kopf.replace("]","")
        Kopf = Kopf.split(";")

        for i in range(11,len(Signal_1)-1): #11 bei 6D   #-1 -> always delete the last line because the last line of the Qualisys is not always complete
            zwischen_speicher = Signal_1.values[i]
            zwischen_speicher = str(zwischen_speicher)
            zwischen_speicher = zwischen_speicher.replace("\\t",";")
            zwischen_speicher = zwischen_speicher.replace("", "")
            zwischen_speicher = zwischen_speicher.replace("'","")
            zwischen_speicher = zwischen_speicher.replace("[","")
            zwischen_speicher = zwischen_speicher.replace("]","")
            zwischen_speicher_ar = zwischen_speicher.split(";")
            Signal_zw = []
            for i in range(0,len(zwischen_speicher_ar)):
                if(zwischen_speicher_ar[i]==''):
                    zwischen_speicher_ar[i] = 0
                Signal_zw.append(float(zwischen_speicher_ar[i]))
            Signal.append(Signal_zw)
        Signal_1 = np.array(Signal)


        if(delet_Null==1):
            count=[]
            for i in range(0,len(Signal_1)):
                if((np.count_nonzero(Signal_1[i,30]==0))>=1): #122 bei 6D,
                    count.append(i)

            Signal_1 = np.delete(Signal_1,count,0)
        return Signal_1, Kopf,first_timestamp_qualisys

    @staticmethod
    def x_y_z_translation_and_rotation(Signal,mean):

        re_Signal=[]


        RM_1 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])  # Rotation um die X-Achse um 270 Grad
        RM_2 = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])  # Rotation um die Y-Achse um 270 Grad
        # RM_3 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]) #Ratation um die Z-Achse um 90 Grad
        diff_wireshark_1 = ((Signal[:, 20]) - np.nanmean(Signal[:, 3]))
        diff_wireshark_2 = ((Signal[:, 21]) - np.nanmean(Signal[:, 4]))
        diff_wireshark_3 = ((Signal[:, 22]) - np.nanmean(Signal[:, 5]))
        x_y_z_Wert = ([diff_wireshark_1, diff_wireshark_2, diff_wireshark_3])
        x_y_z_Wert = np.array(x_y_z_Wert)




        if(mean==1): # RotataionsMatrix mittelwert gebildet
            R_11 = (np.nanmean(Signal[:, 10]));R_12 = (np.nanmean(Signal[:, 11]));R_13 = (np.nanmean(Signal[:, 12]))
            R_21 = (np.nanmean(Signal[:, 13]));R_22 = (np.nanmean(Signal[:, 14]));R_23 = (np.nanmean(Signal[:, 15]))
            R_31 = (np.nanmean(Signal[:, 16]));R_32 = (np.nanmean(Signal[:, 17]));R_33 = (np.nanmean(Signal[:, 18]))
            #Rotationsmatrix RM
            RM = np.array(([[R_11,R_12,R_13],[R_21,R_22,R_23],[R_31,R_32,R_33]]))

            #print("transpose(RM) @ RM:",np.transpose(RM) @ RM)

            re_Signal = x_y_z_Wert
            #re_Signal = RM_1 @ re_Signal
            #re_Signal = RM_2 @ re_Signal
            re_Signal = RM @ re_Signal

            re_Signal =np.transpose(re_Signal)

        if(mean==0): # RotationsMatrix alle werte einzeln berechnet
            for i in range(0,len(Signal)):
                R_11 = (Signal[i, 10]);R_12 = (Signal[i, 11]);R_13 = (Signal[i, 12])
                R_21 = (Signal[i, 13]);R_22 = (Signal[i, 14]);R_23 = (Signal[i, 15])
                R_31 = (Signal[i, 16]);R_32 = (Signal[i, 17]);R_33 = (Signal[i, 18])

                re_Signal = x_y_z_Wert
                re_Signal = RM_1 @ re_Signal
                re_Signal = RM_2 @ re_Signal
            re_Signal = np.array(re_Signal)
        return re_Signal

    def save_qualisys(path,Kopf,qualisys_Daten):
        torch.save([Kopf,qualisys_Daten],path)