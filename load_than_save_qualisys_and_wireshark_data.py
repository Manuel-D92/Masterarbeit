import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import shutil
import qualisys as quali
import Signal_resampling as Signal_r
import wireshark as wire
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit


class load_than_save_qualisys_and_wireshark_data():
    @staticmethod
    def load_than_save_qualisys_and_wireshark_data_compare():
        ff_wireshark = np.load(
            'C:\\Users\\User\\Desktop\\dauserml_Messungen_2020_07_22\\cali_2020_09_01\\messung_cali_1.pcapng.cal.npy')
        timestamp_wireshark = np.transpose(np.load(
            'C:\\Users\\User\\Desktop\\dauserml_Messungen_2020_07_22\\cali_2020_09_01\\messung_cali_1.pcapng.timestamp.npy'))

        # timestamp_wireshark_abgezogen = np.nonzero(timestamp_wireshark_abgezogen)
        ff_wireshark = wire.wireshark.convert_Signal_wireshark(Signal_1=ff_wireshark)
        ff_wireshark = np.array(ff_wireshark)

        # wireshark = (np.exp(-1j*np.deg2rad(-27)) / 4.65)*ff_wireshark

        Signal_1, Kopf, first_timestamp_wireshark_qualisys = quali.qualisys.load_convert_qulisys_Data(
            'C:\\Users\\dauserml\\Desktop\\dauserml_Messungen_2020_07_22\\cali_2020_09_01\\messung_cali_1_6D.tsv',
            delet_Null=0);
        quali.qualisys.save_qualisys(path=('C:\\Users\\dauserml\\Desktop\\dauserml_Messungen_2020_07_22\\measurements7'),
                               Kopf=Kopf, qualisys_Daten=Signal_1)
        timestamp_qualisys = Signal_1[:, 1]
        print((timestamp_wireshark[0, 1] - first_timestamp_wireshark_qualisys))
        timestamp_wireshark[:, 0] = timestamp_wireshark[:, 0] + (
                    timestamp_wireshark[0, 1] - first_timestamp_wireshark_qualisys)
        timestamp_wireshark_int = timestamp_wireshark[:, 1]
        timestamp_wireshark = timestamp_wireshark[:, 0]

        antennen_signale = Signal_r.Signal_resampling.resampl( tm_timestamp=timestamp_wireshark,
                                                     timestamp_qualisys=timestamp_qualisys, tm_wireshark=ff_wireshark)
        antennen_signal_main_1_betrag = np.sqrt((antennen_signale[:, [0]] ** 2 + antennen_signale[:, [1]] ** 2))
        antennen_signal_frame_1_betrag = np.sqrt((antennen_signale[:, [2]] ** 2 + antennen_signale[:, [3]] ** 2))
        antennen_signal_main_2_betrag = np.sqrt((antennen_signale[:, [4]] ** 2 + antennen_signale[:, [5]] ** 2))
        antennen_signal_frame_2_betrag = np.sqrt((antennen_signale[:, [6]] ** 2 + antennen_signale[:, [7]] ** 2))
        antennen_signal_main_3_betrag = np.sqrt((antennen_signale[:, [8]] ** 2 + antennen_signale[:, [9]] ** 2))
        antennen_signal_frame_3_betrag = np.sqrt((antennen_signale[:, [10]] ** 2 + antennen_signale[:, [11]] ** 2))
        antennen_signal_main_4_betrag = np.sqrt((antennen_signale[:, [12]] ** 2 + antennen_signale[:, [13]] ** 2))
        antennen_signal_frame_4_betrag = np.sqrt((antennen_signale[:, [14]] ** 2 + antennen_signale[:, [15]] ** 2))
        antennen_signal_main_5_betrag = np.sqrt((antennen_signale[:, [16]] ** 2 + antennen_signale[:, [17]] ** 2))
        antennen_signal_frame_5_betrag = np.sqrt((antennen_signale[:, [18]] ** 2 + antennen_signale[:, [19]] ** 2))
        antennen_signal_main_6_betrag = np.sqrt((antennen_signale[:, [20]] ** 2 + antennen_signale[:, [21]] ** 2))
        antennen_signal_frame_6_betrag = np.sqrt((antennen_signale[:, [22]] ** 2 + antennen_signale[:, [23]] ** 2))
        antennen_signal_main_7_betrag = np.sqrt((antennen_signale[:, [24]] ** 2 + antennen_signale[:, [25]] ** 2))
        antennen_signal_frame_7_betrag = np.sqrt((antennen_signale[:, [26]] ** 2 + antennen_signale[:, [27]] ** 2))
        antennen_signal_main_8_betrag = np.sqrt((antennen_signale[:, [28]] ** 2 + antennen_signale[:, [29]] ** 2))
        antennen_signal_frame_8_betrag = np.sqrt((antennen_signale[:, [30]] ** 2 + antennen_signale[:, [31]] ** 2))

        antennen_signal_main_1_Komplex = (np.real(antennen_signale[:, 0]) + np.imag(antennen_signale[:, 1])) * (
                np.exp(-1j * np.deg2rad(-27)) / 4.65)
        antennen_signal_main_1_imag = antennen_signal_main_1_Komplex.imag
        antennen_signal_frame_1_Komplex = (np.real(antennen_signale[:, 2]) + np.imag(antennen_signale[:, 3])) * (
                np.exp(-1j * np.deg2rad(-27)) / 4.65)
        antennen_signal_frame_1_imag = antennen_signal_frame_1_Komplex.imag
        antennen_signal_main_2_Komplex = (np.real(antennen_signale[:, 4]) + np.imag(antennen_signale[:, 5])) * (
                np.exp(-1j * np.deg2rad(-27)) / 4.65)
        antennen_signal_main_2_imag = antennen_signal_main_2_Komplex.imag
        antennen_signal_frame_2_Komplex = (np.real(antennen_signale[:, 6]) + np.imag(antennen_signale[:, 7])) * (
                np.exp(-1j * np.deg2rad(-27)) / 4.65)
        antennen_signal_frame_2_imag = antennen_signal_frame_2_Komplex.imag
        antennen_signal_main_3_Komplex = (np.real(antennen_signale[:, 8]) + np.imag(antennen_signale[:, 9])) * (
                np.exp(-1j * np.deg2rad(-27)) / 4.65)
        antennen_signal_main_3_imag = antennen_signal_main_3_Komplex.imag
        antennen_signal_frame_3_Komplex = (np.real(antennen_signale[:, 10]) + np.imag(antennen_signale[:, 11])) * (
                np.exp(-1j * np.deg2rad(-27)) / 4.65)
        antennen_signal_frame_3_imag = antennen_signal_frame_3_Komplex.imag
        antennen_signal_main_4_Komplex = (np.real(antennen_signale[:, 12]) + np.imag(antennen_signale[:, 13])) * (
                np.exp(-1j * np.deg2rad(-27)) / 4.65)
        antennen_signal_main_4_imag = antennen_signal_main_4_Komplex.imag
        antennen_signal_frame_4_Komplex = (np.real(antennen_signale[:, 14]) + np.imag(antennen_signale[:, 15])) * (
                np.exp(-1j * np.deg2rad(-27)) / 4.65)
        antennen_signal_frame_4_imag = antennen_signal_frame_4_Komplex.imag
        antennen_signal_main_5_Komplex = (np.real(antennen_signale[:, 16]) + np.imag(antennen_signale[:, 17])) * (
                np.exp(-1j * np.deg2rad(-27)) / 4.65)
        antennen_signal_main_5_imag = antennen_signal_main_5_Komplex.imag
        antennen_signal_frame_5_Komplex = (np.real(antennen_signale[:, 18]) + np.imag(antennen_signale[:, 19])) * (
                np.exp(-1j * np.deg2rad(-27)) / 4.65)
        antennen_signal_frame_5_imag = antennen_signal_frame_5_Komplex.imag
        antennen_signal_main_6_Komplex = (np.real(antennen_signale[:, 20]) + np.imag(antennen_signale[:, 21])) * (
                np.exp(-1j * np.deg2rad(-27)) / 4.65)
        antennen_signal_main_6_imag = antennen_signal_main_6_Komplex.imag
        antennen_signal_frame_6_Komplex = (np.real(antennen_signale[:, 22]) + np.imag(antennen_signale[:, 23])) * (
                np.exp(-1j * np.deg2rad(-27)) / 4.65)
        antennen_signal_frame_6_imag = antennen_signal_frame_6_Komplex.imag
        antennen_signal_main_7_Komplex = (np.real(antennen_signale[:, 24]) + np.imag(antennen_signale[:, 25])) * (
                np.exp(-1j * np.deg2rad(-27)) / 4.65)
        antennen_signal_main_7_imag = antennen_signal_main_7_Komplex.imag
        antennen_signal_frame_7_Komplex = (np.real(antennen_signale[:, 26]) + np.imag(antennen_signale[:, 27])) * (
                np.exp(-1j * np.deg2rad(-27)) / 4.65)
        antennen_signal_frame_7_imag = antennen_signal_frame_7_Komplex.imag
        antennen_signal_main_8_Komplex = (np.real(antennen_signale[:, 28]) + np.imag(antennen_signale[:, 29])) * (
                np.exp(-1j * np.deg2rad(-27)) / 4.65)
        antennen_signal_main_8_imag = antennen_signal_main_8_Komplex.imag
        antennen_signal_frame_8_Komplex = (np.real(antennen_signale[:, 30]) + np.imag(antennen_signale[:, 31])) * (
                np.exp(-1j * np.deg2rad(-27)) / 4.65)
        antennen_signal_frame_8_imag = antennen_signal_frame_8_Komplex.imag

        # timestamp_wireshark,ff_wireshark = Signal_resampling.resampl_self_made(timestamp_wireshark_in=timestamp_wireshark[0,:],Signal_in=ff_wireshark,decimal_digits=2)
        # timestamp_wireshark = np.transpose(timestamp_wireshark)
        # timestamp_qualisys,Signal_1 = Signal_resampling.resampl_self_made(timestamp_wireshark_in=Signal_1[:,1],Signal_in=Signal_1,decimal_digits=2)

        plt.figure(1)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        plt.subplot(4, 2, 1)
        plt.plot(Signal_1[:, 1], (Signal_1[:, 20]))
        plt.xlabel("x vom wearable")
        plt.subplot(4, 2, 2)
        plt.plot(Signal_1[:, 1], Signal_1[:, 21])
        plt.xlabel("y vom wearable")
        plt.subplot(4, 2, 3)
        plt.plot(Signal_1[:, 1], Signal_1[:, 22])
        plt.xlabel("z vom wearable")

        plt.subplot(4, 2, 4)
        plt.plot(Signal_1[:, 1], Signal_1[:, 3])
        plt.xlabel("x vom shelf")
        plt.subplot(4, 2, 5)
        plt.plot(Signal_1[:, 1], Signal_1[:, 4])
        plt.xlabel("y vom shelf")
        plt.subplot(4, 2, 6)
        plt.plot(Signal_1[:, 1], Signal_1[:, 5])
        plt.xlabel("z vom shelf")

        # Signal = qualisys.x_y_z_translation_and_rotation(Signal_1,0)
        Signal_mean = quali.qualisys.x_y_z_translation_and_rotation(Signal_1, 1)
        antennen_signale_betrag = np.hstack((
                                            Signal_mean, antennen_signal_frame_1_betrag, antennen_signal_frame_2_betrag,
                                            antennen_signal_frame_3_betrag, antennen_signal_frame_4_betrag,
                                            antennen_signal_frame_5_betrag, antennen_signal_frame_6_betrag,
                                            antennen_signal_frame_7_betrag, antennen_signal_frame_8_betrag,
                                            antennen_signal_main_1_betrag, antennen_signal_main_2_betrag,
                                            antennen_signal_main_3_betrag, antennen_signal_main_4_betrag,
                                            antennen_signal_main_5_betrag, antennen_signal_main_6_betrag,
                                            antennen_signal_main_7_betrag, antennen_signal_main_8_betrag))
        antennen_signale_imag = np.hstack((Signal_mean, np.transpose(np.vstack((antennen_signal_frame_1_imag,
                                                                                antennen_signal_frame_2_imag,
                                                                                antennen_signal_frame_3_imag,
                                                                                antennen_signal_frame_4_imag,
                                                                                antennen_signal_frame_5_imag,
                                                                                antennen_signal_frame_6_imag,
                                                                                antennen_signal_frame_7_imag,
                                                                                antennen_signal_frame_8_imag,
                                                                                antennen_signal_main_1_imag,
                                                                                antennen_signal_main_2_imag,
                                                                                antennen_signal_main_3_imag,
                                                                                antennen_signal_main_4_imag,
                                                                                antennen_signal_main_5_imag,
                                                                                antennen_signal_main_6_imag,
                                                                                antennen_signal_main_7_imag,
                                                                                antennen_signal_main_8_imag)))))

        Sekunde = 12  # 14.27
        print("qualisys xyz=", Signal_mean[int(100 * Sekunde), :], " mm \n", "antennen\n",
              ff_wireshark[int(100 * Sekunde + 512), :])

        plt.figure(34)
        plt.subplot(8, 1, 1)
        plt.plot(timestamp_wireshark, ff_wireshark[:, [0, 1]])
        plt.subplot(8, 1, 2)
        plt.plot(timestamp_wireshark, ff_wireshark[:, [2, 3]])
        plt.subplot(8, 1, 3)
        plt.plot(timestamp_wireshark, ff_wireshark[:, [4, 5]])
        plt.subplot(8, 1, 4)
        plt.plot(timestamp_wireshark, ff_wireshark[:, [6, 7]])
        plt.subplot(8, 1, 5)
        plt.plot(timestamp_wireshark, ff_wireshark[:, [8, 9]])
        plt.subplot(8, 1, 6)
        plt.plot(timestamp_wireshark, ff_wireshark[:, [10, 11]])
        plt.subplot(8, 1, 7)
        plt.plot(timestamp_wireshark, ff_wireshark[:, [12, 13]])
        plt.subplot(8, 1, 8)
        plt.plot(timestamp_wireshark, ff_wireshark[:, [14, 15]])
        plt.figure(35)
        plt.subplot(8, 1, 1)
        plt.plot(timestamp_wireshark, ff_wireshark[:, [16, 17]])
        plt.subplot(8, 1, 2)
        plt.plot(timestamp_wireshark, ff_wireshark[:, [18, 19]])
        plt.subplot(8, 1, 3)
        plt.plot(timestamp_wireshark, ff_wireshark[:, [20, 21]])
        plt.subplot(8, 1, 4)
        plt.plot(timestamp_wireshark, ff_wireshark[:, [22, 23]])
        plt.subplot(8, 1, 5)
        plt.plot(timestamp_wireshark, ff_wireshark[:, [24, 25]])
        plt.subplot(8, 1, 6)
        plt.plot(timestamp_wireshark, ff_wireshark[:, [26, 27]])
        plt.subplot(8, 1, 7)
        plt.plot(timestamp_wireshark, ff_wireshark[:, [28, 29]])
        plt.subplot(8, 1, 8)
        plt.plot(timestamp_wireshark, ff_wireshark[:, [30, 31]])

        print("dgdg", np.sqrt((ff_wireshark[:, [0]] ** 2 + ff_wireshark[:, [1]] ** 2)))

        plt.figure(37)
        plt.subplot(8, 1, 1)
        plt.plot(timestamp_wireshark, np.sqrt((ff_wireshark[:, [0]] ** 2 + ff_wireshark[:, [1]] ** 2)))
        plt.subplot(8, 1, 2)
        plt.plot(timestamp_wireshark, np.sqrt((ff_wireshark[:, [2]] ** 2 + ff_wireshark[:, [3]] ** 2)))
        plt.subplot(8, 1, 3)
        plt.plot(timestamp_wireshark, np.sqrt((ff_wireshark[:, [4]] ** 2 + ff_wireshark[:, [5]] ** 2)))
        plt.subplot(8, 1, 4)
        plt.plot(timestamp_wireshark, np.sqrt((ff_wireshark[:, [6]] ** 2 + ff_wireshark[:, [7]] ** 2)))
        plt.subplot(8, 1, 5)
        plt.plot(timestamp_wireshark, np.sqrt((ff_wireshark[:, [8]] ** 2 + ff_wireshark[:, [9]] ** 2)))
        plt.subplot(8, 1, 6)
        plt.plot(timestamp_wireshark, np.sqrt((ff_wireshark[:, [10]] ** 2 + ff_wireshark[:, [11]] ** 2)))
        plt.subplot(8, 1, 7)
        plt.plot(timestamp_wireshark, np.sqrt((ff_wireshark[:, [12]] ** 2 + ff_wireshark[:, [13]] ** 2)))
        plt.subplot(8, 1, 8)
        plt.plot(timestamp_wireshark, np.sqrt((ff_wireshark[:, [14]] ** 2 + ff_wireshark[:, [15]] ** 2)))

        plt.figure(38)
        plt.subplot(8, 1, 1)
        plt.plot(timestamp_wireshark, np.sqrt((ff_wireshark[:, [16]] ** 2 + ff_wireshark[:, [17]] ** 2)))
        plt.subplot(8, 1, 2)
        plt.plot(timestamp_wireshark, np.sqrt((ff_wireshark[:, [18]] ** 2 + ff_wireshark[:, [19]] ** 2)))
        plt.subplot(8, 1, 3)
        plt.plot(timestamp_wireshark, np.sqrt((ff_wireshark[:, [20]] ** 2 + ff_wireshark[:, [21]] ** 2)))
        plt.subplot(8, 1, 4)
        plt.plot(timestamp_wireshark, np.sqrt((ff_wireshark[:, [22]] ** 2 + ff_wireshark[:, [23]] ** 2)))
        plt.subplot(8, 1, 5)
        plt.plot(timestamp_wireshark, np.sqrt((ff_wireshark[:, [24]] ** 2 + ff_wireshark[:, [25]] ** 2)))
        plt.subplot(8, 1, 6)
        plt.plot(timestamp_wireshark, np.sqrt((ff_wireshark[:, [26]] ** 2 + ff_wireshark[:, [27]] ** 2)))
        plt.subplot(8, 1, 7)
        plt.plot(timestamp_wireshark, np.sqrt((ff_wireshark[:, [28]] ** 2 + ff_wireshark[:, [29]] ** 2)))
        plt.subplot(8, 1, 8)
        plt.plot(timestamp_wireshark, np.sqrt((ff_wireshark[:, [30]] ** 2 + ff_wireshark[:, [31]] ** 2)))

        plt.figure(70)
        plt.plot(timestamp_qualisys, antennen_signale[:, [0, 1]])

        plt.figure(3)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        plt.subplot(2, 1, 1)
        plt.plot(Signal_1[:, 1], Signal_mean[:, 0])
        plt.subplot(2, 1, 2)
        plt.plot(timestamp_wireshark, ff_wireshark[:, [0, 1]])
        plt.xlabel("x")
        plt.figure(4)
        plt.subplot(2, 1, 1)
        plt.plot(Signal_1[:, 1], Signal_mean[:, 1])
        plt.subplot(2, 1, 2)
        plt.plot(timestamp_wireshark, ff_wireshark[:, [0, 1]])
        plt.xlabel("y")
        plt.figure(5)
        plt.subplot(2, 1, 1)
        plt.plot(Signal_1[:, 1], Signal_mean[:, 2])
        plt.subplot(2, 1, 2)
        plt.plot(timestamp_wireshark, ff_wireshark[:, [0, 1]])
        plt.xlabel("z")

        postions_look_up_table = []
        tm_antennen_signale = []
        tm_antennen_signale_betrag = []
        tm_antennen_signale_imag = []
        tm_timestamp = []
        for i in range(0, len(Signal_mean)):  # delet NAN rows and timestamp higher 0
            if (np.isnan(Signal_mean[i, 0]) == False and round(Signal_mean[i, 0]) >= 0):
                postions_look_up_table.append(Signal_mean[i, :])
                tm_antennen_signale.append(antennen_signale[i, :])
                tm_antennen_signale_betrag.append(antennen_signale_betrag[i, :])
                tm_antennen_signale_imag.append(antennen_signale_imag[i, :])
                tm_timestamp.append(Signal_1[i, 1])
        postions_look_up_table = np.array(postions_look_up_table) / 1000
        tm_antennen_signale_betrag = np.array(tm_antennen_signale_betrag)
        tm_antennen_signale_imag = np.array(tm_antennen_signale_imag)
        timestamp_look_up_table_compare = np.array(tm_timestamp)
        postions_look_up_table_compare = np.round(postions_look_up_table, 5)
        antennen_signale_betrag_look_up_table_compare = np.hstack((np.round(
            tm_antennen_signale_betrag[:, [0, 1, 2]] / 1000, 5), tm_antennen_signale_betrag[:,
                                                                 [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                                                                  18]]))
        antennen_signale_imag_compre = np.hstack((np.round(tm_antennen_signale_imag[:, [0, 1, 2]] / 1000, 5),
                                                  tm_antennen_signale_imag[:,
                                                  [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]))

        torch.save(postions_look_up_table_compare,
                   'C:\\Users\\dauserml\\Desktop\\dauserml_Messungen_2020_07_22\\cali_2020_09_01\\x_y_z_position_look_up_table_compare.npy')
        torch.save(antennen_signale_betrag_look_up_table_compare,
                   'C:\\Users\\dauserml\\Desktop\\dauserml_Messungen_2020_07_22\\cali_2020_09_01\\orginale_antennen_signale_look_up_table_compare.npy')
        torch.save(timestamp_look_up_table_compare,
                   'C:\\Users\\dauserml\\Desktop\\dauserml_Messungen_2020_07_22\\cali_2020_09_01\\timestamp_look_up_table_compare.npy')
        torch.save(antennen_signale_imag_compre,
                   'C:\\Users\\dauserml\\Desktop\\dauserml_Messungen_2020_07_22\\cali_2020_09_01\\antennen_signale_imag_compre.npy')
        return 0

    def load_than_save_qualisys_and_wireshark_data_to_learning(self,path, File, which_klass, number_Signal):
        print('File:' ,File)
        ff_wireshark = np.load(
            '%s\\' % path + 'Used_Data\\%s.pcapng.cal.npy' % File)
        # ff_wireshark = np.delete(ff_wireshark,-1,axis=0)
        timestamp_wireshark = np.transpose(np.load(
            '%s\\' % path + 'Used_Data\\%s.pcapng.timestamp_wireshark.npy' % File))

        # timestamp_wireshark_abgezogen = np.nonzero(timestamp_wireshark_abgezogen)
        ff_wireshark = wire.wireshark.convert_Signal_wireshark(Signal_1=ff_wireshark)
        ff_wireshark = np.array(ff_wireshark)

        Signal_1, Kopf, first_timestamp_wireshark_qualisys = quali.qualisys.load_convert_qulisys_Data(
            '%s\\' % path + 'Used_Data\\%s_6D.tsv' % File,
            delet_Null=0);

        timestamp_qualisys = Signal_1[:, 1]
        print("difference between wireshark and qualisys timestamp: ",
              (timestamp_wireshark[0, 1] - first_timestamp_wireshark_qualisys))
        timestamp_wireshark[:, 0] = timestamp_wireshark[:, 0] + (
                timestamp_wireshark[0, 1] - first_timestamp_wireshark_qualisys)
        timestamp_wireshark_int = timestamp_wireshark[:, 1]
        timestamp_wireshark = timestamp_wireshark[:, 0]

        Signal_mean = quali.qualisys.x_y_z_translation_and_rotation(Signal_1, 1)
        Signal_yaw_pitch_roll_residual_wearable = Signal_1[:,23:27]-Signal_1[:,6:10]

        antennen_signale = Signal_r.Signal_resampling.resampl(tm_timestamp=timestamp_wireshark,
                                                     timestamp_qualisys=timestamp_qualisys, tm_wireshark=ff_wireshark)


        ## self Resampling Funktion
        #timestamp_wireshark,antennen_signale = Signal_r.Signal_resampling.resampl_self_made(timestamp_wireshark_in=timestamp_wireshark,Signal_in=ff_wireshark,decimal_digits=2)
        #timestamp_qualisys, Signal_mean = Signal_r.Signal_resampling.resampl_self_made(timestamp_wireshark_in=timestamp_qualisys, Signal_in=Signal_mean, decimal_digits=2)

        #tm_wireshark,tm_Signal_mean,tm_timestampwireshark,tm_timestamp_qualisys=[],[],[],[]
        #for i in range(0,len(timestamp_wireshark)):
        #    for j in range(0,len(timestamp_qualisys)):
        #        if(int(timestamp_wireshark[i]*(10**2)) == int(timestamp_qualisys[j]*(10**2))):
        #            tm_wireshark.append(antennen_signale[i,:])
        #            tm_timestampwireshark.append(timestamp_wireshark[i,:])
        #            tm_Signal_mean.append(Signal_mean[j,:])
        #            tm_timestamp_qualisys.append(timestamp_qualisys[j,:])
        #timestamp_wireshark=np.array(tm_wireshark)
        #antennen_signale=np.array(tm_timestampwireshark)
        #Signal_mean=np.array(tm_Signal_mean)
        #timestamp_qualisys=np.array(tm_timestamp_qualisys)

        #if(int(timestamp_wireshark[0]*(10**2))-int(timestamp_qualisys[0]*(10**2))<0):
        #    for i in range(0,int(timestamp_qualisys[0]*(10**2))-int(timestamp_wireshark[0]*(10**2))):
        #        timestamp_wireshark=np.delete(timestamp_wireshark,0,axis=0)
        #        antennen_signale=np.delete(antennen_signale,0,axis=0)
        #else:
        #    for i in range(0,int(timestamp_wireshark[0]*(10**2))-int(timestamp_qualisys[0]*(10**2))):
        #        timestamp_qualisys=np.delete(timestamp_qualisys,0,axis=0)
        #        Signal_mean=np.delete(Signal_mean,0,axis=0)
#
        #te = int(timestamp_wireshark[-1] * (10 ** 2)) - int(timestamp_qualisys[-1] * (10 ** 2))
#
        #if (int(timestamp_wireshark[-1] * (10 ** 2)) - int(timestamp_qualisys[-1] * (10 ** 2)) > 0):
        #    for i in range(0, int(timestamp_wireshark[-1] * (10 ** 2)) -int(timestamp_qualisys[-1] * (10 ** 2))):
        #        timestamp_wireshark=np.delete(timestamp_wireshark, -1, axis=0)
        #        antennen_signale=np.delete(antennen_signale, -1, axis=0)
        #else:
        #    for i in range(0, int(timestamp_qualisys[-1] * (10 ** 2)) -int(timestamp_wireshark[-1] * (10 ** 2))):
        #        timestamp_qualisys=np.delete(timestamp_qualisys, -1, axis=0)
        #        Signal_mean=np.delete(Signal_mean, -1, axis=0)
#

        komplex_orentation_faktor = self.komplex_orentation_faktor

        antennen_signal_main_1_Komplex,antennen_signal_frame_1_Komplex,antennen_signal_main_2_Komplex,antennen_signal_frame_2_Komplex,antennen_signal_main_3_Komplex,antennen_signal_frame_3_Komplex,antennen_signal_main_4_Komplex,antennen_signal_frame_4_Komplex,antennen_signal_main_5_Komplex,antennen_signal_frame_5_Komplex,antennen_signal_main_6_Komplex,antennen_signal_frame_6_Komplex,antennen_signal_main_7_Komplex,antennen_signal_frame_7_Komplex,antennen_signal_main_8_Komplex,antennen_signal_frame_8_Komplex  =[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]

        for i in range(0,len(antennen_signale)):
            antennen_signal_main_1_Komplex.append(np.complex((antennen_signale[i, 0]) ,(
                antennen_signale[i, 1])) * komplex_orentation_faktor)
            antennen_signal_frame_1_Komplex.append(np.complex((antennen_signale[i, 2]), (
                antennen_signale[i, 3])) * komplex_orentation_faktor)
            antennen_signal_main_2_Komplex.append(np.complex((antennen_signale[i, 4]), (
                antennen_signale[i, 5])) * komplex_orentation_faktor)
            antennen_signal_frame_2_Komplex.append(np.complex((antennen_signale[i, 6]), (
                antennen_signale[i, 7])) * komplex_orentation_faktor)
            antennen_signal_main_3_Komplex.append(np.complex((antennen_signale[i, 8]), (
                antennen_signale[i, 9])) * komplex_orentation_faktor)
            antennen_signal_frame_3_Komplex.append(np.complex((antennen_signale[i, 10]), (
                antennen_signale[i, 11])) * komplex_orentation_faktor)
            antennen_signal_main_4_Komplex.append(np.complex((antennen_signale[i, 12]), (
                antennen_signale[i, 13])) * komplex_orentation_faktor)
            antennen_signal_frame_4_Komplex.append(np.complex((antennen_signale[i, 14]), (
                antennen_signale[i, 15])) * komplex_orentation_faktor)
            antennen_signal_main_5_Komplex.append(np.complex((antennen_signale[i, 16]), (
                antennen_signale[i, 17])) * komplex_orentation_faktor)
            antennen_signal_frame_5_Komplex.append(np.complex((antennen_signale[i, 18]), (
                antennen_signale[i, 19])) * komplex_orentation_faktor)
            antennen_signal_main_6_Komplex.append(np.complex((antennen_signale[i, 20]), (
                antennen_signale[i, 21])) * komplex_orentation_faktor)
            antennen_signal_frame_6_Komplex.append(np.complex((antennen_signale[i, 22]), (
                antennen_signale[i, 23])) * komplex_orentation_faktor)
            antennen_signal_main_7_Komplex.append(np.complex((antennen_signale[i, 24]), (
                antennen_signale[i, 25])) * komplex_orentation_faktor)
            antennen_signal_frame_7_Komplex.append(np.complex((antennen_signale[i, 26]), (
                antennen_signale[i, 27])) * komplex_orentation_faktor)
            antennen_signal_main_8_Komplex.append(np.complex((antennen_signale[i, 28]), (
                antennen_signale[i, 29])) * komplex_orentation_faktor)
            antennen_signal_frame_8_Komplex.append(np.complex((antennen_signale[i, 30]), (
                antennen_signale[i, 31])) * komplex_orentation_faktor)

        antennen_signal_main_1_Komplex= np.array(antennen_signal_main_1_Komplex)
        antennen_signal_frame_1_Komplex= np.array(antennen_signal_frame_1_Komplex)
        antennen_signal_main_2_Komplex= np.array(antennen_signal_main_2_Komplex)
        antennen_signal_frame_2_Komplex= np.array(antennen_signal_frame_2_Komplex)
        antennen_signal_main_3_Komplex= np.array(antennen_signal_main_3_Komplex)
        antennen_signal_frame_3_Komplex= np.array(antennen_signal_frame_3_Komplex)
        antennen_signal_main_4_Komplex= np.array(antennen_signal_main_4_Komplex)
        antennen_signal_frame_4_Komplex= np.array(antennen_signal_frame_4_Komplex)
        antennen_signal_main_5_Komplex = np.array(antennen_signal_main_5_Komplex)
        antennen_signal_frame_5_Komplex= np.array(antennen_signal_frame_5_Komplex)
        antennen_signal_main_6_Komplex= np.array(antennen_signal_main_6_Komplex)
        antennen_signal_frame_6_Komplex= np.array(antennen_signal_frame_6_Komplex)
        antennen_signal_main_7_Komplex= np.array(antennen_signal_main_7_Komplex)
        antennen_signal_frame_7_Komplex= np.array(antennen_signal_frame_7_Komplex)
        antennen_signal_main_8_Komplex= np.array(antennen_signal_main_8_Komplex)
        antennen_signal_frame_8_Komplex= np.array(antennen_signal_frame_8_Komplex)

        test = np.sqrt((antennen_signal_frame_1_Komplex.real**2)+(antennen_signal_frame_1_Komplex.imag**2))
        print(test)
        antennen_signale_Betrag = np.abs(antennen_signal_frame_1_Komplex)
        print(antennen_signale_Betrag)

        antennen_signale_Betrag = (np.transpose(np.vstack((np.abs(antennen_signal_frame_1_Komplex),np.abs(antennen_signal_frame_2_Komplex),
                                                           np.abs(antennen_signal_frame_3_Komplex),np.abs(antennen_signal_frame_4_Komplex),
                                                           np.abs(antennen_signal_frame_5_Komplex),np.abs(antennen_signal_frame_6_Komplex),
                                                           np.abs(antennen_signal_frame_7_Komplex),np.abs(antennen_signal_frame_8_Komplex),
                                                           np.abs(antennen_signal_main_1_Komplex),np.abs(antennen_signal_main_2_Komplex),
                                                           np.abs(antennen_signal_main_3_Komplex),np.abs(antennen_signal_main_4_Komplex),
                                                           np.abs(antennen_signal_main_5_Komplex),np.abs(antennen_signal_main_6_Komplex),
                                                           np.abs(antennen_signal_main_7_Komplex),np.abs(antennen_signal_main_8_Komplex)))))

        antennen_signal_main_1_real = antennen_signal_main_1_Komplex.real
        antennen_signal_main_1_imag = antennen_signal_main_1_Komplex.imag
        antennen_signal_frame_1_real = antennen_signal_frame_1_Komplex.real
        antennen_signal_frame_1_imag = antennen_signal_frame_1_Komplex.imag
        antennen_signal_main_2_real = antennen_signal_main_2_Komplex.real
        antennen_signal_main_2_imag = antennen_signal_main_2_Komplex.imag
        antennen_signal_frame_2_real = antennen_signal_frame_2_Komplex.real
        antennen_signal_frame_2_imag = antennen_signal_frame_2_Komplex.imag
        antennen_signal_main_3_real = antennen_signal_main_3_Komplex.real
        antennen_signal_main_3_imag = antennen_signal_main_3_Komplex.imag
        antennen_signal_frame_3_real = antennen_signal_frame_3_Komplex.real
        antennen_signal_frame_3_imag = antennen_signal_frame_3_Komplex.imag
        antennen_signal_main_4_real = antennen_signal_main_4_Komplex.real
        antennen_signal_main_4_imag = antennen_signal_main_4_Komplex.imag
        antennen_signal_frame_4_real = antennen_signal_frame_4_Komplex.real
        antennen_signal_frame_4_imag = antennen_signal_frame_4_Komplex.imag
        antennen_signal_main_5_real = antennen_signal_main_5_Komplex.real
        antennen_signal_main_5_imag = antennen_signal_main_5_Komplex.imag
        antennen_signal_frame_5_real = antennen_signal_frame_5_Komplex.real
        antennen_signal_frame_5_imag = antennen_signal_frame_5_Komplex.imag
        antennen_signal_main_6_real = antennen_signal_main_6_Komplex.real
        antennen_signal_main_6_imag = antennen_signal_main_6_Komplex.imag
        antennen_signal_frame_6_real = antennen_signal_frame_6_Komplex.real
        antennen_signal_frame_6_imag = antennen_signal_frame_6_Komplex.imag
        antennen_signal_main_7_real = antennen_signal_main_7_Komplex.real
        antennen_signal_main_7_imag = antennen_signal_main_7_Komplex.imag
        antennen_signal_frame_7_real = antennen_signal_frame_7_Komplex.real
        antennen_signal_frame_7_imag = antennen_signal_frame_7_Komplex.imag
        antennen_signal_main_8_real = antennen_signal_main_8_Komplex.real
        antennen_signal_main_8_imag = antennen_signal_main_8_Komplex.imag
        antennen_signal_frame_8_real = antennen_signal_frame_8_Komplex.real
        antennen_signal_frame_8_imag = antennen_signal_frame_8_Komplex.imag


        antennen_signale_Komplex = (np.transpose(np.vstack((antennen_signal_frame_1_real, antennen_signal_frame_1_imag,
                                                            antennen_signal_frame_2_real, antennen_signal_frame_2_imag,
                                                            antennen_signal_frame_3_real, antennen_signal_frame_3_imag,
                                                            antennen_signal_frame_4_real, antennen_signal_frame_4_imag,
                                                            antennen_signal_frame_5_real, antennen_signal_frame_5_imag,
                                                            antennen_signal_frame_6_real, antennen_signal_frame_6_imag,
                                                            antennen_signal_frame_7_real, antennen_signal_frame_7_imag,
                                                            antennen_signal_frame_8_real, antennen_signal_frame_8_imag,
                                                            antennen_signal_main_1_real, antennen_signal_main_1_imag,
                                                            antennen_signal_main_2_real, antennen_signal_main_2_imag,
                                                            antennen_signal_main_3_real, antennen_signal_main_3_imag,
                                                            antennen_signal_main_4_real, antennen_signal_main_4_imag,
                                                            antennen_signal_main_5_real, antennen_signal_main_5_imag,
                                                            antennen_signal_main_6_real, antennen_signal_main_6_imag,
                                                            antennen_signal_main_7_real, antennen_signal_main_7_imag,
                                                            antennen_signal_main_8_real, antennen_signal_main_8_imag))))

        number_Signal_array = np.full((1,len(Signal_mean)),number_Signal)
        xyz_antennen_Signal_Komplex = np.hstack((Signal_mean,Signal_yaw_pitch_roll_residual_wearable, antennen_signale_Komplex))
        time_xyz_antennen_Signal_Komplex = np.hstack(
            (np.transpose(number_Signal_array),np.transpose(np.array(timestamp_qualisys, ndmin=2)), xyz_antennen_Signal_Komplex))
        #time_xyz_antennen_Signal_Komplex = np.hstack(
        #    ((timestamp_qualisys), xyz_antennen_Signal_Komplex))


        tm_time_xyz_antennen_Signal_Komplex = []
        for i in range(0, len(Signal_mean)):  # delet NAN rows and timestamp higher 0
            if (np.isnan(Signal_mean[i, 0]) == False):
                tm_time_xyz_antennen_Signal_Komplex.append(time_xyz_antennen_Signal_Komplex[i, :])
        time_xyz_antennen_Signal_Komplex = np.array(tm_time_xyz_antennen_Signal_Komplex)

        #append which klass
        klass_array = np.full((1,len(time_xyz_antennen_Signal_Komplex)),which_klass)
        time_xyz_antennen_Signal_Komplex_klass = np.hstack((time_xyz_antennen_Signal_Komplex,np.transpose(klass_array)))

        print('The File have '+str(len(tm_time_xyz_antennen_Signal_Komplex))+' Data Points')
        time_xyz_antennen_Signal_Komplex_klass_cut = load_than_save_qualisys_and_wireshark_data.cut_x_y_z(self=self,
            time_xyz_antennen_Signal=time_xyz_antennen_Signal_Komplex_klass)

        #plt.figure(69)
        #plt.subplot(4, 1, 1)
        #plt.plot(time_xyz_antennen_Signal_Komplex[:,0], time_xyz_antennen_Signal_Komplex[:,1:4])
        #plt.subplot(4, 1, 2)
        #plt.plot(time_xyz_antennen_Signal_Komplex[:,0], time_xyz_antennen_Signal_Komplex[:, 4:6])
        #plt.subplot(4, 1, 3)
        #plt.plot(time_xyz_antennen_Signal_Komplex[:,0], time_xyz_antennen_Signal_Komplex[:, 20:22])

        return time_xyz_antennen_Signal_Komplex_klass_cut

    @staticmethod
    def load_File_List_from_folder(path,name_file):
        if (False == os.path.isdir(path + "\\Unused_Data")):
            os.mkdir(os.path.join(path, "Unused_Data"))
        if (False == os.path.isdir(path + "\\Used_Data")):
            os.mkdir(os.path.join(path, "Used_Data"))
        File_list = []
        File_list_cal = []
        File_list_timestamp = []
        File_list_qualisys = []
        for folder in os.listdir(path):
            if (folder.endswith(".pcapng.cal.npy")):
                File_list_cal.append(folder.replace(".pcapng.cal.npy", ""))
                shutil.move((path + "\\" + folder), (path + "\\Used_Data\\" + folder))
            elif (folder.endswith(".pcapng.timestamp_wireshark.npy")):
                File_list_timestamp.append(folder.replace(".pcapng.timestamp_wireshark.npy", ""))
                shutil.move((path + "\\" + folder), (path + "\\Used_Data\\" + folder))
            elif (folder.endswith("_6D.tsv")):
                File_list_qualisys.append(folder.replace("_6D.tsv", ""))
                shutil.move((path + "\\" + folder), (path + "\\Used_Data\\" + folder))
            elif (folder == "Unused_Data" or folder == "Used_Data" or folder == name_file+".csv"):
                print("Unused/Used Folder and all Files exist")
            else:
                shutil.move((path + "\\" + folder), (path + "\\Unused_Data\\" + folder))

        for i in range(0, len(File_list_cal)):
            for k in range(0, len(File_list_timestamp)):
                for m in range(0, len(File_list_qualisys)):
                    if (File_list_cal[i] == File_list_timestamp[k] and File_list_cal[i] == File_list_qualisys[m]):
                        File_list.append(File_list_cal[i])
        return File_list

    @staticmethod
    def get_which_subject_intervened(File_list): #for Classifikation
        File_list_klass=[]
        for i in range(0,len(File_list)):
            print('-1 = Different interventions (Testsequnzen)\n-2 = Outside past the shelf\n0 = No interventions')
            print('Which Subject intervened?\n' +File_list[i]+':')
            input_klass = input()
            File_list_klass.append(int(input_klass))
        return np.array(File_list_klass)

    def cut_x_y_z(self,time_xyz_antennen_Signal):
        tm_time_xyz_antennen_Signal = []
        for i in range(0, len(time_xyz_antennen_Signal)):
            if (self.x_cut[0] < int(round(time_xyz_antennen_Signal[i, 2])) < self.x_cut[1] and self.y_cut[0]  < int(
                    round(time_xyz_antennen_Signal[i, 3])) < self.y_cut[1] and self.z_cut[0] < int(
                    round(time_xyz_antennen_Signal[i, 4])) < self.z_cut[1]):
                tm_time_xyz_antennen_Signal.append(time_xyz_antennen_Signal[i, :])
        return np.array(tm_time_xyz_antennen_Signal)