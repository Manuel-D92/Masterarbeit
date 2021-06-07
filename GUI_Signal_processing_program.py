from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.uic import *
#import pyqtgraph as pg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import wireshark_extract
import look_up_table_model as lutm
import load_qualisys_daten as lqd
import load_than_save_qualisys_and_wireshark_data as ltsqawd
import GUI_start_fingerprinting_table
import wireshark as wire
import qualisys as quali
from scipy.signal import medfilt2d
import GUI_config_fingerprinting_table as GUI_cft

class Gui_Signal_processing_programm(QAbstractItemModel):
    def __init__(self,parent=None):
        super(Gui_Signal_processing_programm,self).__init__(parent)
        self.path= r".\Test"
        ##Load UI Files
        self.w = loadUi(self.path+"\\mainwindow.ui")
        # d = loadUi("C:\\Users\\User\\Documents\\Test\\dialog.ui")
        self.look_up_table_ui = loadUi(self.path+"\\load_from_look_up_table.ui")
        self.wireshark_extract_ui = loadUi(self.path+"\\wireshark_extract.ui")
        self.merge_qualisys_and_wireshark_ui = loadUi(self.path+"\\merge_qualisys_and_wireshark.ui")
        self.generate_data_for_look_up_table_mse_ui = loadUi(self.path+"\\generate_data_for_look_up_table_mse.ui")
        self.wireshark_window_ui = loadUi(self.path+"\\wireshark_window.ui")
        self.qualisys_window_ui = loadUi(self.path+"\\qualisys_window.ui")
        self.Signal=[]
        self.wireshark_window_data=[]
        self.wireshark_window_head=[]
        self.qualisys_window_data=[]
        self.qualisys_window_head=[]

    def gui_signal_processing_main(self):
        look_up_table_size = [self.look_up_table_ui.tableView.maximumWidth(),self.look_up_table_ui.tableView.maximumHeight()]
        self.look_up_table_ui.tableView = QTableWidget(self.look_up_table_ui.tableView)
        self.look_up_table_ui.tableView.setMinimumSize(look_up_table_size[0],look_up_table_size[1])

        #Look_up_Table_Buttons
        self.look_up_table_ui.load_look_up_table.clicked.connect(lambda: Gui_Signal_processing_programm.load_look_up_table(self))
        self.look_up_table_ui.pushButton_browser_look_up_table.clicked.connect(lambda: Gui_Signal_processing_programm.button_browser_look_up_table(self))
        self.look_up_table_ui.pushButton_plot_look_up_table.clicked.connect(lambda: Gui_Signal_processing_programm.button_plot_look_up_table(self))

        #Start Buttons
        self.w.load_from_Look_up_table.clicked.connect(lambda: Gui_Signal_processing_programm.look_up_table(self))
        self.w.pushButton_wireshark_extract.clicked.connect(lambda: Gui_Signal_processing_programm.load_ui_wireshark_extract(self))

        #wireshark extract Buttons
        self.wireshark_extract_ui.pushButton_wireshark_extract_path.clicked.connect(lambda : Gui_Signal_processing_programm.wireshark_extract_path(self))
        self.wireshark_extract_ui.pushButton_wireshark_extract_files.clicked.connect(lambda : Gui_Signal_processing_programm.wireshark_extract_files(self))
        #merge Qualisys and wireshark
        #merge_qualisys_and_wireshark_ui.pushButton_merge_qualisys_and_wireshark
        self.w.pushButton_merge_qualisys_and_wireshark.clicked.connect(lambda: Gui_Signal_processing_programm.merge_qualisys_and_wireshark(self))
        self.merge_qualisys_and_wireshark_ui.pushButton_merge_path_browser.clicked.connect(lambda: Gui_Signal_processing_programm.merge_qualisys_and_wireshark_path_browser(self))
        self.merge_qualisys_and_wireshark_ui.pushButton_start_merge.clicked.connect(lambda: Gui_Signal_processing_programm.merge_qualisys_and_wireshark_start(self))

        #Start for generation Data for Look up table compare with Qualisys and wireshark Data #look_up_table_generation_Data_for_compare_MSE
        self.w.pushButton_compare_look_up_table_to_real_Data.clicked.connect(lambda: Gui_Signal_processing_programm.look_up_table_generation_Data_for_compare_MSE(self))
        self.generate_data_for_look_up_table_mse_ui.pushButton_look_up_table_browser.clicked.connect(lambda : Gui_Signal_processing_programm.look_up_table_path_browser(self))
        self.generate_data_for_look_up_table_mse_ui.pushButton_start_xyz_Daten.clicked.connect(lambda : Gui_Signal_processing_programm.look_up_table_XYZ_Data_Start(self))
        self.generate_data_for_look_up_table_mse_ui.pushButton_generation_look_up_table.clicked.connect(lambda : Gui_Signal_processing_programm.look_up_table_generation_Data_Start(self))

        #Start wireshark window
        self.w.pushButton_wireshark_window.clicked.connect(lambda : Gui_Signal_processing_programm.load_wireshark_window(self))
        self.wireshark_window_ui.pushButton_wireshark_window_browser.clicked.connect(lambda : Gui_Signal_processing_programm.wireshark_window_path_browser(self))
        wireshark_window = [self.wireshark_window_ui.tableView_wireshark_window.maximumWidth(),
                            self.wireshark_window_ui.tableView_wireshark_window.maximumHeight()]
        self.wireshark_window_ui.tableView_wireshark_window = QTableWidget(self.wireshark_window_ui.tableView_wireshark_window)
        self.wireshark_window_ui.tableView_wireshark_window.setMinimumSize(782, 457)
        self.wireshark_window_ui.pushButton_load_wireshark_window.clicked.connect(lambda : Gui_Signal_processing_programm.wireshark_window_load_data(self))
        self.wireshark_window_ui.pushButton_plot_wireshark_window.clicked.connect(lambda : Gui_Signal_processing_programm.plot_wireshark_window(self))

        #Start qualisys window
        self.w.pushButton_qualisys_window.clicked.connect(lambda :Gui_Signal_processing_programm.load_qualisys_window(self))
        self.qualisys_window_ui.pushButton_qualisys_window_browser.clicked.connect(lambda : Gui_Signal_processing_programm.qualisys_window_path_browser(self))

        self.qualisys_window_ui.tableView_qualisys_window = QTableWidget(self.qualisys_window_ui.tableView_qualisys_window)
        self.qualisys_window_ui.tableView_qualisys_window.setMinimumSize(782,457)
        self.qualisys_window_ui.pushButton_load_qualisys_window.clicked.connect(lambda : Gui_Signal_processing_programm.qualisys_window_load_data(self))
        self.qualisys_window_ui.pushButton_plot_qualisys_window.clicked.connect(lambda : Gui_Signal_processing_programm.plot_qualisys_window(self))
        self.qualisys_window_ui.pushButton_plot_qualisys_window_xyz_difference.clicked.connect(lambda : Gui_Signal_processing_programm.plot_qualisys_window_xyz_difference(self))

        self.w.show()
        app.quit

    def button_browser_look_up_table(self):
        filename = QFileDialog.getOpenFileName(None, 'Test Dialog', os.getcwd())
        print(filename)
        if(filename[1]==''):
            self.look_up_table_ui.path_look_up_table.setText("No File are change")
            self.look_up_table_ui.fileName_look_up_table.setText('-')
        elif(filename[0].endswith(".csv")):
            folder = os.path.dirname(filename[0])
            fileName= os.path.basename(filename[0])
            self.look_up_table_ui.path_look_up_table.setText(folder)
            self.look_up_table_ui.fileName_look_up_table.setText(fileName)
        else:
            self.look_up_table_ui.path_look_up_table.setText("No CSV-File are change")
            self.look_up_table_ui.fileName_look_up_table.setText('-')

    def button_plot_look_up_table(self):
        qt_index_plat = self.look_up_table_ui.tableView.selectionModel().selection().indexes()
        index_plot_list=[]
        for i in range(0,len(qt_index_plat)):
            index_plot_list.append([qt_index_plat[i].row(),qt_index_plat[i].column()])

        selection = self.look_up_table_ui.tableView.selectedRanges()[0]
        rows = list(range(selection.topRow(),selection.bottomRow()+1))
        columns = list(range(selection.leftColumn(),selection.rightColumn()+1))
        print(rows,columns)

        path = self.look_up_table_ui.path_look_up_table.text() + '/' + self.look_up_table_ui.fileName_look_up_table.text()
        lookuptable_array =self.Signal #np.loadtxt(path, delimiter=';')
        header_data = ['X-position', 'Y-position', 'Z-position', 'X-angle','Y-angle','Z-angle', 'coil X', 'coil Y', 'coil Z','Frame 1','Frame 2', 'Frame 3', 'Frame 4', 'Frame 5','Frame 6','Frame 7', 'Frame 8', 'Main 1', 'Main 2','Main 3','Main 4', 'Main 5', 'Main 6', 'Main 7','Main 8']
        plt.figure(1)
        for i in range(0,len(columns)):
            plt.figure(i+1)
            plt.plot(lookuptable_array[rows[0]:rows[len(rows)-1]+1,columns[i]])
            plt.title(header_data[columns[i]])
            plt.xlabel('n')
            if(columns[i]<3):
                plt.ylabel('m')
            elif(2<columns[i]<6):
                plt.ylabel('deg')
            elif(5<columns[i]<9):
                plt.ylabel('deg?')
            else:
                plt.ylabel('U')
        plt.show()

    def look_up_table(self):
        self.look_up_table_ui.show()

    def load_look_up_table(self):
        path = self.look_up_table_ui.path_look_up_table.text()+'/'+self.look_up_table_ui.fileName_look_up_table.text()
        print("\nLook up table Path:",path)
        if(True==os.path.isfile(path)):
            horHeaders=[]
            try:
                self.Signal = np.loadtxt(path, delimiter=';')
            except ValueError:
                print("No Look_up_table CSV-File")
            else:
                e=[]
                for k in range(1,40):
                    for s in range(0,len(self.Signal[0,:])):
                        e.append(str(self.Signal[k,s]))
                print(self.Signal[0,:])
                #self.tm_look_up_table_model = look_up_table_model()
                #self.look_up_table_ui.tableView.setModel(self.tm_look_up_table_model)

                #self.tm_look_up_table_model.add_look_up_table(look_up_table(e))
                #self.tm_look_up_table_model.add_look_up_table(look_up_table('Ramesh', 'Delhi','Ramesh', 'Delhi','Ramesh', 'Delhi','Ramesh', 'Delhi','Ramesh', 'Delhi','Ramesh', 'Delhi','Ramesh', 'Delhi','Ramesh', 'Delhi','Ramesh', 'Delhi','Ramesh', 'Delhi','Ramesh', 'Delhi','Ramesh', 'Delhi','Ramesh'))

                #tm_look_up_table_model.add_look_up_table(lutm.look_up_table(self.Signal))
                self.look_up_table_ui.tableView.setRowCount(len(self.Signal[:, 0]))
                self.look_up_table_ui.tableView.setColumnCount(len(self.Signal[0, :]))
                for i, row in enumerate(self.Signal):
                    for j, val in enumerate(row):
                        self.look_up_table_ui.tableView.setItem(i, j, QTableWidgetItem(str(val)))
                self.look_up_table_ui.tableView.setHorizontalHeaderLabels(['X-position', 'Y-position', 'Z-position', 'X-angle','Y-angle','Z-angle', 'coil X', 'coil Y', 'coil Z','Frame 1','Frame 2', 'Frame 3', 'Frame 4', 'Frame 5','Frame 6','Frame 7', 'Frame 8', 'Main 1', 'Main 2','Main 3','Main 4', 'Main 5', 'Main 6', 'Main 7','Main 8'])
                self.look_up_table_ui.tableView.resizeColumnsToContents()
                self.look_up_table_ui.tableView.resizeRowsToContents()
                # user can not edit any cells!
                self.look_up_table_ui.tableView.setEditTriggers(QAbstractItemView.NoEditTriggers)
                print(self.look_up_table_ui.tableView.sizeHint())
                print("fertig")
        else:
            self.look_up_table_ui.look_up_table_ui.tableView.clear()
            print("CSV-Datei is not found!")
        #C:\Users\User\Desktop\Master_Studium\Semester_3\Masterabeit\Simulationstool_Look_up_tables\holzRegal_119KHz

    def load_ui_wireshark_extract(self):
        self.wireshark_extract_ui.show()

    def wireshark_extract_path(self):
        filename = QFileDialog.getExistingDirectory(None, 'Select project folder:', 'F:\\', QtWidgets.QFileDialog.ShowDirsOnly)
        if(os.path.isdir(filename)==True):
            print(filename)
            self.wireshark_extract_ui.path_label_wireshark_extract.setText(filename)
            File_list_wireshark_extract,folder,File_list_wireshark_extract_plot = wireshark_extract.load_File_List_from_folder(filename)
            if(len(File_list_wireshark_extract_plot)!=0):
                print(File_list_wireshark_extract_plot)
                self.wireshark_extract_ui.Text_Browse_wireshark_extract_list.setText(str(len(File_list_wireshark_extract))+'Files are extracted\n'+str(len(File_list_wireshark_extract_plot))+' Files are found\n' + str(File_list_wireshark_extract_plot))#'%d Files are found',len(File_list_wireshark_extract)+
            else:
                self.wireshark_extract_ui.Text_Browse_wireshark_extract_list.setText('No .pcapng File found!')
        else:
            self.wireshark_extract_ui.path_label_wireshark_extract.setText("")
            self.wireshark_extract_ui.Text_Browse_wireshark_extract_list.setText('No Folder change')

    def wireshark_extract_files(self):
        if(len(self.wireshark_extract_ui.path_label_wireshark_extract.text())!=0):
            self.wireshark_extract_ui.Text_Browser_wireshark_finish_info.setText("")
            path_wireshark_extract=self.wireshark_extract_ui.path_label_wireshark_extract.text()
            File_list_wireshark_extract, folder,File_list_wireshark_extract_plot = wireshark_extract.load_File_List_from_folder(path_wireshark_extract)
            if(len(File_list_wireshark_extract) !=0):
                wireshark_extract_doku = wireshark_extract.wireshark_start_extract(flag=1, FILENAME_folder=File_list_wireshark_extract)
                self.wireshark_extract_ui.Text_Browser_wireshark_finish_info.setText(str(wireshark_extract_doku))
            print('File extracted finish')

            #packets_value,actuall_file = wireshark_extract.wireshark_start_extract(flag=1,FILENAME_folder=File_list_wireshark_extract)
            #print(packets_value+' wireshark_extract: '+actuall_file)
        else:
            print("empty")

    def merge_qualisys_and_wireshark(self):
        self.merge_qualisys_and_wireshark_ui.show()

    def merge_qualisys_and_wireshark_path_browser(self):
        filename = QFileDialog.getExistingDirectory(None, 'Select project folder:', 'F:\\',
                                                QtWidgets.QFileDialog.ShowDirsOnly)
        if (os.path.isdir(filename) == True):
            self.merge_qualisys_and_wireshark_ui.lineEdit_merge_path.setText(filename)

    def merge_qualisys_and_wireshark_start(self):
        merge_path = self.merge_qualisys_and_wireshark_ui.lineEdit_merge_path.text()
        save_Name = self.merge_qualisys_and_wireshark_ui.lineEdit_saved_name.text()
        cut_check_box = self.merge_qualisys_and_wireshark_ui.checkBox_cut.checkState()
        if(cut_check_box==2):
            cut_check_box=True
            self.merge_qualisys_and_wireshark_ui.lineEdit_x_cut_from.setDisabled(False)
            self.merge_qualisys_and_wireshark_ui.lineEdit_x_cut_to.setDisabled(False)
            self.merge_qualisys_and_wireshark_ui.lineEdit_y_cut_from.setDisabled(False)
            self.merge_qualisys_and_wireshark_ui.lineEdit_y_cut_to.setDisabled(False)
            self.merge_qualisys_and_wireshark_ui.lineEdit_z_cut_from.setDisabled(False)
            self.merge_qualisys_and_wireshark_ui.lineEdit_z_cut_to.setDisabled(False)
        if(cut_check_box==0):
            cut_check_box = False
            self.merge_qualisys_and_wireshark_ui.lineEdit_x_cut_from.setDisabled(True)
            self.merge_qualisys_and_wireshark_ui.lineEdit_x_cut_to.setDisabled(True)
            self.merge_qualisys_and_wireshark_ui.lineEdit_y_cut_from.setDisabled(True)
            self.merge_qualisys_and_wireshark_ui.lineEdit_y_cut_to.setDisabled(True)
            self.merge_qualisys_and_wireshark_ui.lineEdit_z_cut_from.setDisabled(True)
            self.merge_qualisys_and_wireshark_ui.lineEdit_z_cut_to.setDisabled(True)
        ver = self.merge_qualisys_and_wireshark_ui.lineEdit_x_cut_from.text()
        x_y_z_cut_range_error=False
        a= self.merge_qualisys_and_wireshark_ui.lineEdit_x_cut_from.text()!=''
        if(self.merge_qualisys_and_wireshark_ui.lineEdit_x_cut_from.text()!='' and self.merge_qualisys_and_wireshark_ui.lineEdit_x_cut_to.text()!='' and self.merge_qualisys_and_wireshark_ui.lineEdit_y_cut_from.text()!='' and self.merge_qualisys_and_wireshark_ui.lineEdit_y_cut_to.text()!='' and self.merge_qualisys_and_wireshark_ui.lineEdit_z_cut_from.text()!='' and self.merge_qualisys_and_wireshark_ui.lineEdit_z_cut_to.text()!=''):
            try:
                x_cut_range=[int(self.merge_qualisys_and_wireshark_ui.lineEdit_x_cut_from.text()),int(self.merge_qualisys_and_wireshark_ui.lineEdit_x_cut_to.text())]
                y_cut_range=[int(self.merge_qualisys_and_wireshark_ui.lineEdit_y_cut_from.text()), int(self.merge_qualisys_and_wireshark_ui.lineEdit_y_cut_to.text())]
                z_cut_range=[int(self.merge_qualisys_and_wireshark_ui.lineEdit_z_cut_from.text()), int(self.merge_qualisys_and_wireshark_ui.lineEdit_z_cut_to.text())]
            except:
                print("Please do not enter any letters")
                x_y_z_cut_range_error = True
        elif(cut_check_box == False):
            x_cut_range = [0,0]
            y_cut_range = [0,0]
            z_cut_range = [0,0]
        else:
            print("X-Y-Z-range missing")
            x_y_z_cut_range_error=True
        komplex_orentation_faktor= self.merge_qualisys_and_wireshark_ui.lineEdit_komplex_orentation_faktor.text()
        path_test =merge_path # r"C:\Users\User\Desktop\dauserml_Messungen_2020_07_22\Referenz_alfred"
        # komplexer orentation faktor
        if(komplex_orentation_faktor==0):
            komplex_orentation_faktor = (np.exp(-1j * np.deg2rad(-27)) / 4.65)
        komplex_orentation_faktor = (np.exp(-1j * np.deg2rad(-27)) / 4.65)

        cut_x_y_z_position_higher_then_antenna_null = False

        # Interpolation between NaN Sequence
        interpolation = True
        interpolation_value = 20  # 40*0.00714 = 0.2856s

        # Threshold for Antenna threshold_value > Antenna Signal = 0
        threshold_antenna = False
        threshold_value = 0.0001

        if(x_y_z_cut_range_error==False or cut_check_box == False):
            lsmdqw = lqd.load_and_save_merge_data_qualisys_and_wireshark(merge_path,path_test,save_Name,komplex_orentation_faktor,cut_check_box,x_cut_range,y_cut_range,z_cut_range,cut_x_y_z_position_higher_then_antenna_null,interpolation,interpolation_value,threshold_antenna,threshold_value)
            lsmdqw.covert_and_save_merge_data_qualisys_and_wireshark()
        if(True == os.path.isfile(merge_path + "\\" +save_Name+".csv")):
            print("No plan")
            #lsmdqw.load_merge_data_and_learn()
        else:
            print("CSV-Data not found")

    def look_up_table_generation_Data_for_compare_MSE(self): #look_up_table_with_qualisys_mse
        self.generate_data_for_look_up_table_mse_ui.show()

    def look_up_table_path_browser(self):

        filename = QFileDialog.getOpenFileName(None, 'Test Dialog', os.getcwd())
        folder = os.path.dirname(filename[0])
        fileName = os.path.basename(filename[0])
        if (os.path.isdir(folder) == True):
            self.generate_data_for_look_up_table_mse_ui.lineEdit_look_up_table_path.setText(folder)
            if (fileName.endswith(".pcapng.cal.npy")):
                fileName= (fileName.replace(".pcapng.cal.npy", ""))
                self.generate_data_for_look_up_table_mse_ui.lineEdit_load_name_look_up_table.setText(fileName)
            elif (folder.endswith(".pcapng.timestamp_wireshark.npy")):
                fileName = (fileName.replace(".pcapng.timestamp_wireshark.npy", ""))
                self.generate_data_for_look_up_table_mse_ui.lineEdit_load_name_look_up_table.setText(fileName)
            elif (folder.endswith("_6D.tsv")):
                fileName = (fileName.replace("_6D.tsv", ""))
                self.generate_data_for_look_up_table_mse_ui.lineEdit_load_name_look_up_table.setText(fileName)

    def look_up_table_XYZ_Data_Start(self):
        look_up_table_path = self.generate_data_for_look_up_table_mse_ui.lineEdit_look_up_table_path.text()
        load_Name = self.generate_data_for_look_up_table_mse_ui.lineEdit_load_name_look_up_table.text()
        cut_check_box = self.generate_data_for_look_up_table_mse_ui.checkBox_cut_look_up_table.checkState()
        if (cut_check_box == 2):
            cut_check_box = True
            self.generate_data_for_look_up_table_mse_ui.lineEdit_x_cut_from_look_up_table.setDisabled(False)
            self.generate_data_for_look_up_table_mse_ui.lineEdit_x_cut_to_look_up_table.setDisabled(False)
            self.generate_data_for_look_up_table_mse_ui.lineEdit_y_cut_from_look_up_table.setDisabled(False)
            self.generate_data_for_look_up_table_mse_ui.lineEdit_y_cut_to_look_up_table.setDisabled(False)
            self.generate_data_for_look_up_table_mse_ui.lineEdit_z_cut_from_look_up_table.setDisabled(False)
            self.generate_data_for_look_up_table_mse_ui.lineEdit_z_cut_to_look_up_table.setDisabled(False)
        if (cut_check_box == 0):
            cut_check_box = False
            self.generate_data_for_look_up_table_mse_ui.lineEdit_x_cut_from_look_up_table.setDisabled(True)
            self.generate_data_for_look_up_table_mse_ui.lineEdit_x_cut_to_look_up_table.setDisabled(True)
            self.generate_data_for_look_up_table_mse_ui.lineEdit_y_cut_from_look_up_table.setDisabled(True)
            self.generate_data_for_look_up_table_mse_ui.lineEdit_y_cut_to_look_up_table.setDisabled(True)
            self.generate_data_for_look_up_table_mse_ui.lineEdit_z_cut_from_look_up_table.setDisabled(True)
            self.generate_data_for_look_up_table_mse_ui.lineEdit_z_cut_to_look_up_table.setDisabled(True)
        ver = self.generate_data_for_look_up_table_mse_ui.lineEdit_x_cut_from_look_up_table.text()
        x_y_z_cut_range_error = False
        a =   self.generate_data_for_look_up_table_mse_ui.lineEdit_x_cut_from_look_up_table.text() != ''
        print(self.generate_data_for_look_up_table_mse_ui.lineEdit_load_name_look_up_table.text())
        if (self.generate_data_for_look_up_table_mse_ui.lineEdit_x_cut_from_look_up_table.text() != '' and self.generate_data_for_look_up_table_mse_ui.lineEdit_x_cut_to_look_up_table.text() != '' and self.generate_data_for_look_up_table_mse_ui.lineEdit_y_cut_from_look_up_table.text() != '' and self.generate_data_for_look_up_table_mse_ui.lineEdit_y_cut_to_look_up_table.text() != '' and self.generate_data_for_look_up_table_mse_ui.lineEdit_z_cut_from_look_up_table.text() != '' and self.generate_data_for_look_up_table_mse_ui.lineEdit_z_cut_to_look_up_table.text() != ''):
            try:
                x_cut_range = [int(self.generate_data_for_look_up_table_mse_ui.lineEdit_x_cut_from_look_up_table.text()),
                               int(self.generate_data_for_look_up_table_mse_ui.lineEdit_x_cut_to_look_up_table.text())]
                y_cut_range = [int(self.generate_data_for_look_up_table_mse_ui.lineEdit_y_cut_from_look_up_table.text()),
                               int(self.generate_data_for_look_up_table_mse_ui.lineEdit_y_cut_to_look_up_table.text())]
                z_cut_range = [int(self.generate_data_for_look_up_table_mse_ui.lineEdit_z_cut_from_look_up_table.text()),
                               int(self.generate_data_for_look_up_table_mse_ui.lineEdit_z_cut_to_look_up_table.text())]
            except:
                print("Please do not enter any letters")
                x_y_z_cut_range_error = True
        else:
            print("X-Y-Z-range missing")
            x_y_z_cut_range_error = True

        komplex_orentation_faktor = self.generate_data_for_look_up_table_mse_ui.lineEdit_komplex_orentation_faktor_look_up_table.text()
        # komplexer orentation faktor
        if (komplex_orentation_faktor == 0):
            komplex_orentation_faktor = (np.exp(-1j * np.deg2rad(-27)) / 4.65)
        komplex_orentation_faktor = (np.exp(-1j * np.deg2rad(-27)) / 4.65)
        ltsqawd_tm = ltsqawd.load_than_save_qualisys_and_wireshark_data(path=look_up_table_path,path_test= look_up_table_path,name_file=load_Name,
                                                                        komplex_orentation_faktor=komplex_orentation_faktor,
                                                                        cut=cut_check_box,x_cut=x_cut_range,y_cut=y_cut_range,z_cut=z_cut_range)

        File = 'loop_open_1'
        ltsqawd_tm.load_than_save_qualisys_and_wireshark_data_compare(File=File)
        print('finish xyz Data generation')

    def look_up_table_generation_Data_Start(self):
        look_up_table_path = self.generate_data_for_look_up_table_mse_ui.lineEdit_look_up_table_path.text()
        save_name = self.generate_data_for_look_up_table_mse_ui.lineEdit_saved_name_look_up_table_2.text()
        #GUI_cft.GUI_config_fingerprinting_table(load_path=look_up_table_path+'\\'+"x_y_z_position_look_up_table_compare.npy")
        save_table_name=(look_up_table_path+'/'+save_name)
        look_up_table_path_t = (look_up_table_path+'/'+"x_y_z_position_look_up_table_compare.npy")
        gsft = GUI_start_fingerprinting_table.GUI_start_fingerprinting_table(save_table_name,look_up_table_path_t)
        gsft.main_sft()

    def load_wireshark_window(self):
        self.wireshark_window_ui.show()

    def wireshark_window_path_browser(self):
        file_name = QFileDialog.getOpenFileName(None, 'Test Dialog', os.getcwd())
        folder = os.path.dirname(file_name[0])
        fileName = os.path.basename(file_name[0])
        if (os.path.isdir(folder) == True):
            self.wireshark_window_ui.lineEdit_wireshark_window_path_file.setText(folder+"/"+fileName)

    def wireshark_window_load_data(self):
        file_name = self.wireshark_window_ui.lineEdit_wireshark_window_path_file.text()
        test  = os.path.isfile(file_name)
        if(os.path.isfile(file_name)==True):
            if (file_name.endswith(".pcapng.cal.npy")):
                try:
                    file = np.load(file_name)
                    ff_wireshark = wire.wireshark.convert_Signal_wireshark(Signal_1=file)
                    self.wireshark_window_data = np.array(ff_wireshark)
                    File_list_cal = (file_name.replace(".pcapng.cal.npy", ""))
                    timestamp_wireshark = np.transpose(np.load(
                        File_list_cal + '.pcapng.timestamp_wireshark.npy'))
                except:
                    print("No pcapng.cal.npy or pcapng.timestamp_wireshark.npy -File are Found")
            elif (file_name.endswith(".pcapng.timestamp_wireshark.npy")):
                try:
                    timestamp_wireshark = np.transpose(np.load(file_name))
                    File_list_cal = (file_name.replace(".pcapng.timestamp_wireshark.npy", ""))
                    file = np.load(file_name+".pcapng.cal.npy")
                    self.wireshark_window_data = wire.wireshark.convert_Signal_wireshark(Signal_1=file)
                    self.wireshark_window_data = np.array(self.wireshark_window_data)
                except:
                    print("No pcapng.cal.npy or pcapng.timestamp_wireshark.npy -File are Found")
            #test = np.transpose(timestamp_wireshark[:, 0])
            self.wireshark_window_data = np.concatenate((timestamp_wireshark, self.wireshark_window_data),axis=1)
            self.wireshark_window_ui.tableView_wireshark_window.setRowCount(len(self.wireshark_window_data[:, 0]))
            self.wireshark_window_ui.tableView_wireshark_window.setColumnCount(len(self.wireshark_window_data[0, :]))
            for i, row in enumerate(self.wireshark_window_data):
                for j, val in enumerate(row):
                    self.wireshark_window_ui.tableView_wireshark_window.setItem(i, j, QTableWidgetItem(str(val)))
            self.wireshark_window_head = ['Time steps','Unix Time stamps', 'Antenna Main 1 reel', 'Antenna Main 1 imaginary', 'Antenna Frame 1 reel', 'Antenna Frame 1 imaginary',
                 'Antenna Main 2 reel', 'Antenna Main 2 imaginary', 'Antenna Frame 2 reel', 'Antenna Frame 2 imaginary',
                 'Antenna Main 3 reel', 'Antenna Main 3 imaginary', 'Antenna Frame 3 reel', 'Antenna Frame 3 imaginary',
                 'Antenna Main 4 reel', 'Antenna Main 4 imaginary', 'Antenna Frame 4 reel', 'Antenna Frame 4 imaginary',
                 'Antenna Main 5 reel', 'Antenna Main 5 imaginary', 'Antenna Frame 5 reel', 'Antenna Frame 5 imaginary',
                 'Antenna Main 6 reel', 'Antenna Main 6 imaginary', 'Antenna Frame 6 reel', 'Antenna Frame 6 imaginary',
                 'Antenna Main 7 reel', 'Antenna Main 7 imaginary', 'Antenna Frame 7 reel', 'Antenna Frame 7 imaginary',
                 'Antenna Main 8 reel', 'Antenna Main 8 imaginary', 'Antenna Frame 8 reel', 'Antenna Frame 8 imaginary',]
            self.wireshark_window_ui.tableView_wireshark_window.setHorizontalHeaderLabels(self.wireshark_window_head)
            self.wireshark_window_ui.tableView_wireshark_window.resizeColumnsToContents()
            self.wireshark_window_ui.tableView_wireshark_window.resizeRowsToContents()
            # user can not edit any cells!
            self.wireshark_window_ui.tableView_wireshark_window.setEditTriggers(QAbstractItemView.NoEditTriggers)
        else:
            self.wireshark_window_ui.tableView_wireshark_window.clear()

    def plot_wireshark_window(self):
        header_data = self.wireshark_window_head
        lookuptable_array = self.wireshark_window_data
        for t in range(0,2):
            qt_index_pqw = self.wireshark_window_ui.tableView_wireshark_window.selectionModel().selection().indexes()
            index_plot_list = []
            for i in range(0, len(qt_index_pqw)):
                index_plot_list.append([qt_index_pqw[i].row(), qt_index_pqw[i].column()])

            selection = self.wireshark_window_ui.tableView_wireshark_window.selectedRanges()[0]
            rows = list(range(selection.topRow(), selection.bottomRow() + 1))
            columns = list(range(selection.leftColumn(), selection.rightColumn() + 1))
            print(rows, columns)

            # np.loadtxt(path, delimiter=';')
            plt.figure(1)
            for i in range(0, len(columns)):
                plt.figure(i + 1)
                plt.plot(lookuptable_array[rows[0]:rows[len(rows) - 1] + 1, columns[i]])
                plt.title(header_data[columns[i]])
                plt.xlabel('n')
                if (columns[i] < 1):
                    plt.ylabel('s')
                elif (0 < columns[i] < 2):
                    plt.ylabel('timestamp')
                elif (2 < columns[i] ):
                    plt.ylabel('U')
                else:
                    plt.ylabel('U')
            lookuptable_array = medfilt2d(lookuptable_array,kernel_size=[17,1])
            #lookuptable_array = np.where(((-0.00015) > lookuptable_array) | (lookuptable_array > 0.00015), lookuptable_array, 0)

        plt.show()

    def load_qualisys_window(self):
        self.qualisys_window_ui.show()

    def qualisys_window_path_browser(self):
        file_name = QFileDialog.getOpenFileName(None, 'Test Dialog', os.getcwd())
        folder = os.path.dirname(file_name[0])
        fileName = os.path.basename(file_name[0])
        if (os.path.isdir(folder) == True):
            self.qualisys_window_ui.lineEdit_qualisys_window_path_file.setText(folder + "/" + fileName)

    def qualisys_window_load_data(self):
        file_name = self.qualisys_window_ui.lineEdit_qualisys_window_path_file.text()
        if (os.path.isfile(file_name) == True):
            if (file_name.endswith("_6D.tsv")):
                try:
                    self.qualisys_window_data, self.qualisys_window_head, first_timestamp_wireshark_qualisys = quali.qualisys.load_convert_qulisys_Data(
                        file_name,
                        delet_Null=0);
                except:
                    print("No 6D.tsv -File are Found")
            # test = np.transpose(timestamp_wireshark[:, 0])
            self.qualisys_window_ui.tableView_qualisys_window.setRowCount(len(self.qualisys_window_data[:, 0]))
            self.qualisys_window_ui.tableView_qualisys_window.setColumnCount(len(self.qualisys_window_data[0, :]))
            for i, row in enumerate(self.qualisys_window_data):
                for j, val in enumerate(row):
                    self.qualisys_window_ui.tableView_qualisys_window.setItem(i, j, QTableWidgetItem(str(val)))
            self.qualisys_window_ui.tableView_qualisys_window.setHorizontalHeaderLabels(np.array(self.qualisys_window_head))
            self.qualisys_window_ui.tableView_qualisys_window.resizeColumnsToContents()
            self.qualisys_window_ui.tableView_qualisys_window.resizeRowsToContents()
            # user can not edit any cells!
            self.qualisys_window_ui.tableView_qualisys_window.setEditTriggers(QAbstractItemView.NoEditTriggers)
        else:
            self.qualisys_window_ui.tableView_qualisys_window.clear()

    def plot_qualisys_window(self):
        qt_index_pqw = self.qualisys_window_ui.tableView_qualisys_window.selectionModel().selection().indexes()
        index_plot_list = []
        for i in range(0, len(qt_index_pqw)):
            index_plot_list.append([qt_index_pqw[i].row(), qt_index_pqw[i].column()])

        selection = self.qualisys_window_ui.tableView_qualisys_window.selectedRanges()[0]
        rows = list(range(selection.topRow(), selection.bottomRow() + 1))
        columns = list(range(selection.leftColumn(), selection.rightColumn() + 1))
        print(rows, columns)

        lookuptable_array = self.qualisys_window_data  # np.loadtxt(path, delimiter=';')
        header_data = self.qualisys_window_head
        plt.figure(1)
        for i in range(0, len(columns)):
            plt.figure(i + 1)
            plt.plot(lookuptable_array[rows[0]:rows[len(rows) - 1] + 1, columns[i]])
            plt.title(header_data[columns[i]])
            plt.xlabel('n')
            if (columns[i] < 3):
                plt.ylabel('m')
            elif (2 < columns[i] < 6):
                plt.ylabel('deg')
            elif (5 < columns[i] < 9):
                plt.ylabel('deg?')
            else:
                plt.ylabel('U')
        plt.show()

    def plot_qualisys_window_xyz_difference(self):
        check_rotation = self.qualisys_window_ui.checkBox.checkState()
        header_data = self.qualisys_window_head
        lookuptable_array = self.qualisys_window_data  # np.loadtxt(path, delimiter=';')
        if(check_rotation==0):
            diff_qualisys_x = ((lookuptable_array[:, 20]) - np.nanmean(lookuptable_array[:, 3]))
            diff_qualisys_y = ((lookuptable_array[:, 21]) - np.nanmean(lookuptable_array[:, 4]))
            diff_qualisys_z = ((lookuptable_array[:, 22]) - np.nanmean(lookuptable_array[:, 5]))
            x_y_z_Wert = ([diff_qualisys_x, diff_qualisys_y, diff_qualisys_z])
            lookuptable_array = np.transpose(np.array(x_y_z_Wert))
        elif(check_rotation==2):
            lookuptable_array = quali.qualisys.x_y_z_translation_and_rotation(lookuptable_array,1)

        plt.figure(1)
        plt.plot(lookuptable_array[:,0])
        plt.title("X Position")
        plt.xlabel('n')
        plt.ylabel('m')
        plt.figure(2)
        plt.plot(lookuptable_array[:, 1])
        plt.title("Y Position")
        plt.xlabel('n')
        plt.ylabel('m')
        plt.figure(3)
        plt.plot(lookuptable_array[:, 2])
        plt.title("Z Position")
        plt.xlabel('n')
        plt.ylabel('m')
        plt.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gspp = Gui_Signal_processing_programm()
    gspp.gui_signal_processing_main()
    app.quit
    sys.exit(app.exec_())

