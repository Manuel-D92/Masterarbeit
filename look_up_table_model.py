
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

class look_up_table(object):
    def __init__(self,X_position,Y_position,Z_position,X_angle,Y_angle,Z_angle, X_coil, Y_coil, Z_coil,Frame_1,Frame_2,Frame_3,Frame_4,Frame_5,Frame_6,Frame_7,Frame_8,Main_1,Main_2,Main_3,Main_4, Main_5, Main_6, Main_7,Main_8):
        self.X_position = X_position
        self.Y_position = Y_position
        self.Z_position = Z_position
        self.X_angle = X_angle
        self.Y_angle = Y_angle
        self.Z_angle = Z_angle
        self.X_coil = X_coil
        self.Y_coil = Y_coil
        self.Z_coil = Z_coil
        self.Frame_1 = Frame_1
        self.Frame_2 = Frame_2
        self.Frame_3 = Frame_3
        self.Frame_4 = Frame_4
        self.Frame_5 = Frame_5
        self.Frame_6 = Frame_6
        self.Frame_7 = Frame_7
        self.Frame_8 = Frame_8
        self.Main_1 = Main_1
        self.Main_2 = Main_2
        self.Main_3 = Main_3
        self.Main_4 = Main_4
        self.Main_5 = Main_5
        self.Main_6 = Main_6
        self.Main_7 = Main_7
        self.Main_8 = Main_8

class look_up_table_model(QAbstractItemView):

    Row_Batch_Count = 15
    def __init__(self):
        super(look_up_table_model, self).__init__()
        self.headers = ['X-position', 'Y-position', 'Z-position', 'X-angle','Y-angle','Z-angle', 'coil X', 'coil Y', 'coil Z','Frame 1','Frame 2', 'Frame 3', 'Frame 4', 'Frame 5','Frame 6','Frame 7', 'Frame 8', 'Main 1', 'Main 2','Main 3','Main 4', 'Main 5', 'Main 6', 'Main 7','Main 8']
        self.tm_look_up_table = []
        self.rowsLoaded = look_up_table_model.Row_Batch_Count

    def rowCount(self,index=QModelIndex()):
        if not self.tm_look_up_table:
            return 0
        if len(self.tm_look_up_table)<= self.rowsLoaded:
            return len(self.tm_look_up_table)
        else:
            return self.rowsLoaded

    def canFetchMore(self,index=QModelIndex()):
        if len(self.tm_look_up_table)> self.rowsLoaded:
            return True
        else:
            return False

    def fetchMore(self,index=QModelIndex()):
        reminder = len(self.tm_look_up_table)-self.rowsLoaded
        itemsToFetch = min(reminder,look_up_table_model.Row_Batch_Count)
        self.beginInsertRows(QModelIndex(),self.rowsLoaded,self.rowsLoaded+itemsToFetch-1)
        self.rowsLoaded += itemsToFetch
        self.endInsertRows()

    def add_look_up_table(self,look_up_table):
        self.beginResetModel()
        self.tm_look_up_table.append(look_up_table)
        self.endResetModel()

    def columnCount(self,index=QModelIndex()):
        return len(self.headers)

    def data(self,index,role=Qt.DisplayRole):
        col= index.column()
        look_up_table = self.tm_look_up_table[index.row()]
        if role == self.Qt.DisplayRole:
            if col == 0:
                return QVariant(look_up_table.X_position)
            elif col ==1:
                return QVariant(look_up_table.Y_position)
            elif col ==2:
                return QVariant(look_up_table.Z_position)
            elif col == 3:
                return QVariant(look_up_table.X_angel)
            elif col ==4:
                return QVariant(look_up_table.Y_angel)
            elif col ==5:
                return QVariant(look_up_table.Z_angel)
            elif col == 6:
                return QVariant(look_up_table.X_coil)
            elif col ==7:
                return QVariant(look_up_table.Y_coil)
            elif col ==8:
                return QVariant(look_up_table.Z_coil)
            elif col == 9:
                return QVariant(look_up_table.Frame_1)
            elif col ==10:
                return QVariant(look_up_table.Frame_2)
            elif col ==11:
                return QVariant(look_up_table.Frame_3)
            elif col == 12:
                return QVariant(look_up_table.Frame_4)
            elif col ==13:
                return QVariant(look_up_table.Frame_5)
            elif col ==14:
                return QVariant(look_up_table.Frame_6)
            elif col ==15:
                return QVariant(look_up_table.Frame_7)
            elif col ==16:
                return QVariant(look_up_table.Frame_8)
            elif col ==17:
                return QVariant(look_up_table.Main_1)
            elif col ==18:
                return QVariant(look_up_table.Main_2)
            elif col ==19:
                return QVariant(look_up_table.Main_3)
            elif col == 20:
                return QVariant(look_up_table.Main_4)
            elif col ==21:
                return QVariant(look_up_table.Main_5)
            elif col ==22:
                return QVariant(look_up_table.Main_6)
            elif col ==23:
                return QVariant(look_up_table.Main_7)
            elif col ==24:
                return QVariant(look_up_table.Main_8)
            return QVariant()

    def headerData(self,section,orientation,role=Qt.DisplayRole):
        if role!= Qt.DisplayRole:
            return QVariant()
        if orientation== Qt.Horizontal:
            return QVariant(self.headers[section])
        return QVariant(int(section+1))
