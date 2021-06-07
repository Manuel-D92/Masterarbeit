#include "look_up_table_model.h"

look_up_table_model::look_up_table_model(QObject *parent) : QAbstractTableModel(parent)
{
}

// Create a method to populate the model with data:
void look_up_table_model::populateData(const QList<QString> &contactName,const QList<QString> &contactPhone,const QList<int> &contactnum, const QList<float> &frame_1,const QList<float> &X_Position,const QVector<QList<float>> &Data)
{
    tm_contact_name.clear();
    tm_contact_name = contactName;
    tm_contact_phone.clear();
    tm_contact_phone = contactPhone;
    tm_contact_numb.clear();
    tm_contact_numb = contactnum;
    Frame_1.clear();
    Frame_1 = frame_1;
    tm_X_Position.clear();
    tm_X_Position = X_Position;
    tm_Data.clear();
    tm_Data = Data;
    return;
}

int look_up_table_model::rowCount(const QModelIndex &parent) const
{
    Q_UNUSED(parent);
    return tm_contact_name.length();
}

int look_up_table_model::columnCount(const QModelIndex &parent) const
{
    Q_UNUSED(parent);
    return 25;
}

QVariant look_up_table_model::data(const QModelIndex &index, int role) const
{
    if (!index.isValid() || role != Qt::DisplayRole) {
        return QVariant();
    }
    if (index.column() == 0) {
        return tm_Data[look_up_table_model::X_Position][index.row()];
    } else if (index.column()==1){
        return tm_Data[look_up_table_model::Y_Position][index.row()];
    } else if (index.column()==2){
        return tm_Data[look_up_table_model::Z_Position][index.row()];
    } else if (index.column()==3){
        return tm_Data[look_up_table_model::X_Angel][index.row()];
    } else if (index.column()==4){
        return tm_Data[look_up_table_model::Y_Angel][index.row()];
    } else if (index.column()==5){
        return tm_Data[look_up_table_model::Z_Angel][index.row()];
    } else if (index.column()==6){
        return tm_Data[look_up_table_model::X_Coil][index.row()];
    } else if (index.column()==7){
        return tm_Data[look_up_table_model::Y_Coil][index.row()];
    } else if (index.column()==8){
        return tm_Data[look_up_table_model::Z_Coil][index.row()];
    } else if (index.column()==9){
        return tm_Data[look_up_table_model::Frame_antenne_1][index.row()];
    } else if (index.column()==10){
        return tm_Data[look_up_table_model::Frame_antenne_2][index.row()];
    } else if (index.column()==11){
        return tm_Data[look_up_table_model::Frame_antenne_3][index.row()];
    } else if (index.column()==12){
        return tm_Data[look_up_table_model::Frame_antenne_4][index.row()];
    } else if (index.column()==13){
        return tm_Data[look_up_table_model::Frame_antenne_5][index.row()];
    } else if (index.column()==14){
        return tm_Data[look_up_table_model::Frame_antenne_6][index.row()];
    } else if (index.column()==15){
        return tm_Data[look_up_table_model::Frame_antenne_7][index.row()];
    } else if (index.column()==16){
        return tm_Data[look_up_table_model::Frame_antenne_8][index.row()];
    } else if (index.column()==17){
        return tm_Data[look_up_table_model::Main_antenne_1][index.row()];
    } else if (index.column()==18){
        return tm_Data[look_up_table_model::Main_antenne_2][index.row()];
    }else if (index.column()==19){
        return tm_Data[look_up_table_model::Main_antenne_3][index.row()];
    }else if (index.column()==20){
        return tm_Data[look_up_table_model::Main_antenne_4][index.row()];
    }else if (index.column()==21){
        return tm_Data[look_up_table_model::Main_antenne_5][index.row()];
    }else if (index.column()==22){
        return tm_Data[look_up_table_model::Main_antenne_6][index.row()];
    }else if (index.column()==23){
        return tm_Data[look_up_table_model::Main_antenne_7][index.row()];
    }else if (index.column()==24){
        return tm_Data[look_up_table_model::Main_antenne_8][index.row()];
    }
    return QVariant();
}

QVariant look_up_table_model::headerData(int section, Qt::Orientation orientation, int role) const
{
    if (role == Qt::DisplayRole && orientation == Qt::Horizontal) {
        if (section == 0) {
            return QString("X_Position");
        } else if (section == 1) {
            return QString("Y_Position");
        } else if (section ==2){
            return QString("Z_Position");
        } else if (section == 3){
            return QString("X_Angel");
        } else if (section == 4){
            return QString("Y_Angel");
        } else if (section == 5){
            return QString("Z_Angel");
        } else if (section == 6){
            return QString("X_Coil");
        } else if (section == 7){
            return QString("Y_Coil");
        } else if (section == 8){
            return QString("Z_Coil");
        } else if (section == 9){
            return QString("Frame_1");
        } else if (section == 10){
            return QString("Frame_2");
        } else if (section == 11){
            return QString("Frame_3");
        } else if (section == 12){
            return QString("Frame_4");
        } else if (section == 13){
            return QString("Frame_5");
        } else if (section == 14){
            return QString("Frame_6");
        } else if (section == 15){
            return QString("Frame_7");
        } else if (section == 16){
            return QString("Frame_8");
        } else if (section == 17){
            return QString("Main_1");
        } else if (section == 18){
            return QString("Main_2");
        } else if (section == 19){
            return QString("Main_3");
        } else if (section == 20){
            return QString("Main_4");
        } else if (section == 21){
            return QString("Main_5");
        } else if (section == 22){
            return QString("Main_6");
        } else if (section == 23){
            return QString("Main_7");
        } else if (section == 24){
            return QString("Main_8");
        }
    }
    return QVariant();
}
