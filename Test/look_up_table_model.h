#ifndef LOOK_UP_TABLE_MODEL_H
#define LOOK_UP_TABLE_MODEL_H

#include <QAbstractItemModel>

class look_up_table_model : public QAbstractTableModel
{
    Q_OBJECT
    Q_ENUMS(tes)
public:
    static const int X_Position      =0 ;
    static const int Y_Position      =1 ;
    static const int Z_Position      =2 ;
    static const int X_Angel         =3 ;
    static const int Y_Angel         =4 ;
    static const int Z_Angel         =5 ;
    static const int X_Coil          =6 ;
    static const int Y_Coil          =7 ;
    static const int Z_Coil          =8 ;
    static const int Frame_antenne_1 =9 ;
    static const int Frame_antenne_2 =10;
    static const int Frame_antenne_3 =11;
    static const int Frame_antenne_4 =12;
    static const int Frame_antenne_5 =13;
    static const int Frame_antenne_6 =14;
    static const int Frame_antenne_7 =15;
    static const int Frame_antenne_8 =16;
    static const int Main_antenne_1  =17;
    static const int Main_antenne_2  =18;
    static const int Main_antenne_3  =19;
    static const int Main_antenne_4  =20;
    static const int Main_antenne_5  =21;
    static const int Main_antenne_6  =22;
    static const int Main_antenne_7  =23;
    static const int Main_antenne_8  =24;

    look_up_table_model(QObject *parent = 0);

    void populateData(const QList<QString> &contactName,const QList<QString> &contactPhone, const QList<int> &contactnum, const QList<float> &frame_1,const QList<float> &X_Position,const QVector<QList<float>> &Data);

    int rowCount(const QModelIndex &parent = QModelIndex()) const Q_DECL_OVERRIDE;
    int columnCount(const QModelIndex &parent = QModelIndex()) const Q_DECL_OVERRIDE;

    QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const Q_DECL_OVERRIDE;
    QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const Q_DECL_OVERRIDE;

private:
    QMap<float,float> test;
    QList<QString> tm_contact_name;
    QList<QString> tm_contact_phone;
    QList<int> tm_contact_numb;
    QList<float> tm_X_Position;
    QList<float> Frame_1;
    QVector<QList<float>> tm_Data;

};

#endif // LOOK_UP_TABLE_MODEL_H
