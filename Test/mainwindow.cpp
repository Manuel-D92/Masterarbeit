#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "look_up_table_model.h"


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    QList<float> value;
    value.append(0.0);

    QList<QString> contactNames;
    QList<QString> contactPhoneNums;
    QList<int> contactnum;
    QList<float> frame_1;
    QList<float> X_Position;
    QList<float> Y_Position;
    QList<float> Z_Position;
    QVector<QList<float>> k;
    QList<QList<float>> f;



    look_up_table_model::X_Coil;
    // Create some data that is tabular in nature:
    contactNames.append("Thomas");
    contactNames.append("Richard");

    contactPhoneNums.append("123-456-7890");
    contactPhoneNums.append("222-333-4444");
    contactPhoneNums.append("333-444-5555");
    contactnum.append(1);
    contactnum.append(3);
    contactnum.append(2);
    frame_1.append(0.0);
    frame_1.append(0.0);
    frame_1.append(0.0);
    X_Position.append(0.1);
    X_Position.append(1.1);
    X_Position.append(2.1);
    //k << X_Position<<frame_1;
    k<<frame_1<<X_Position<<frame_1<<X_Position<<frame_1<<X_Position<<frame_1<<X_Position<<frame_1<<X_Position<<frame_1<<X_Position<<frame_1<<X_Position<<frame_1<<X_Position<<frame_1<<X_Position<<frame_1<<X_Position<<frame_1<<X_Position<<frame_1<<X_Position<<frame_1;
    //k[1].append(frame_1);


    // Create model:
    look_up_table_model *look_up_model = new look_up_table_model();

    // Populate model with data:
    look_up_model->populateData(contactNames,contactPhoneNums,contactnum,frame_1,X_Position,k);

    // Connect mod  el to table view:
    //ui->pushButton_wireshark_extract->setDisabled()

    //int e = event->KeyPress;
//  //  connect(event->KeyPress,SIGNAL(),)
    //ui->setupUi(this);
    //
    //setWindowFlags(windowFlags() | Qt::WindowStaysOnTopHint);
    //setWindowState(Qt::WindowMaximized);
}

MainWindow::~MainWindow()
{
    delete ui;
}


