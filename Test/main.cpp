#include "mainwindow.h"
#include <QApplication>

#include <QList>
#include <QVariant>



int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();


}
