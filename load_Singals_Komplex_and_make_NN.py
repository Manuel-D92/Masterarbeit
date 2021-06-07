import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score
from sklearn.multiclass import OneVsOneClassifier,OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import torch

Training= 1 #Training=1 oder laden einse Trainierten Netz=0
average = 0 # average=0 -> load Signal without average; average=1 -> load Signal make average; average = 2-> load average Signal

Average_count = 6
scaler = StandardScaler()

class Signal():
    def Load_Signal(self,i):
        Signal_1= np.load('C:\\Users\\dauserml\\Desktop\\dauserml_Messungen_2020_07_22\\wareable_2020_08_14\\linke_hand\\Feld_1_%s_linke_Hand.npy'%i);
        Signal_2= np.load('C:\\Users\\dauserml\\Desktop\\dauserml_Messungen_2020_07_22\\wareable_2020_08_14\\linke_hand\\Feld_2_%s_linke_Hand.npy'%i);
        Signal_3= np.load('C:\\Users\\dauserml\\Desktop\\dauserml_Messungen_2020_07_22\\wareable_2020_08_14\\linke_hand\\Feld_3_%s_linke_Hand.npy'%i);
        Signal_4= np.load('C:\\Users\\dauserml\\Desktop\\dauserml_Messungen_2020_07_22\\wareable_2020_08_14\\linke_hand\\Feld_4_%s_linke_Hand.npy'%i);
        Signal_5= np.load('C:\\Users\\dauserml\\Desktop\\dauserml_Messungen_2020_07_22\\wareable_2020_08_14\\linke_hand\\Feld_5_%s_linke_Hand.npy'%i);
        Signal_6= np.load('C:\\Users\\dauserml\\Desktop\\dauserml_Messungen_2020_07_22\\wareable_2020_08_14\\linke_hand\\Feld_6_%s_linke_Hand.npy'%i);
        Signal_7= np.load('C:\\Users\\dauserml\\Desktop\\dauserml_Messungen_2020_07_22\\wareable_2020_08_14\\linke_hand\\Ruhezustand_nach_Feld_%s.npy' % i);
        #Signal_8= np.load('C:\\Users\\dauserml\\Desktop\\dauserml_Messungen_2020_07_22\\coil_10_mal_10\\kein_Eingriff_%s.npy' % i);
        #Signal_9= np.load('C:\\Users\\dauserml\\Desktop\\dauserml_Messungen_2020_07_22\\coil_10_mal_10\\ausen_vorbei_%s.npy' % i);

        S_1,y_1,time_1 = Signal.convert_Signal(self=0,Signal_1=Signal_1,Klasse=0) #Feld 1
        S_2,y_2,time_2 = Signal.convert_Signal(self=0,Signal_1=Signal_2,Klasse=1) #Feld 2
        S_3,y_3,time_3 = Signal.convert_Signal(self=0,Signal_1=Signal_3,Klasse=2) #Feld 3
        S_4,y_4,time_4 = Signal.convert_Signal(self=0,Signal_1=Signal_4,Klasse=3) #Feld 4
        S_5,y_5,time_5 = Signal.convert_Signal(self=0,Signal_1=Signal_5,Klasse=4) #Feld 5
        S_6,y_6,time_6 = Signal.convert_Signal(self=0,Signal_1=Signal_6,Klasse=5) #Feld 6
        S_7,y_7,time_7 = Signal.convert_Signal(self=0,Signal_1=Signal_7,Klasse=6) #Ruhezustand
        #S_8,y_8,time_8 = Signal.convert_Signal(self=0,Signal_1=Signal_8,Klasse=7) #kein Eingriff
        #S_9,y_9,time_9 = Signal.convert_Signal(self=0,Signal_1=Signal_9,Klasse=8) #außen

        Signal_g = np.concatenate((S_1, S_2, S_3, S_4, S_5, S_6, S_7))  # ,S_8,S_9))
        Y = np.concatenate((y_1, y_2, y_3, y_4, y_5, y_6, y_7))  # , y_8, y_9))
        #time_stamp = np.concatenate((time_1, time_2, time_3, time_4, time_5, time_6, time_7))  # , time_8, time_9))

        if(average):
            S_1_average, y_1_average, time_1_average = Signal.Average_1(S_1, y_1, time_1, Average_count ,i)
            S_2_average, y_2_average, time_2_average = Signal.Average_1(S_2, y_2, time_2, Average_count ,i)
            S_3_average, y_3_average, time_3_average = Signal.Average_1(S_3, y_3, time_3, Average_count ,i)
            S_4_average, y_4_average, time_4_average = Signal.Average_1(S_4, y_4, time_4, Average_count ,i)
            S_5_average, y_5_average, time_5_average = Signal.Average_1(S_5, y_5, time_5, Average_count ,i)
            S_6_average, y_6_average, time_6_average = Signal.Average_1(S_6, y_6, time_6, Average_count ,i)
            S_7_average, y_7_average, time_7_average = Signal.Average_1(S_7, y_7, time_7, Average_count ,i)
            #S_8_average, y_8_average, time_8_average = Signal.Average(S_8, y_8, time_8, 6)
            #S_9_average, y_9_average, time_9_average = Signal.Average(S_9, y_9, time_9, 6)

            Signal_g = np.concatenate((S_1_average, S_2_average, S_3_average, S_4_average, S_5_average, S_6_average, S_7_average))  # ,S_8,S_9))
            Y = np.concatenate((y_1_average, y_2_average, y_3_average, y_4_average, y_5_average, y_6_average, y_7_average))  # , y_8, y_9))
            #time_stamp = np.concatenate((time_1_average, time_2_average, time_3_average, time_4_average, time_5_average, time_6_average, time_7_average))  # , time_8_average, time_9_average))

            #S_1_average_arr = np.array(S_1_average)   %%Zum überprüfen vom richtigen signal zum average Signal
            #S_1_arr = np.array(S_1)
            #plt.figure(10)
            #plt.plot(time_1_average,S_1_average_arr[:,0])
            #plt.plot(time_1,S_1_arr[:,0])   %%end


        return Signal_g,Y

    def Average(X,Y,time_stamps,iter,Anzahl_Signal):
        for k in range(0,iter):
            print('Average:  Average Zahl= %s,'%(k+1)+'Klasse= %s,'%Y[0]+ 'Signalnummer= %s...'%Anzahl_Signal)
            X_re = []
            YY = []
            time_stamps_re = []
            for i in range(0, int(math.ceil(len(X) / 2)) - 1):
                XX = []
                for j in range (0,len(X[0])):
                    XX.append(np.mean([X[2*i][j],X[2*i+1][j],X[2*i+2][j]]))
                X_re.append(XX)
            for i in range (0,int(math.ceil(len(X)/2))-1):
                YY.append(Y[2*i+1])
                time_stamps_re.append(time_stamps[2*i+1])
            X = X_re;Y=YY;time_stamps=time_stamps_re
        return X_re,YY,time_stamps_re

    def Average_1(X,Y,time_stamps,iter,Anzahl_Signal):
        for k in range(0,iter):
            print('Average:  Average Zahl= %s,'%(k+1)+'Klasse= %s,'%Y[0]+ 'Signalnummer= %s...'%Anzahl_Signal)
            X_re = []
            YY = []
            time_stamps_re = []
            for i in range(0, int(math.ceil(len(X) / 2)) - 9):
                XX = []
                for j in range (0,len(X[0])):
                    #XX.append(np.mean([X[1 * i][j], X[1 * i + 1][j], X[1 * i + 2][j]]))
                    XX.append(np.mean([X[2*i][j],X[2*i+1][j],X[2*i+2][j],X[2*i+3][j],X[2*i+4][j],X[2*i+5][j],X[2*i+6][j],X[2*i+7][j],X[2*i+8][j],X[2*i+9][j]]))
                X_re.append(XX)
            for i in range (0,int(math.ceil(len(X)/2))-9):
                YY.append(Y[2*i+1])
                time_stamps_re.append(time_stamps[2*i+1])
            X = X_re;Y=YY;time_stamps=time_stamps_re
        return X_re,YY,time_stamps_re

    def Average_Test_Sequenz(X,iter):
        #print('Average Test Sequenz...')
        for k in range(0,iter):
            X_re = []
            for i in range(0, int(math.ceil(len(X) / 2)) - 2):
                XX = []
                for j in range(0, len(X[0])):
                    #gausfaktor = 8
                    #XX.append((((X[1 * i][j]) *(1+gausfaktor) + (X[1 * i + 1][j]) *(26+gausfaktor) + (X[1 * i + 2][j]) *(1+gausfaktor))/(28+3*gausfaktor)))
                    XX.append(np.mean([X[2*i][j],X[2*i+1][j],X[2*i+2][j]]))
                X_re.append(XX)
            X = X_re
        return X

    def Average_Test_Sequenz_1(X,iter):
        #print('Average Test Sequenz...')
        for k in range(0,iter):
            X_re = []
            for i in range(0, int(math.ceil(len(X) / 2)) - 9):
                XX = []
                for j in range(0, len(X[0])):
                    #medianfilter=
                    #XX.append(np.median([X[1 * i][j], X[1 * i + 1][j], X[1 * i + 2][j], X[1 * i + 3][j], X[1 * i + 4][j], X[1 * i + 5][j], X[1 * i + 6][j], X[1 * i + 7][j], X[1 * i + 8][j], X[1 * i + 9][j]]))
                    #Gausfilter=
                    #gausfaktor = 8
                    #XX.append((((X[1 * i][j]) *(4+gausfaktor) + (X[1 * i + 1][j]) *(12+gausfaktor) + (X[1 * i + 2][j]) *(20+gausfaktor) + (X[1*i+3][j]) *(28+gausfaktor) + (X[1*i+4][j]) *(36+gausfaktor) + (X[1*i+5][j]) *(28+gausfaktor) + (X[1*i+6][j]) *(20+gausfaktor) + (X[1*i+7][j]) *(12+gausfaktor) + (X[1*i+8][j]) *(4+gausfaktor))/(164+9*gausfaktor)))
                    #kompriemieren=
                    XX.append(np.mean([X[2*i][j],X[2*i+1][j],X[2*i+2][j],X[2*i+3][j],X[2*i+4][j],X[2*i+5][j],X[2*i+6][j],X[2*i+7][j],X[2*i+8][j],X[2*i+9][j]]))
                X_re.append(XX)
            X = X_re
        return X

    def Load_Test_Sequenz(self):
        test = np.load(
            'C:\\Users\\dauserml\\Desktop\\dauserml_Messungen_2020_07_22\\wareable_2020_08_14\\Sequenzen_linke_hand\\Sequenz_Feld_5_4_linke_Hand.npy');
        #test = test[2700:4700, :, :]
        X_test,time_test = Signal.convert_X_Test(self=0, Signal=test)
        X_test = Signal.Average_Test_Sequenz_1(X_test,Average_count)
        return X_test,test

    def convert_Signal(self,Signal_1,Klasse):
        XX=[];YY=[]
        sample_rate =  1/((np.shape(Signal_1)[0]-1)/np.real(Signal_1[-1,0,0]))
        lengh_Singal = len(Signal_1)
        time_s = np.linspace(0,lengh_Singal*sample_rate,lengh_Singal)
        for i in range(0, len(Signal_1)): #Signal_1.shape[0]): len(Signal_1)
            X = []
            for j in range(0,Signal_1.shape[2]-2):
                X.append(Signal_1.real[i, 1, j])
                X.append(Signal_1.imag[i, 1, j])
            XX.append(X)
        for i in range(0, len(Signal_1)): #len(Signal_1)
            YY.append(Klasse)
        return XX,YY,time_s

    def convert_X_Test(self, Signal):
        XX = []
        sample_rate = 1 / ((np.shape(Signal)[0] - 1) / np.real(Signal[-1, 0, 0]))
        lengh_Singal = len(Signal)
        time_s = np.linspace(0, lengh_Singal * sample_rate, lengh_Singal)
        for i in range(0, len(Signal)-1):  # Signal_1.shape[0]):
            X = []
            for j in range(0, Signal.shape[2] - 2):
                X.append(Signal.real[i, 1, j])
                X.append(Signal.imag[i, 1, j])
            XX.append(X)
        return XX,time_s

    def cut_Signal(self, Signal):
        Signal_cutting =[]
        for i in range (0,len(Signal)):
            k=0
            for j in range (0,len(Signal[1])-2): # letzte Antenne 16 ausschalten zu großes Rauschen!!

                if(((Signal[i,j]>=0.002) or (Signal[i,j]<=-0.002)) and k==0):
                    Signal_cutting.append(Signal[i,:])
                    k=i

        return Signal_cutting

if(Training):
    print('Load Data...')
    X=0;Y=0
    if(average==0):
        X, Y = torch.load(('C:\\Users\\dauserml\\Desktop\\dauserml_Messungen_2020_07_22\\wareable_2020_08_14\\linke_hand\\alle_Daten_zusammengefasst_ohne_Average'))
        if(X is 0):
            X, Y = Signal.Load_Signal(self=0, i=1)
            for i in range(2, 10):  # 10
                X_i, Y_i= Signal.Load_Signal(self=0, i=i)
                X = np.concatenate((X, X_i), axis=0)
                Y = np.concatenate((Y, Y_i), axis=0)
                torch.save([X, Y], ('C:\\Users\\dauserml\\Desktop\\dauserml_Messungen_2020_07_22\\wareable_2020_08_14\\linke_hand\\alle_Daten_zusammengefasst_ohne_Average'))
    if(average==1):
        X, Y = Signal.Load_Signal(self=0, i=1)
        for i in range(2, 10):  # 10
            X_i, Y_i = Signal.Load_Signal(self=0, i=i)
            X = np.concatenate((X, X_i), axis=0)
            Y = np.concatenate((Y, Y_i), axis=0)
        torch.save([X,Y],('C:\\Users\\dauserml\\Desktop\\dauserml_Messungen_2020_07_22\\wareable_2020_08_14\\linke_hand\\Average_daten_%s_30.npy'%Average_count))
    if(average==2):
        X,Y = torch.load(('C:\\Users\\dauserml\\Desktop\\dauserml_Messungen_2020_07_22\\wareable_2020_08_14\\linke_hand\\Average_daten_%s_30.npy'%Average_count))

    #plt.figure(9)
    #plt.subplot(4,2,1);plt.plot(X[:,[0,1]]);plt.subplot(4,2,2);plt.plot(X[:,[2,3]]);plt.subplot(4,2,3);plt.plot(X[:,[4,5]]);plt.subplot(4,2,4);plt.plot(X[:,[6,7]]);plt.subplot(4,2,5);plt.plot(X[:,[8,9]]);plt.subplot(4,2,6);plt.plot(X[:,[10,11]]);plt.subplot(4,2,7);plt.plot(X[:,[12,13]]);plt.subplot(4,2,8);plt.plot(X[:,[14,15]])
    #plt.figure(10)
    #plt.subplot(4,2,1);plt.plot(X[:,[16,17]]);plt.subplot(4,2,2);plt.plot(X[:,[18,19]]);plt.subplot(4,2,3);plt.plot(X[:,[20,21]]);plt.subplot(4,2,4);plt.plot(X[:,[22,23]]);plt.subplot(4,2,5);plt.plot(X[:,[24,25]]);plt.subplot(4,2,6);plt.plot(X[:,[26,27]]);plt.subplot(4,2,7);plt.plot(X[:,[28,29]]);plt.subplot(4,2,8);plt.plot(X[:,[30#,31]])
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    #X_train=X;Y_train=Y;

    print('Number of Size',len(X))

    for j in range(0,1):
        for i in range(0,1):
            print('Begin learn NN...',"i=",i,"j=" ,j)
            #clf = RandomForestClassifier(n_estimators=2,min_samples_split=3,min_samples_leaf=2)
            #clf = OneVsOneClassifier(SGDClassifier(random_state=20,max_iter=8000))
            #clf = make_pipeline(StandardScaler(),MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(80+j*10  ,80+i), random_state=50, max_iter=1000))
            #clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(20+j*10,15+i), random_state=0, max_iter=2000)
            #clf = make_pipeline(MinMaxScaler(),OneVsOneClassifier(MultinomialNB()))
            clf = make_pipeline(MinMaxScaler(), OneVsOneClassifier(GaussianNB()))
            #GaussianNB.score()

            #clf = make_pipeline(StandardScaler(), DecisionTreeClassifier())
            clf.fit(X_train,Y_train)
            #print(clf.named_steps['gaussiannb'].__doc__)  # ,clf.named_steps['gaussiannb'].__doc__.sigma)
            #sigma =  clf.named_steps['gaussiannb'].sigma_
            #theta =  clf.named_steps['gaussiannb'].theta_
            #epsilon =  clf.named_steps['gaussiannb'].epsilon_
            #class_prior =  clf.named_steps['gaussiannb'].class_prior_
            #print( repr(clf.named_steps['gaussiannb'].sigma_)) #, clf.named_steps['gaussiannb'].classes_.theta_
            torch.save(clf, ("C:\\Users\\dauserml\\Desktop\\dauserml_Messungen_2020_07_22\\save_Model\\2020_08_20\\wareable_clf_bayesian_linke_10_mal_10_%s"%(j*10+20)+"_%s"%(i+15)))
            print('End learn NN...')

    y_pre_t = clf.predict(X_test)
    print('confusion_matrix= ',metrics.confusion_matrix(Y_test, y_pre_t))
    #print('r2_score=', r2_score(Y_test, y_pre_t))
    print('Score = ',clf.score(X_test,Y_test))
    Y_pre = clf.predict(X_test)
    #print(clf.score(Y_test,Y_pre))


clf=torch.load("C:\\Users\\dauserml\\Desktop\\dauserml_Messungen_2020_07_22\\save_Model\\2020_08_20\\wareable_clf_bayesian_linke_10_mal_10_20_15")

X_Test,X_Test_1 = Signal.Load_Test_Sequenz(self=0)

Y_pre = clf.predict(X_Test)
#print(clf.score(X_Test,Y_pre))



X_Test = np.array(X_Test)

er = clf.predict(X_Test)

#print(cross_val_score(clf,X_train,Y,cv=4,scoring='accuracy'))

print(er)
print('1->',np.count_nonzero(er==0),'2->',np.count_nonzero(er==1),'3->',np.count_nonzero(er==2),'4->',np.count_nonzero(er==3),'5->',np.count_nonzero(er==4),'6->',np.count_nonzero(er==5),'7_gleich_0->',np.count_nonzero(er==6),'8_kein Eingriff->',np.count_nonzero(er==7),'9_außen vorbei->',np.count_nonzero(er==8))
klassifikation=[]
klassifikation = [(np.count_nonzero(er==0),np.count_nonzero(er==1),np.count_nonzero(er==2),np.count_nonzero(er==3),np.count_nonzero(er==4),np.count_nonzero(er==5))]
print("Es wurde eine Klassifizierung im Fach:",np.nanargmax(klassifikation)+1,"erkannt.")



test_programm=1
if(test_programm==1):
    print('Test Programm:')
    for i in range(1,7):
        Ruhezustand = []
        klassifikation = []
        for j in range(1,9):
            test = np.load( 'C:\\Users\\dauserml\\Desktop\\dauserml_Messungen_2020_07_22\\wareable_2020_08_14\\Sequenzen_linke_hand\\Sequenz_Feld_%s'%i+'_%s_linke_Hand.npy'%j);
            #test = test[2000:4000, :, :]
            X_test,time_test = Signal.convert_X_Test(self=0, Signal=test)
            #X_test = Signal.Average_Test_Sequenz(X_test, Average_count)
            X_test = np.array(X_test)
            cut = Signal.cut_Signal(self=0, Signal=X_test)
            X_Test = np.array(cut)
            er = clf.predict(X_Test)
            klassifikation.append(np.nanargmax([(np.count_nonzero(er == 0), np.count_nonzero(er == 1), np.count_nonzero(er == 2),
                                  np.count_nonzero(er == 3), np.count_nonzero(er == 4), np.count_nonzero(er == 5))])+1)
            Ruhezustand.append(np.count_nonzero(er == 6))
            print('Signal_%s'%i,'_%s'%j,'= 1->', np.count_nonzero(er == 0), '2->', np.count_nonzero(er == 1), '3->', np.count_nonzero(er == 2),
                  '4->', np.count_nonzero(er == 3), '5->', np.count_nonzero(er == 4), '6->', np.count_nonzero(er == 5),
                  '7_gleich_0->', np.count_nonzero(er == 6), '8_kein Eingriff->', np.count_nonzero(er == 7),
                  '9_außen vorbei->', np.count_nonzero(er == 8))
        print("%s"%i,"= Es wurde eine Klassifizierung im Fach:", klassifikation, "erkannt. Anzahl Ruhezustand:",)


a = np.load( 'C:\\Users\\dauserml\\Desktop\\dauserml_Messungen_2020_07_22\\wareable_2020_08_14\\Sequenzen_linke_hand\\Sequenz_Feld_5_4_linke_Hand.npy');
b,time_test =  Signal.convert_X_Test(self=0,Signal=a)
X=Signal.Average_Test_Sequenz(b, Average_count)
X = np.array(X)
cut = Signal.cut_Signal(self=0,Signal=X)
plt.figure(9)
plt.subplot(4,2,1);plt.plot(X[:,[0,1]]);plt.subplot(4,2,2);plt.plot(X[:,[2,3]]);plt.subplot(4,2,3);plt.plot(X[:,[4,5]]);plt.subplot(4,2,4);plt.plot(X[:,[6,7]]);plt.subplot(4,2,5);plt.plot(X[:,[8,9]]);plt.subplot(4,2,6);plt.plot(X[:,[10,11]]);plt.subplot(4,2,7);plt.plot(X[:,[12,13]]);plt.subplot(4,2,8);plt.plot(X[:,[14,15]])
plt.figure(10)
plt.subplot(4,2,1);plt.plot(X[:,[16,17]]);plt.subplot(4,2,2);plt.plot(X[:,[18,19]]);plt.subplot(4,2,3);plt.plot(X[:,[20,21]]);plt.subplot(4,2,4);plt.plot(X[:,[22,23]]);plt.subplot(4,2,5);plt.plot(X[:,[24,25]]);plt.subplot(4,2,6);plt.plot(X[:,[26,27]]);plt.subplot(4,2,7);plt.plot(X[:,[28,29]]);plt.subplot(4,2,8);plt.plot(X[:,[30,31]])

cut = np.array(cut)
X=cut
#plt.figure(11)
#plt.subplot(4,2,1);plt.plot(X[:,[0,1]]);plt.subplot(4,2,2);plt.plot(X[:,[2,3]]);plt.subplot(4,2,3);plt.plot(X[:,[4,5]]);plt.subplot(4,2,4);plt.plot(X[:,[6,7]]);plt.subplot(4,2,5);plt.plot(X[:,[8,9]]);plt.subplot(4,2,6);plt.plot(X[:,[10,11]]);plt.subplot(4,2,7);plt.plot(X[:,[12,13]]);plt.subplot(4,2,8);plt.plot(X[:,[14,15]])
#plt.figure(12)
#plt.subplot(4,2,1);plt.plot(X[:,[16,17]]);plt.subplot(4,2,2);plt.plot(X[:,[18,19]]);plt.subplot(4,2,3);plt.plot(X[:,[20,21]]);plt.subplot(4,2,4);plt.plot(X[:,[22,23]]);plt.subplot(4,2,5);plt.plot(X[:,[24,25]]);plt.subplot(4,2,6);plt.plot(X[:,[26,27]]);plt.subplot(4,2,7);plt.plot(X[:,[28,29]]);plt.subplot(4,2,8);plt.plot(X[:,[30,31]])


if(1):
    Signal = a
    fig = plt.figure(1)
    plt.subplot(4,2,1)
    plt.plot(Signal.real[:,1,0]);plt.plot(Signal.imag[:,1,0])
    plt.subplot(4,2,2)
    plt.plot(Signal.real[:,1,1]);plt.plot(Signal.imag[:,1,1])
    plt.subplot(4,2,3)
    plt.plot(Signal.real[:,1,2]);plt.plot(Signal.imag[:,1,2])
    plt.subplot(4,2,4)
    plt.plot(Signal.real[:,1,3]);plt.plot(Signal.imag[:,1,3])
    plt.subplot(4,2,5)
    plt.plot(Signal.real[:,1,4]);plt.plot(Signal.imag[:,1,4])
    plt.subplot(4,2,6)
    plt.plot(Signal.real[:,1,5]);plt.plot(Signal.imag[:,1,5])
    plt.subplot(4,2,7)
    plt.plot(Signal.real[:,1,6]);plt.plot(Signal.imag[:,1,6])
    plt.subplot(4,2,8)
    plt.plot(Signal.real[:,1,7]);plt.plot(Signal.imag[:,1,7])
    plt.suptitle('Antennen 1 - 8')

    plt.figure(2)
    plt.subplot(4,2,1)
    plt.plot(Signal.real[:,1,8]);plt.plot(Signal.imag[:,1,8])
    plt.subplot(4,2,2)
    plt.plot(Signal.real[:,1,9]);plt.plot(Signal.imag[:,1,9])
    plt.subplot(4,2,3)
    plt.plot(Signal.real[:,1,10]);plt.plot(Signal.imag[:,1,10])
    plt.subplot(4,2,4)
    plt.plot(Signal.real[:,1,11]);plt.plot(Signal.imag[:,1,11])
    plt.subplot(4,2,5)
    plt.plot(Signal.real[:,1,12]);plt.plot(Signal.imag[:,1,12])
    plt.subplot(4,2,6)
    plt.plot(Signal.real[:,1,13]);plt.plot(Signal.imag[:,1,13])
    plt.subplot(4,2,7)
    plt.plot(Signal.real[:,1,14]);plt.plot(Signal.imag[:,1,14])
    plt.subplot(4,2,8)
    plt.plot(Signal.real[:,1,15]);plt.plot(Signal.imag[:,1,15])
    plt.suptitle('Antennen 9 -16')

#plt.figure(4)
#plt.plot(X_Test[:,1]);plt.plot(X_Test[:,0])

#X,Y = torch.load(('C:\\Users\\dauserml\\Desktop\\dauserml_Messungen_2020_07_22\\wareable_2020_08_14\\linke_hand\\Average_daten_%s.npy'%Average_count))
#plt.figure(5)
#plt.plot(X[:,1]);plt.plot(X[:,0])

#plt.figure(3)
#plt.subplot(2,1,1)
#plt.plot(Signal.imag[:,1,16]);plt.plot(Signal.real[:,1,16])
#plt.subplot(2,1,2)
#plt.plot(Signal.imag[:,1,17]);plt.plot(Signal.real[:,1,17])
#plt.suptitle('Strom Feedback')
plt.show()