import numpy as np
import Classification_Sequence as CS
import Test_Tensorflow as TT
import pandas as pd
from sklearn.preprocessing import Normalizer,RobustScaler
from keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import seaborn as sns
from sklearn.metrics import balanced_accuracy_score


x_y_z_to_Class =False


path = r"C:\Users\dauserml\Documents\2020_12_03"
#path = r"C:\Users\dauserml\Documents\2020_12_03_Training\02_Manuel_03_12_2020"
filename_training = "\\Julian_Test.csv"#"\\all_Files_C_R.csv"

model_Reg = tf.keras.models.load_model(r"C:\Users\dauserml\Desktop\ResNet_Time_Series_all_01_NN_from_16_Dec_2020_08_46\MSE_Time_Step_16/MSE.h5",compile = False)
model_Reg_classification= tf.keras.models.load_model(r"C:\Users\dauserml\Documents\2020_11_16_und_2020_12_03\xyz_to_Classes_Manuel_Training\NN_from_14_Dec_2020_17_09\sparse_categorical_crossentropy_Batch_128/sparse_categorical_crossentropy.h5",compile = False)

# Trainings Daten
all_or_range_selection_Data = True  # all=True, range selection=False
training_Data_from = 0  # Only "all_or_range_selection_Data = False"
training_Data_to = 20000  # Only "all_or_range_selection_Data = False"
time = 1  # 1= Timesteps
X_Signal_number = 0  # 0= Signal delimitation
Input_from = 9  # Input Signal from 9 to 40 (9=frame1_real,10=frame1_imag,11=frame2_real,....,40=main8_imag)
Input_to = 41
# Signal (5=Yaw,6=Roll,7=Pitch,8=Residual) (41= For Classification, witch shelf)
Output_from = 2  # Ouput Signal from 41 -> Class
Output_to = 5

def cut_Signal_into_sequencen_train(self, X_train, Y_train, X_Signal_number_train,Class):
    if (self.sequence_training == True):
        X_train_cut_tm = [];
        X_train_cut_NaN = [];
        Y_train_cut_tm = [];
        Y_train_cut_NaN = [];
        X_Signal_number_train_cut_tm = [];
        Class_tm=[]
        X_Signal_number_train_cut_NaN = []
        over_under_x = 0
        over_under_y = 0
        over_under_z = 0
        flag = 0
        for c in range(0, len(Y_train)):
            if (self.sequence_training_cut_x[0] < Y_train[c, 0] < self.sequence_training_cut_x[1]
                    and self.sequence_training_cut_y[0] < Y_train[c, 1] < self.sequence_training_cut_y[1]
                    and self.sequence_training_cut_z[0] < Y_train[c, 2] < self.sequence_training_cut_z[1]):
                X_train_cut_tm.append(X_train[c, :])
                Y_train_cut_tm.append(Y_train[c, :])
                X_Signal_number_train_cut_tm.append(X_Signal_number_train[c, :])
                Class_tm.append(Class[c])
                flag = 1
            #elif (flag == 1):
            ######else:
            ######    if (self.sequence_training_cut_x[0] > Y_train[c, 0]):
            ######        # under
            ######        over_under_x = self.sequence_training_cut_x[0] - 10
            ######    elif (Y_train[c, 0] > self.sequence_training_cut_x[1]):
            ######        # over
            ######        over_under_x = self.sequence_training_cut_x[1] + 10
            ######    else:
            ######        over_under_x = Y_train[c, 0]
#####
            ######    if (self.sequence_training_cut_y[0] > Y_train[c, 1]):
            ######        # under
            ######        over_under_y = self.sequence_training_cut_y[0] - 10
            ######    elif (Y_train[c, 1] > self.sequence_training_cut_y[1]):
            ######        # over
            ######        over_under_y = self.sequence_training_cut_y[1] + 10
            ######    else:
            ######        over_under_y = Y_train[c, 1]
#####
            ######    if (self.sequence_training_cut_z[0] > Y_train[c, 2]):
            ######        # under
            ######        over_under_z = self.sequence_training_cut_z[0] - 10
            ######    elif (Y_train[c, 2] > self.sequence_training_cut_z[1]):
            ######        # over
            ######        over_under_z = self.sequence_training_cut_z[1] + 10
            ######    else:
            ######        over_under_z = Y_train[c, 2]
#####
            ######    #X_Signal_number_train = X_Signal_number_train + 1
            ######    ## Not an NaN Training
            ######    X_train_cut_tm.append(X_train[c, :])
            ######    Y_train[c, :] = [over_under_x, over_under_y, over_under_z]
            ######    Y_train_cut_tm.append(Y_train[c, :])
            ######    X_Signal_number_train_cut_tm.append(X_Signal_number_train[c, :])
                #####X_Signal_number_train = X_Signal_number_train + 1
                ####### Not an NaN Training
                #####X_train_cut_NaN.append(X_train[c, :])
                #####Y_train[c, :] = [over_under_x, over_under_y, over_under_z]
                #####Y_train_cut_NaN.append(Y_train[c, :])
                #####X_Signal_number_train_cut_NaN.append(X_Signal_number_train[c, :])
                flag = 0
            ###else:
            ###    if (self.sequence_training_cut_x[0] < Y_train[c, 0]):
            ###        # under
            ###        over_under_x = self.sequence_training_cut_x[0] - 10
            ###    elif (Y_train[c, 0] < self.sequence_training_cut_x[1]):
            ###        # over
            ###        over_under_x = self.sequence_training_cut_x[1] + 10
            ###    if (self.sequence_training_cut_y[0] < Y_train[c, 1]):
            ###        # under
            ###        over_under_y = self.sequence_training_cut_y[0] - 10
            ###    elif (Y_train[c, 1] < self.sequence_training_cut_y[1]):
            ###        # over
            ###        over_under_y = self.sequence_training_cut_y[1] + 10
            ###    if (self.sequence_training_cut_z[0] < Y_train[c, 2]):
            ###        # under
            ###        over_under_z = self.sequence_training_cut_z[0] - 10
            ###    elif (Y_train[c, 2] < self.sequence_training_cut_x[1]):
            ###        # over
            ###        over_under_z = self.sequence_training_cut_z[1] + 10
##
            ###    ## Not an NaN Training
            ###    X_train_cut_NaN.append(X_train[c, :])
            ###    Y_train[c, :] = [over_under_x, over_under_y, over_under_z]
            ###    Y_train_cut_NaN.append(Y_train[c, :])
            ###    X_Signal_number_train_cut_NaN.append(X_Signal_number_train[c, :])
            # X_Signal_number_train = X_Signal_number_train + 1
            ### Not an NaN Training
            # X_train_cut_NaN.append(X_train[c, :])
            # Y_train[c, :] = np.nan
            # Y_train_cut_NaN.append(Y_train[c, :])
            # X_Signal_number_train_cut_NaN.append(X_Signal_number_train[c,:])
        #last_X_Number_train_value = X_Signal_number_train[-1]
        #X_Signal_number_train_cut_NaN = np.array(X_Signal_number_train_cut_NaN)
        #X_Signal_number_train_cut_NaN[:, :] = (last_X_Number_train_value + 1)
        #X_train_cut_tm = X_train_cut_tm + X_train_cut_NaN
        #Y_train_cut_tm = Y_train_cut_tm + Y_train_cut_NaN
        #X_Signal_number_train = np.vstack((np.array(X_Signal_number_train_cut_tm), X_Signal_number_train_cut_NaN))
        X_train = np.array(X_train_cut_tm)
        Y_train = np.array(Y_train_cut_tm)
        Class = np.array(Class)
        X_Signal_number_train = np.array(X_Signal_number_train_cut_tm)
    return X_train, Y_train, X_Signal_number_train,Class


time_xyz_antennen_Signal_Komplex_all_Files = np.loadtxt(path+filename_training,delimiter=';')

X_Signal_number=(time_xyz_antennen_Signal_Komplex_all_Files[:,X_Signal_number:(X_Signal_number+1)].astype(int))
Class = (time_xyz_antennen_Signal_Komplex_all_Files[:,41:42].astype(int))
X_1 = time_xyz_antennen_Signal_Komplex_all_Files[:,Input_from:Input_to]#[:,9:]#
Y_1 = (time_xyz_antennen_Signal_Komplex_all_Files[:,Output_from:Output_to].astype(int))#[:,2:5]#

conf = CS.config_Test_Tensorflow
tensor = CS.Tensorflow(conf.creat_dataset_time_steps,conf.time_steps_geo_folge,conf.create_dataset_overlap,conf.create_dataset_offset,conf.sequence_training,
                    conf.sequence_training_cut_x,conf.sequence_training_cut_y,conf.sequence_training_cut_z,conf.sequence_test,conf.sequence_test_cut_x,
                    conf.sequence_test_cut_y,conf.sequence_test_cut_z,conf.path, conf.path_test_sequenz,conf.filename_training,
                    conf.filename_test_sequenz, conf.save_dir, conf.all_or_range_selection_Data,
                    conf.training_Data_from, conf.training_Data_to,
                    conf.time, conf.X_Signal_number, conf.Input_from, conf.Input_to, conf.Output_from,
                    conf.Output_to, conf.trainings_epochs, conf.batch_size,
                    conf.verbose, conf.metrics, conf.learning_rate, conf.optimizer_name, conf.optimizer,
                    conf.split_test_size, conf.split_KFold_set, conf.split_random_state,
                    conf.split_shuffle, conf.loss_funktions, conf.scaler,conf.threshold_antenna,conf.threshold_value, conf.pickle, conf.png,conf.kernel_init,conf.bias_init,conf.save)

#X_test, Y_test, X_Signal_number_train, Class_train = CS.Tensorflow.cut_Signal_into_sequencen_test_or_train(self=tensor,X_test=X_1,Y_test=Y_1,Class_test=Class,X_Signal_number_test=X_Signal_number,test_or_train=True)

normalizer_x_Reg = Normalizer()  ### Wichtig! Aufpassen das der Richtige Scaler verwendet wird!!!

X_test_normalize_Reg = normalizer_x_Reg.fit_transform(X_1)
X_test_normalize_time_Reg = pd.DataFrame(X_test_normalize_Reg)
Y_test_time_series_normalize_Reg = pd.DataFrame(Y_1)
X_test_time_series_Reg, Y_test_time_series_Reg, Class_test_Reg,X_Signal_number_Reg = CS.Tensorflow.shape_the_Signal(self=tensor,
                                                                               X_Signal_number_train=X_Signal_number,
                                                                               X_train_time_series_normalize=X_test_normalize_time_Reg,
                                                                               Y_train_time_series_normalize=Y_test_time_series_normalize_Reg,
                                                                               Class_train=Class,
                                                                               i=(1 - 1))
#X_test_normalize_Reg = normalizer_x_Reg.fit_transform(X_test_time_series_Reg)
#X_test_time_series_normalize_Reg = pd.DataFrame(X_test_normalize_Reg)

normalizer_y_Reg = Normalizer()
Y_test_normalize_Reg = normalizer_y_Reg.fit_transform(Y_test_time_series_Reg)
#Y_test_time_series_normalize_Reg = pd.DataFrame(Y_test_normalize_Reg)
model_Reg.compile(loss=tensor.loss_funktions[0],
              optimizer=tensor.optimizer[0],metrics=tensor.metrics)
print("x-test-reg"+str(len(X_test_time_series_Reg)))
print(str(len(Y_test_time_series_Reg)))
print(str(len(Y_1)))
Y_pred_Reg = model_Reg.predict(X_test_time_series_Reg)
plt.figure(88)
plt.subplot(3,1,1)
plt.plot(Y_pred_Reg[:,0])
plt.plot(Y_test_normalize_Reg[:,0])
plt.subplot(3, 1, 2)
plt.plot(Y_pred_Reg[:, 1])
plt.plot(Y_test_normalize_Reg[:, 1])
plt.subplot(3, 1, 3)
plt.plot(Y_pred_Reg[:, 2])
plt.plot(Y_test_normalize_Reg[:, 2])


X_test_time_series_Reg, Y_test_time_series_Reg, X_Signal_number_Reg, Class_test_Reg = CS.Tensorflow.cut_Signal_into_sequencen_test_or_train(self=tensor,X_test=X_test_time_series_Reg,Y_test=Y_test_time_series_Reg,Class_test=Class_test_Reg,X_Signal_number_test=X_Signal_number_Reg,test_or_train=True)



#ergeb_Reg =  model_Reg.evaluate(X_test_time_series_Reg,Y_test_normalize_Reg)
#print("Evaluate Regression:"+str(ergeb_Reg))
#try:
#    Y_pred_Reg = pd.DataFrame(Y_pred_Reg[2])# googlenet!!!!!! aufpassen
#except:
#    print("no googlenet")
Y_pred_Reg = pd.DataFrame(Y_pred_Reg)
print("y_pred"+str(len(Y_pred_Reg)))
Y_test_time_series_Reg=pd.DataFrame(Y_test_time_series_Reg)
X_test_time_series_reg_class, Y_test_time_series_Reg, Class_test_reg_class,X_Signal_number_trr = CS.Tensorflow.shape_the_Signal(self=tensor,
                                                                               X_Signal_number_train=X_Signal_number_Reg,
                                                                               X_train_time_series_normalize=Y_pred_Reg,
                                                                               Y_train_time_series_normalize=Y_test_time_series_Reg,
                                                                               Class_train=Class_test_Reg,
                                                                               i=(1 - 1))


#Y_pred_Reg = normalizer_y_Reg.inverse_transform(Y_pred_Reg)

#X_test_normalize_reg_class = normalizer_x.fit_transform(X_test_time_series_Reg)


####Für NN_01 Regression auf Klassifikation
#X_test_normalize_reg_class = normalizer_x.fit_transform(Y_test)
#X_test_time_series_normalize_reg_class = pd.DataFrame(X_test_normalize_reg_class)

#if(x_y_z_to_Class==True):
#    X_test_time_series_reg_class, Y_test_time_series_reg_class, Class_test_reg_class,X_Signal_number_tr = CS.Tensorflow.shape_the_Signal(self=tensor,
#                                                                                   X_Signal_number_train=X_Signal_number_train,
#                                                                                   X_train_time_series_normalize=X_test_time_series_normalize_reg_class,
#                                                                                   Y_train_time_series_normalize=Y_test_time_series_normalize,
#                                                                                   Class_train=Class_train,
#                                                                                   i=(1 - 1))



print("Y_1: "+str(len(Y_1)))
#print("Y_test: "+str(len(Y_test)))
#print("Y_test_time_series: "+str(len(Y_test_time_series)))
print("Y_test_time_series_reg_class"+str(len(Y_test_time_series_Reg)))

#diff = (Y_1[7:1344,:]-Y_test_time_series)
#er = np.mean(diff)
#print("error: "+str(er))


plt.figure(3)
plt.subplot(3,1,1)
plt.plot(Y_1[:,2],"o")


#plt.subplot(3,1,2)
#plt.plot(Y_test[:,2],"o")
#plt.subplot(3,1,3)
#plt.plot(Y_test_time_series[:,2],"o")

CS.Tensorflow.plot_2D_Classification(tensor,model_Reg_classification,X_test_time_series_reg_class,Y_test_time_series_Reg,Class_test_reg_class,2,"Test",path)

model_Reg_classification.compile(loss=tensor.loss_funktions[0],
              optimizer=tensor.optimizer[0],metrics=tensor.metrics)


Y_pred_reg_class = np.argmax(model_Reg_classification.predict(X_test_time_series_reg_class), axis=-1)


Y_pred_reg_class = pd.DataFrame(Y_pred_reg_class,columns=["Y_pred_reg_class"])
Class_true_Reg = pd.DataFrame(Class_test_reg_class,columns=["Class_test_reg_class"])

confusion_matrix_reg_class = pd.crosstab(Y_pred_reg_class["Y_pred_reg_class"],Class_true_Reg["Class_test_reg_class"])

#group_names=["kein Eingriff","Eingriff","Ausgriff","Klasse 1","Klasse 2","Klasse 3","Klasse 4","Klasse 5","Klasse 6"]

#print("Confusion antenna2class: "+confusion_matrix)
#sns.heatmap(confusion_matrix/np.sum(confusion_matrix),annot=True,vmax=120, fmt=".2%")

#Y_Pred_Class=model.predict(X_test_time_series)

#print("Confusion reg2class: "+confusion_matrix)
plt.figure(6)
sns.heatmap(confusion_matrix_reg_class,annot=True,vmax=120, fmt="d")

#Y_Pred_Class=model.predict(X_test_time_series)

bal_erg_reg_class = balanced_accuracy_score(Y_pred_reg_class,Class_test_reg_class)
print("balanced accuracy score: "+str(bal_erg_reg_class))

erg_reg_class = model_Reg_classification.evaluate(X_test_time_series_reg_class,Class_test_reg_class)
print("evaluate: "+str(erg_reg_class))

plt.show()