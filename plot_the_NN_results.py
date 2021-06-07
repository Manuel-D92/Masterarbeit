from sklearn import preprocessing
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

scaler = preprocessing.Normalizer()
creat_dataset_time_steps = 32
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)
def shape_the_Signal(X_Signal_number_train,X_train_time_series_normalize,Y_train_time_series_normalize,i):
    count = 0
    X_train_time_series = []
    Y_train_time_series = []
    for j in range(int(X_Signal_number_train[0]),int(X_Signal_number_train[-1] + 1)):
        of = 0 + count
        count = count + np.count_nonzero(X_Signal_number_train[:] == j)
        to = count
        if(creat_dataset_time_steps*(2**i)<np.count_nonzero(X_Signal_number_train[:] == j)):
            X_train_time_series_tm, Y_train_time_series_tm = create_dataset(X_train_time_series_normalize[of:to][:],
                                                                            Y_train_time_series_normalize[of:to][:],
                                                                            time_steps=creat_dataset_time_steps * (2 ** i))
            if(j==int(X_Signal_number_train[0])):
                X_train_time_series = X_train_time_series_tm
                Y_train_time_series = Y_train_time_series_tm
            else:
                X_train_time_series= np.vstack((X_train_time_series,X_train_time_series_tm))
                Y_train_time_series = np.vstack((Y_train_time_series, Y_train_time_series_tm))
    return X_train_time_series,Y_train_time_series


path = r"C:\Users\dauserml\Documents\2020_09_25\Messung_1_Test"
time_xyz_antennen_Signal_Komplex_all_Files = np.loadtxt(path+"\\all_Files_Z_4000_mm_Mit_seperater_Signal_trennung.csv",delimiter=';')
X_Signal_number=time_xyz_antennen_Signal_Komplex_all_Files[:,0]
X_1 = time_xyz_antennen_Signal_Komplex_all_Files[:,9:41]#[:,4:]#
Y_1 = time_xyz_antennen_Signal_Komplex_all_Files[:,2:5]#[:,1:4]#
normalizer_x = scaler
X_normalize= normalizer_x.fit_transform(X_1)
normalizer_y = scaler
Y_normalize= normalizer_y.fit_transform(Y_1)
X_time_series_normalize = pd.DataFrame(X_normalize)
Y_time_series_normalize=pd.DataFrame(Y_normalize)
print("Shape the Signal Wait...")
X_time_series,Y_time_series= shape_the_Signal(X_Signal_number_train=X_Signal_number,
                                              X_train_time_series_normalize=X_time_series_normalize,
                                              Y_train_time_series_normalize=Y_time_series_normalize,i=0)
print("Shape finished")

path_NN = (r"C:\Users\dauserml\Documents\2020_09_25\Messung_1_Test\Neuronale_Netze_Time_Series_Generator_nur_mit_X\NN_from_20_Oct_2020_14_07\MAPE")

model = tf.keras.models.load_model(path_NN+".h5")

Y_pred_time_series =model.predict(X_time_series)
plt.figure(7)
plt.subplot(2,1,1)
plt.title('X Position')
plt.plot(Y_pred_time_series[:, 0])
plt.plot(Y_time_series[:, 0])
plt.legend(["X_pred","X_true"])
plt.subplot(2,1,2)
plt.plot(Y_pred_time_series[:, 0]-Y_time_series[:, 0])
plt.figure(8)
plt.subplot(2, 1, 1)
plt.title('Y Position')
plt.plot(Y_pred_time_series[:, 1])
plt.plot(Y_time_series[:, 1])
plt.legend(["Y_pred", "Y_true"])
plt.subplot(2, 1, 2)
plt.plot(Y_pred_time_series[:, 1] - Y_time_series[:, 1])
plt.figure(9)
plt.subplot(2, 1, 1)
plt.title('Z Position')
plt.plot(Y_pred_time_series[:, 2])
plt.plot(Y_time_series[:, 2])
plt.legend(["Z_pred", "Z_true"])
plt.subplot(2, 1, 2)
plt.plot(Y_pred_time_series[:, 2] - Y_time_series[:, 2])
plt.show()