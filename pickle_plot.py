import pickle
import matplotlib.pyplot as plt


from pandas import DataFrame
from pandas import concat

# create sequence
length = 10
sequence = [i/float(length) for i in range(length)]
# create X/y pairs
df = DataFrame(sequence)
df = concat([df, df.shift(1)], axis=1)
df.dropna(inplace=True)
# convert to LSTM friendly format
values = df.values
X, y = values[:, 0], values[:, 1]
X = X.reshape(len(X), 1, 1)
# configure network
# configure network
n_batch = len(X)
n_epoch = 1000
n_neurons = 10
# design network


#path = r"C:\Users\dauserml\Documents\2020_09_25\Messung_1_Test\Neuronale_Netze_Time_Series_Generator_nur_mit_X\NN_from_27_Oct_2020_14_55\MSE"

#fig_training_x = pickle.load(open(path+"\\MSE_Training_X.pkl","rb"))
#fig_training_y = pickle.load(open(path+"\\MSE_Training_Y.pkl","rb"))
#fig_training_z = pickle.load(open(path+"\\MSE_Training_Z.pkl","rb"))
#fig_test_x = pickle.load(open(path+"\\MSE_Test_X.pkl","rb"))
#fig_test_y = pickle.load(open(path+"\\MSE_Test_Y.pkl","rb"))
#fig_test_z = pickle.load(open(path+"\\MSE_Test_Z.pkl","rb"))
#plt.show()