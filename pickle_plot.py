import pickle
import matplotlib.pyplot as plt
import numpy as np

#path = r"C:\Users\dauserml\Documents\2020_09_25\Messung_1_Test\Neuronale_Netze_Time_Series_Generator_nur_mit_X\NN_from_27_Oct_2020_15_23\MSE"




import random

from itertools import count

import time

import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

from mpl_toolkits import mplot3d

from matplotlib.patches import Circle, PathPatch

from matplotlib.text import TextPath

from matplotlib.transforms import Affine2D

import mpl_toolkits.mplot3d.art3d as art3d



from sklearn.model_selection import train_test_split, TimeSeriesSplit, KFold
from sklearn import preprocessing
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
import time as time_1
import pandas as pd
import os


time_xyz_antennen_Signal_Komplex_all_Files = np.loadtxt(r'C:\Users\dauserml\Documents\2020_11_16\2020_11_16_files_Z_400.csv',delimiter=';')

time_12 = time_xyz_antennen_Signal_Komplex_all_Files[:,1]
X_Signal_number = (time_xyz_antennen_Signal_Komplex_all_Files[:, 0])
X_1 = time_xyz_antennen_Signal_Komplex_all_Files[:,9:41]#[:,9:]#
Y_1 = time_xyz_antennen_Signal_Komplex_all_Files[:,2:5]#[:,2:5

#plt.style.use('fivethirtyeight')

time_values=[]

x_values = []

y_values = []

z_values = []

q_values = []

w_values = []

index = count()


def animate(i):
    size=4
    x = next(index)
    k=0
    sprung = int(140/size)
    for t in range(0,sprung):
        time_values.append(time_12[t+sprung*x])

        x_values.append(Y_1[t+sprung*x,0])

        y_values.append(Y_1[t+sprung*x,1])

        z_values.append(Y_1[t+sprung*x,2])


    #if(X_Signal_number[x]>k):
    #    time_values.pop(0)
    #    x_values.pop(0)
    #    y_values.pop(0)
    #    z_values.pop(0)
    #    plt.cla()
    #    k=k+1

    #ax.plot3D(x_values, y_values, z_values, 'blue', linestyle=':')

    plt.subplot(3,1,1)
    plt.plot(time_values,x_values,'red')
    plt.legend(["X Position"])
    plt.subplot(3, 1, 2)
    plt.plot(time_values,y_values,'blue')
    plt.legend(["Y Position"])
    plt.subplot(3,1,3)
    plt.plot(time_values,z_values,'green')
    plt.legend(["Z Position"])

    time.sleep(1/size)


fig = plt.figure()

#ax = plt.axes(projection='3d')

ani = FuncAnimation(plt.gcf(), animate, 1000)

plt.tight_layout()

plt.show()

#fig_training_x = pickle.load(open(path+"\\MSE_Training_X.pkl","rb"))
#fig_training_y = pickle.load(open(path+"\\MSE_Training_Y.pkl","rb"))
#fig_training_z = pickle.load(open(path+"\\MSE_Training_Z.pkl","rb"))
#fig_test_x = pickle.load(open(path+"\\MSE_Test_X.pkl","rb"))
#fig_test_y = pickle.load(open(path+"\\MSE_Test_Y.pkl","rb"))
#fig_test_z = pickle.load(open(path+"\\MSE_Test_Z.pkl","rb"))
#plt.show()