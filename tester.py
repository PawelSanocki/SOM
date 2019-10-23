import numpy as np
import cv2
from scipy.io import loadmat
a = dict()
img = np.array(loadmat("C:\\Users\\sanoc\\OneDrive\\Pulpit\\Salinas.mat", mdict = a))
print(a[0].shape)