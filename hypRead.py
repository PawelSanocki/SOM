import spectral.io.envi as envi
import numpy as np
import MySOM
import cv2
from spectral import open_image
from os.path import join

def segmentImage(folderPath, imageFile, n_iter, learn_rate, threshold, dx = 8, dy = 8, dz = 8):
    #image_path = "C:\\Users\\sanoc\\OneDrive\\Pulpit\\f080806t01p00r15.tar\\f080806t01p00r15\\f080806t01p00r15rdn_c_obs.hdr"
    #img = envi.open(image_path) # image to be converted
    img = open_image(folderPath + "\\" + imageFile + ".lan")
    #print("file opened")
    #print(img.shape)
    my_som = MySOM.Som(dim_x = dx,dim_y = dy,dim_z = dz,input_dim = img.shape[2],learning_rate = learn_rate, learn_iter = n_iter)
    #my_som.load_weights()
    #path = "C:\\Users\\sanoc\\OneDrive\\Pulpit\\f080806t01p00r15.tar\\f080806t01p00r15" # folder with images 10-band
    my_som.train_with_threshold_hyperspectral(threshold, folderPath)
    #my_som.save_weights()
    result = my_som.convert_image(img)
    check = cv2.imwrite(join("Wyniki", imageFile, str(my_som.d_x) + str(my_som.d_x) + str(my_som.d_x) + "lr" + str(int(1//my_som.lr)) + "li" + str(my_som._learn_iterations) + ".png"),result)
    cv2.imwrite("Wyniki\\wynik.png", result)
    print(check)
    cv2.imshow(str(my_som.d_x) + str(my_som.d_x) + str(my_som.d_x) + "lr" + str(int(1//my_som.lr)) + "li" + str(my_som._learn_iterations), result)
    cv2.waitKey()

