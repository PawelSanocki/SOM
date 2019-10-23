import numpy as np
import tensorflow as tf
import os
from os.path import join
import cv2
import spectral.io.envi as envi
from spectral import open_image


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Som:
    def __init__(self, dim_x = 2, dim_y = 3, dim_z = 4, input_dim = 3, learn_iter = 10, learning_rate = 0.01, sigma = None, time_constant = None):
        self.d_x = dim_x
        self.d_y = dim_y
        self.d_z = dim_z
        if sigma == None:
            sigma = max(dim_x, dim_y, dim_z) / 2.0
        else:
            sigma = float(sigma)
        if time_constant == None:
            time_constant = learn_iter / np.log(sigma)
        if learning_rate == None:
            learning_rate = float(learning_rate)
        self.input_dim = input_dim
        self._learn_iterations = learn_iter
        self.lr = learning_rate
        self._graph = tf.Graph()
        with self._graph.as_default():
            #tf.enable_eager_execution()
            self._weights = tf.Variable(tf.random.normal(shape = [dim_x, dim_y, dim_z, input_dim], name = "weights"))
            self._input_vec = tf.compat.v1.placeholder("float", [input_dim], name = "input_vec")
            self._learning_iteration = tf.compat.v1.placeholder("float", name = "learning_iteration")
            self.input_tensor = tf.stack([tf.stack([tf.stack([self._input_vec for i in range(dim_z)]) for j in range(dim_y)]) for k in range(dim_x)])

            distances_weights = tf.math.sqrt(tf.math.reduce_sum(tf.math.pow((self._weights - self.input_tensor),2.0), axis = 3))
            self.bmu_distance = tf.math.reduce_min(distances_weights)

            #self.bmu_location = self.find_minimum(distances_weights)
            linear_location = tf.math.argmin(
                tf.reshape(
                    tf.reshape(distances_weights, (self.d_x, self.d_y * self.d_z)), (self.d_x * self.d_y * self.d_z, 1)))
            self.bmu_location = [(linear_location // self.d_z // self.d_y) % self.d_x, (linear_location // self.d_z) % self.d_y, linear_location % self.d_z]

            sigma_t = (sigma * tf.math.exp(tf.math.negative((self._learning_iteration / time_constant))))
            learning_rate_t = (learning_rate * tf.math.exp(tf.math.negative((self._learning_iteration / time_constant))))

            bmu_influence = tf.stack([tf.math.exp(tf.math.negative(tf.math.divide(tf.math.pow(distances_weights,2), tf.math.multiply(tf.math.pow(sigma_t, 2), 2.0)))) for i in range(input_dim)], axis = 3)
            
            new_weights = (self._weights + ((bmu_influence * learning_rate_t) * (self.input_tensor - self._weights)))
            self._training_op = tf.compat.v1.assign(self._weights, new_weights)
            
            self._sess = tf.compat.v1.Session()
            init_op = tf.compat.v1.global_variables_initializer()

            self._sess.run(init_op)
            self.saver = tf.compat.v1.train.Saver({"weights": self._weights})
            print("som created")


    def train(self, input_vecs):
        for iter in range(self._learn_iterations):
            for input_vec in input_vecs:
                self._sess.run(self._training_op, feed_dict={self._input_vec: input_vec, self._learning_iteration: iter})
    def convert(self, input_vec):
        return self._sess.run(self.bmu_location, feed_dict={self._input_vec: input_vec})
    def show_bmu_distance(self, input_vec):
        return self._sess.run(self.bmu_distance, feed_dict={self._input_vec: input_vec})
    def find_minimum(self, tensor):
        coords = [0,0,0]
        for i in self.gen_spots():
            coords = tf.cond( tf.less(tensor[i], tensor[coords]), lambda: i, lambda: coords)
        return coords
    def gen_spots(self):
        for i in range(self.d_x):
            for j in range(self.d_y):
                for k in range(self.d_z):
                    yield [i, j, k]
    def find_minimum_fast(self, tensor, bmu_candidats):
        coords = [0,0,0]
        for i in range(self.d_x):
            for vect in bmu_candidats:
                print (vect)
                if tensor[coords] > tensor[i,vect[0], vect[1]]:
                    coords = vect
        return coords
    def train_image(self):
        path = "C:\\Users\\sanoc\\OneDrive\\Pulpit\\SomTensorflow\\data"
        filenames = [join(path, name) for name in os.listdir(path)]
        for iter in range(self._learn_iterations):
            for filename in filenames:
                img = cv2.imread(filename)
                for i in self.generator_image(img.shape[0], img.shape[1]):
                    input_vec = img[i]
                    self._sess.run(self._training_op, feed_dict={self._input_vec: input_vec, self._learning_iteration: iter})
    def convert_image(self, img):
        converted_img = np.zeros((img.shape[0], img.shape[1], 3))
        for i in self.generator_image(img.shape[0], img.shape[1]):
            if (i[0]%100 == 0 and i[1] == 0):
                print("Progress: " + str(np.floor(100.0 * i[0] / img.shape[0])) + "%")
            input_vec = img[i]
            a = self._sess.run(self.bmu_location, feed_dict={self._input_vec: input_vec})
            converted_img[i] = a
        print("converted")
        return converted_img * int(256 / max(self.d_x, self.d_y, self.d_z))
    def generator_image(self, cols, rows):
        for i in range(cols):
            for j in range(rows):
                yield i, j
    def save_weights(self):
        self.saver.save(self._sess, "model\\model.ckpt")
        print("saved")
    def load_weights(self):
        self.saver.restore(self._sess, "model\\model.ckpt")
        print("Loaded")
    def train_with_threshold_rgb(self, threshold):
        path = "C:\\Users\\sanoc\\OneDrive\\Pulpit\\SomTensorflow\\data"
        filenames = [join(path, name) for name in os.listdir(path)]
        centroids = np.zeros((1, self.input_dim))
        for filename in filenames:
            img = cv2.imread(filename)
            for i in self.generator_image(img.shape[0], img.shape[1]):
                input_vec = img[i]
                if (np.min(np.linalg.norm(input_vec - centroids[range(centroids.shape[0])])) > threshold):
                    np.append(centroids, input_vec)
        for iter in range(self._learn_iterations):
            for input_vec in centroids:
                self._sess.run(self._training_op, feed_dict={self._input_vec: input_vec, self._learning_iteration: iter})
    def train_with_threshold_hyperspectral(self, threshold, path):
        filenames = [join(path, name) for name in os.listdir(path)]
        centroids = np.zeros((1, self.input_dim))
        for filename in filenames:
            if filename.find(".lan") == -1: #hdr
                continue
            #img = envi.open(filename)
            img = open_image(filename)
            if img.shape[2] != self.input_dim:
                continue
            for i in self.generator_image(img.shape[0], img.shape[1]):
                input_vec = img[i]
                if (np.min(np.linalg.norm(input_vec - centroids[range(centroids.shape[0])])) > threshold):
                    np.append(centroids, input_vec)
        for iter in range(self._learn_iterations):
            if (iter%1000 == 0):
                print("Iteration: " + str(iter//1000) + " / " + str(self._learn_iterations//1000))
            for input_vec in centroids:
                self._sess.run(self._training_op, feed_dict={self._input_vec: input_vec, self._learning_iteration: iter})
        print("training done")
