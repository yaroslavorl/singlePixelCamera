import numpy as np
import cv2


def soft_tresh(img, param):
    return np.sign(img) * np.maximum(np.abs(img) - param, 0.)


def least_squares(img):
    return np.sum(img ** 2)/2


def mse(img, coeff, num_patterns):

    error = 0
    for i in range(num_patterns):
        error += least_squares(idct_matrices[i].dot(img) - y[i]) + coeff*np.linalg.norm(img, 1)

    return error/num_patterns


def gradient(img, img_size, num_patterns, coeff):
    grad = np.zeros(img_size)

    for i in range(num_patterns):
        grad += np.dot(idct_matrices[i].T, idct_matrices[i].dot(img) - y[i]) + coeff*np.sign(img)

    return 2/num_patterns * grad


def make_dct_matrix(n):

    matrix = np.zeros((n, n))
    for i in range(1, n):
        for j in range(n):
            matrix[i][j] = (2*j + 1) * i*np.pi/(2*n)
            
    matrix = np.sqrt(2/n) * np.cos(matrix)
    matrix[0] = 1/np.sqrt(n)


    return matrix


def make_measurements_patterns(num_patterns, sample, part):

    y_patterns = np.zeros((num_patterns, part))

    for i in range(num_patterns):
        y_patterns[i] = img_orig.flatten()[sample[i]]

    return y_patterns


def make_measurements_matrix(num_patterns, size_img, sample, y_patterns, part):

    img_patterns = np.zeros((num_patterns, size_img))
    img_dct = np.zeros((num_patterns, size_img))
    idct_matrices = np.zeros((num_patterns, part, size_img))

    for i in range(num_patterns):
        img_patterns[i][sample[i]] = y_patterns[i]

        img_dct[i] = dct_matrix.dot(img_patterns[i])
        idct_matrices[i] = idct_matrix[sample[i]]

    return idct_matrices, img_patterns[0]




def gradient_descent(img, img_size, num_patterns, coeff=0.01, learning_rate=0.9, epochs=15):

    t = 1
    new_img = img.copy()

    for epoch in range(epochs):

        old_img = img.copy()
        new_img = new_img - learning_rate * gradient(new_img, img_size, num_patterns, coeff)
        img = soft_tresh(new_img, coeff/learning_rate)

        t_0 = t
        t = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        new_img = img + ((t_0 - 1) / t) * (img - old_img)

        print('Эпоха: ', epoch, 'Ошибка: ', mse(img, coeff, num_patterns))

    return img


num_patterns = 50
img_orig = cv2.imread('home.jpeg', 0)
img_orig = img_orig.astype(np.float64) / 255

size_2d_img = img_orig.shape
size_img = img_orig.flatten().shape[0]

dct_matrix = make_dct_matrix(size_img)
idct_matrix = dct_matrix.T

percent = 0.1
part = int(size_img * percent)
sample = np.random.choice(size_img, (num_patterns, part))

y = make_measurements_patterns(num_patterns, sample, part)
idct_matrices, img = make_measurements_matrix(num_patterns, size_img, sample, y, part)

img = gradient_descent(img, size_img, num_patterns)

cv2.imwrite('test.jpeg', dct_matrix.T.dot(img * 255).reshape(size_2d_img))
