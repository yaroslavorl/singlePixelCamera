import numpy as np
import cv2


def single_pixel_detector(img, pattern):
    """Simulation single pixel detector

    :param img: 1-D image vector
    :param pattern: 1-D pattern vector
    :return: scalar
    """
    return pattern.dot(img)


def make_dct_matrix(n):

    matrix = np.zeros((n, n))
    for i in range(1, n):
        for j in range(n):
            matrix[i][j] = (2*j + 1) * i*np.pi/(2*n)

    matrix = np.sqrt(2/n) * np.cos(matrix)
    matrix[0] = 1/np.sqrt(n)

    return matrix


def make_patterns(img_size, num_patterns):

    patterns = np.random.rand(num_patterns, img_size, img_size)
    patterns = np.where(patterns > 0.1, 0, 1)

    return patterns.reshape(num_patterns, img_size ** 2)


def make_sparse_mask(img_size, percent):
    mask = np.random.rand(img_size, img_size)

    return np.where(mask > percent, 0, 1)


def least_squares(img):
    return np.sum(img ** 2)/2


def loss(img, param):
    return least_squares(P.dot(img) - m) + param*np.linalg.norm(img, 1)


def gradient(img, param):
    return 2/num_patterns * P.T.dot(np.dot(P, img) - m) + param*np.sign(img)


def gradient_descent(img, param=0.05, learning_rate=0.01, epochs=10000):

    for epoch in range(epochs):
        img = img - learning_rate * gradient(img, param)

        print('Эпоха: ', epoch, 'Ошибка: ', loss(img, param))

    return img


num_patterns = 100
percent_pixel = 0.1
img_orig = cv2.imread('test.jpeg', 0)
img_orig = img_orig.astype(np.float64) / 255

size_2d_img = img_orig.shape
size_img = img_orig.flatten().shape[0]

dct_matrix = make_dct_matrix(size_img)

patterns = make_patterns(size_2d_img[0], num_patterns)
m = single_pixel_detector(img_orig.flatten(), patterns)
img = make_sparse_mask(size_2d_img[0], percent_pixel) * img_orig

cv2.imwrite('0.jpeg', img * 255)

img = dct_matrix.dot(img.flatten())
idct_matrix = dct_matrix.T
P = patterns.dot(idct_matrix)

img = gradient_descent(img)
cv2.imwrite('1.jpeg', dct_matrix.T.dot(img * 255).reshape(size_2d_img))
