import numpy as np
import cv2



def least_squares(img):
    return np.sum(img ** 2)/2


def loss(img, param):
    return least_squares(idct_matrix.dot(img) - y) + param*np.linalg.norm(img, 1)


def gradient(img, param):
    return 2*idct_matrix.T.dot(np.dot(idct_matrix, img) - y) + param*np.sign(img)


def make_dct_matrix(n):

    matrix = np.zeros((n, n))
    for i in range(1, n):
        for j in range(n):
            matrix[i][j] = (2*j + 1) * i*np.pi/(2*n)
        print(i)

    matrix = np.sqrt(2/n) * np.cos(matrix)
    matrix[0] = 1/np.sqrt(n)

    return matrix


def soft_tresh(img, param):
    return np.sign(img) * np.maximum(np.abs(img) - param, 0.)


def gradient_descent(img, learning_rate=0.9, param=0.1, epochs=100):

   # learning_rate = 1 / np.linalg.norm(idct_matrix, ord=2) ** 2

    for epoch in range(epochs):
        img = soft_tresh(img - learning_rate * gradient(img, param), param/learning_rate)

        print('Эпоха: ', epoch, 'Ошибка: ', loss(img, param))

    return img


img_orig = cv2.imread('home128.jpeg', 0)
img_orig = img_orig.astype(np.float64) / 255

size_2d_img = img_orig.shape
size_img = img_orig.flatten().shape[0]

dct_matrix = make_dct_matrix(size_img)


percent = 0.3
part = int(size_img * percent)
sample = np.random.choice(size_img, part, replace=False)


y = img_orig.flatten()[sample]
img = np.zeros(size_img)
img[sample] = y

cv2.imwrite('0.jpeg', img.reshape(size_2d_img) * 255)
img = dct_matrix.dot(img)

idct_matrix = dct_matrix.T
idct_matrix = idct_matrix[sample]

img = gradient_descent(img)

cv2.imwrite('1.jpeg', dct_matrix.T.dot(img * 255).reshape(size_2d_img))
