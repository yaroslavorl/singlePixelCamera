import numpy as np
import torch
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

    patterns = np.random.rand(num_patterns, img_size ** 2)
    patterns = np.where(patterns > 0.5, 1, -1)

    return patterns


def make_sparse_mask(img_size, percent):

    mask = np.random.rand(img_size, img_size)

    return np.where(mask > percent, 0, 1)


def soft_tresh(img, param):
    return torch.sign(img) * torch.maximum(torch.abs(img) - param, torch.tensor(0))


def least_squares(img):
    return torch.sum(img ** 2)/2


def loss(img, param):
    return least_squares(torch.matmul(P, img) - m)/num_patterns + param*torch.norm(img, 1)


def gradient(img, param):
    return 2/num_patterns * torch.matmul(P.T, torch.matmul(P, img) - m) + param*torch.sign(img)


def gradient_descent(img, param=0.01, learning_rate=0.1, epochs=13):

    t = 1
    new_img = img.clone()

    for epoch in range(epochs):

        old_img = img.clone()
        new_img = new_img - learning_rate * gradient(new_img, param)
        img = soft_tresh(new_img, param / learning_rate)

        t_0 = t
        t = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        new_img = img + ((t_0 - 1) / t) * (img - old_img)

        print('Эпоха: ', epoch)

    return img


num_patterns = 3000
percent_pixel = 0.1

img_orig = cv2.imread('lena_0.jpeg', 0)
img_orig = img_orig.astype(np.float64) / 255

size_2d_img = img_orig.shape
size_img = img_orig.flatten().shape[0]

print('Заполнение базиса')
dct_matrix = make_dct_matrix(size_img)
print('Базис заполнен')

patterns = make_patterns(size_2d_img[0], num_patterns)
print('Паттерны созданы')
m = single_pixel_detector(img_orig.flatten(), patterns)
img = make_sparse_mask(size_2d_img[0], percent_pixel) * img_orig

cv2.imwrite('0.jpeg', img * 255)

img = dct_matrix.dot(img.flatten())
idct_matrix = dct_matrix.T
P = patterns.dot(idct_matrix)


P = torch.from_numpy(P).to(device=torch.device('cuda'))
img = torch.from_numpy(img).to(device=torch.device('cuda'))
m = torch.from_numpy(m).to(device=torch.device('cuda'))

img = gradient_descent(img)

cv2.imwrite('1.jpeg', dct_matrix.T.dot(img.cpu() * 255).reshape(size_2d_img))

