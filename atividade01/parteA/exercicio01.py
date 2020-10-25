# Referencias
# https://www.imageeprocessing.com/2017/11/nearest-neighbor-interpolation.html
# https://gist.github.com/KeremTurgutlu/68feb119c9dd148285be2e247267a203
# https://chao-ji.github.io/jekyll/update/2018/07/19/BilinearResize.html

import numpy as np
import cv2
from cv2 import imread
import matplotlib.pyplot as plt

#a
def show_img(img, title='image'):
    img = img[:, :, ::-1]
    plt.imshow(img)
    plt.title(title)
    plt.axis(False)
    plt.show()

def interpolate_neighbors_loop(size_new, row_positions, col_positions, img):
    img_resized = np.empty((size_new[0], size_new[1], 3), dtype=np.uint8)

    for i in range(len(row_positions)):
        for j in range(len(col_positions)):
            img_resized[i][j] = img[row_positions[i]][col_positions[j]]

    return img_resized

def interpolate_neighbors_vectorized(row_positions, col_positions, img):
    return img[row_positions, :][:, col_positions]

def interpolate_neighbors(img, size_new, fator, method='vectorized'):
    row_positions = (np.ceil((np.arange(size_new[0]) + 1) / fator) - 1).astype(int)
    col_positions = (np.ceil((np.arange(size_new[1]) + 1) / fator) - 1).astype(int)

    if method == 'vectorized':
        return interpolate_neighbors_vectorized(row_positions, col_positions, img)
    elif method == 'loop':
        return interpolate_neighbors_loop(size_new, row_positions, col_positions, img)
    else:
        print('Método não disponível')
        return

def interpolate_bilinear(img, size_new, fator):
    img_resized = np.empty((size_new[0], size_new[1], 3), dtype=np.uint8)

    for row in range(size_new[0]):
        for col in range(size_new[1]):
            x_new = (1/fator) * col
            y_new = (1/fator) * row

            x1 = np.floor(x_new).astype(int)
            y1 = np.floor(y_new).astype(int)
            x2 = np.ceil(x_new).astype(int)
            y2 = np.ceil(y_new).astype(int)

            #limite final das posições da imagem
            if x2 == img.shape[:2][0]:
                x2 -= 1
            if y2 == img.shape[:2][1]:
                y2 -= 1

            dx = x_new - x1
            dy = y_new - y1

            A = img[y1][x1]
            B = img[y1][x2]
            C = img[y2][x1]
            D = img[y2][x2]

            px1 = (1 - dx) * (1 - dy) * A
            px2 = dx * (1 - dy) * B
            px3 = (1 - dx) * dy * C
            px4 = dx * dy * D

            img_resized[row][col] = px1 + px2 + px3 + px4

    return img_resized

def mudanca_resolucao(img, fator, method='vizinho'):
    size_original = np.array(img.shape[:2])
    size_new = (size_original * fator).astype(int)

    img_resized = None

    if method == 'vizinho':
        img_resized = interpolate_neighbors(img, size_new, fator)
    elif method == 'bilinear':
        img_resized = interpolate_bilinear(img, size_new, fator)
    else:
        print('Método não disponível')
        pass

    return img_resized

def mudanca_resolucao_path(img_path, fator, method='vizinho'):
    img = imread(img_path)

    return mudanca_resolucao(img, fator, method)

if __name__ == "__main__":
    img_folder = 'atividade01/parteA/imagens/'
    img_path = img_folder + 'lenna.png'

    fatores = [2, 4, 8, 16]

    ### b
    # img_reduzidas = []

    # print(f'Tamanho original da imagem {imread(img_path).shape[:2]}')

    # for fator in fatores:
    #     fator = 1/fator
    #     img_reduzidas.append(mudanca_resolucao_path(img_path, fator, 'vizinho'))

    # for fator, img in zip(fatores, img_reduzidas):
    #     img_aumentada = mudanca_resolucao(img, fator, 'vizinho')
    #     show_img(img_aumentada, title=f'Recuperação de imagem reduzida pelo fator {fator}, tamanho reduzido = {img.shape[:2]}')

    ### c
    # img_reduzidas = []

    # print(f'Tamanho original da imagem {imread(img_path).shape[:2]}')

    # for fator in fatores:
    #     fator = 1/fator
    #     img_reduzidas.append(mudanca_resolucao_path(img_path, fator, 'bilinear'))

    # for fator, img in zip(fatores, img_reduzidas):
    #     img_aumentada = mudanca_resolucao(img, fator, 'bilinear')
    #     show_img(img_aumentada, title=f'Recuperação de imagem reduzida pelo fator {fator}, tamanho reduzido = {img.shape[:2]}')


    # fator = 2
    # test = mudanca_resolucao_path(img_folder + 'lenna.png', 1/fator, 'vizinho')
    # # test = mudanca_resolucao(test, fator, 'bilinear')
    # show_img(test, title=f'Imagem reduzida pelo fator {fator}')