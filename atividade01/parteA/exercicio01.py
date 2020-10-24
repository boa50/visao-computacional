# Referencias
# https://www.imageeprocessing.com/2017/11/nearest-neighbor-interpolation.html
# https://gist.github.com/KeremTurgutlu/68feb119c9dd148285be2e247267a203
# https://chao-ji.github.io/jekyll/update/2018/07/19/BilinearResize.html

import numpy as np
import cv2
from cv2 import imread, imshow

#a
def show_img(img, title='image'):
    imshow(title, img)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()

def interpolate_neighbors_loop(size_new, row_positions, col_positions, blue, green, red):
    b = np.zeros((size_new[0], size_new[1]))
    g = np.zeros((size_new[0], size_new[1]))
    r = np.zeros((size_new[0], size_new[1]))
    for i in range(len(row_positions)):
        for j in range(len(col_positions)):
            b[i][j] = blue[row_positions[i]][col_positions[j]]
            g[i][j] = green[row_positions[i]][col_positions[j]]
            r[i][j] = red[row_positions[i]][col_positions[j]]

    return b, g, r

def interpolate_neighbors_vectorized(row_positions, col_positions, blue, green, red):
    b = blue[row_positions, :][:, col_positions]
    g = green[row_positions, :][:, col_positions]
    r = red[row_positions, :][:, col_positions]

    return b, g, r

def interpolate_neighbors(img, size_new, fator, method='vectorized'):
    row_positions = (np.ceil((np.arange(size_new[0]) + 1) / fator) - 1).astype(int)
    col_positions = (np.ceil((np.arange(size_new[1]) + 1) / fator) - 1).astype(int)

    blue = img[:, :, 0]
    green = img[:, :, 1]
    red = img[:, :, 2]

    if method == 'vectorized':
        b, g, r = interpolate_neighbors_vectorized(row_positions, col_positions, blue, green, red)
    elif method == 'loop':
        b, g, r = interpolate_neighbors_loop(size_new, row_positions, col_positions, blue, green, red)
    else:
        print('Método não disponível')
        return

    img_resized = np.zeros((size_new[0], size_new[1], 3), dtype=np.uint8)
    img_resized[:, :, 0] = b
    img_resized[:, :, 1] = g
    img_resized[:, :, 2] = r

    return img_resized

def interpolate_bilinear(img, size_new, fator):
    blue = img[:, :, 0]
    green = img[:, :, 1]
    red = img[:, :, 2]

    b = np.empty(size_new, dtype=np.uint8)

    for row in range(size_new[0]):
        for col in range(size_new[1]):
            x_new = (1/fator) * row
            y_new = (1/fator) * col

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

            A = blue[x1][y1]
            B = blue[x2][y1]
            C = blue[x1][y2]
            D = blue[x2][y2]

            b1 = (1 - dx) * (1 - dy) * A
            b2 = dx * (1 - dy) * B
            b3 = (1 - dx) * dy * C
            b4 = dx * dy * D

            b[row][col] = b1 + b2 + b3 + b4

    return b

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

    fator_reducoes = [1/2, 1/4, 1/8, 1/16]
    fator_aumentos = [2, 4, 8, 16]

    #b
    # img_reduzidas = []

    # for fator in fator_reducoes:
    #     img_reduzidas.append(mudanca_resolucao_path(img_folder + 'lenna.png', fator))

    # for fator, img in zip(fator_aumentos, img_reduzidas):
    #     img_aumentada = mudanca_resolucao(img, fator)
    #     show_img(img_aumentada, title=f'Imagem reduzida pelo fator {fator}')

    #c
    # img_reduzidas = []

    # for fator in fator_reducoes:
    #     img_reduzidas.append(mudanca_resolucao_path(img_folder + 'lenna.png', fator, 'bilinear'))

    # for fator, img in zip(fator_aumentos, img_reduzidas):
    #     img_aumentada = mudanca_resolucao(img, fator, 'bilinear')
    #     show_img(img_aumentada, title=f'Imagem reduzida pelo fator {fator}')

    fator = 0.5
    test = mudanca_resolucao_path(img_folder + 'lenna.png', fator, 'bilinear')
    show_img(test, title=f'Imagem reduzida pelo fator {fator}')