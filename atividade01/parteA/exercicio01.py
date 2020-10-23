# Referencias
# https://www.imageeprocessing.com/2017/11/nearest-neighbor-interpolation.html
# https://gist.github.com/KeremTurgutlu/68feb119c9dd148285be2e247267a203

import numpy as np
import cv2
from cv2 import imread, imshow

#a
def show_img(img, title='image'):
    imshow(title, img)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()

def interpolate_loop(size_new, row_positions, col_positions, blue, green, red):
    b = np.zeros((size_new[0], size_new[1]))
    g = np.zeros((size_new[0], size_new[1]))
    r = np.zeros((size_new[0], size_new[1]))
    for i in range(len(row_positions)):
        for j in range(len(col_positions)):
            b[i][j] = blue[row_positions[i]][col_positions[j]]
            g[i][j] = green[row_positions[i]][col_positions[j]]
            r[i][j] = red[row_positions[i]][col_positions[j]]

    return b, g, r

def interpolate_vectorized(row_positions, col_positions, blue, green, red):
    b = blue[row_positions, :][:, col_positions]
    g = green[row_positions, :][:, col_positions]
    r = red[row_positions, :][:, col_positions]

    return b, g, r

def interpolate(img, size_new, row_positions, col_positions, method='vectorized'):
    blue = img[:, :, 0]
    green = img[:, :, 1]
    red = img[:, :, 2]

    if method == 'vectorized':
        b, g, r = interpolate_vectorized(row_positions, col_positions, blue, green, red)
    elif method == 'loop':
        b, g, r = interpolate_loop(size_new, row_positions, col_positions, blue, green, red)
    else:
        print('Método não disponível')
        return

    img_resized = np.zeros((size_new[0], size_new[1], 3), dtype=np.uint8)
    img_resized[:, :, 0] = b
    img_resized[:, :, 1] = g
    img_resized[:, :, 2] = r

    return img_resized

def mudanca_resolucao(img, fator, method='vizinho'):
    size_original = np.array(img.shape[:2])
    size_new = (size_original * fator).astype(int)

    img_resized = None

    if method == 'vizinho':
        row_positions = (np.ceil((np.arange(size_new[0]) + 1) / fator) - 1).astype(int)
        col_positions = (np.ceil((np.arange(size_new[1]) + 1) / fator) - 1).astype(int)

        img_resized = interpolate(img, size_new, row_positions, col_positions)
    elif method == 'bilinear':
        print('Método não implementado')
        pass
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
    img_reduzidas = []

    for fator in fator_reducoes:
        img_reduzidas.append(mudanca_resolucao_path(img_folder + 'lenna.png', fator, 'bilinear'))

    for fator, img in zip(fator_aumentos, img_reduzidas):
        img_aumentada = mudanca_resolucao(img, fator, 'bilinear')
        show_img(img_aumentada, title=f'Imagem reduzida pelo fator {fator}')
