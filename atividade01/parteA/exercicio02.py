import numpy as np
import cv2
from cv2 import imread
import matplotlib.pyplot as plt

### a
def show_img(imgs, titles):
    width = 4 * len(imgs)
    height = 4
    plt.figure(figsize=(width, height))

    for i, (img, title) in enumerate(zip(imgs, titles)):
        img = img[:, :, ::-1]

        plt.subplot(1, len(imgs), i+1)
        plt.imshow(img)
        plt.title(title)
        plt.axis(False)

    plt.show()

def img_padding(img, filter_shape):
    padding = filter_shape[0] - 1
    start = int((filter_shape[0] - 1)/2)
    end = -start

    img_padded = np.zeros(np.array(img.shape) + [padding, padding, 0], dtype=np.uint8)
    img_padded[start:end, start:end, :] = img

    return img_padded

def filtra_espacial(img_path, filtro):
    img = imread(img_path)
    rows = img.shape[0]
    cols = img.shape[1]
    colors = img.shape[2]

    filter_shape = filtro.shape
    filtro = filtro[:, :, None] * np.ones(colors, dtype=int)[None, None, :]

    img_padded = img_padding(img, filter_shape)
    img_filtered = np.empty(img.shape, dtype=np.uint8)

    for y in range(rows):
        for x in range(cols):
            part = img_padded[y:y + filter_shape[0], x:x + filter_shape[1]]
            filtered = np.sum(part * filtro, axis=(0, 1), dtype=np.uint8)
            img_filtered[y, x] = filtered
    
    return img_filtered

if __name__ == "__main__":
    img_folder = 'atividade01/parteA/imagens/'
    img_path = img_folder + 'lenna.png'

    ### b
    # filtro1 = 1/9 * np.array([
    #     [1, 1, 1],
    #     [1, 1, 1],
    #     [1, 1, 1]])

    # filtro2 = 1/16 * np.array([
    #     [1, 2, 1],
    #     [2, 4, 2],
    #     [1, 2, 1]])

    # filtro3 = np.array([
    #     [ 0, -1,  0],
    #     [-1,  5, -1],
    #     [ 0, -1,  0]])

    # filtro4 = np.array([
    #     [-1, -1, -1],
    #     [-1,  8, -1],
    #     [-1, -1, -1]])

    # filtro5 = np.array([
    #     [ 1,  2,  1],
    #     [ 0,  0,  0],
    #     [-1, -2, -1]])

    # filtros = [filtro1, filtro2, filtro3, filtro4, filtro5]

    # imgs = []
    # for filtro in filtros:
    #     imgs.append(filtra_espacial(img_path, filtro))

    # titles = ['Filtro 1', 'Filtro 2', 'Filtro 3', 'Filtro 4', 'Filtro 5']
    # show_img(imgs, titles)

    ### c
    # filtro3x3 = (1/(3*3)) * np.ones((3,3))
    # filtro11x11 = (1/(11*11)) * np.ones((11,11))
    # filtro17x17 = (1/(17*17)) * np.ones((17,17))
    # filtro35x35 = (1/(35*35)) * np.ones((35,35))

    # filtros = [filtro3x3, filtro11x11, filtro17x17, filtro35x35]

    # imgs = []
    # for filtro in filtros:
    #     imgs.append(filtra_espacial(img_path, filtro))

    # titles = ['Filtro 3x3', 'Filtro 11x11', 'Filtro 17x17', 'Filtro 35x35']
    # show_img(imgs, titles)



    # filtro11x11 = (1/(11*11)) * np.ones((11,11))
    # img = filtra_espacial(img_path, filtro11x11)
    # show_img([img], [''])