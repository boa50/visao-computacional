import numpy as np
import cv2
from cv2 import imread
import matplotlib.pyplot as plt

def show_img(imgs, titles, rgb=True):
    width = 4 * len(imgs)
    height = 4
    plt.figure(figsize=(width, height))

    for i, (img, title) in enumerate(zip(imgs, titles)):
        if rgb:
            img = img[:, :, ::-1]

        plt.subplot(1, len(imgs), i+1)
        plt.imshow(img)
        plt.title(title)
        plt.axis(False)

    plt.show()

### a
def converte_dominio_frequencia(filtro):

    return

def shift_freq_center(filtro):
    center = filtro.shape[0]//2

    row = 0
    filtro_t = np.empty(filtro.shape, dtype=np.complex)
    temp = np.empty(filtro.shape, dtype=np.complex)
    for i in range(center, center + filtro.shape[0]):
        idx = i % filtro.shape[0]
        temp[idx, :] = filtro[row, :]
        row += 1

    col = 0
    for j in range(center, center + filtro.shape[0]):
        idx = j % filtro.shape[0]
        filtro_t[:, idx] = temp[:, col]
        col += 1

    return filtro_t

if __name__ == "__main__":
    img_folder = 'atividade01/parteA/imagens/'

    h1 = (1/25) * np.ones((5,5))

    # h2 = np.array([
    #     [ 1,  2,  1],
    #     [ 0,  0,  0],
    #     [-1, -2, -1]])

    h1f = np.fft.fft2(h1)

    print(h1f)
    print()

    h1f_t = shift_freq_center(h1f)

    print(h1f_t)
    print()

    show_img([h1f_t.astype(int)], [''], rgb=False)

    ### b
    # img_paths = [img_folder + 'minhas/floresta.jpg',
    #             img_folder + 'minhas/montanha.jpg',
    #             img_folder + 'minhas/praia.jpg'] 

    # imgs = []
    # for path in img_paths:
    #     imgs.append(imread(path))

    # show_img(imgs, ['', '', ''])