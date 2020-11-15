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
            cmap = None
            img = img[:, :, ::-1]
        else:
            cmap = 'gray'

        plt.subplot(1, len(imgs), i+1)
        plt.imshow(img, cmap=cmap)
        plt.title(title)
        plt.axis(False)

    plt.show()

def global_thresholding(img, threshold=150, limite=0.5):
    iteracoes = 0

    while True:
        iteracoes += 1
        
        G1 = img[img > threshold]
        G2 = img[img <= threshold]

        if len(G1) == 0:
            G1 = np.array([0])
        if len(G2) == 0:
            G2 = np.array([0])

        m1 = G1.mean()
        m2 = G2.mean()

        threshold_new = (m1 + m2) / 2

        if abs(threshold_new - threshold) < limite:
            break
        else:
            threshold = threshold_new

    new_img = (img > threshold_new)

    return new_img, iteracoes, threshold_new

if __name__ == "__main__":
    img_folder = 'atividade01/parteB/imagens/'
    img_path = img_folder + 'fingerPrint.png'

    img = imread(img_path, 0)

    limite = 0.5

    thresholds = [0, 50, 100, 127, 150, 200, 255]

    for threshold in thresholds:
        new_img, iteracoes, threshold_new = global_thresholding(img, threshold=threshold, limite=limite)
        print(f'Para um threshold inicial de {threshold} foram necessárias {iteracoes} iterações para chegar ao fim.')

    show_img([new_img], [f'Imagem binarizada com threshold final de {int(threshold_new)}'], rgb=False)