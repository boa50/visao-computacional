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

if __name__ == "__main__":
    img_folder = 'atividade01/parteA/imagens/'
    img_paths = [img_folder + 'minhas/floresta.jpg',
                img_folder + 'minhas/montanha.jpg',
                img_folder + 'minhas/praia.jpg'] 

    imgs = []
    for path in img_paths:
        imgs.append(imread(path))

    show_img(imgs, ['', '', ''])