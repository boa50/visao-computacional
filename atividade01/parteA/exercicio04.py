import numpy as np
import cv2
from cv2 import imread, getGaussianKernel
import matplotlib.pyplot as plt

from exercicio03 import *

if __name__ == "__main__":
    img_folder = 'atividade01/parteA/imagens/'

    img_paths = [img_folder + 'noiseball.png',
                img_folder + 'footBallOrig.png']
    
    img_shape = imread(img_paths[0], 0).shape

    ### a
    titles = ['Noiseball', 'Módulo de noiseball', 'FootBallOrig', 'Módulo de footBallOrig']
    for i, path in enumerate(img_paths):
        imgs = []
        img = imread(path, 0)
        imgs.append(img)

        img_freq = converte_dominio_frequencia(img, (36, 36))
        img_mod = get_modulo(img_freq)
        imgs.append(img_mod)

        start = 0 + i*2
        end = 2 + i*2
        show_img(imgs, titles[start:end], rgb=False)

    ### b
    filtro_shape = (5,5)
    filtro_elementos = np.prod(filtro_shape)
    filtro_desvio_padrao = 50
    filtro = getGaussianKernel(filtro_elementos, filtro_desvio_padrao).reshape(filtro_shape)
    show_img([filtro], ['Filtro Gaussiano utilizado'], rgb=False)

    noiseball = imread(img_paths[0], 0)
    noiseball = np.expand_dims(noiseball, -1)
    noiseball_filtered = convolve_freq(noiseball, filtro)
    show_img([noiseball_filtered], ['Noiseball filtrada'], rgb=False)