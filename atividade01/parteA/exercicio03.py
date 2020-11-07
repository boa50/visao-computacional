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

def converte_dominio_frequencia(filtro, tamanho=None):
    if tamanho == None:
        return np.fft.fft2(filtro)
    else:
        return np.fft.fftn(filtro, s=tamanho)

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

def img_normalize(img):
    return ((img - img.min()) * (1/(img.max() - img.min()) * 255)).astype('uint8')

def get_modulo(filtro):
    ### (módulo = raiz quadrada da parte real ao quadrado mais parte imaginária ao quadrado)
    ### (o ângulo de fase 
    ##### [cosseno do ângulo = parte real dividido pelo módulo, 
    ##### seno do ângulo = parte imaginária dividida pelo ângulo])
    ### (aula 03 2:10)

    modulo = shift_freq_center(filtro)
    ### cálculo do módulo
    modulo = np.sqrt(modulo.real**2 + modulo.imag**2)
    ### normalização entre 0 e 255
    modulo = img_normalize(modulo)

    return modulo

if __name__ == "__main__":
    img_folder = 'atividade01/parteA/imagens/'

    h1 = (1/25) * np.ones((5,5))

    h2 = np.array([
        [ 1,  2,  1],
        [ 0,  0,  0],
        [-1, -2, -1]])

    filtros = [h1, h2]

    ### a
    # imgs = []
    # for filtro in filtros:
    #     filtro_freq = converte_dominio_frequencia(filtro, tamanho=(256, 256))
    #     filtro_mod = get_modulo(filtro_freq)
    #     imgs.append(filtro_mod)

    # titles = ['Módulo de h1', 'Módulo de h2']

    # show_img(imgs, titles, rgb=False)

    ### b 
    ### (aplicação dos filtros = multiplica os dois no domínio da frequência depois calcula a transformada inversa)
    ### (a multiplicação deve ser pixel a pixel, usar o parâmetro s (shape))
    ### (aula 03 1:04)
    img_paths = [img_folder + 'minhas/floresta.jpg',
                img_folder + 'minhas/montanha.jpg',
                img_folder + 'minhas/praia.jpg'] 

    for path in img_paths:
        imgs = []
        img = imread(path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shape = img.shape[:2]
        
        img_freq = converte_dominio_frequencia(img)
        
        for filtro in filtros:
            filtro_freq = converte_dominio_frequencia(filtro, tamanho=shape)
            # filtro_freq = filtro_freq[:, :, None] * np.ones(1, dtype=int)[None, None, :]
            
            img_new = np.empty((256, 256, 3), dtype=np.complex128)
            for c in range(3):
                conv_freq = img_freq[:, :, c] * filtro_freq
                img_new[:, :, c] = np.fft.ifft2(conv_freq)

            # conv_freq = img_freq * filtro_freq
            # img_new = np.fft.ifft2(conv_freq)

            # img = img_normalize(img_new)
            img = get_modulo(img_new)
            
            imgs.append(img)

        show_img(imgs, ['', ''], rgb=False)