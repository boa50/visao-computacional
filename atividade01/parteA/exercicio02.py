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

def filtra_espacial(img_path, filtro, average_filter=False):
    img = imread(img_path)
    rows = img.shape[0]
    cols = img.shape[1]
    colors = img.shape[2]

    filtro = np.rot90(filtro, 2)
    filter_shape = filtro.shape
    filtro = filtro[:, :, None] * np.ones(colors, dtype=int)[None, None, :]

    img_padded = img_padding(img, filter_shape)
    img_filtered = np.empty(img.shape)

    for y in range(rows):
        for x in range(cols):

            ### Remover as bordas pretas dos filtros grandes
            y_diff = y - filter_shape[0]//2
            x_diff = x - filter_shape[1]//2

            if y_diff <= 0:
                y_start = filter_shape[0]//2 + 1
            else:
                y_start = y

            y_end = y + filter_shape[0]
            y_end = min(y_end, rows)

            if x_diff <= 0:
                x_start = filter_shape[1]//2 + 1
            else:
                x_start = x

            x_end = x + filter_shape[1]
            x_end = min(x_end, cols)

            part = img_padded[y_start:y_end, x_start:x_end]

            y_filter_diff = y_end - y_start
            x_filter_diff = x_end - x_start

            if average_filter:
                filter_norm = y_filter_diff * x_filter_diff 
            else:
                filter_norm = 1

            filtered = np.sum(part * filtro[:y_filter_diff, :x_filter_diff, :] / filter_norm, axis=(0, 1))

            img_filtered[y, x] = filtered
    
    ### Remover valores extremos
    img_filtered = img_filtered.clip(min=0, max=255).astype('uint8')

    return img_filtered

if __name__ == "__main__":
    img_folder = 'atividade01/parteA/imagens/'
    img_path = img_folder + 'lenna.png'
    img_original = imread(img_path)

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

    # imgs = [img_original]
    # for filtro in filtros:
    #     imgs.append(filtra_espacial(img_path, filtro))

    # titles = ['Original', 'Filtro 1', 'Filtro 2', 'Filtro 3', 'Filtro 4', 'Filtro 5']
    # show_img(imgs, titles)

    ### c
    # filtro3x3 = np.ones((3,3))
    # filtro11x11 = np.ones((11,11))
    # filtro17x17 = np.ones((17,17))
    # filtro35x35 = np.ones((35,35))

    # filtros = [filtro3x3, filtro11x11, filtro17x17, filtro35x35]

    # imgs = [img_original]
    # for filtro in filtros:
    #     imgs.append(filtra_espacial(img_path, filtro, average_filter=True))

    # titles = ['Original', 'Filtro 3x3', 'Filtro 11x11', 'Filtro 17x17', 'Filtro 35x35']
    # show_img(imgs, titles)


    ### testes
    # filtro11x11 = np.ones((11,11))
    # meu = filtra_espacial(img_path, filtro11x11, average_filter=True)
    # filtro = np.array([
    #     [ 1,  2,  1],
    #     [ 0,  0,  0],
    #     [-1, -2, -1]])
    # meu = filtra_espacial(img_path, filtro)

    # open_cv = cv2.blur(img_original, (35,35))
    # open_cv = cv2.filter2D(img_original, 0, filtro)
    # show_img([meu, open_cv], ['meu', 'opencv'])

    # imgs = [img_original]
    # # for filtro in filtros:
    #     # imgs.append(cv2.filter2D(img_original, 0, filtro))
    # imgs.append(cv2.blur(img_original, (3,3)))
    # imgs.append(cv2.blur(img_original, (11,11)))
    # imgs.append(cv2.blur(img_original, (17,17)))
    # imgs.append(cv2.blur(img_original, (35,35)))

    # titles = ['Original', 'Filtro 3x3', 'Filtro 11x11', 'Filtro 17x17', 'Filtro 35x35']
    # show_img(imgs, titles)