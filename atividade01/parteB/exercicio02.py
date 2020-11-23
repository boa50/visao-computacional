import numpy as np
import cv2
from cv2 import imread
import matplotlib.pyplot as plt

from exercicio01 import show_img

if __name__ == "__main__":
    img_folder = 'atividade01/parteB/imagens/'
    img_path = img_folder + 'blocks.png'
    img = imread(img_path, 0)

    ### (a)
    # Que fração dos pixels da imagem são brancos?
    ones = np.sum(img)/255
    total = np.prod(img.shape)
    fracao = (ones/total) * 100
    print('Há {:.2f}% de pixels brancos na imagem.'.format(fracao))
    
    # Quantos objetos a imagem possui?
    ### hierarchy [Next, Previous, First_Child, Parent]
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    object_count = 0
    for obj in hierarchy[0]:
        if obj[-1] == -1:
            object_count += 1

    print('A imagem possui {} objetos.'.format(object_count))


    ### (b)
    # Quantos buracos há na imagem? 
    hole_count = 0
    for obj in hierarchy[0]:
        if obj[-1] != -1:
            hole_count += 1

    print('A imagem possui {} buracos.'.format(hole_count))
    
    
    # Quantos objetos têm mais de um buraco?
    obj_hole_count = {}
    for obj in hierarchy[0]:
        if obj[-1] != -1:
            idx = obj[-1]
            if idx in obj_hole_count:
                obj_hole_count[idx] += 1
            else:
                obj_hole_count[idx] = 1

    many_holes_count = 0
    for key in obj_hole_count.keys():
        if obj_hole_count[key] > 1:
            many_holes_count += 1

    print('Na imagem {} objetos possuem mais de um buraco.'.format(many_holes_count))
    
    
    ### (c)
    # Quantos quadrados e círculos a imagem possui?
    quadrados = 0
    circulos = 0
    for contour, obj in zip(contours, hierarchy[0]):
        if obj[-1] == -1:
            if len(contour) == 4:
                quadrados += 1
            else:
                circulos += 1

    print('A imagem possui {} quadrados e {} círculos.'.format(quadrados, circulos))
    
    
    ### (d)
    # Identifique os quadrados que possuem buracos e os círculos que não possuem buracos?
    img = imread(img_path)

    obj_has_hole = {}
    for i, obj in enumerate(hierarchy[0]):
        if obj[-1] != -1:
            idx = obj[-1]
            obj_has_hole[idx] = 1
        else:
            obj_has_hole[i] = -1

    for idx, contour in enumerate(contours):
        plot = False
        
        if idx in obj_has_hole.keys():
            if len(contour) == 4:
                if obj_has_hole[idx] == 1:
                    plot = True
                    color = (0, 255, 0) 
            else:
                if obj_has_hole[idx] == -1:
                    plot = True
                    color = (0, 0, 255) 

        if plot:
            start_point = tuple(contour.min(axis=0)[0])
            end_point = tuple(contour.max(axis=0)[0])
            thickness = 2
            img = cv2.rectangle(img, start_point, end_point, color, thickness)


    show_img([img], ['Quadrados com buracos (em verde) e círculos sem buracos (em vermelho)'], rgb=True)