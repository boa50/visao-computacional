import numpy as np
import cv2
from cv2 import imread, imshow

#a
def show_img(img):
    imshow('image',img)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()

def mudanca_resolucao(img_path, fator):
    img = imread(img_path)
    show_img(img)

if __name__ == "__main__":
    mudanca_resolucao('atividade01/parteA/imagens/lenna.png', 'b')
