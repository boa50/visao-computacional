import numpy as np
import os
import os.path
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from sklearn.cluster import KMeans, MiniBatchKMeans
import random
import imutils

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, Dense, Dropout, LeakyReLU, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

from tensorflow.keras.utils import to_categorical
from tensorflow.compat.v1 import ConfigProto, InteractiveSession

def config_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    sess = InteractiveSession(config=config)


def load_image_pair(file_name, path='atividade03/cityscapes_data/cityscapes_data/train',
             rotation=0.0, flip=False, size=(256, 200)):
    
    # Path da imagem do dataset de treino ou validação. 
    # Por default o parâmetro está pegando o path de treino.
    image_path = os.path.join(path, file_name)
    
    # Neste bloco leia a Imagem neste bloco como RGB numpy array.
    imgs_concat = cv2.imread(image_path)
    imgs_concat = cv2.cvtColor(imgs_concat, cv2.COLOR_BGR2RGB)
    
    # Neste bloco separe imagem de entrada e imagem segmentada.
    width = imgs_concat.shape[1] // 2
    seg = imgs_concat[:, width:, :]
    img = imgs_concat[:, :width, :]
    
    # Este boco irar equalizar os canais RGB da imagem - Já implementado
    for i in range(3):
        zimg = img[:,:,i]
        zimg = cv2.equalizeHist(zimg)
        img[:,:,i] = zimg
    
    # Neste bloco dê um resize nas duas imagens para o size passado como paâmetro na função.
    # Esse será o tamanho da entrada da nossa rede neural.
    img = cv2.resize(img, size)
    seg = cv2.resize(seg, size)
    
    
    # Neste bloco aplique uma rotação na imagem (isso será usado mais a frente em um generator).
    # Por default este parâmetro está com rotação de 0 graus, ou seja, não vai rotacionar a princípio.
    img = imutils.rotate(img, rotation)
    seg = imutils.rotate(seg, rotation)

   
    # Neste bloco as duas imagens sofrem um flip horizontal. Por default está False. Ou seja, não vai flippar,
    # a menos que peçamos para tal. Servirar par ao genrator mais a frente.
    if flip:
        img = img[:,::-1,:]
        seg = seg[:,::-1,:]
    
    return img/255, seg/255 # Ao final já normaliza as duas imagens entre 0 e 1!

def colors_to_class_layers(seg):
    
    # Deixa imagem no formato necessário para passar no Kmeans -- já codificado
    s = seg.reshape((seg.shape[0]*seg.shape[1],3))
    
    # Passa no Kmenas treinado -- já codificado
    s = km.predict(s)
    
    # Pega a imagem gerada pelo agrupamento do Kmeans no format linhas x colunas  --- já codificado
    s = s.reshape((seg.shape[0], seg.shape[1]))
    
    # Pega número de clusters e cria um array-tensor com o número de mapas com a quantidade de 
    # classes (clusters) -- já codificado
    
    n = len(km.cluster_centers_)    
    classes = np.zeros((seg.shape[0], seg.shape[1], n))
    
    # Neste bloco, implemente um for sobre as classes (clusters), crie uma cópia de s e verifique 
    # quais pixels dele tem valor igual ao índice que representa a classe da vez. Quando for igual 
    # atribua o valor 1 a exste pixel e quando não atribua zero. Assim teremso mapas para as classes 
    # sendo imagens binárias (com zeros e uns). Depoois preencha os n mapas no array classes e retorne ele.
    
    for i in range(n):
        copy = np.where(s == i, 1, 0).copy()
        classes[:, :, i] = copy
       
        
    return classes

def layers_to_rgb_image(layers):
    
    # Lista de tuplas RGB para representar as cores das regiões segmentadas. 
    # 13 cores e mais o preto para quando não for nenhuma das classes de interesse
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255),
             (255,255,255), (200,50,0),(50,200,0), (50,0,200), (200,200,50), (0,50,200),
             (0,200,50), (0,0,0)]
    
    # Imagem de saída.
    img_out = np.zeros((layers.shape[0], layers.shape[1], 3))
    
    for i in range(layers.shape[2]): # Percorre as n layers, no caso aqui 13 classes
        
        c = layers[:,:,i] # Pega a layer da vez
        col = colors[i] # Pega uma cor pra essa layer/classe
        
        for j in range(3): # percorre os canais RGB
            img_out[:,:,j] += col[j]*c # Pega a layer binária e multiplica pela cor que ela deverá ter
            
    img_out = img_out/255.0 # Normaliza imagem resultante entre 0 e 1 e, em seguida retorna
    
    return img_out

def Generate(path='atividade03/cityscapes_data/cityscapes_data/train', batch_size=32,
            maxangle=10.0):
    
    # Maxangle é o angulo máximo que a imagem pode ser rotacionada. Por default é 10.
    
    ### PARA TESTES
    # image_files = os.listdir(path)[:64]
    image_files = os.listdir(path)
    
    while True:
        
        # Listas para imagem e versão segmentada
        imgs=[]
        segs_layers=[]
        
        for i in range(batch_size):# For para geração do batch/lote
            
            file = random.sample(image_files, 1)[0] # Sorteia um arquivo entre todos.
            
            # Neste bloco, implemente uma forma de aleatoriamente a imagem da vez ser flippada 
            # na Horizontal ou não, bem como sortear um angulo para rotacionar entre -maxangle e maxangle
            # que, por default, é 10 e pode manter esse angulo máximo para a atividade.
            
            angle = random.uniform(-maxangle, maxangle)
            flip = bool(random.getrandbits(1))
            
            # Usa a função load_image_pair implementada lá em cima
            # recebe o arquivo da vez, o path de treinamento ou validação, o angulo
            # para rotacionar e a flag flip para flippar na horizontal ou não.
            img, seg = load_image_pair(file, path, rotation=angle, flip=flip)
            
            # Retora o mapa de layers da imagem segmentada
            seg_layers = colors_to_class_layers(seg)
            
            imgs.append(img)
            segs_layers.append(seg_layers)
            
            
        # Retorna (yield) imagens com augmentation ou não, bem como sua versão segmentada no formato de layers
        yield np.array(imgs), np.array(segs_layers) 

def plot_curvas(dados, labels, title):
    fig = plt.figure(figsize=[16,9])
    ax = fig.add_subplot(111)

    ax.plot(range(1, len(dados[0]) + 1), dados[0], c='r', label=labels[0])
    ax.plot(range(1, len(dados[1]) + 1), dados[1], c='b', label=labels[1])

    ax.set_title(title, fontdict={'fontsize': 20})
    ax.set_xlabel('Épocas', fontdict={'fontsize': 15})
    ax.legend(fontsize=12)
    plt.show()

if __name__ == "__main__":
    config_gpu()
    train_path='atividade03/cityscapes_data/cityscapes_data/train'
    val_path='atividade03/cityscapes_data/cityscapes_data/val'

    ### PARA TESTES
    # i = 0

    colors = []
    for img_path in os.listdir(train_path):
        img = load_image_pair(img_path)
        seg = img[1]
        shape = seg.shape
        img_reshaped = np.reshape(seg, (shape[0] * shape[1], 3))
        colors.append(img_reshaped)

        ### PARA TESTES
        # if i < 63:
        #     i += 1
        # else:
        #     break

    colors = np.array(colors) # Já codificado

    # transforma colors
    img_qtd = len(colors)
    img_shape = colors.shape[1]
    colors = np.reshape(colors, (img_qtd * img_shape, 3))
    colors.shape

    km_path = 'atividade03/saves/kmeans.pickle'
    if os.path.isfile(km_path):
        km = pickle.load(open(km_path, 'rb'))
    else:
        km = MiniBatchKMeans(n_clusters=13)
        km.fit(colors)
        pickle.dump((km), open(km_path, 'wb'))


    # input_layer = Input(shape=(200, 256, 3))

    # x1 = BatchNormalization()(input_layer)
    # x1 = Conv2D(64, 12, activation="relu", padding="same")(x1)
    # x1 = Conv2D(128, 12, activation="relu", padding="same")(x1)
    # p1 = MaxPooling2D()(x1)


    # x2 = Conv2D(128, 9, activation="relu", padding="same")(p1)
    # x2 = Conv2D(128, 9, activation="relu", padding="same")(x2)
    # p2 = MaxPooling2D()(x2)


    # x3 = Conv2D(128, 6, activation="relu", padding="same")(p2)
    # x3 = Conv2D(128, 6, activation="relu", padding="same")(x3)
    # p3 = MaxPooling2D()(x3)


    # x4 = Conv2D(128, 3, activation="relu", padding="same")(p3)
    # x4 = Conv2D(128, 3, activation="relu", padding="same")(x4)


    # x5 = UpSampling2D()(x4)
    # x5 = concatenate([x3, x5])
    # x5 = Conv2D(128, 6, activation="relu", padding="same")(x5)
    # x5 = Conv2D(128, 6, activation="relu", padding="same")(x5)


    # x6 = UpSampling2D()(x5)
    # x6 = concatenate([x2, x6])
    # x6 = Conv2D(128, 9, activation="relu", padding="same")(x6)
    # x6 = Conv2D(128, 9, activation="relu", padding="same")(x6)

    # x7 = UpSampling2D()(x6)
    # x7 = concatenate([x1, x7])
    # x7 = Conv2D(128, 12, activation="relu", padding="same")(x7)
    # x7 = Conv2D(64, 12, activation="relu", padding="same")(x7)

    # # x7 = Conv2D(?, 6, activation="relu", padding="same")(x7)
    # x7 = Conv2D(13, 6, activation="softmax", padding="same")(x7)


    # model = Model(input_layer, x7)

    # optmizer = Adam(lr=0.0001)
    # model.compile(optimizer=optmizer,
    #             loss="categorical_crossentropy", 
    #             metrics=["accuracy"])



    model = load_model('atividade03/saves/model_94+04.h5')

    callback_list = [ModelCheckpoint("atividade03/saves/model_best.h5", save_best_only=True, verbose=0),
                    ModelCheckpoint("atividade03/saves/model_98+{epoch:02d}.h5", verbose=0),
                    CSVLogger('atividade03/saves/history.csv', append=True)]

    BATCH_SIZE = 16

    train_gen = Generate(batch_size=BATCH_SIZE)
    val_gen = Generate(path=val_path, batch_size=BATCH_SIZE)

    ### PARA TESTES
    # test_number = 64
    # train_qtd = len(os.listdir(train_path)[:test_number])
    # val_qtd = len(os.listdir(val_path)[:test_number])
    train_qtd = len(os.listdir(train_path))
    val_qtd = len(os.listdir(val_path))

    STEPS_PER_EPOCH = train_qtd // BATCH_SIZE
    VALIDATION_STEPS = val_qtd // BATCH_SIZE

    EPOCHS = 100

    history = model.fit(train_gen, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH,
                        validation_data=val_gen, validation_steps=VALIDATION_STEPS,
                        callbacks=callback_list)

    # accuracy = history.history['accuracy']
    # loss = history.history['loss']
    # val_accuracy = history.history['val_accuracy']
    # val_loss = history.history['val_loss']

    # dados = [accuracy, val_accuracy]
    # labels = ['Treino', 'Validação']
    # title = 'Acurácia'
    # plot_curvas(dados, labels, title)

    # dados = [loss, val_loss]
    # labels = ['Treino', 'Validação']
    # title = 'Loss'
    # plot_curvas(dados, labels, title)


    # model = load_model('atividade03/saves/model_best.h5')

    # test_gen = Generate(val_path)

    # for imgs, segs in test_gen:
        
    #     i = 0
    #     max_plots = 20
    #     for val_img, val_seg in zip(imgs, segs):
    #         plt.figure(figsize=[24,9])

    #         img = layers_to_rgb_image(val_img)
    #         plt.subplot(131)
    #         plt.title('Imagem Real')
    #         plt.imshow(img)

    #         pred = model.predict(np.expand_dims(val_img, 0))
    #         img = layers_to_rgb_image(pred[0])
    #         plt.subplot(132)
    #         plt.title('Segmentação Predita')
    #         plt.imshow(img)
            
    #         img = layers_to_rgb_image(val_seg)
    #         plt.subplot(133)
    #         plt.title('Segmentação Verdadeira')
    #         plt.imshow(img)
            
    #         plt.show()

    #         if i < (max_plots - 1):
    #             i += 1
    #         else:
    #             break

    #     break