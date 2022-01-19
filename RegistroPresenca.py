import numpy as np
import cv2
import face_recognition
import os
from datetime import datetime

# Lista para carregar imagens automaticamente da pasta
path = 'presencaImagens'
imagens = []
classeNomes = []
lista = os.listdir(path)
print(lista)

for cls in lista:
    imgAtual = cv2.imread(f'{path}/{cls}')
    imagens.append(imgAtual)
    classeNomes.append(os.path.splitext(cls)[0])
print(classeNomes)


# Função para computar os encodings
def localizarEncodings(imagens):
    listaEncode = []
    for img in imagens:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        listaEncode.append(encode)
    return listaEncode


# Função para registrar a lista de presença
def marcarPresenca(nome):
    with open('Presenca.csv', 'r+') as f:
        listaDados = f.readlines()
        nomeLista = []
        for linha in listaDados:
            entrada = linha.split(',')
            nomeLista.append(entrada[0])
        if nome not in nomeLista:
            horaAtual = datetime.now()
            horaString = horaAtual.strftime('%H:%M:%S')
            f.writelines(f' \n{nome}, {horaString}')


encodeListaConhecida = localizarEncodings(imagens)
print('Encode realizado')

# Inicializar Web-Cam e coletar cada Frame
capt = cv2.VideoCapture(0)

while True:
    success, img = capt.read()

    # Exibir a imagem em 25% de seu tamanho original
    imgResize = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgResize = cv2.cvtColor(imgResize, cv2.COLOR_BGR2RGB)
    rostoFrame = face_recognition.face_locations(imgResize)
    encodeFrame = face_recognition.face_encodings(imgResize, rostoFrame)

    # Iterar entre os rostos detectados e comparar
    for encodeRosto, localRosto in zip(encodeFrame, rostoFrame):
        match = face_recognition.compare_faces(encodeListaConhecida, encodeRosto)
        disRosto = face_recognition.face_distance(encodeListaConhecida, encodeRosto)
        print(disRosto)
        # Achar o menor elemento na lista para fazer o match
        matchIndice = np.argmin(disRosto)

        #Se a distancia do rosto for menor que 0.50 exibir como desconhecido e não marcar presença
        #Pode ser alterado quanto menor (ex: <0.40, <0.30) menor sera a tolerancia da distancia do rosto
        if disRosto[matchIndice] < 0.50:
            nome = classeNomes[matchIndice].upper()
            marcarPresenca(nome)
        else: nome = 'Desconhecido'
        # print(nome)
        # Criar marcação retangular e exibir a imagem em 100% do tamanho
        y1, x2, y2, x1 = localRosto
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, nome, (x1 + 6, y2 - 6), cv2.QT_FONT_NORMAL, 1, (255, 255, 255), 2)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)
