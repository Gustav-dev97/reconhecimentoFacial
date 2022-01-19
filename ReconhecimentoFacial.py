import cv2
import face_recognition

# Carregar Imagem principal Steve Jobs
imgSteve = face_recognition.load_image_file('Recursos/SteveJobs.jpg')

# Converter imagem de BGR para RGB
imgSteve = cv2.cvtColor(imgSteve, cv2.COLOR_BGR2RGB)

# Carregar Imagem Trainer Steve Jobs
imgTrainer = face_recognition.load_image_file('Recursos/SteveJobsTrainer.jpg')

# Converter imagem de BGR para RGB
imgTrainer = cv2.cvtColor(imgTrainer, cv2.COLOR_BGR2RGB)

# Detectar Rostos
localFace = face_recognition.face_locations(imgSteve)[0]
localFaceTrainer = face_recognition.face_locations(imgTrainer)[0]

# Gerar Encode de Rostos detectados
codSteve = face_recognition.face_encodings(imgSteve)[0]
encodeTrainer = face_recognition.face_encodings(imgTrainer)[0]

# Criar demarcação no rosto detectado
cv2.rectangle(imgSteve, (localFace[3], localFace[0]), (localFace[1], localFace[2]), (255, 0, 255), 2)
cv2.rectangle(imgTrainer, (localFaceTrainer[3], localFaceTrainer[0]), (localFaceTrainer[1], localFaceTrainer[2]),
              (255, 0, 255), 2)

# Comparar similaridades dos rostos codificados
resultado = face_recognition.compare_faces([codSteve], encodeTrainer)

# Comparar rostos baseado  distância
distanciaRosto = face_recognition.face_distance([codSteve], encodeTrainer)

# Exibir resultados na imagem
cv2.putText(imgTrainer, f' {resultado}, {round(distanciaRosto[0], 3)}', (50, 50), cv2.QT_FONT_NORMAL, 1, (0, 0, 255), 2)

print(resultado, distanciaRosto)

cv2.imshow('Steve Jobs', imgSteve)
cv2.imshow('Steve Jobs Trainer', imgTrainer)
cv2.waitKey(0)
