import face_recognition
import cv2

cap = cv2.VideoCapture(0)

known_image = face_recognition.load_image_file('foto34.jpg')
known_encoding = face_recognition.face_encodings(known_image)[0]
class_name = 'Bruno'

while True:
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(frame_rgb, model='hog')
    unknown_encodings_list = face_recognition.face_encodings(frame_rgb, face_locations)

    matches = list()
    for unknown_encoding in unknown_encodings_list:
        results = face_recognition.compare_faces([known_encoding], unknown_encoding, tolerance=0.6)
        matches.append(results[0])

    for face_tuple, match in zip(face_locations, matches):
        square_color = (50, 205, 50)
        text_color = (255, 255, 255)
        cima, direita, baixo, esquerda = face_tuple

        cv2.rectangle(frame, (esquerda, cima), (direita, baixo), square_color, 2)

        if match:
            name = class_name
        else:
            name = 'Desconhecido'

        cv2.rectangle(frame, (esquerda - 1, baixo), (direita + 1, baixo + 30), square_color, -1)
        cv2.putText(frame, name, (esquerda + 6, baixo + 25), cv2.FONT_HERSHEY_DUPLEX, 1.0, text_color, 1)


    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()