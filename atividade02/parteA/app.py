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
        cv2.rectangle(frame, (face_tuple[3], face_tuple[0]), (face_tuple[1], face_tuple[2]), (0,255,0), 2)

        if match:
            cv2.putText(frame, class_name, (face_tuple[3] + 6, face_tuple[2] - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,255,255), 1)
        else:
            cv2.putText(frame, 'Desconhecido', (face_tuple[3] + 6, face_tuple[2] - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,255,255), 1)


    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()