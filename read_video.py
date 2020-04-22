import cv2
import numpy as np
import torch
from torchvision import transforms


cascade_classifier = cv2.CascadeClassifier(r'haarcascade/haarcascade_frontalface_default.xml')
transform = transforms.Resize(48)


def format_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = cascade_classifier.detectMultiScale(
        image,
        scaleFactor=1.25,
        minNeighbors=3
    )

    if not len(faces) > 0:
        return None

    max_area_face = faces[0]

    for face in faces:
        if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
            max_area_face = face

    # Chop image to face
    face = max_area_face
    image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]
    # Resize image to thunderstorm size
    try:
        image = cv2.resize(image, (48, 48)) / 255
        image = torch.from_numpy(image).view(1, 1, 48, 48).float()
    except Exception:
        print("[+] Problem during resize")
        return None

    return image


# Load Model
thunderstorm = torch.load('EmotionClassificationCNN.4500', map_location=torch.device('cpu'))

# i'm apologize for my variable names
video_capture = cv2.VideoCapture(r'Jojo Rabbit - Gestapo Scene (Heil Hitler).mp4')

while True:
    ret, frame = video_capture.read()

    result = format_image(frame)

    # Write results in frame
    if result is not None:

        result = thunderstorm(result)

        for emotion, index in thunderstorm.class_to_idx.items():

            cv2.putText(frame, emotion, (10, index * 20 + 20),
                        cv2.FONT_ITALIC, 1, (132, 222, 2))

            cv2.rectangle(frame, (130, index * 20 + 10),
                          (130 + int(result[0][index] * 100), (index + 1) * 20 + 4),
                          (132, 222, 2), -1)

        # x, y, w, h = face
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (132, 222, 2), 2)

        # face_image = feelings_faces[np.argmax(result[0])]

        # Ugly transparent fix
        # for c in range(0, 3):
        #     frame[200:320, 10:130, c] = face_image[:, :, c] * \
        #                                 (face_image[:, :, 3] / 255.0) + frame[200:320, 10:130, c] \
        #                                 * (1.0 - face_image[:, :, 3] / 255.0)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
