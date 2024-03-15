import os
import cv2
import pickle
import numpy as np

IMAGE_DIRECTORY = "../images/celeb-data"
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
recognizer = cv2.face.LBPHFaceRecognizer_create()


def get_collections(directory=IMAGE_DIRECTORY):
    collections = {}
    for root, dirs, files in os.walk(directory, topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            dir_path = os.path.dirname(file_path)
            dir_name = os.path.basename(dir_path)
            if not dir_name in collections:
                collections[dir_name] = []
            collections[dir_name].append(file_path)
    return collections


def face_rects_detection(gray_img, scaleFactor=1.1, minNeighbors=5):
    faces = face_classifier.detectMultiScale(
        gray_img, scaleFactor, minNeighbors, minSize=(40, 40)
    )
    return faces


def exact_one_gray_face(file_path, reshape=None):
    img = cv2.imread(file_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = face_rects_detection(gray_img)
    if len(rects) > 0:
        rect = rects[0]
        x, y, w, h = rect
        crop_image = gray_img[y: y + h, x: x + w]
        if reshape is not None:
            crop_image = cv2.resize(crop_image, reshape)
        return crop_image


def get_gray_collections(dir_collections, reshape=None):
    gray_collections = {}
    for key, files in dir_collections.items():
        if key not in gray_collections:
            gray_collections[key] = []
        for file in files:
            face_img = exact_one_gray_face(file, reshape)
            if face_img is not None:
                gray_collections[key].append(face_img)
    return gray_collections


def trainLBPH():
    collections = get_collections()
    gray_collections = get_gray_collections(collections)

    X = []
    y = []
    label2Id = {}
    id2Label = {}
    for idx, label in enumerate(gray_collections.keys()):
        label2Id[label] = idx
        id2Label[idx] = label
    for label, gray_imgs in gray_collections.items():
        for gray_img in gray_imgs:
            X.append(gray_img)
            y.append(label2Id[label])
    with open("labels.pickle", "wb") as f:
        pickle.dump(label2Id, f)
    print("Training....")
    recognizer.train(X, np.array(y))
    recognizer.save("trainer.yml")


def updateLBPH(lbph, faces, name, id2Label):
    newId = max(id2Label) + 1
    id2Label[newId] = name
    label2Id = {v: k for k, v in id2Label.items()}
    with open("labels.pickle", "wb") as f:
        pickle.dump(label2Id, f)
    ids = np.array([newId for _ in faces])
    lbph.update(faces, ids)
    lbph.save("trainer.yml")


if __name__ == "__main__":
    trainLBPH()
