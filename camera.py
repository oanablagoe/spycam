from cv2 import *
from time import sleep


def get_face():
    camera = VideoCapture(0)
    sleep(3)
    s, img = camera.read()
    camera.release()
    face_crop(img)


def face_crop(image):
    face_cascade = CascadeClassifier('haarcascade_frontalface_default.xml')

    img = image
    minisize = (img.shape[1], img.shape[0])
    miniframe = resize(img, minisize)
    faces = face_cascade.detectMultiScale(miniframe)
    for f in faces:
        x, y, w, h = [v for v in f]
        sub_face = img[y:y+h+20, x:x+w+20]
        face_file_name = "out.jpg"
        imwrite(face_file_name, sub_face)
        print("face output")
        return sub_face

#import cv2
#from random import randint
#def facechop(image):
#    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#
#    img = cv2.imread("img/" + image)
#    minisize = (img.shape[1],img.shape[0])
#    miniframe = cv2.resize(img, minisize)
#    faces = face_cascade.detectMultiScale(miniframe)
#    for f in faces:
#        x, y, w, h = [ v for v in f ]
#        sub_face = img[y:y+h+20, x:x+w+20]
#        face_file_name = "out/" + str(randint(0, 90000)) + image
#        cv2.imwrite(face_file_name, sub_face)
#
#from os import listdir
#from os.path import isfile, join
#onlyfiles = [f for f in listdir("img/") if isfile(join("img/", f))]
#
#for f in onlyfiles:
#    if ".DS_Store" in f:
#        continue
#    facechop(f)
#

