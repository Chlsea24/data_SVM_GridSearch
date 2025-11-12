import cv2
import math
import numpy as np
import os
import urllib.request

nose_cascade_path = "haarcascade_mcs_nose.xml"
if not os.path.exists(nose_cascade_path):
    url = "https://raw.githubusercontent.com/atduskgreg/opencv-processing/master/lib/cascade-files/haarcascade_mcs_nose.xml"
    urllib.request.urlretrieve(url, nose_cascade_path)

def euclidean(p1, p2):
    return math.dist(p1, p2)

def midpoint(box):
    x, y, w, h = box
    return (x + w//2, y + h//2)

def extract_facial_metrics(image_path):
    face_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade   = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    nose_cascade  = cv2.CascadeClassifier(nose_cascade_path)

    img = cv2.imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    if len(faces) == 0: return None

    x, y, w, h = faces[0]
    roi = gray[y:y+h, x:x+w]

    eyes = eye_cascade.detectMultiScale(roi, 1.1, 10)
    eyes = [(x+ex, y+ey, ew, eh) for ex,ey,ew,eh in eyes if ey < h/2][:2]

    noses = nose_cascade.detectMultiScale(roi, 1.3, 5)
    noses = [(x+nx, y+ny, nw, nh) for nx,ny,nw,nh in noses[:1]]

    mouths = mouth_cascade.detectMultiScale(roi, 1.7, 20)
    mouths = [(x+mx, y+my, mw, mh) for mx,my,mw,mh in mouths if my > h/2][:1]

    if not (len(eyes)==2 and len(noses)==1 and len(mouths)==1): return None

    mata = sorted([midpoint(eyes[0]), midpoint(eyes[1])], key=lambda p:p[0])
    mata_kiri, mata_kanan = mata
    hidung = midpoint(noses[0])
    mulut = mouths[0]
    mulut_kiri  = (mulut[0], mulut[1]+mulut[3]//2)
    mulut_kanan = (mulut[0]+mulut[2], mulut[1]+mulut[3]//2)

    return np.array([
        euclidean(mata_kiri, mata_kanan),             
        euclidean(mulut_kiri, mulut_kanan),           
        abs(mata_kiri[1] - mata_kanan[1]),            
        abs(euclidean(mata_kiri, hidung) - euclidean(mata_kanan, hidung)),
        euclidean(mata_kiri, mulut_kiri),
        euclidean(mata_kanan, mulut_kanan),
        euclidean(mata_kiri, mulut_kiri)/(euclidean(mata_kanan, mulut_kanan)+1e-6),
        math.degrees(math.atan2(mulut_kiri[1]-mata_kiri[1], mulut_kiri[0]-mata_kiri[0])),
        math.degrees(math.atan2(mulut_kanan[1]-mata_kanan[1], mulut_kanan[0]-mata_kanan[0]))
    ]).reshape(1,-1)

def extract_numeric_vector(image_path):
    return extract_facial_metrics(image_path)
