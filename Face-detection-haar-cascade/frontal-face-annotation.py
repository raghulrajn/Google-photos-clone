import cv2
import os
import json
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
IMAGE_PATH = os.path.join(os.getcwd(),"faces")
adj = 0

data = {"shapes": [{"label": "face","shape_type": "rectangle" }]}
from tqdm import tqdm

for img in tqdm(os.listdir(IMAGE_PATH)):
    img_name = str(img)
    img = cv2.imread(os.path.join(IMAGE_PATH,img))
    img = cv2.resize(img, (600, 600), fx = 0.1, fy = 0.1
                 ,interpolation = cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05,
	                                    minNeighbors=5, 
                                        minSize=(30, 30),
	                                    flags=cv2.CASCADE_SCALE_IMAGE)

    with open(os.path.join(os.getcwd(),"faces-annotation/"+img_name.split(".")[0]+".json"),"w") as f:
        for (x, y, w, h) in faces:
            x = x-adj +10
            y = y-adj
            cv2.rectangle(img, (x, y), (x + w +adj, y + h+adj), (255, 255, 0), 2)
            xmin, xmax = int(x), int(x+ w+ adj)
            ymin, ymax = int(y), int(w+ h+ adj)
        points = [[xmin,ymin], [xmax, ymax]]
        data = {"shapes": [{"label": "face","shape_type": "rectangle" }]}
        data["shapes"][0]['points'] = points
        json.dump(data,f)
        # cv2.imshow('img', img)

    if cv2.waitKey(0) & 0xFF == 27:
        break

    cv2.destroyAllWindows()