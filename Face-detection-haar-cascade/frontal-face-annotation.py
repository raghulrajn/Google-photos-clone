import cv2
import os
import json
import numpy as np
from tqdm import tqdm

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
IMAGE_PATH = os.path.join(os.getcwd(),"faces")

data = {"shapes": [{"label": "face","shape_type": "rectangle" }]}


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

    try:
        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)[0]
        x,y,w,h = faces[0],faces[1],faces[2],faces[3]
    except IndexError as e:
        x,y,w,h = [0,0,1,1]
    cv2.imwrite("faces_size/"+img_name,img)

    with open(os.path.join(os.getcwd(),"faces-annotation/"+img_name.split(".")[0]+".json"),"w") as f:
        cv2.rectangle(img, (x, y-50), (x + w , y + h+50), (255, 255, 0), 2)
        xmin, xmax = int(x), int(x+ w)
        ymin, ymax = int(y), int(y+ h)
        points = [[xmin,ymin], [xmax, ymax]]
        data = {"shapes": [{"label": "face","shape_type": "rectangle" }]}
        data["shapes"][0]['points'] = points
        json.dump(data,f)
        
    # cv2.imshow("img", img)

    if cv2.waitKey(0) & 0xFF == 27:
        break

    cv2.destroyAllWindows()
