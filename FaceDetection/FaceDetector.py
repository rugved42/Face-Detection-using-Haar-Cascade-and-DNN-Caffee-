import cv2 
import numpy as np  
import glob 
import os
from os.path import dirname, join
from PIL import Image
import natsort 
import json
import sys 
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
print(sys.argv[1],len(sys.argv[1]))
image_list = []
read_images = []   
imagenames_list = []
json_list = []
for folder in sorted(glob.glob(sys.argv[1]+'/*.jpg')):
    imagenames_list.append(folder) 
# print(imagenames_list)   
imagenames_list = natsort.natsorted(imagenames_list)
# print(imagenames_list)  
for image in imagenames_list:
    read_images.append(cv2.imread(image))
prototxt_path = 'model_files/face_detector/deploy.prototxt'
caffemodel_path = 'model_files/face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel'
# Read the model
model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
for i in range(len(read_images)):
    img = read_images[i]
    img_name = str(imagenames_list[i])
    # print(img_name)
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    model.setInput(blob)
    detect = model.forward()
    # print(detect.shape)
    idx = np.argsort(detect)[::-1][0]
    # print(idx.shape)
    for i in range(0, detect.shape[2]):
        d = detect[0, 0, i, 3:7]
        wh = np.array([w, h, w, h])
        box = d * wh
        (x, y, ex, ey) = box.astype("int")
        confidence = detect[0, 0, i, 2]
        if (confidence > 0.5):
            cv2.rectangle(img, (x, y), (ex, ey), (255, 0, 0), 2)
            # print(x, y, ex-x, ey-y)
            # cv2.imwrite("results_tmp/"+ img_name[len(sys.argv[1]):],img)
            bbox = ([int(x),int(y),int(ex-x),int(ey-y)])
            if bbox != []:
                element = {"iname": img_name[len(sys.argv[1]):], "bbox": bbox} 
                json_list.append(element)
        else: 
            pass
output_json = "results.json"
with open(output_json, 'w') as f:
    json.dump(json_list, f)
