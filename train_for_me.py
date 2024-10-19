import time
from ultralytics import YOLO
import cv2
import uuid
import os

# creating dir for our images

# labels = ['Face Up','Face Down','Face Left','Face Right']
# classes = 5;

# IMAGE_PATH = f'{os.getcwd()}/images/collectedimages'

# if not os.path.exists(IMAGE_PATH):
#     os.makedirs(IMAGE_PATH)

# for labels in labels:
#     path = f"{IMAGE_PATH}/label"
#     if not os.path.exists(path):
#         os.mkdir(path)



# # capturing images from web-cam

# for label in labels:
    
#     cap = cv2.VideoCapture(cv2.CAP_ANDROID)
#     print('Collecting Images {}'.format(labels))
#     time.sleep(5)

#     for imgnum in range(classes):
#         print("Collecting {}".format(imgnum))
#         ret,frame = cap.read()
#         image_name = os.path.join(IMAGE_PATH,label,label+'.'+'{}.jpg'.format(str(imgnum)))
#         print(frame)
#         cv2.imwrite(image_name,frame)
#         cv2.imshow('frame',frame)
#         time.sleep(2)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


# cap.release()
# cv2.destroyAllWindows()

model = YOLO('yolov8n.pt',verbose=True)

# model.train(data="./annotate me.v3i.yolov8/data.yaml",epochs=100,imgsz=640)

# test = model.val(data="./annotate me.v3i.yolov8/data.yaml",epochs=100,imgsz=640)

device = 'cuda'

model = YOLO("./runs/detect/train37/weights/best.pt")
# path ="VID_20240719_170632.mp4"


# This shows the result....
results = model.predict(source=path, show = True)  




