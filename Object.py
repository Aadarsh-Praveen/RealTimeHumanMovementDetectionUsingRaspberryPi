import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow.compat.v1 as tf
import zipfile
from datetime import datetime
import json
import requests
import os
from requests.auth import HTTPBasicAuth

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
# import ujson as json
import pandas as pd
import numpy as np
import time
import sys
from firebase import firebase


config = tf.ConfigProto()

config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import cv2

# cam = 0 # Use  local webcam.
# cam = "http://192.168.0.101:8081/video"s
# cap = cv2.VideoCapture(cam)

# cap = cv2.VideoCapture(1)

sys.path.append("..")

fixefixed_interval = 3
firebase = firebase.FirebaseApplication('https://humandec-f9578.firebaseio.com', None)
count=1

# In[3]:

from utils import label_map_util

from utils import visualization_utils as vis_util

def diffImg(t0, t1, t2):  
    d1 = cv2.absdiff(t2, t1)
    d2 = cv2.absdiff(t1, t0)
    return cv2.bitwise_and(d1, d2)

#url ='rtsp://admin:888888@192.168.0.164:10554/tcp/av0_0'

threshold = 100000  # Threshold for triggering "motion detection"
cap = cv2.VideoCapture(0)  # Lets initialize capture on webcam

# Read three images first:
t_minus = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
t = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
t_plus = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
# Lets use a time check so we only take 1 pic per sec
timeCheck = datetime.now().strftime('%Ss')



  
  	
  
 
  
# In[4]:

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# ## Download Model

# In[5]:

opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())


# In[6]:

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# In[7]:

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# In[8]:

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

# In[9]:

PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3)]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

count = 1

# In[10]:

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while(cap.isOpened()):
            
            # Read first and next image
            ret,image = cap.read()
            ret2,image2 = cap.read()
            
            
            if image is not None and image2 is not None and ret is True and ret2 is True:
            
                cv2.imshow('frame', image)

                if cv2.countNonZero(diffImg(t_minus, t, t_plus)) > threshold and timeCheck != datetime.now().strftime(
                        '%Ss'):
                    dimg = image
                    # cv2.imwrite(path + datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss%f') + '.jpg', dimg)
                    image_np = dimg
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    # Each box represents a part of the image where a particular object was detected.
                    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    # Each score represent how level of confidence for each of the objects.
                    # Score is shown on the result image, together with the class label.
                    scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                    # Actual detection.
                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                    # Visualization of the results of a detection.
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8)

                    df = pd.DataFrame(boxes.reshape(100, 4), columns=['y_min', 'x_min', 'y_max', 'x_max'])
                    df1 = pd.DataFrame(classes.reshape(100, 1), columns=['classes'])
                    df2 = pd.DataFrame(scores.reshape(100, 1), columns=['scores'])
                    df5 = pd.concat([df, df1, df2], axis=1)
                    df6 = df5.loc[df5['classes'] == 1]
                    df7 = df6.loc[df6['scores'] > 0.50]

                    if int(len(df7.index)) > 0:
                        people = int(len(df7.index))
                    else:
                        people = 0

                    columns = ['LocationID','DeviceID','DeviceTime', 'Class', 'Count']
                    index = [0]
                    timenow = datetime.now().strftime(" (%Y%m%d %H:%M:%S)")
                    df_ = pd.DataFrame(index=index, columns=columns)
                    df_.loc[0, 'LocationID'] = '1'
                    df_.loc[0, 'DeviceID'] = '32'
                    df_.loc[0, 'DeviceTime'] = timenow
                    df_.loc[0, 'Class'] = 1
                    df_.loc[0, 'Count'] = people


                    jn = df_.to_json(orient='records', lines=True)

                    jn1 = json.loads(jn)
                    print(jn)
                    
                    
                    firebase.put('', 'IOT guard/Location 1', jn1)

                    print("pixelDiff:" + str(cv2.countNonZero(diffImg(t_minus, t, t_plus))))
                    
                    if cv2.countNonZero(diffImg(t_minus, t, t_plus)) > threshold and timeCheck != datetime.now().strftime('%Ss'):
                        if df_.loc[0, 'Count']==1:
                            dimg=cap.read()[1]
                            #cv2.imwrite(datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss%f') + '.jpg',dimg)
                            cv2.imwrite("images1"+ '.jpg',dimg)
  
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        #	break

                timeCheck = datetime.now().strftime('%Ss')
                # Read next image
                t_minus = t
                t = t_plus
                t_plus = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
            
            if ret is False:
                cap.release()
                cv2.destroyAllWindows()
                cap = cv2.VideoCapture(0)
                

# In[ ]:



# libraries to be imported 
import smtplib 
from email.mime.multipart import MIMEMultipart 
from email.mime.text import MIMEText 
from email.mime.base import MIMEBase 
from email import encoders 
   
fromaddr = "bharathkarthick05@gmail.com"
toaddr = "bharathkarthick05@gmail.com"
   
# instance of MIMEMultipart 
msg = MIMEMultipart() 
  
# storing the senders email address   
msg['From'] = fromaddr 
  
# storing the receivers email address  
msg['To'] = toaddr 
  
# storing the subject  
msg['Subject'] = "Subject of the Mail"
  
# string to store the body of the mail 
body = "person detection in location 1"
  
# attach the body with the msg instance 
msg.attach(MIMEText(body, 'plain')) 
  
# open the file to be sent  
filename = "images1.jpg"
attachment = open("images1.jpg", "rb") 
  
# instance of MIMEBase and named as p 
p = MIMEBase('application', 'octet-stream') 
  
# To change the payload into encoded form 
p.set_payload((attachment).read()) 
  
# encode into base64 
encoders.encode_base64(p) 
   
p.add_header('Content-Disposition', "attachment; filename= %s" % filename) 
  
# attach the instance 'p' to instance 'msg' 
msg.attach(p) 
  
# creates SMTP session 
s = smtplib.SMTP('smtp.gmail.com', 587) 
  
# start TLS for security 
s.starttls() 
  
# Authentication 
s.login(fromaddr, "bharathrocks69") 
  
# Converts the Multipart msg into a string 
text = msg.as_string() 
  
# sending the mail 
s.sendmail(fromaddr, toaddr, text) 
  
# terminating the session 
s.quit() 
