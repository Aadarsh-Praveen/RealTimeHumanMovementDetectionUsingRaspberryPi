# USAGE
# python3 pi_face_recognition.py --cascade haarcascade_frontalface_default.xml --encodings encodings.pickle

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
import requests
import smtplib
import email
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from time import sleep
import RPi.GPIO as GPIO
from gpiozero import LED
from gpiozero import MCP3008
import requests
from gpiozero import Button


lt=0
lt1=0
lt2=0
lt3=0
lt4=0

n1=0
n2=0
n3=0
n4=0
count=0
name=' '

sender = 'rasp@iotclouddata.com'
password = 'Rasp1234$'
receiver1 = 'buvanesh2310@gmail.com'



DIR = './Database/'
FILE_PREFIX = 'image'



# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
	help = "path to where the face cascade resides")
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
args = vars(ap.parse_args())

# load the known faces and embeddings along with OpenCV's Haar
# cascade for face detection
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(args["encodings"], "rb").read())
detector = cv2.CascadeClassifier(args["cascade"])

# initialize the video stream and allow the camera sensor to warm up
print("Bus Monitoring System")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
# time.sleep(2.0)

# start the FPS counter
fps = FPS().start()

def send_mail1():
    print ('Sending E-Mail')    
    # Save image to file
    filename = '1.jpeg'

    # Sending mail
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = receiver1
    msg['Subject'] = 'Detected'
    
    body = 'Picture is Attached.'
    msg.attach(MIMEText(body, 'plain'))
    attachment = open(filename, 'rb')
    part = MIMEBase('application', 'octet-stream')
    part.set_payload((attachment).read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', 'attachment; filename= %s' % filename)
    msg.attach(part)
    server = smtplib.SMTP('smtpout.asia.secureserver.net', 3535)
    server.starttls()
    server.login(sender, password)
    text = msg.as_string()
    server.sendmail(sender, receiver1, text)
    server.quit()

def send_mail2():
    print ('Sending E-Mail')    
    # Save image to file
    filename = '2.jpeg'

    # Sending mail
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = receiver1
    msg['Subject'] = 'Detected'
    
    body = 'Picture is Attached'
    msg.attach(MIMEText(body, 'plain'))
    attachment = open(filename, 'rb')
    part = MIMEBase('application', 'octet-stream')
    part.set_payload((attachment).read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', 'attachment; filename= %s' % filename)
    msg.attach(part)
    server = smtplib.SMTP('smtpout.asia.secureserver.net', 3535)
    server.starttls()
    server.login(sender, password)
    text = msg.as_string()
    server.sendmail(sender, receiver1, text)
    server.quit()

def send_mail3():
    print ('Sending E-Mail')    
    # Save image to file
    filename = '3.jpeg'

    # Sending mail
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = receiver1
    msg['Subject'] = 'Detected'
    
    body = 'Picture is Attached.'
    msg.attach(MIMEText(body, 'plain'))
    attachment = open(filename, 'rb')
    part = MIMEBase('application', 'octet-stream')
    part.set_payload((attachment).read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', 'attachment; filename= %s' % filename)
    msg.attach(part)
    server = smtplib.SMTP('smtpout.asia.secureserver.net', 3535)
    server.starttls()
    server.login(sender, password)
    text = msg.as_string()
    server.sendmail(sender, receiver1, text)
    server.quit()

def send_mail4():
    print ('Sending E-Mail')    
    # Save image to file
    filename = '4.jpeg'

    # Sending mail
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = receiver1
    msg['Subject'] = 'Detected'
    
    body = 'Picture is Attached.'
    msg.attach(MIMEText(body, 'plain'))
    attachment = open(filename, 'rb')
    part = MIMEBase('application', 'octet-stream')
    part.set_payload((attachment).read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', 'attachment; filename= %s' % filename)
    msg.attach(part)
    server = smtplib.SMTP('smtpout.asia.secureserver.net', 3535)
    server.starttls()
    server.login(sender, password)
    text = msg.as_string()
    server.sendmail(sender, receiver1, text)
    server.quit()


# loop over frames from the video file stream
while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=500)
        # convert the input frame from (1) BGR to grayscale (for face#
        # detection) and (2) from BGR to RGB (for face recognition)#
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # detect faces in the grayscale frame
        rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                minNeighbors=5, minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE)

        # OpenCV returns bounding box coordinates in (x, y, w, h) order
        # but we need them in (top, right, bottom, left) order, so we
        # need to do a bit of reordering

        boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

        # compute the facial embeddings for each face bounding box#
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []

        # loop over the facial embeddings
        for encoding in encodings:
                # attempt to match each face in the input image to our known
                # encodings
                matches = face_recognition.compare_faces(data["encodings"],
                        encoding)
                name = "Unknown"

                # check to see if we have found a match
                if True in matches:
                        # find the indexes of all matched faces then initialize a
                        # dictionary to count the total number of times each face
                        # was matched
                        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                        counts = {}

                        # loop over the matched indexes and maintain a count for
                        # each recognized face face
                        for i in matchedIdxs:
                                name = data["names"][i]
                                counts[name] = counts.get(name, 0) + 1#

                        # determine the recognized face with the largest number
                        # of votes (note: in the event of an unlikely tie Python
                        # will select first entry in the dictionary)
                        name = max(counts, key=counts.get)

                # update the list of names
                names.append(name)

        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(boxes, names):
                # draw the predicted face name on the image
                cv2.rectangle(frame, (left, top), (right, bottom),
                        (0, 255, 0), 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2)

        lt=lt+1
        print('Time:{:3d}'.format(lt))
        print('Count:{:3d}'.format(count))
        #print(count)
        

        #print('Time1:{:3d}'.format(lt1))
        #print('Time2:{:3d}'.format(lt2))
        #print('Time3:{:3d}'.format(lt3))
        #print('Time4:{:3d}'.format(lt4))

        if lt > 30:
                break

        if n1 == 1:
                if lt1 < 31:
                        lt1=lt1+1
        if n1 == 2:
                lt1=lt1+1

        if n2 == 1:
                if lt2 < 31:
                        lt2=lt2+1

        if n2 == 2:
                lt2=lt2+1
        if n3 == 1:
                if lt3 < 31:
                        lt3=lt3+1

        if n3 == 2:
                lt3=lt3+1

        if n4 == 1:
                if lt4 < 31:
                        lt4=lt4+1
        if n4 == 2:
                lt4=lt4+1

        if name=='1':
                if lt1 == 0:
                        if n1==0:
                                count=count+1
                                send_mail1()
                                n1=1
        if name=='2':
                if lt2 == 0:
                        if n2==0:
                                count=count+1
                                send_mail2()
                                n2=1

        if name=='3':
                if lt3 == 0:
                        if n3==0:
                                count=count+1
                                send_mail3()
                                n3=1
        if name=='4':
                if lt4 == 0:
                        if n4==0:
                                count=count+1
                                send_mail4()
                                n4=1


        # display the image to our screen
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop#
        if key == ord("q"):
            break


        # update the FPS counter
        fps.update()

r =requests.get('http://www.iotclouddata.com/20log/279/iot20.php?A=Count=' + str(count))

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()