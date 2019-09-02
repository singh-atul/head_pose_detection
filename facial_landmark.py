#!/usr/bin/python
from __future__ import division
import dlib
import cv2
import numpy as np
import math


def get_distance(x1,y1,x2,y2):
    return math.sqrt((y2-y1)**2+(x2-x1)**2)

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = get_distance(landmark.part(36).x,landmark.part(36).y,landmark.part(39).x,landmark.part(39).y)
    B = get_distance(landmark.part(37).x,landmark.part(37).y,landmark.part(41).x,landmark.part(41).y)

	# compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = get_distance(landmark.part(38).x,landmark.part(38).y,landmark.part(40).x,landmark.part(40).y)
    # compute the eye aspect ratio
    left_ear = (A + B) / (2.0 * C)

    A = get_distance(landmark.part(42).x,landmark.part(42).y,landmark.part(45).x,landmark.part(45).y)
    B = get_distance(landmark.part(43).x,landmark.part(43).y,landmark.part(47).x,landmark.part(47).y)

	# compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = get_distance(landmark.part(44).x,landmark.part(44).y,landmark.part(46).x,landmark.part(46).y)
    
    right_ear = (A + B) / (2.0 * C)
    
    # return the eye aspect ratio
    return left_ear + right_ear

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat_2')
count = 0
while(True):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    det = detector(gray)
    for d in det:
        x1 = d.left()
        x2 = d.right()
        y1 = d.bottom()
        y2 = d.top()
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        
        y_mid = (y1+y2)//2
        
        landmark = predictor(gray,d)
        #print (get_distance(landmark.part(32).x,landmark.part(32).y,x2,y_mid))
        cv2.line(frame,(landmark.part(30).x,landmark.part(30).y),(landmark.part(27).x,landmark.part(27).y),(0,255,0),2)
        theta = math.atan2(landmark.part(30).y-landmark.part(27).y,landmark.part(30).x-landmark.part(27).x) * 180 / 3.14
        direction = "Straight"
        if theta>100:
            direction = "Left"
            print("Left")
        elif theta<80:
            direction = "Right"
            print("Right")
        else:
            print("Straight")
        cv2.putText(frame, '{} : {}'.format(direction,round(theta,2)), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), lineType=cv2.LINE_AA) 
        
        eye_open = eye_aspect_ratio(landmark)
        print(eye_open/2)
        if eye_open/2 > 3.1:
            count = count+1
        cv2.putText(frame, str(count), (frame.shape[0],40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), lineType=cv2.LINE_AA) 
        #cv2.circle(frame,(x1,y_mid), 2, (255,0,255), -1)
        #cv2.circle(frame,(x2,y_mid), 2, (255,0,255), -1)

        for n in range(68):
            cv2.circle(frame,(landmark.part(n).x,landmark.part(n).y), 2, (0,0,255), -1)

    
    
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()






'''
def resize(img, width=None, height=None, interpolation=cv2.INTER_AREA):
    global ratio
    w, h = img.shape

    if width is None and height is None:
        return img
    elif width is None:
        ratio = height / h
        width = int(w * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized
    else:
        ratio = width / w
        height = int(h * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

camera = cv2.VideoCapture(1)

predictor_path = 'shape_predictor_68_face_landmarks.dat_2'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

while True:

    ret, frame = camera.read()
    if ret == False:
        print('Failed to capture frame from camera. Check camera index in cv2.VideoCapture(0) \n')
        break

    frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_resized = resize(frame_grey, width=120)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(frame_resized, 1)
    if len(dets) > 0:
        for k, d in enumerate(dets):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy array
            shape = predictor(frame_resized, d)
            shape = shape_to_np(shape)

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                cv2.circle(frame, (int(x/ratio), int(y/ratio)), 3, (255, 255, 255), -1)
            #cv2.rectangle(frame, (int(d.left()/ratio), int(d.top()/ratio)),(int(d.right()/ratio), int(d.bottom()/ratio)), (0, 255, 0), 1)

    cv2.imshow("image", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        camera.release()
        break

'''