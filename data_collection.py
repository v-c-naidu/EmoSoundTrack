import numpy as np
import mediapipe as mp
import cv2 as cv
import keras

cap=cv.VideoCapture(0)

holistic=mp.solutions.holistic
hands=mp.solutions.hands
holis=holistic.Holistic()
drawing=mp.solutions.drawing_utils
name=input("Enter emotion name: ")
x=[]
size=0
while True:
    lst=[]
    ret,frame=cap.read()
    frame=cv.flip(frame,1)
    res=holis.process(cv.cvtColor(frame,cv.COLOR_BGR2RGB))

    if res.face_landmarks:
        for i in res.face_landmarks.landmark:
            lst.append(i.x-res.face_landmarks.landmark[1].x)
            lst.append(i.y-res.face_landmarks.landmark[1].y)
        if res.left_hand_landmarks:
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x-res.left_hand_landmarks.landmark[8].x)
                lst.append(i.y-res.left_hand_landmarks.landmark[8].y)
        else:
            for i in range(42):
                lst.append(0.0)
        if res.right_hand_landmarks:
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x-res.right_hand_landmarks.landmark[8].x)
                lst.append(i.y-res.right_hand_landmarks.landmark[8].y)
        else:
            for i in range(42):
                lst.append(0.0)
        x.append(lst)
        size+=1



    drawing.draw_landmarks(frame,res.face_landmarks,holistic.FACEMESH_CONTOURS)
    drawing.draw_landmarks(frame,res.left_hand_landmarks,hands.HAND_CONNECTIONS)
    drawing.draw_landmarks(frame,res.right_hand_landmarks,hands.HAND_CONNECTIONS)
    cv.imshow("frame",frame)

    if cv.waitKey(1)==27 or size>99:
        cv.destroyAllWindows()
        cap.release()
        break
np.save(f"{name}.npy",np.array(x))
