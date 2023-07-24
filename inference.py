import cv2 as cv
import numpy as np
import mediapipe as mp
from keras.models import load_model

model=load_model("model.h5")
labels=np.load("labels.npy")

holistic=mp.solutions.holistic
hands=mp.solutions.hands
holis=holistic.Holistic()
drawing=mp.solutions.drawing_utils 

cap=cv.VideoCapture(0)

while True:
    lst=[]
    ret,frame=cap.read()
    frame=cv.flip(frame,1)
    frame=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    res=holis.process(frame)

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
        lst=np.array(lst[:936]).reshape(1,-1)
        # pred=model.predict(lst)
        pred = labels[np.argmax(model.predict(lst))]
        print(pred)
        cv.putText(frame, pred, (50,50),cv.FONT_ITALIC, 1, (255,0,0),2)



    drawing.draw_landmarks(frame,res.face_landmarks,holistic.FACEMESH_CONTOURS)
    drawing.draw_landmarks(frame,res.left_hand_landmarks,hands.HAND_CONNECTIONS)
    drawing.draw_landmarks(frame,res.right_hand_landmarks,hands.HAND_CONNECTIONS)
    cv.imshow("frame",frame)

    if cv.waitKey(1)==27:
        cv.destroyAllWindows()
        cap.release()
        break