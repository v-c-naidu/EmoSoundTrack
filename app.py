import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
from keras.models import load_model
import cv2 as cv
import numpy as np
import mediapipe as mp
import webbrowser

model=load_model("model.h5")
labels=np.load("labels.npy")

holistic=mp.solutions.holistic
hands=mp.solutions.hands
holis=holistic.Holistic()
drawing=mp.solutions.drawing_utils 
if "run" not in st.session_state:
    st.session_state["run"]="True"
try:
    emotion=np.load("emotion.npy")[0]
except:
    emotion=""

if not emotion:
    st.session_state["run"]!="True"
else:
    st.session_state["run"]!="False"
class EmotionProcessor:
    def recv(self,frame):
        frame=frame.to_ndarray(format="bgr24")
        lst=[]
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
            emotion=pred
            np.save("emotion.npy",np.array([pred]))
            cv.putText(frame, pred, (50,50),cv.FONT_ITALIC, 1, (255,0,0),2)



        drawing.draw_landmarks(frame,res.face_landmarks,holistic.FACEMESH_CONTOURS)
        drawing.draw_landmarks(frame,res.left_hand_landmarks,hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frame,res.right_hand_landmarks,hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(frame,format="bgr24")
lang=st.text_input("Language")
singer=st.text_input("singer")
if lang and singer and st.session_state["run"]!="False":
    webrtc_streamer(key="key",desired_playing_state=True,video_processor_factory=EmotionProcessor)


btn=st.button("Recommend me songs")
if btn:
    if not emotion:
        st.warning("Please let me capture the emotion")
        st.session_state["run"]!="True"
    else:
        webbrowser.open(f"https://www.youtube.com/results?search_query={lang}+{emotion}+song+{singer}")
        # np.save("emotion.npy",np.array([pred]))
        st.session_state["run"]!="False"
