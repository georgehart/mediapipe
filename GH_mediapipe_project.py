#!/usr/bin/env python
# coding: utf-8

# # Biering Sorensen Test ... some trials

# <div style="background-color:#EAE9FF; padding: 10px; border-radius: 5px;">
# Goal    : Study of Google Mediapipe </br>
# author  : <b>Georges Hart</b></br>
# date    : 21 oktober '25</br>
# 
# comments: none ... yet :-)
# 
# 
# </div>

# ## Mediapipe from Google
# ---

# <div style="background-color: lightyellow; padding: 10px; border-radius: 5px;">
# <b>MediaPipe</b></br> is an open-source, cross-platform framework developed by Google for building and deploying applied Machine Learning (ML) pipelines, particularly for processing live and streaming multimedia data like video, audio, and sensor information. It uses a graph-based dataflow architecture where processing tasks (called "Calculators") are connected to form a pipeline, allowing for real-time performance on various devices including mobile (Android/iOS), web, desktop, and edge devices. It is especially well-known for providing a collection of ready-to-use solutions like hand tracking, pose estimation, and face detection, enabling developers to quickly integrate complex computer vision and ML features into their applications.
# 
# </br><b>To read :</b> [link](http://localhost:8888/doc/tree/venv_22102025/pdf/Understanding%20the%20Biering-S%C3%B8rensen%20test.pdf)    
# </div>

# In[7]:


#!pip install sys


# In[8]:


# !pip install mediapipe


# In[9]:


from IPython.display import YouTubeVideo


# ## General information about Google mediapipe

# In[10]:


# For example, for the URL https://www.youtube.com/watch?v=yOP_FY2KTm8_Is, the ID is yOP_FY2KTm8
YouTubeVideo('yOP_FY2KTm8', width=800, height=600)


# # HANDS

#  source: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker

# ![hand_mediapipe.png](attachment:de790bb1-2905-4cb6-98d4-fdce11826fcf.png)

# In[11]:


'''  -----------------------------------------------------------------------

Title  : Videocapture with MediaPipe (Google)
                     Hands
         ____________________________________

Author : George Hart
Date   : 22 oktober 2025
Release: 0.1

Source : https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker

Comment: 





---------------------------------------------------------------------------- '''

import mediapipe as mp
import videosource
from videosource import WebcamSource

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=4)


def main():
    source = WebcamSource()

    with mp_hands.Hands(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as hands:

        for idx, (frame, frame_rgb) in enumerate(source):

            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec,
                    )

            source.show(frame)


if __name__ == "__main__":
    main()


# ## Installed libraries (see requirements.txt)

# In[12]:


# !pip list
# !pip freeze >requirements.txt


# ## System PATH ...

# In[13]:


import sys
print (sys.path)


# In[ ]:




