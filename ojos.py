from datetime import datetime
from typing import final
import cv2
import mediapipe as mp
import numpy as np
import time, datetime
from pygame import mixer


def eye_aspect_ratio(coordinates):
    d_A = np.linalg.norm(np.array(coordinates[1]) - np.array(coordinates[5]))
    d_B = np.linalg.norm(np.array(coordinates[2]) - np.array(coordinates[4]))
    d_C = np.linalg.norm(np.array(coordinates[0]) - np.array(coordinates[3]))

    return (d_A + d_B) / (2 * d_C)

cap = cv2.VideoCapture(0)

mp_face_mesh = mp.solutions.face_mesh
index_left_eye = [33, 160, 158, 133, 153, 144]
index_right_eye = [362, 385, 387, 263, 373, 380]
EAR_THRESH = 0.24
parpadeo = False
aux_counter = 0
conteo_sue = 0
start_sue = 0
start_sue2 = 0
final_sue = 0 
time_sue = 0
micro_sue = False
start_time = 0
final = 0
algo = 0
mixer.init()
mixer.music.load("alarma.mp3")

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1
) as face_mesh:
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        coordinates_left_eye = []
        coordinates_right_eye = []


        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                for index in index_left_eye:
                    x = int(face_landmarks.landmark[index].x * width)
                    y = int(face_landmarks.landmark[index].y * height)
                    coordinates_left_eye.append([x,y])
                    cv2.circle(frame, (x, y), 2, (0, 255, 255), 1)
                    cv2.circle(frame, (x, y), 1, (128, 0, 250), 1)
                    
                for index in index_right_eye:
                    x = int(face_landmarks.landmark[index].x * width)
                    y = int(face_landmarks.landmark[index].y * height)
                    coordinates_right_eye.append([x,y])
                    cv2.circle(frame, (x, y), 2, (128, 0, 250), 1)
                    cv2.circle(frame, (x, y), 1, (0, 255, 255), 1)

            ear_left_eye = eye_aspect_ratio(coordinates_left_eye)
            ear_right_eye = eye_aspect_ratio(coordinates_right_eye)
            ear = round((ear_right_eye + ear_left_eye) / 2, 3)
            #print(ear)
            
             
            if start_time == 0:
                start_time = time.time()
            #print(start_time)
            alarma_encendida = None
            # OJOS CERRADOS
             
            while ear > EAR_THRESH and parpadeo == True:
                aux_counter +=1
                print(aux_counter)
                parpadeo = False
                alarma_encendida = True 
                print("salio")

            if alarma_encendida == True and ear > EAR_THRESH:
                start_sue = time.time()
                final += 1
                print(final)
                print("estoy")
                alarma_encendida = False
            
                
            if algo >= 110:
                print("final")
                mixer.music.play()
                final = 0 
                algo = 0
                print("saliooooooooooooooooo")
                    
            while ear <= EAR_THRESH and parpadeo == False:
                final_sue = time.time()
                alarma_encendida = False
                start_sue = 0
                start_time = 0
                parpadeo = True
                print(parpadeo)
                algo = 0
            
            algo += 1
             
            time_sue = round(start_sue - final_sue, 0)
            if time_sue >= 3:
                micro_sue = True
                conteo_sue +=1
                start_sue = 0
                final_sue = 0
                print(conteo_sue)
                print("microsueño")
                print(time_sue)
            micro_sue = False


        cv2.imshow("Frame", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

cap.release()
cv2.destroyAllWindows()

