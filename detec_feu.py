import cv2
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import keras
import os
import torch
import tempfile


#Chargement du modèle permettant de détecter le port du masque
model = torch.hub.load('ultralytics/yolov5', 'custom', 'best.onnx')


#detction sur des images statiques
def detection_incendie(our_image):
    result =model(our_image)                        #chargement de l'image
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img,1)
    nb_detec = len(result.xyxy[0])                  #prediction
    for i in range(nb_detec) :
        conf = result.xyxy[0][i][4]                 #recuperation de la classe et de la confiance de la prediction
        cat = result.xyxy[0][i][5]
        #tracé des boites
        if conf > 0.3 :
            cv2.rectangle(img, (int(result.xyxy[0][i][0]), int(result.xyxy[0][i][1])), (int(result.xyxy[0][i][2]), int(result.xyxy[0][i][3])), (0, 0, 255), 2)
            cv2.rectangle(img, (int(result.xyxy[0][0][0]), int(result.xyxy[0][0][1])), (int(result.xyxy[0][0][2]), int(result.xyxy[0][0][1])+15), (255, 255, 255), -1)
            if (cat == 0 ):
                #affichage de la prediction
                # cv2.rectangle(image, (startX, startY), (endX, endY),(0, 0, 255), 2)
                cv2.putText(img, "FLAME "+str(round(float(conf), 3)),(int(result.xyxy[0][i][0]), int(result.xyxy[0][i][1])+15) ,cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 0, 0), 1)
            else:
                # cv2.rectangle(image, (startX, startY), (endX, endY),(0, 255, 0), 2)
                cv2.putText(img, "SMOKE "+str(round(float(conf), 3)), (int(result.xyxy[0][i][0]), int(result.xyxy[0][i][1])+15), cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 0, 0), 1)
    
    return img 

def main():
    """Fire Detection App"""

    st.title("Fire Detection App")

    ###PARTIE WEBCAM###

    st.title("Webcam Live Feed")        #ouverture de la webcam quand la checkbox 'run' est cochée
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    while run:
        #chargement et préparation des différentes frame 
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        capture = cv2.resize(frame, (224, 224))
        #prediction
        result_vid = model(capture)
        print(result_vid)
        if len(result_vid.xyxy[0]) > 0 :
            #récupération des info de la prediction
            conf = result_vid.xyxy[0][0][4]
            cat = result_vid.xyxy[0][0][5]
            #creation des boites
            if conf > 0.3 :
                cv2.rectangle(frame, (int(result_vid.xyxy[0][0][0]), int(result_vid.xyxy[0][0][1])), (int(result_vid.xyxy[0][0][2]), int(result_vid.xyxy[0][0][3])), (0, 0, 255), 2)
                cv2.rectangle(frame, (int(result_vid.xyxy[0][0][0]), int(result_vid.xyxy[0][0][1])), (int(result_vid.xyxy[0][0][2]), int(result_vid.xyxy[0][0][1])+15), (255, 255, 255), -1)
                #affichage de la prediction
                if (cat == 0 ):
                    # cv2.rectangle(image, (startX, startY), (endX, endY),(0, 0, 255), 2)
                    cv2.putText(frame, "FLAME "+str(round(float(conf), 3)),(int(result_vid.xyxy[0][0][0]), int(result_vid.xyxy[0][0][1])+15) ,cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 0, 0), 1)
                else:
                    # cv2.rectangle(image, (startX, startY), (endX, endY),(0, 255, 0), 2)
                    cv2.putText(frame, "SMOKE "+str(round(float(conf), 3)), (int(result_vid.xyxy[0][0][0]), int(result_vid.xyxy[0][0][1])+15), cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 0, 0), 1)
        FRAME_WINDOW.image(frame)
    else:
        st.write('Stopped')

    ### Front et choix de l'action voulue ###

    activities = ["Detection","About"]
    choice = st.sidebar.selectbox("Select Activty",activities)

    if choice == 'Detection':
        ### Détection via upload de fichier (image ou video) ###
        st.subheader("Fire Detection")
        image_file = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])
        video_file = st.file_uploader("Upload Video",type=['mp4'])
        if image_file is not None :
            ### PARTIE IMAGE ###
            our_image = Image.open(image_file)
            st.text("Original Image")
            # st.write(type(our_image))
            st.image(our_image)

            # Fire Detection
            if st.button("Process"):
                result_img=detection_incendie(our_image)
                st.image(result_img)

        if video_file is not None :
            ### PARTIE VIDEO ###
            st.title("Video Feed")
            live = st.checkbox('launch')
            FRAME_WINDOW = st.image([])
            #chargement de la video dans un fichier temporaire pour pouvoir la lire avec OpenCV
            stframe = st.empty()
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(video_file.read())
            cap = cv2.VideoCapture(tfile.name)

            result = cv2.VideoWriter('detec_vid.mp4', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, (224,224))

            while(live):
                #préparation des frame
                _, frame = cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                FRAME_WINDOW.image(frame)
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                capture = cv2.resize(frame, (224, 224))
                #même schéma que pour la webcam ici
                result_vid = model(capture)
                print(result_vid)
                if len(result_vid.xyxy[0]) > 0 :
                    conf = result_vid.xyxy[0][0][4]
                    cat = result_vid.xyxy[0][0][5]
                    if conf > 0.3 :
                        cv2.rectangle(frame, (int(result_vid.xyxy[0][0][0]), int(result_vid.xyxy[0][0][1])), (int(result_vid.xyxy[0][0][2]), int(result_vid.xyxy[0][0][3])), (0, 0, 255), 2)
                        cv2.rectangle(frame, (int(result_vid.xyxy[0][0][0]), int(result_vid.xyxy[0][0][1])), (int(result_vid.xyxy[0][0][2]), int(result_vid.xyxy[0][0][1])+15), (255, 255, 255), -1)
                        if(cat == 0) :    
                            # cv2.rectangle(image, (startX, startY), (endX, endY),(0, 0, 255), 2)
                            cv2.putText(frame, "FLAME "+str(round(float(conf), 3)),(int(result_vid.xyxy[0][0][0]), int(result_vid.xyxy[0][0][1])+15) ,cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 0, 0), 1)
                        else:
                            # cv2.rectangle(image, (startX, startY), (endX, endY),(0, 255, 0), 2)
                            cv2.putText(frame, "SMOKE "+str(round(float(conf), 3)), (int(result_vid.xyxy[0][0][0]), int(result_vid.xyxy[0][0][1])+15), cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 0, 0), 1)
                    
                #lecture de la vidéo
                FRAME_WINDOW.image(frame)
            else:
                st.write('Stopped')
            

    # front placeholder

    elif choice == 'About':
        st.subheader("About Fire Detection App")
        st.text("fait par Quentin Guichoux")
        st.success("y'a pas grand chose à voir ici, l'autre page est mieux")


if __name__ == '__main__':
        main()  