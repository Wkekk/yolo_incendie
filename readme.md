# Brief détection d'incendie avec l'algorithme YOLOV5

## L'objectif du brief

L'objecti de ce brief est la création d'une application streamlit permettant la détection d'incendie sur des images fixes, des vidéos ou via une capture par webcam. Pour faire celà nous avons utilisé l'algorithme YOLOV5 après avoir labelisé un jeu d'image d'incendie pour l'entrainer.

## Le déroulé du projet

J'ai commencé par labeliser des images d'incendie pour entrainer l'algorithme YOLOV5, pour cela j'ai utilisé le site [makesense.ai](https://www.makesense.ai/). J'ai créé deux classe différentes utile à la détection d'incendie : `Flame` et `Smoke` que j'ai attribué à différentes parties des images du jeu de données (les flammes et la fumée, respectivement).

Une fois le jeu d'entrainement créé j'ai fais tourn,er l'algorithme YOLOV5 sur [google collab](https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb), je l'ai fais tourner avec 60 epoch pour avoir une précision convenable.

Une fois le meilleur modèle sauvegardé j'ai créé l'application streamlit voulu pour la détection.

##Le code

L'application est codé sous python en se basant sur un des travaux précédent le [brief détection de masque](https://github.com/Wkekk/mask_detection_v2).

Pour la lancer il faut exécuter la commande `streamlit run detec_feu.py` dans un terminal.

## Les bibliothèques requise :

	- cv2
	- streamlit
	- numpy
	- PIL
	- tensorflow
	- keras
	- os
	- torch
	- tempfile

### Le chargement du modèle :

le modèle est chargé avec la commande `model = torch.hub.load('ultralytics/yolov5', 'custom', 'best.onnx')
le fichier ``best.onnx` contient le modèle et ses poids

### La détection sur une image statique

La détection se fait via la fonction `detection_incendie` dans laquelle nous appliquons une prédiction du modèle sur l'image chargée et en récupérons les coordonnées des rectangles permettant de délimiter la zone détectée, ainsi que la classe à laquelle appartient la détection. Ensuite nous recréons ces rectangles sur  l'image via streamlit pour l'affichage.

### La détection via webcam et sur une vidéo

La détection via webcam et sur vidéo fonctionne de la même façon que pour les images statiques (à quelque changement près pour la partie traitement) mais sur un flot continue d'images.


## Performance :

Les données sur les performance du modèle se trouvent dans le dossier `analytics`