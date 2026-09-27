# ✋ Hand Shape Manipulator

Déplacez, agrandissez et faites tourner des formes géométriques **avec vos doigts**, devant votre webcam.
Le suivi des mains en temps réel utilise [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker), et l'affichage OpenCV.

## Gestes

| Geste | Action |
|---|---|
| Pincer (pouce + index) sur une forme | Attraper et déplacer la forme |
| Pincer avec les 2 mains, puis écarter ou rapprocher | Agrandir ou rétrécir les formes sélectionnées |
| Pincer avec les 2 mains, puis tourner | Faire tourner les formes sélectionnées |
| `D` | Tout désélectionner |
| `Q` | Quitter |

## Installation

```bash
pip install -r requirements.txt
python hand_shape_manipulator.py
```

Au premier lancement, le modèle `hand_landmarker.task` (environ 7,5 Mo) est téléchargé automatiquement.

## Fonctionnement

1. Chaque image de la webcam est inversée comme un miroir, puis analysée par MediaPipe (jusqu'à 2 mains, 21 points clés par main).
2. Un **pincement** est détecté quand la distance normalisée entre le pouce (point 4) et l'index (point 8) passe sous `0.055`.
3. Déplacement : la forme touchée au moment du pincement suit le point milieu entre le pouce et l'index.
4. Deux mains qui pincent : les formes sélectionnées changent de taille selon la variation de **distance** entre les deux pinces, et tournent selon la variation d'**angle** entre elles.

Six formes sont proposées : cercle, rectangle, triangle, pentagone, hexagone et étoile.

## Technologies

Python · OpenCV · MediaPipe Tasks · NumPy
