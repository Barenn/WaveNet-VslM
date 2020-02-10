# WaveNet-VslM

Une implémentation éco+ de WaveNet pour le Projet de Veille sur les Médias.
L'objectif était de réduire au maximum la charge sur la RAM et sur les processeurs pour faire tourner le modèle sur un ordinateur peu performant.

## Importations

Bien entendu il y a un certain nombre de packages à importer. Pas encore de requirements.txt.

## Données

Pour récupérer les données, il faut créer un folder "Data", puis créer dedans 2 folders "Train" et "Validation". Il suffit ensuite d'executer le script __fma_scrapping.py__ pour collecter puis traiter les données.

## Entraînement

Pour l'entraînement du modèle, le fichier est __train.py__.

## TODO

- Réduire encore la base d'entraînement en échantillonant au fil des epochs.
