# -*- coding: utf-8 -*-

"""
Fonctions utilitaires pour Projet VslM - ENSAI 2020
Author : BERNARD Renan

Ce fichier .py definit des fonctions utiles pour le projet VslM.
"""

import numpy as np

def generate_sound(signal, model, nb_of_points):
    """
    Cette fonction permet de predire un certain nombre de points a
    l'aide du modele WaveNet de ce repo.
    Si ce signal est au format des .npy de ce repo, il faut proceder
    ainsi :
    x = np.load(FILE_PREDICT, allow_pickle=True)[0]
    x = np.reshape(x, (1, INPUT_SIZE, 1))

    Args:
        signal (numpy.array): le numpy.array contenant le signal, les
        points doivent etre des entiers entre 0 et 255.
        nb_of_points (int): le nombre de points a predire.
    Returns:
        numpy.array: le signal predit.
    """
    # On fait a chaque fois une prediction, puis on decale de 1.
    # La prediction se fait en prenant comme dernier point d'entre la
    # prediction precedente.
    for i in range(nb_of_points):
        pred = model.predict(signal[:, i:, :])
        # On recupere l'indice de la valeur max de la prediction.
        pred = np.argmax(pred)
        pred = np.reshape(pred, (1, 1, 1))
        # Puis on l'ajoute a la fin du signal.
        signal = np.concatenate((signal, pred), axis=1)
    return np.squeeze(signal)
