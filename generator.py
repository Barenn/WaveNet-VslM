# -*- coding: utf-8 -*-

"""
Generateurs de donnees pour Projet VslM- ENSAI 2020
Author : BERNARD Renan

Ce fichier .py definit la classe permettant la generation des donnes.
"""

import os
import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    """
    Genere les donnees au format necessaire pour le reseau WaveNet.
    """

    def __init__(self, path_to_files, batch_size=32,
                 dim=(1024, 1), n_classes=256, shuffle=True):
        """
        Initialisation.

        Args:
            path_to_files (str): le chemin vers les donnes.
            batch_size (int): batch size pour l'entrainement.
            dim (tuple): la dimension des donnees d'entree (les x).
            n_classes (int): le nombre de classe, notre WaveNet predit
            un entier entre 0 et 255, donc 256 classes.
            shuffle (bool): si les donnees doivent etre melangees a
            chaque epoch.
        """
        self.path_to_files = path_to_files
        self.dim = dim
        self.batch_size = batch_size
        self.list_ids = [
            fn for fn in os.listdir(path_to_files)
            if fn.endswith(".npy")]
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """
        Le nombre de batches par epoch.
        """
        return int(np.floor(len(self.list_ids) / self.batch_size))

    def __getitem__(self, index):
        """
        Genere une batch.

        Args:
            index (int): l'index de la batch courrante.
        Returns:
            numpy.array: contient les x de la batch.
            numpy.array: contient les y de la batch.
        """
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of ids
        list_ids_temp = [self.list_ids[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_ids_temp)

        return X, y

    def on_epoch_end(self):
        """
        Mise a jour des index a chaque epoch.
        """
        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids_temp):
        """
        Genere une batch de donnees au format voulu, en utilisant les
        ids des donnees a aggreger.

        Args:
            list_ids_temp (list): contient la liste des ids des fichiers
            a aller chercher pour creer la batch.
        Returns:
            numpy.array: contient les x de la batch.
            numpy.array: contient les y de la batch.
        """
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, self.n_classes), dtype=int)

        # Generate data
        for i, id in enumerate(list_ids_temp):
            file_current = self.path_to_files + id
            X[i,], y[i] = np.load(file_current, allow_pickle=True)
        return X, y
