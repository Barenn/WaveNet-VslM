# -*- coding: utf-8 -*-

"""
FreeMusicArchive Scrapping pour projet VslM - ENSAI 2020
Author : BERNARD Renan

Ce fichier .py collecte les donnees au format .mp3 sur le site
FreeMusicArchive puis les stocke dans un format optimal pour
l'entrainement du reseau.
"""

import os
import requests
import numpy as np

from bs4 import BeautifulSoup
from pydub import AudioSegment

# Les parametres :
RECEPTIVE_FIELD_SIZE = 4096
STRIDE_STEP = 32

def fma_scrapping():
    """
    Cette fonction correspond a l'execution du scrapping, au traitement,
    et au stockage des donnees.
    """
    sample_nb = 0
    urls = get_all_mp3_urls(
        "https://freemusicarchive.org/music/Blue_Dot_Sessions")
    for url in urls:
        sound_name = url.split("/")[-1]
        # Le fichier .mp3 est telecharge localement.
        # Des qu'un nouveau fichier est dl, le precdent est supprimme.
        download_mp3_file(url)
        audio_array = read(";/current_sound.mp3")
        audio_array = audio_array[1].mean(axis=1)
        # On ne prend qu'un point sur 10.L
        # La frequence d'echantillonage est divisee par 10.
        audio_array = audio_array[1::10]
        audio_array = scale_audio_int16_to_float64(audio_array)
        audio_array = scale_audio_float64_to_uint8(audio_array)
        X, Y = get_audio_sample_batches(
            audio=audio_array,
            receptive_field_size=RECEPTIVE_FIELD_SIZE,
            stride_step=STRIDE_STEP)
        for i in range(len(X)):
            data_current = (X[i], Y[i])
            if np.random.binomial(n=1, p=0.1):
                np.save(
                    "Data/Validation/" + str(sample_nb) + ".npy",
                    data_current)
                print(
                    "Sample :", sample_nb, "-",
                    sound_name, "- Validation.")
            else:
                np.save(
                    "Data/Train/" + str(sample_nb) +".npy",
                    data_current)
                print(
                    "Sample :", sample_nb, "-", sound_name, "- Train.")
            sample_nb += 1

def download_mp3_file(url):
    """
    Cette fonction telecharge le fichier .mp3 correspond a l'url donnee.

    Args:
        url (stirng): url d'un fichier .mp3.
    Returns:
        None.
    """
    os.system(
        "wget -O " + "current_sound.mp3" + " " + url +
        " >/dev/null 2>&1")
    return None

def get_all_mp3_urls(url_page):
    """
    Recupere l'ensemble des urls correspondant a un fichier .mp3 contenu
    dans la page html correspond a l'url donnee.

    Args:
        url_page (str): url de la page html dans laquelle chercher les
        urls.
    Returns:
        list: la liste contenant les urls au format cible.
    """
    urls = []
    html = requests.get(url_page).text
    bs = BeautifulSoup(html, features="lxml")
    for link in bs.find_all('a'):
        if link.has_attr('href'):
            if link.attrs['href'][-3:] == "mp3":
                urls.append(link.attrs['href'])
    return urls

def read(f, normalized=False):
    """
    Cette fonction transforme un fichier .mp3 en un numpy.array.

    Args:
        f (str): le chemin vers le fichier .mp3.
        normalized (bool): si le numpy.array doit etre normalise ou non.
    Returns:
        int: la frequence d'echantillonage du son.
        numpy.array: contient le son en question, la dimension depend du
        nombre de channels.
    """
    a = AudioSegment.from_mp3(f)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    if normalized:
        return a.frame_rate, np.float32(y) / 2**15
    else:
        return a.frame_rate, y

def scale_audio_int16_to_float64(arr):
    """
    Cette fonction transforme un numpy.array d'entiers 16 bits
    [-2^15, 2^15 - 1] en un flottant entre 1 et -1.

    Args:
        arr (numpy.array): contient le numpy.array d'entiers 16 bits a
        transformer.
    Returns:
        numpy.array: le numpy.array transforme.
    """
    vmax = np.iinfo(np.int16).max
    vmin = np.iinfo(np.int16).min
    arr = arr.astype(np.float64)
    return 2 * ((arr - vmin) / (vmax - vmin)) - 1

def scale_audio_float64_to_uint8(arr):
    """
    Cette fonction tranforme un numpy.array de flottant entre -1 et 1 en
    un numpy.array d'entiers entre 0 et 255.

    Args:
        arr (numpy.array): contient le numpy.array d'entiers a
        transformer.
    Returns:
        numpy.array: le numpy.array transforme.
    """
    vmax = np.iinfo(np.uint8).max
    arr = ((arr + 1) / 2) * vmax
    arr = arr.astype(np.uint8)
    return arr

def to_one_hot(xt):
    """
    Cette fonction transformer un entier entre 0 et 255 en un one_hot
    vecteur, ou la valeur 1 correspond à la valeur de l'entier.

    Args:
        xt (int): l'entier cible.
    Returns:
        np.array: le vecteur one-hot-encoded.
    """
    return np.eye(256, dtype="uint8")[xt]

def get_audio_sample_batches(audio, receptive_field_size,
                             stride_step=32):
    """
    Cette fonction transforme un numpy.array correspondant a un son,
    en un ensemble d'echantillons pouvant etre feed en entre du reseau.

    Args:
        audio (numpy.array): le vecteur correspondant au son.
        receptive_field_size (int): la taille du vecteur qui peut etre
        lu par le reseau. Ca correspond au nombre d'elements lu par le
        reseau.
        strid_step (int): le decalage entre les echantillons extraits.
    Returns:
        numpy.array: contient l'ensemble des vecteurs qui correspondent
        a l'entree du reseau.
        numpy.array: contient les y correspondant au point suivant les
        x precedent, one hot encoded.
    """
    offset = 0
    X = []
    Y = []
    while offset + receptive_field_size < len(audio):
        if np.random.binomial(n=1, p=0.01):
        # On ne garde que 1% des vecteurs, choisis aleatoirement.
            X.append(
                audio[
                    offset:offset + receptive_field_size
                ].reshape(receptive_field_size, 1))
            y_current = audio[offset + receptive_field_size + 1]
            # Le y correpond au point suivant.
            y_current = to_one_hot(y_current)
            Y.append(y_current)
        offset += stride_step
    return np.array(X), np.array(Y)

if __name__ == '__main__':
    fma_scrapping()
