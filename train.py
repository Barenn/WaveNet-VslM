# -*- coding: utf-8 -*-

"""
Entrainement de WaveNet - ENSAI 2020
Author : BERNARD Renan

Ce fichier .py execute l'entrainement du reseau WaveNet.
"""

from keras.callbacks import ModelCheckpoint

from model import build_wavenet_model
from generator import DataGenerator

EXPERIMENT_NAME = "exp1"
TRAIN_PATH = "./Data/Train/"
VALID_PATH = "./Data/Validation/"
INPUT_SIZE = 4096
NUM_FILTERS = 16
KERNEL_SIZE = 2
NUM_RESIDUAL_BLOCKS = 9
BATCH_SIZE = 256
EPOCHS = 20
FILE_PREDICT_CALLBACK = "./Data/Validation/14.npy"

checkpoint_path = "weights_" + EXPERIMENT_NAME + ".hdf5"

checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor='val_loss',
    verbose=0,
    save_best_only=True,
    mode='auto',
    period=1)

print("Building model...")
model = build_wavenet_model(
    input_size=INPUT_SIZE,
    num_filters=NUM_FILTERS,
    kernel_size=KERNEL_SIZE,
    num_residual_blocks=NUM_RESIDUAL_BLOCKS)
print("OK...")

training_generator = DataGenerator(
    path_to_files=TRAIN_PATH,
    batch_size=BATCH_SIZE,
    dim=(INPUT_SIZE, 1),
    n_classes=256,
    shuffle=True)

validation_generator = DataGenerator(
    path_to_files=VALID_PATH,
    batch_size=BATCH_SIZE,
    dim=(INPUT_SIZE, 1),
    n_classes=256,
    shuffle=True)

model.fit_generator(
    generator=training_generator,   
    validation_data=validation_generator,
    use_multiprocessing=True,
    workers=4,
    epochs=EPOCHS,
    verbose=1,
    callbacks=[checkpoint])
