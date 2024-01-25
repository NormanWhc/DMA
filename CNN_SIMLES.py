import numpy as np
from rdkit import Chem

from keras import layers, models, optimizers
from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Dropout, BatchNormalization, Reshape, LeakyReLU
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
import tensorflow as tf

SMILES_CHARS = [' ',
                  '#', '%', '(', ')', '+', '-', '.', '/', '_', '_', '_', '_', '_', '_', '_', '_'
                  '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                  '=', '@',
                  'A', 'B', 'C', 'F', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P',
                  'R', 'S', 'T', 'V', 'X', 'Z',
                  '[', '\\', ']',
                  'a', 'b', 'c', 'e', 'g', 'i', 'l', 'n', 'o', 'p', 'r', 's',
                  't', 'u', 'd']
smi2index = dict( (c,i) for i,c in enumerate( SMILES_CHARS ) )
index2smi = dict( (i,c) for i,c in enumerate( SMILES_CHARS ) )
def smiles_encoder( smiles, maxlen=1024 ):
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles( smiles ))
    X = np.zeros( ( maxlen, len( SMILES_CHARS ) ) )
    for i, c in enumerate( smiles ):
        X[i, smi2index[c] ] = 1
    return X

input_shape = (1024, 64, 1)
input_tensor = layers.Input(input_shape)

conv1 = layers.Conv2D(1, (3,3), padding='same', activation='relu')(input_tensor)
pooling1 = layers.MaxPool2D(name='imlatent_layer')(conv1)

conv2 = layers.Conv2D(1, (3,3), padding='same', activation='relu')(pooling1)
pooling2 = layers.MaxPool2D(name='imlatent_layer2')(conv2)

conv3 = layers.Conv2D(1, (3,3), padding='same', activation='relu')(pooling2)
pooling3 = layers.MaxPool2D(name='imlatent_layer3')(conv3)

conv4 = layers.Conv2D(1, (3,3), padding='same', activation='relu')(pooling3)
pooling4 = layers.MaxPool2D(name='imlatent_layer4')(conv4)

flatten1 = layers.Flatten()(pooling4)
dense1 = layers.Dense(128, activation='relu',name='latent_layer')(flatten1)

latent_tensor = dense1

dense2 = layers.Dense(64 * 4, activation='relu')(latent_tensor)
reshaped = tf.reshape(dense2, [-1, 64, 4, 1])

upsample1 = layers.UpSampling2D()(reshaped)
deconv1 = layers.Conv2DTranspose(1, (3,3), padding='same', activation='sigmoid')(upsample1)

upsample2 = layers.UpSampling2D()(deconv1)
deconv2 = layers.Conv2DTranspose(1, (3,3), padding='same', activation='sigmoid')(upsample2)

upsample3 = layers.UpSampling2D()(deconv2)
deconv3 = layers.Conv2DTranspose(1, (3,3), padding='same', activation='sigmoid')(upsample3)

upsample4 = layers.UpSampling2D()(deconv3)
deconv4 = layers.Conv2DTranspose(1, (3,3), padding='same', activation='sigmoid')(upsample4)

output_tensor = deconv4

ae = models.Model(input_tensor, output_tensor)

model = models.load_model("./model/CNN.ckpt", compile=False)

def cnn_smiles(smi_list):
    drug_vector_list = []
    for i in smi_list:
        encoded = smiles_encoder(i)
        feature_layer_model = models.Model(inputs=ae.input, outputs=ae.get_layer('latent_layer').output)
        feature_output = feature_layer_model.predict(encoded.reshape(1, 1024, 64, 1))
        drug_vector_list.append(feature_output[0])

    return np.array(drug_vector_list)