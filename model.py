from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis as PA
import pandas as pd
import numpy as np
# from tensorflow.keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Flatten, concatenate, Reshape,BatchNormalization, MultiHeadAttention, Concatenate, Dropout
import keras.backend as K
# from keras.layers.recurrent import SimpleRNN
# from keras import layers
# from keras.optimizers import Adam
# from CapsuleLayer import Capsule
import matplotlib.pyplot as plt
# from matplotlib import pyplot as plt
from bert import bert
import matplotlib
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
import tensorflow as tf
from sklearn.model_selection import train_test_split
from Capsule_MPNN import *
from ReadData import *
from keras.optimizers.legacy import Adam
import tensorflow as tf
from sklearn.model_selection import train_test_split
# import pickle
def split(protein, drug, y, drug_encoder, name,dti):
    if not os.path.exists(name):
        os.makedirs(name)
    if dti == 'data/ncDR.csv':
        # train_p, test_p, train_d, test_d, train_y, test_y = train_test_split(protein, drug[0], y, test_size=0.2,
        #                                                                      random_state=1, stratify=y)  # 1
        if drug_encoder == "MPNN":
            train_p = protein[:7130]
            train_d0 = drug[0][0][:7130]
            train_d1 = drug[0][1][:7130]
            train_d2 = drug[0][2][:7130]
            train_y = y[:7130]
            test_p = protein[7130:]
            test_d0 = drug[0][0][7130:]
            test_d1 = drug[0][1][7130:]
            test_d2 = drug[0][2][7130:]
            test_y = y[7130:]
            test_d = (tf.ragged.constant(test_d0, dtype=tf.float32), tf.ragged.constant(test_d1, dtype=tf.float32),
                      tf.ragged.constant(test_d2, dtype=tf.int64))
            train_d = (train_d0, train_d1, train_d2)
        else:
            train_p = protein[:7130, :]
            train_d = drug[0][:7130, :]
            train_y = y[:7130]
            test_p = protein[7130:, :]
            test_d = drug[0][7130:, :]
            test_y = y[7130:]
        with open(name + "/sample_summary.txt", "w") as fw:
            fw.write("train_protein\ttest_protein\ttrain_drug\ttest_drug\ttrain_y\ttest_y\n")
            # fw.write(str(train_p.shape) + "\t" + str(test_p.shape) + "\t" + str(train_d.shape) + "\t" + str(
            #     test_d.shape) + "\t" + str(len(train_y)) + "\t" + str(len(test_y)))
        # print(drug[0], "drug[0]",train_d,"train_d",test_d,"test_d")
    if dti == 'data/RNAInter.csv':

        if drug_encoder == "MPNN":
            train_p = protein[:9182, :]
            train_d0 = drug[0][0][:9182, :]
            train_d1 = drug[0][1][:9182, :]
            train_d2 = drug[0][2][:9182, :]
            train_y = y[:9182]
            test_p = protein[9182:, :]
            test_d0 = drug[0][0][9182:, :]
            test_d1 = drug[0][1][9182:, :]
            test_d2 = drug[0][2][9182:, :]
            test_y = y[9182:]
            test_d = (tf.ragged.constant(test_d0, dtype=tf.float32), tf.ragged.constant(test_d1, dtype=tf.float32),
                      tf.ragged.constant(test_d2, dtype=tf.int64))
            train_d = (train_d0, train_d1, train_d2)
        else:
            train_p = protein[:9182, :]
            train_d = drug[0][:9182, :]
            train_y = y[:9182]
            test_p = protein[9182:, :]
            test_d = drug[0][9182:, :]
            test_y = y[9182:]
        with open(name + "/sample_summary.txt", "w") as fw:
            fw.write("train_protein\ttest_protein\ttrain_drug\ttest_drug\ttrain_y\ttest_y\n")
            # fw.write(str(train_p.shape) + "\t" + str(test_p.shape) + "\t" + str(train_d.shape) + "\t" + str(
            #     test_d.shape) + "\t" + str(len(train_y)) + "\t" + str(len(test_y)))
        # print(drug[0], "drug[0]",train_d,"train_d",test_d,"test_d")
    if dti == 'data/SM2miR1.csv':
        if drug_encoder == "MPNN":
            train_p = protein[:2140, :]
            train_d0 = drug[0][0][:2140, :]
            train_d1 = drug[0][1][:2140, :]
            train_d2 = drug[0][2][:2140, :]
            train_y = y[:2140]
            test_p = protein[2140:, :]
            test_d0 = drug[0][0][2140:, :]
            test_d1 = drug[0][1][2140:, :]
            test_d2 = drug[0][2][2140:, :]
            test_y = y[2140:]
            test_d = (tf.ragged.constant(test_d0, dtype=tf.float32), tf.ragged.constant(test_d1, dtype=tf.float32),
                      tf.ragged.constant(test_d2, dtype=tf.int64))
            train_d = (train_d0, train_d1, train_d2)
        else:
            train_p = protein[:2140, :]
            train_d = drug[0][:2140, :]
            train_y = y[:2140]
            test_p = protein[2140:, :]
            test_d = drug[0][2140:, :]
            test_y = y[2140:]
        with open(name + "/sample_summary.txt", "w") as fw:
            fw.write("train_protein\ttest_protein\ttrain_drug\ttest_drug\ttrain_y\ttest_y\n")
            # fw.write(str(train_p.shape) + "\t" + str(test_p.shape) + "\t" + str(train_d.shape) + "\t" + str(
            #     test_d.shape) + "\t" + str(len(train_y)) + "\t" + str(len(test_y)))
        # print(drug[0], "drug[0]",train_d,"train_d",test_d,"test_d")
    if dti == 'data/SM2miR2.csv':
        if drug_encoder == "MPNN":
            train_p = protein[:3140, :]
            train_d0 = drug[0][0][:3140, :]
            train_d1 = drug[0][1][:3140, :]
            train_d2 = drug[0][2][:3140, :]
            train_y = y[:3140]
            test_p = protein[3140:, :]
            test_d0 = drug[0][0][3140:, :]
            test_d1 = drug[0][1][3140:, :]
            test_d2 = drug[0][2][3140:, :]
            test_y = y[3140:]
            test_d = (tf.ragged.constant(test_d0, dtype=tf.float32), tf.ragged.constant(test_d1, dtype=tf.float32),
                      tf.ragged.constant(test_d2, dtype=tf.int64))
            train_d = (train_d0, train_d1, train_d2)
        else:
            train_p = protein[:3140, :]
            train_d = drug[0][:3140, :]
            train_y = y[:3140]
            test_p = protein[3140:, :]
            test_d = drug[0][3140:, :]
            test_y = y[3140:]
        with open(name + "/sample_summary.txt", "w") as fw:
            fw.write("train_protein\ttest_protein\ttrain_drug\ttest_drug\ttrain_y\ttest_y\n")
            # fw.write(str(train_p.shape) + "\t" + str(test_p.shape) + "\t" + str(train_d.shape) + "\t" + str(
            #     test_d.shape) + "\t" + str(len(train_y)) + "\t" + str(len(test_y)))
        # print(drug[0], "drug[0]",train_d,"train_d",test_d,"test_d")
    if dti == 'data/SM2miR3.csv':
        if drug_encoder == "MPNN":
            train_p = protein[:3618, :]
            train_d0 = drug[0][0][:3618, :]
            train_d1 = drug[0][1][:3618, :]
            train_d2 = drug[0][2][:3618, :]
            train_y = y[:3618]
            test_p = protein[3618:, :]
            test_d0 = drug[0][0][3618:, :]
            test_d1 = drug[0][1][3618:, :]
            test_d2 = drug[0][2][3618:, :]
            test_y = y[3618:]
            test_d = (tf.ragged.constant(test_d0, dtype=tf.float32), tf.ragged.constant(test_d1, dtype=tf.float32),
                      tf.ragged.constant(test_d2, dtype=tf.int64))
            train_d = (train_d0, train_d1, train_d2)
        else:
            train_p = protein[:3618, :]
            train_d = drug[0][:3618, :]
            train_y = y[:3618]
            test_p = protein[3618:, :]
            test_d = drug[0][3618:, :]
            test_y = y[3618:]
        with open(name + "/sample_summary.txt", "w") as fw:
            fw.write("train_protein\ttest_protein\ttrain_drug\ttest_drug\ttrain_y\ttest_y\n")
            # fw.write(str(train_p.shape) + "\t" + str(test_p.shape) + "\t" + str(train_d.shape) + "\t" + str(
            #     test_d.shape) + "\t" + str(len(train_y)) + "\t" + str(len(test_y)))
        # print(drug[0], "drug[0]",train_d,"train_d",test_d,"test_d")
    else:
        if drug_encoder == "fingerprint":
            train_p, test_p, train_d, test_d, train_y, test_y = train_test_split(protein, drug[0], y, test_size=0.2,
                                                                                 random_state=1, stratify=y)  # 1
            with open(name + "/sample_summary.txt", "w") as fw:
                fw.write("train_protein\ttest_protein\ttrain_drug\ttest_drug\ttrain_y\ttest_y\n")
                fw.write(str(train_p.shape) + "\t" + str(test_p.shape) + "\t" + str(train_d.shape) + "\t" + str(
                    test_d.shape) + "\t" + str(len(train_y)) + "\t" + str(len(test_y)))
            # print(drug[0], "drug[0]",train_d,"train_d",test_d,"test_d")
        if drug_encoder == "mole":
            train_p, test_p, train_d, test_d, train_y, test_y = train_test_split(protein, drug[0], y, test_size=0.2,
                                                                                 random_state=1, stratify=y)  # 1
            with open(name + "/sample_summary.txt", "w") as fw:
                fw.write("train_protein\ttest_protein\ttrain_drug\ttest_drug\ttrain_y\ttest_y\n")
                # fw.write(str(train_p.shape) + "\t" + str(test_p.shape) + "\t" + str(train_d.shape) + "\t" + str(
                #     test_d.shape) + "\t" + str(len(train_y)) + "\t" + str(len(test_y)))
            # print(drug[0], "drug[0]",train_d,"train_d",test_d,"test_d")
        if drug_encoder == "smole":
            train_p, test_p, train_d, test_d, train_y, test_y = train_test_split(protein, drug[0], y, test_size=0.2,
                                                                                 random_state=1, stratify=y)  # 1
            with open(name + "/sample_summary.txt", "w") as fw:
                fw.write("train_protein\ttest_protein\ttrain_drug\ttest_drug\ttrain_y\ttest_y\n")
                # fw.write(str(train_p.shape) + "\t" + str(test_p.shape) + "\t" + str(train_d.shape) + "\t" + str(
                #     test_d.shape) + "\t" + str(len(train_y)) + "\t" + str(len(test_y)))
            # print(drug[0], "drug[0]",train_d,"train_d",test_d,"test_d")
        if drug_encoder == "chemberta":
            train_p, test_p, train_d, test_d, train_y, test_y = train_test_split(protein, drug[0], y, test_size=0.2,
                                                                                 random_state=1, stratify=y)  # 1
            with open(name + "/sample_summary.txt", "w") as fw:
                fw.write("train_protein\ttest_protein\ttrain_drug\ttest_drug\ttrain_y\ttest_y\n")
                # fw.write(str(train_p.shape) + "\t" + str(test_p.shape) + "\t" + str(train_d.shape) + "\t" + str(
                #     test_d.shape) + "\t" + str(len(train_y)) + "\t" + str(len(test_y)))
            # print(drug[0], "drug[0]",train_d,"train_d",test_d,"test_d")
        if drug_encoder == "smolechemfusion":
            train_p, test_p, train_d, test_d, train_y, test_y = train_test_split(protein, drug[0], y, test_size=0.2,
                                                                                 random_state=1, stratify=y)  # 1
            with open(name + "/sample_summary.txt", "w") as fw:
                fw.write("train_protein\ttest_protein\ttrain_drug\ttest_drug\ttrain_y\ttest_y\n")
                # fw.write(str(train_p.shape) + "\t" + str(test_p.shape) + "\t" + str(train_d.shape) + "\t" + str(
                #     test_d.shape) + "\t" + str(len(train_y)) + "\t" + str(len(test_y)))
            # print(drug[0], "drug[0]",train_d,"train_d",test_d,"test_d")

        if drug_encoder == "smolemolefusion":
            train_p, test_p, train_d, test_d, train_y, test_y = train_test_split(protein, drug[0], y, test_size=0.2,
                                                                                 random_state=1, stratify=y)  # 1
            with open(name + "/sample_summary.txt", "w") as fw:
                fw.write("train_protein\ttest_protein\ttrain_drug\ttest_drug\ttrain_y\ttest_y\n")
                # fw.write(str(train_p.shape) + "\t" + str(test_p.shape) + "\t" + str(train_d.shape) + "\t" + str(
                #     test_d.shape) + "\t" + str(len(train_y)) + "\t" + str(len(test_y)))
            # print(drug[0], "drug[0]",train_d,"train_d",test_d,"test_d")

        if drug_encoder == "chemmolefusion":
            train_p, test_p, train_d, test_d, train_y, test_y = train_test_split(protein, drug[0], y, test_size=0.2,
                                                                                 random_state=1, stratify=y)  # 1
            with open(name + "/sample_summary.txt", "w") as fw:
                fw.write("train_protein\ttest_protein\ttrain_drug\ttest_drug\ttrain_y\ttest_y\n")
                # fw.write(str(train_p.shape) + "\t" + str(test_p.shape) + "\t" + str(train_d.shape) + "\t" + str(
                #     test_d.shape) + "\t" + str(len(train_y)) + "\t" + str(len(test_y)))
            # print(drug[0], "drug[0]",train_d,"train_d",test_d,"test_d")

        if drug_encoder == "onehot":
            train_p, test_p, train_d, test_d, train_y, test_y = train_test_split(protein, drug[0], y, test_size=0.2,
                                                                                 random_state=1, stratify=y)  # 1
            with open(name + "/sample_summary.txt", "w") as fw:
                fw.write("train_protein\ttest_protein\ttrain_drug\ttest_drug\ttrain_y\ttest_y\n")
                # fw.write(str(train_p.shape) + "\t" + str(test_p.shape) + "\t" + str(train_d.shape) + "\t" + str(
                #     test_d.shape) + "\t" + str(len(train_y)) + "\t" + str(len(test_y)))
            # print(drug[0], "drug[0]",train_d,"train_d",test_d,"test_d")
        if drug_encoder == "CNN":
            train_p, test_p, train_d, test_d, train_y, test_y = train_test_split(protein, drug[0], y, test_size=0.2,
                                                                                 random_state=1, stratify=y)  # 1
            with open(name + "/sample_summary.txt", "w") as fw:
                fw.write("train_protein\ttest_protein\ttrain_drug\ttest_drug\ttrain_y\ttest_y\n")
                # fw.write(str(train_p.shape) + "\t" + str(test_p.shape) + "\t" + str(train_d.shape) + "\t" + str(
                #     test_d.shape) + "\t" + str(len(train_y)) + "\t" + str(len(test_y)))
            # print(drug[0], "drug[0]",train_d,"train_d",test_d,"test_d")
        if drug_encoder == "MPNN":
            train_p, test_p, train_d0, test_d0, train_d1, test_d1, train_d2, test_d2, train_y, test_y = train_test_split(
                protein, drug[0][0], drug[0][1], drug[0][2], y, test_size=0.2, random_state=1, stratify=y)  #
            with open(name + "/sample_summary.txt", "w") as fw:
                fw.write(
                    "train_protein\ttest_protein\ttrain_drug_d0\ttest_drug_d0\ttrain_drug_d1\ttest_drug_d1\ttrain_drug_d2\ttest_drug_d2\ttrain_y\ttest_y\n")
                fw.write(str(train_p.shape) + "\t" + str(test_p.shape) + "\t" + str(len(train_d0)) + "\t" + str(
                    len(test_d0)) + "\t" + str(len(train_d1)) + "\t" + str(len(test_d1)) + "\t" + str(
                    len(train_d2)) + "\t" + str(len(test_d2)) + "\t" + str(len(train_y)) + "\t" + str(len(test_y)))
            # print(train_p.shape,test_p.shape,len(train_d0),len(test_d0),len(train_d1),len(test_d1),len(train_d2),len(test_d2),len(train_y),len(test_y))
            # train_d=(tf.ragged.constant(train_d0, dtype=tf.float32),tf.ragged.constant(train_d1, dtype=tf.float32),tf.ragged.constant(train_d2, dtype=tf.int64))
            test_d = (tf.ragged.constant(test_d0, dtype=tf.float32), tf.ragged.constant(test_d1, dtype=tf.float32),
                      tf.ragged.constant(test_d2, dtype=tf.int64))
            train_d = (train_d0, train_d1, train_d2)
            # print(drug[0], "drug[0]")
    train_y = to_categorical(train_y, num_classes=2)
    test_y = to_categorical(test_y, num_classes=2)
    return train_p, test_p, train_d, test_d, train_y, test_y

def model_onehot_MPNN_capsule(atom_dim=29, bond_dim=7, seq_len=1024, target_dense=200, kernel_size=5, num_capsule=2,
                              routings=3,
                              batch_size=64,
                              message_units=64,
                              message_steps=4,
                              num_attention_heads=8,
                              dense_units=512
                              ):
    sequence_input_1 = Input(shape=(seq_len, 6))
    model_p = Flatten()(sequence_input_1)
    model_p = Dense(target_dense, activation='relu')(model_p)
    model_p = BatchNormalization()(model_p)

    atom_features = layers.Input((atom_dim), dtype="float32", name="atom_features")
    bond_features = layers.Input((bond_dim), dtype="float32", name="bond_features")
    pair_indices = layers.Input((2), dtype="int32", name="pair_indices")
    molecule_indicator = layers.Input((), dtype="int32", name="molecule_indicator")
    x = MessagePassing(message_units, message_steps)(
        [atom_features, bond_features, pair_indices]
    )

    model_d = TransformerEncoderReadout(
        num_attention_heads, message_units, dense_units, batch_size
    )([x, molecule_indicator])

    model = concatenate([model_p, model_d])
    model = Reshape((-1, 8))(model)
    primarycaps = PrimaryCap(model, dim_vector=8, n_channels=8, kernel_size=kernel_size, strides=1, padding='valid')
    capsule = Capsule(num_capsule=num_capsule, dim_capsule=16, routings=routings, share_weights=True)(primarycaps)
    length = Length()(capsule)
    model = keras.Model(inputs=[sequence_input_1, [atom_features, bond_features, pair_indices, molecule_indicator]],
                        outputs=length)
    # plot_model(model,to_file='onehot_MPNN_capsule/onehot_MPNN_capsule.png',show_shapes=True)
    model.summary()

    return model


def model_bert_MPNN_capsule(atom_dim=29, bond_dim=7, seq_len=1000, target_dense=400, kernel_size=5, num_capsule=2,
                            routings=3,
                            batch_size=64,
                            message_units=64,
                            message_steps=4,
                            num_attention_heads=8,
                            dense_units=512
                            ):
    sequence_input_1 = Input(shape=(768))
    # model_p=Flatten()(sequence_input_1)
    model_p = Dense(target_dense, activation='relu')(sequence_input_1)
    model_p = BatchNormalization()(model_p)

    atom_features = layers.Input((atom_dim), dtype="float32", name="atom_features")
    bond_features = layers.Input((bond_dim), dtype="float32", name="bond_features")
    pair_indices = layers.Input((2), dtype="int32", name="pair_indices")
    molecule_indicator = layers.Input((), dtype="int32", name="molecule_indicator")
    x = MessagePassing(message_units, message_steps)(
        [atom_features, bond_features, pair_indices]
    )

    model_d = TransformerEncoderReadout(
        num_attention_heads, message_units, dense_units, batch_size
    )([x, molecule_indicator])

    model = concatenate([model_p, model_d])
    model = Reshape((-1, 8))(model)
    primarycaps = PrimaryCap(model, dim_vector=8, n_channels=8, kernel_size=kernel_size, strides=1, padding='valid')
    capsule = Capsule(num_capsule=num_capsule, dim_capsule=16, routings=routings, share_weights=True)(primarycaps)
    length = Length()(capsule)
    model = keras.Model(inputs=[sequence_input_1, [atom_features, bond_features, pair_indices, molecule_indicator]],
                        outputs=length)
    # plot_model(model,to_file='onehot_MPNN_capsule/onehot_MPNN_capsule.png',show_shapes=True)
    model.summary()

    return model


def model_bert_MPNN_dense(atom_dim=29, bond_dim=7, seq_len=1000, target_dense=200,
                          batch_size=64,
                          message_units=64,
                          message_steps=4,
                          num_attention_heads=8, dense_units_2=512,
                          dense_units=512):
    sequence_input_1 = Input(shape=(768))
    model_p = Dense(200, activation='relu')(sequence_input_1)
    model_p = BatchNormalization()(model_p)

    atom_features = layers.Input((atom_dim), dtype="float32", name="atom_features")
    bond_features = layers.Input((bond_dim), dtype="float32", name="bond_features")
    pair_indices = layers.Input((2), dtype="int32", name="pair_indices")
    molecule_indicator = layers.Input((), dtype="int32", name="molecule_indicator")
    x = MessagePassing(message_units, message_steps)(
        [atom_features, bond_features, pair_indices]
    )
    model_d = TransformerEncoderReadout(
        num_attention_heads, message_units, dense_units, batch_size
    )([x, molecule_indicator])

    model = concatenate([model_p, model_d])
    x = Dense(dense_units_2, activation='relu')(model)
    output = Dense(2, activation='sigmoid')(x)

    model = keras.Model(inputs=[sequence_input_1, [atom_features, bond_features, pair_indices, molecule_indicator]],
                        outputs=output)
    # plot_model(model,to_file='bert_MPNN_dense/bert_MPNN_dense.png',show_shapes=True)
    model.summary()

    return model


from tensorflow import keras


def model_onehot_MPNN_dense(atom_dim=29, bond_dim=7, seq_len=1024, target_dense=200,
                            batch_size=64,
                            message_units=64,
                            message_steps=4,
                            num_attention_heads=8, dense_units_2=512,
                            dense_units=512):
    sequence_input_1 = Input(shape=(seq_len, 6))
    # model_p=Conv1D(filters=filter_num,kernel_size=kernel_size,padding="same",activation="relu")(sequence_input_1)
    # model_p=BatchNormalization()(model_p)
    # model_p=GlobalAveragePooling1D()(model_p)
    model_p = Flatten()(sequence_input_1)
    model_p = Dense(target_dense, activation='relu')(model_p)
    model_p = BatchNormalization()(model_p)

    atom_features = layers.Input((atom_dim), dtype="float32", name="atom_features")
    bond_features = layers.Input((bond_dim), dtype="float32", name="bond_features")
    pair_indices = layers.Input((2), dtype="int32", name="pair_indices")
    molecule_indicator = layers.Input((), dtype="int32", name="molecule_indicator")
    x = MessagePassing(message_units, message_steps)(
        [atom_features, bond_features, pair_indices]
    )

    model_d = TransformerEncoderReadout(
        num_attention_heads, message_units, dense_units, batch_size
    )([x, molecule_indicator])

    model = concatenate([model_p, model_d])

    x = Dense(dense_units_2, activation='relu')(model)
    output = Dense(2, activation='sigmoid')(x)

    model = keras.Model(inputs=[sequence_input_1, [atom_features, bond_features, pair_indices, molecule_indicator]],
                        outputs=output)
    # plot_model(model,to_file='onehot_MPNN_dense/onehot_MPNN_dense.png',show_shapes=True)
    model.summary()

    return model


import tensorflow as tf

def custom_f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = (true_positives + K.epsilon()) / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = (true_positives + K.epsilon()) / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
def model_onehot_fingerprint_dense_tuner(hp):
    target_dense = hp.Int('target_dense', min_value=32, max_value=512, step=8)
    drug_dense = hp.Int('target_dense', min_value=32, max_value=512, step=8)
    adam = Adam(learning_rate=1e-4)
    # sequence_input_1 = Input(shape=(param['seq_len'], 21))
    sequence_input_1 = Input(shape=(1024,6))
    # model_p=Conv1D(filters=filter_num,kernel_size=kernel_size,padding="same",activation="relu")(sequence_input_1)
    # model_p=BatchNormalization()(model_p)·
    # model_p=GlobalAveragePooling1D()(model_p)
    model_p = Flatten()(sequence_input_1)
    model_p = Dense(target_dense, activation='relu')(model_p)
    model_p = BatchNormalization()(model_p)

    # sequence_input_2 = Input(shape=(1024))
    sequence_input_2 = Input(shape=(1024))
    # cnn2=Conv1D(filters=filter_num,kernel_size=kernel_size,padding="same",activation="relu")(sequence_input_2)
    # cnn2=MaxPooling1D(pool_size=2,strides=2)(cnn2)
    model_d = Dense(drug_dense, activation='relu')(sequence_input_2)
    model_d = BatchNormalization()(model_d)

    model = concatenate([model_p, model_d])

    # cnn1=Dropout(0.25)(cnn1)
    # x=LSTM(num_lstm,return_sequences=True)(model)
    x = Dense(20, activation='relu')(model)
    output = Dense(2, activation='sigmoid')(x)

    model = Model(inputs=[sequence_input_1, sequence_input_2], outputs=output)
    # plot_model(model,to_file='onehot_fingerprint_dense/onehot_fingerprint_dense.png',show_shapes=True)
    model.compile(loss="binary_crossentropy", optimizer=adam,
                    metrics=[custom_f1, 'accuracy', 'AUC',
                             tf.keras.metrics.Precision(),
                             tf.keras.metrics.Recall(),
                             tf.keras.metrics.TruePositives(),
                             tf.keras.metrics.TrueNegatives(),
                             tf.keras.metrics.FalsePositives(),
                             tf.keras.metrics.FalseNegatives()])
    model.summary()

    return model

def model_onehot_fingerprint_dense(param):
    # sequence_input_1 = Input(shape=(param['seq_len'], 21))
    sequence_input_1 = Input(shape=(param['seq_len'],6))
    # model_p=Conv1D(filters=filter_num,kernel_size=kernel_size,padding="same",activation="relu")(sequence_input_1)
    # model_p=BatchNormalization()(model_p)·
    # model_p=GlobalAveragePooling1D()(model_p)
    model_p = Flatten()(sequence_input_1)
    model_p = Dense(param['target_dense'], activation='relu')(model_p)
    model_p = BatchNormalization()(model_p)

    # sequence_input_2 = Input(shape=(1024))
    sequence_input_2 = Input(shape=(param['seq_len']))
    # cnn2=Conv1D(filters=filter_num,kernel_size=kernel_size,padding="same",activation="relu")(sequence_input_2)
    # cnn2=MaxPooling1D(pool_size=2,strides=2)(cnn2)
    model_d = Dense(param['drug_dense'], activation='relu')(sequence_input_2)
    model_d = BatchNormalization()(model_d)

    model = concatenate([model_p, model_d])

    # cnn1=Dropout(0.25)(cnn1)
    # x=LSTM(num_lstm,return_sequences=True)(model)
    x = Dense(20, activation='relu')(model)
    output = Dense(2, activation='sigmoid')(x)

    model = Model(inputs=[sequence_input_1, sequence_input_2], outputs=output)
    # plot_model(model,to_file='onehot_fingerprint_dense/onehot_fingerprint_dense.png',show_shapes=True)
    model.summary()

    return model


def model_bert_fingerprint_dense(param):
    sequence_input_1 = Input(shape=(768))
    model_p = Dense(param['target_dense'], activation='relu')(sequence_input_1)
    model_p = BatchNormalization()(model_p)

    sequence_input_2 = Input(shape=(1024))
    # cnn2=Conv1D(filters=filter_num,kernel_size=kernel_size,padding="same",activation="relu")(sequence_input_2)
    # cnn2=MaxPooling1D(pool_size=2,strides=2)(cnn2)
    model_d = Dense(param['drug_dense'], activation='relu')(sequence_input_2)
    model_d = BatchNormalization()(model_d)

    model = concatenate([model_p, model_d])

    # cnn1=Dropout(0.25)(cnn1)
    # x=LSTM(num_lstm,return_sequences=True)(model)
    model = Dense(20, activation='relu')(model)
    # x=Dense(50,activation='relu')(model)
    output = Dense(2, activation='sigmoid')(model)

    model = Model(inputs=[sequence_input_1, sequence_input_2], outputs=output)
    # plot_model(model,to_file='bert_fingerprint_dense/bert_fingerprint_dense.png',show_shapes=True)
    model.summary()

    return model


def model_onehot_fingerprint_capsule(param):  #
    # sequence_input_1 = Input(shape=(seq_len,21))
    # model_p=Conv1D(filters=filter_num,kernel_size=kernel_size,padding="same",activation="relu")(sequence_input_1)
    # model_p=BatchNormalization()(model_p)
    # model_p=GlobalAveragePooling1D()(model_p)
    sequence_input_1 = Input(shape=(param['seq_len'], 6))
    # model_p=Conv1D(filters=filter_num,kernel_size=kernel_size,padding="same",activation="relu")(sequence_input_1)
    # model_p=BatchNormalization()(model_p)
    # model_p=GlobalAveragePooling1D()(model_p)
    model_p = Flatten()(sequence_input_1)
    model_p = Dense(param['target_dense'], activation='relu')(model_p)
    model_p = BatchNormalization()(model_p)

    sequence_input_2 = Input(shape=(1024))
    # cnn2=Conv1D(filters=filter_num,kernel_size=kernel_size,padding="same",activation="relu")(sequence_input_2)
    # cnn2=MaxPooling1D(pool_size=2,strides=2)(cnn2)
    model_d = Dense(param['drug_dense'], activation='relu')(sequence_input_2)
    model_d = BatchNormalization()(model_d)

    model = concatenate([model_p, model_d])
    model = Reshape((-1, 8))(model)

    primarycaps = PrimaryCap(model, dim_vector=8, n_channels=8, kernel_size=param['kernel_size'], strides=1,
                             padding='valid')
    capsule = Capsule(num_capsule=param['num_capsule'], dim_capsule=16, routings=param['routings'], share_weights=True)(
        primarycaps)

    length = Length()(capsule)

    # cnn1=Dropout(0.25)(cnn1)
    # x=LSTM(num_lstm,return_sequences=True)(model)
    # x=Dense(20,activation='relu')(model)
    # output=Dense(2,activation='sigmoid')(x)

    model = Model(inputs=[sequence_input_1, sequence_input_2], outputs=length)
    # plot_model(model,to_file='onehot_fingerprint_capsule/onehot_fingerprint_capsule.png',show_shapes=True)
    model.summary()
    # model.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])

    return model


def model_bert_fingerprint_capsule(param):
    # sequence_input_1 = Input(shape=(1024))
    # model_p=Dense(200,activation='relu')(sequence_input_1)
    # model_p = BatchNormalization()(model_p)

    sequence_input_1 = Input(name="input_1",shape=(768))
    model_p = Dense(param['target_dense'], activation='relu')(sequence_input_1)
    model_p = BatchNormalization()(model_p)

    sequence_input_2 = Input(name="input_2",shape=(1024))
    # cnn2=Conv1D(filters=filter_num,kernel_size=kernel_size,padding="same",activation="relu")(sequence_input_2)
    # cnn2=MaxPooling1D(pool_size=2,strides=2)(cnn2)
    model_d = Dense(param['drug_dense'],name="drug_dense",activation='relu')(sequence_input_2)
    model_d = BatchNormalization()(model_d)

    model = concatenate([model_p, model_d])
    model = Reshape((-1, 8))(model)

    # primarycaps = PrimaryCap(model, dim_vector=8, n_channels=8, kernel_size=kernel_size, strides=1, padding='valid')
    # capsule = Capsule(num_capsule=num_capsule, dim_capsule = 16, routings = routings, share_weights=True)(primarycaps)
    primarycaps = PrimaryCap(model, dim_vector=8, n_channels=8, kernel_size=param['kernel_size'], strides=1,
                             padding='valid')
    capsule = Capsule(name="capsule",num_capsule=param['num_capsule'], dim_capsule=16, routings=param['routings'], share_weights=True)(
        primarycaps)
    length = Length()(capsule)

    # cnn1=Dropout(0.25)(cnn1)
    # x=LSTM(num_lstm,return_sequences=True)(model)
    # x=Dense(20,activation='relu')(model)
    # output=Dense(2,activation='sigmoid')(x)
    model = Model(inputs=[sequence_input_1, sequence_input_2], outputs=length)
    # plot_model(model,to_file='bert_fingerprint_capsule/bert_fingerprint_capsule.png',show_shapes=True)
    model.summary()

    return model

def model_genalm_fingerprint_capsule(param):
    # sequence_input_1 = Input(shape=(1024))
    # model_p=Dense(200,activation='relu')(sequence_input_1)
    # model_p = BatchNormalization()(model_p)

    sequence_input_1 = Input(name="input_1",shape=(32000))
    model_p = Dense(param['target_dense'], activation='relu')(sequence_input_1)
    model_p = BatchNormalization()(model_p)

    sequence_input_2 = Input(name="input_2",shape=(1024))
    # cnn2=Conv1D(filters=filter_num,kernel_size=kernel_size,padding="same",activation="relu")(sequence_input_2)
    # cnn2=MaxPooling1D(pool_size=2,strides=2)(cnn2)
    model_d = Dense(param['drug_dense'],name="drug_dense",activation='relu')(sequence_input_2)
    model_d = BatchNormalization()(model_d)

    model = concatenate([model_p, model_d])
    model = Reshape((-1, 8))(model)

    # primarycaps = PrimaryCap(model, dim_vector=8, n_channels=8, kernel_size=kernel_size, strides=1, padding='valid')
    # capsule = Capsule(num_capsule=num_capsule, dim_capsule = 16, routings = routings, share_weights=True)(primarycaps)
    primarycaps = PrimaryCap(model, dim_vector=8, n_channels=8, kernel_size=param['kernel_size'], strides=1,
                             padding='valid')
    capsule = Capsule(name="capsule",num_capsule=param['num_capsule'], dim_capsule=16, routings=param['routings'], share_weights=True)(
        primarycaps)
    length = Length()(capsule)

    # cnn1=Dropout(0.25)(cnn1)
    # x=LSTM(num_lstm,return_sequences=True)(model)
    # x=Dense(20,activation='relu')(model)
    # output=Dense(2,activation='sigmoid')(x)
    model = Model(inputs=[sequence_input_1, sequence_input_2], outputs=length)
    # plot_model(model,to_file='bert_fingerprint_capsule/bert_fingerprint_capsule.png',show_shapes=True)
    model.summary()

    return model
def model_genalm_MPNN_capsule(atom_dim=29, bond_dim=7, seq_len=1000, target_dense=200, kernel_size=5, num_capsule=2,
                            routings=3,
                            batch_size=64,
                            message_units=64,
                            message_steps=4,
                            num_attention_heads=8,
                            dense_units=512
                            ):
    sequence_input_1 = Input(shape=(32000))
    # model_p=Flatten()(sequence_input_1)
    model_p = Dense(target_dense, activation='relu')(sequence_input_1)
    model_p = BatchNormalization()(model_p)

    atom_features = layers.Input((atom_dim), dtype="float32", name="atom_features")
    bond_features = layers.Input((bond_dim), dtype="float32", name="bond_features")
    pair_indices = layers.Input((2), dtype="int32", name="pair_indices")
    molecule_indicator = layers.Input((), dtype="int32", name="molecule_indicator")
    x = MessagePassing(message_units, message_steps)(
        [atom_features, bond_features, pair_indices]
    )

    model_d = TransformerEncoderReadout(
        num_attention_heads, message_units, dense_units, batch_size
    )([x, molecule_indicator])

    model = concatenate([model_p, model_d])
    model = Reshape((-1, 8))(model)
    primarycaps = PrimaryCap(model, dim_vector=8, n_channels=8, kernel_size=kernel_size, strides=1, padding='valid')
    capsule = Capsule(num_capsule=num_capsule, dim_capsule=16, routings=routings, share_weights=True)(primarycaps)
    length = Length()(capsule)
    model = keras.Model(inputs=[sequence_input_1, [atom_features, bond_features, pair_indices, molecule_indicator]],
                        outputs=length)
    # plot_model(model,to_file='onehot_MPNN_capsule/onehot_MPNN_capsule.png',show_shapes=True)
    model.summary()

    return model
def model_genalm_fingerprint_dense(param):
    sequence_input_1 = Input(shape=(32000))
    model_p = Dense(param['target_dense'], activation='relu')(sequence_input_1)
    model_p = BatchNormalization()(model_p)

    sequence_input_2 = Input(shape=(1024))
    # cnn2=Conv1D(filters=filter_num,kernel_size=kernel_size,padding="same",activation="relu")(sequence_input_2)
    # cnn2=MaxPooling1D(pool_size=2,strides=2)(cnn2)
    model_d = Dense(param['drug_dense'], activation='relu')(sequence_input_2)
    model_d = BatchNormalization()(model_d)

    model = concatenate([model_p, model_d])

    # cnn1=Dropout(0.25)(cnn1)
    # x=LSTM(num_lstm,return_sequences=True)(model)
    model = Dense(20, activation='relu')(model)
    # x=Dense(50,activation='relu')(model)
    output = Dense(2, activation='sigmoid')(model)

    model = Model(inputs=[sequence_input_1, sequence_input_2], outputs=output)
    # plot_model(model,to_file='bert_fingerprint_dense/bert_fingerprint_dense.png',show_shapes=True)
    model.summary()
    model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

    return model
def model_genalm_MPNN_dense(atom_dim=29, bond_dim=7, seq_len=1000, target_dense=200,
                          batch_size=64,
                          message_units=64,
                          message_steps=4,
                          num_attention_heads=8, dense_units_2=512,
                          dense_units=512):
    sequence_input_1 = Input(shape=(32000))
    model_p = Dense(200, activation='relu')(sequence_input_1)
    model_p = BatchNormalization()(model_p)

    atom_features = layers.Input((atom_dim), dtype="float32", name="atom_features")
    bond_features = layers.Input((bond_dim), dtype="float32", name="bond_features")
    pair_indices = layers.Input((2), dtype="int32", name="pair_indices")
    molecule_indicator = layers.Input((), dtype="int32", name="molecule_indicator")
    x = MessagePassing(message_units, message_steps)(
        [atom_features, bond_features, pair_indices]
    )
    model_d = TransformerEncoderReadout(
        num_attention_heads, message_units, dense_units, batch_size
    )([x, molecule_indicator])

    model = concatenate([model_p, model_d])
    x = Dense(dense_units_2, activation='relu')(model)
    output = Dense(2, activation='sigmoid')(x)

    model = keras.Model(inputs=[sequence_input_1, [atom_features, bond_features, pair_indices, molecule_indicator]],
                        outputs=output)
    # plot_model(model,to_file='bert_MPNN_dense/bert_MPNN_dense.png',show_shapes=True)
    model.summary()

    return model

def model_bert_mole_capsule(param):
    sequence_input_1 = Input(shape=(768))
    model_p = Flatten()(sequence_input_1)
    model_p = Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(model_p)
    # model_p = Dropout(0.5)(model_p)
    model_p = BatchNormalization()(model_p)

    sequence_input_2 = Input(shape=(300))
    model_d = Flatten()(sequence_input_2)
    model_d = Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(model_d)
    # model_d = Dropout(0.5)(model_d)
    model_d = BatchNormalization()(model_d)


    model = concatenate([model_p, model_d])
    model = Reshape((-1, 8))(model)

    # primarycaps = PrimaryCap(model, dim_vector=8, n_channels=8, kernel_size=kernel_size, strides=1, padding='valid')
    # capsule = Capsule(num_capsule=num_capsule, dim_capsule = 16, routings = routings, share_weights=True)(primarycaps)
    primarycaps = PrimaryCap(model, dim_vector=8, n_channels=8, kernel_size=param['kernel_size'], strides=1,
                             padding='valid')
    capsule = Capsule(name="capsule",num_capsule=param['num_capsule'], dim_capsule=16, routings=param['routings'], share_weights=True)(
        primarycaps)
    length = Length()(capsule)

    # cnn1=Dropout(0.25)(cnn1)
    # x=LSTM(num_lstm,return_sequences=True)(model)
    # x=Dense(20,activation='relu')(model)
    # output=Dense(2,activation='sigmoid')(x)
    model = Model(inputs=[sequence_input_1, sequence_input_2], outputs=length)
    # plot_model(model,to_file='bert_fingerprint_capsule/bert_fingerprint_capsule.png',show_shapes=True)
    model.summary()
    model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

    return model
def model_bert_mole_dense(param):
    sequence_input_1 = Input(shape=(768))
    model_p = Flatten()(sequence_input_1)
    model_p = Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(model_p)
    # model_p = Dropout(0.5)(model_p)
    model_p = BatchNormalization()(model_p)

    sequence_input_2 = Input(shape=(300))
    model_d = Flatten()(sequence_input_2)
    model_d = Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(model_d)
    # model_d = Dropout(0.5)(model_d)
    model_d = BatchNormalization()(model_d)

    model = concatenate([model_p, model_d])
    model = Dense(20, activation='relu')(model)
    output = Dense(2, activation='sigmoid')(model)

    model = Model(inputs=[sequence_input_1, sequence_input_2], outputs=output)
    model.summary()
    model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

    return model

def model_bert_mole_attention(param):

    sequence_input_1 = Input(shape=768)
    model_p = Flatten()(sequence_input_1)
    model_p = Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(model_p)
    # model_p = Dropout(0.5)(model_p)
    model_p = BatchNormalization()(model_p)

    sequence_input_2 = Input(shape=300)
    model_d = Flatten()(sequence_input_2)
    model_d = Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(model_d)
    # model_d = Dropout(0.5)(model_d)
    model_d = BatchNormalization()(model_d)

    model_p_reshaped = Reshape((1, -1))(model_p)
    model_d_reshaped = Reshape((1, -1))(model_d)

    merged = Concatenate(axis=1)([model_p_reshaped, model_d_reshaped])

    attention_heads = 4
    attention_size = 128

    multiheaded_attention = MultiHeadAttention(
        num_heads=attention_heads,
        key_dim=attention_size,
        value_dim=attention_size,
        dropout=0.2
    )(merged, merged)

    multiheaded_attention = Flatten()(multiheaded_attention)
    next_layer = Dense(64, activation='relu')(multiheaded_attention)
    output = Dense(2, activation='sigmoid')(next_layer)
    model = Model(inputs=[sequence_input_1, sequence_input_2], outputs=output)

    return model

def model_onehot_mole_dense(param):
    # sequence_input_1 = Input(shape=(param['seq_len'], 21))
    sequence_input_1 = Input(shape=(param['seq_len'],6))
    # model_p=Conv1D(filters=filter_num,kernel_size=kernel_size,padding="same",activation="relu")(sequence_input_1)
    # model_p=BatchNormalization()(model_p)·
    # model_p=GlobalAveragePooling1D()(model_p)
    model_p = Flatten()(sequence_input_1)
    model_p = Dense(param['target_dense'], activation='relu')(model_p)
    model_p = BatchNormalization()(model_p)

    # sequence_input_2 = Input(shape=(1024))
    sequence_input_2 = Input(shape=(300))
    # cnn2=Conv1D(filters=filter_num,kernel_size=kernel_size,padding="same",activation="relu")(sequence_input_2)
    # cnn2=MaxPooling1D(pool_size=2,strides=2)(cnn2)
    model_d = Dense(param['drug_dense'], activation='relu')(sequence_input_2)
    model_d = BatchNormalization()(model_d)

    model = concatenate([model_p, model_d])

    # cnn1=Dropout(0.25)(cnn1)
    # x=LSTM(num_lstm,return_sequences=True)(model)
    x = Dense(20, activation='relu')(model)
    output = Dense(2, activation='sigmoid')(x)

    model = Model(inputs=[sequence_input_1, sequence_input_2], outputs=output)
    # plot_model(model,to_file='onehot_fingerprint_dense/onehot_fingerprint_dense.png',show_shapes=True)
    model.summary()

    return model

def model_onehot_mole_capsule(param):  #
    # sequence_input_1 = Input(shape=(seq_len,21))
    # model_p=Conv1D(filters=filter_num,kernel_size=kernel_size,padding="same",activation="relu")(sequence_input_1)
    # model_p=BatchNormalization()(model_p)
    # model_p=GlobalAveragePooling1D()(model_p)
    sequence_input_1 = Input(shape=(param['seq_len'], 6))
    # model_p=Conv1D(filters=filter_num,kernel_size=kernel_size,padding="same",activation="relu")(sequence_input_1)
    # model_p=BatchNormalization()(model_p)
    # model_p=GlobalAveragePooling1D()(model_p)
    model_p = Flatten()(sequence_input_1)
    model_p = Dense(param['target_dense'], activation='relu')(model_p)
    model_p = BatchNormalization()(model_p)

    sequence_input_2 = Input(shape=(300))
    # cnn2=Conv1D(filters=filter_num,kernel_size=kernel_size,padding="same",activation="relu")(sequence_input_2)
    # cnn2=MaxPooling1D(pool_size=2,strides=2)(cnn2)
    model_d = Dense(param['drug_dense'], activation='relu')(sequence_input_2)
    model_d = BatchNormalization()(model_d)

    model = concatenate([model_p, model_d])
    model = Reshape((-1, 8))(model)

    primarycaps = PrimaryCap(model, dim_vector=8, n_channels=8, kernel_size=param['kernel_size'], strides=1,
                             padding='valid')
    capsule = Capsule(num_capsule=param['num_capsule'], dim_capsule=16, routings=param['routings'], share_weights=True)(
        primarycaps)

    length = Length()(capsule)

    # cnn1=Dropout(0.25)(cnn1)
    # x=LSTM(num_lstm,return_sequences=True)(model)
    # x=Dense(20,activation='relu')(model)
    # output=Dense(2,activation='sigmoid')(x)

    model = Model(inputs=[sequence_input_1, sequence_input_2], outputs=length)
    # plot_model(model,to_file='onehot_fingerprint_capsule/onehot_fingerprint_capsule.png',show_shapes=True)
    model.summary()
    # model.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])

    return model
def model_bert_onehot_capsule(param):
    # sequence_input_1 = Input(shape=(1024))
    # model_p=Dense(200,activation='relu')(sequence_input_1)
    # model_p = BatchNormalization()(model_p)

    sequence_input_1 = Input(name="input_1",shape=(768))
    model_p = Dense(param['target_dense'], activation='relu')(sequence_input_1)
    model_p = BatchNormalization()(model_p)

    sequence_input_2 = Input(name="input_2",shape=(31))
    # cnn2=Conv1D(filters=filter_num,kernel_size=kernel_size,padding="same",activation="relu")(sequence_input_2)
    # cnn2=MaxPooling1D(pool_size=2,strides=2)(cnn2)
    model_d = Dense(param['drug_dense'],name="drug_dense",activation='relu')(sequence_input_2)
    model_d = BatchNormalization()(model_d)

    model = concatenate([model_p, model_d])
    model = Reshape((-1, 8))(model)

    # primarycaps = PrimaryCap(model, dim_vector=8, n_channels=8, kernel_size=kernel_size, strides=1, padding='valid')
    # capsule = Capsule(num_capsule=num_capsule, dim_capsule = 16, routings = routings, share_weights=True)(primarycaps)
    primarycaps = PrimaryCap(model, dim_vector=8, n_channels=8, kernel_size=param['kernel_size'], strides=1,
                             padding='valid')
    capsule = Capsule(name="capsule",num_capsule=param['num_capsule'], dim_capsule=16, routings=param['routings'], share_weights=True)(
        primarycaps)
    length = Length()(capsule)

    # cnn1=Dropout(0.25)(cnn1)
    # x=LSTM(num_lstm,return_sequences=True)(model)
    # x=Dense(20,activation='relu')(model)
    # output=Dense(2,activation='sigmoid')(x)
    model = Model(inputs=[sequence_input_1, sequence_input_2], outputs=length)
    # plot_model(model,to_file='bert_fingerprint_capsule/bert_fingerprint_capsule.png',show_shapes=True)
    model.summary()
    # model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

    return model
def model_bert_onehot_dense(param):
    sequence_input_1 = Input(shape=(768))
    model_p = Dense(param['target_dense'], activation='relu')(sequence_input_1)
    model_p = BatchNormalization()(model_p)

    sequence_input_2 = Input(shape=(31))
    # cnn2=Conv1D(filters=filter_num,kernel_size=kernel_size,padding="same",activation="relu")(sequence_input_2)
    # cnn2=MaxPooling1D(pool_size=2,strides=2)(cnn2)
    model_d = Dense(param['drug_dense'], activation='relu')(sequence_input_2)
    model_d = BatchNormalization()(model_d)

    model = concatenate([model_p, model_d])

    # cnn1=Dropout(0.25)(cnn1)
    # x=LSTM(num_lstm,return_sequences=True)(model)
    model = Dense(20, activation='relu')(model)
    # x=Dense(50,activation='relu')(model)
    output = Dense(2, activation='sigmoid')(model)

    model = Model(inputs=[sequence_input_1, sequence_input_2], outputs=output)
    # plot_model(model,to_file='bert_fingerprint_dense/bert_fingerprint_dense.png',show_shapes=True)
    model.summary()
    # model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

    return model

def model_onehot_onehot_capsule(param):  #
    # sequence_input_1 = Input(shape=(seq_len,21))
    # model_p=Conv1D(filters=filter_num,kernel_size=kernel_size,padding="same",activation="relu")(sequence_input_1)
    # model_p=BatchNormalization()(model_p)
    # model_p=GlobalAveragePooling1D()(model_p)
    sequence_input_1 = Input(shape=(param['seq_len'], 6))
    # model_p=Conv1D(filters=filter_num,kernel_size=kernel_size,padding="same",activation="relu")(sequence_input_1)
    # model_p=BatchNormalization()(model_p)
    # model_p=GlobalAveragePooling1D()(model_p)
    model_p = Flatten()(sequence_input_1)
    model_p = Dense(param['target_dense'], activation='relu')(model_p)
    model_p = BatchNormalization()(model_p)

    sequence_input_2 = Input(shape=(31))
    # cnn2=Conv1D(filters=filter_num,kernel_size=kernel_size,padding="same",activation="relu")(sequence_input_2)
    # cnn2=MaxPooling1D(pool_size=2,strides=2)(cnn2)
    model_d = Dense(param['drug_dense'], activation='relu')(sequence_input_2)
    model_d = BatchNormalization()(model_d)

    model = concatenate([model_p, model_d])
    model = Reshape((-1, 8))(model)

    primarycaps = PrimaryCap(model, dim_vector=8, n_channels=8, kernel_size=param['kernel_size'], strides=1,
                             padding='valid')
    capsule = Capsule(num_capsule=param['num_capsule'], dim_capsule=16, routings=param['routings'], share_weights=True)(
        primarycaps)

    length = Length()(capsule)

    # cnn1=Dropout(0.25)(cnn1)
    # x=LSTM(num_lstm,return_sequences=True)(model)
    # x=Dense(20,activation='relu')(model)
    # output=Dense(2,activation='sigmoid')(x)

    model = Model(inputs=[sequence_input_1, sequence_input_2], outputs=length)
    # plot_model(model,to_file='onehot_fingerprint_capsule/onehot_fingerprint_capsule.png',show_shapes=True)
    model.summary()
    # model.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])

    return model
def model_onehot_onehot_dense(param):
    # sequence_input_1 = Input(shape=(param['seq_len'], 21))
    sequence_input_1 = Input(shape=(param['seq_len'],6))
    # model_p=Conv1D(filters=filter_num,kernel_size=kernel_size,padding="same",activation="relu")(sequence_input_1)
    # model_p=BatchNormalization()(model_p)·
    # model_p=GlobalAveragePooling1D()(model_p)
    model_p = Flatten()(sequence_input_1)
    model_p = Dense(param['target_dense'], activation='relu')(model_p)
    model_p = BatchNormalization()(model_p)

    # sequence_input_2 = Input(shape=(1024))
    sequence_input_2 = Input(shape=(31))
    # cnn2=Conv1D(filters=filter_num,kernel_size=kernel_size,padding="same",activation="relu")(sequence_input_2)
    # cnn2=MaxPooling1D(pool_size=2,strides=2)(cnn2)
    model_d = Dense(param['drug_dense'], activation='relu')(sequence_input_2)
    model_d = BatchNormalization()(model_d)

    model = concatenate([model_p, model_d])

    # cnn1=Dropout(0.25)(cnn1)
    # x=LSTM(num_lstm,return_sequences=True)(model)
    x = Dense(20, activation='relu')(model)
    output = Dense(2, activation='sigmoid')(x)

    model = Model(inputs=[sequence_input_1, sequence_input_2], outputs=output)
    # plot_model(model,to_file='onehot_fingerprint_dense/onehot_fingerprint_dense.png',show_shapes=True)
    model.summary()

    return model
def model_bert_CNN_capsule(param):
    # sequence_input_1 = Input(shape=(1024))
    # model_p=Dense(200,activation='relu')(sequence_input_1)
    # model_p = BatchNormalization()(model_p)

    sequence_input_1 = Input(name="input_1",shape=(768))
    model_p = Dense(param['target_dense'], activation='relu')(sequence_input_1)
    model_p = BatchNormalization()(model_p)

    sequence_input_2 = Input(name="input_2",shape=(128))
    # cnn2=Conv1D(filters=filter_num,kernel_size=kernel_size,padding="same",activation="relu")(sequence_input_2)
    # cnn2=MaxPooling1D(pool_size=2,strides=2)(cnn2)
    model_d = Dense(param['drug_dense'],name="drug_dense",activation='relu')(sequence_input_2)
    model_d = BatchNormalization()(model_d)

    model = concatenate([model_p, model_d])
    model = Reshape((-1, 8))(model)

    # primarycaps = PrimaryCap(model, dim_vector=8, n_channels=8, kernel_size=kernel_size, strides=1, padding='valid')
    # capsule = Capsule(num_capsule=num_capsule, dim_capsule = 16, routings = routings, share_weights=True)(primarycaps)
    primarycaps = PrimaryCap(model, dim_vector=8, n_channels=8, kernel_size=param['kernel_size'], strides=1,
                             padding='valid')
    capsule = Capsule(name="capsule",num_capsule=param['num_capsule'], dim_capsule=16, routings=param['routings'], share_weights=True)(
        primarycaps)
    length = Length()(capsule)

    # cnn1=Dropout(0.25)(cnn1)
    # x=LSTM(num_lstm,return_sequences=True)(model)
    # x=Dense(20,activation='relu')(model)
    # output=Dense(2,activation='sigmoid')(x)
    model = Model(inputs=[sequence_input_1, sequence_input_2], outputs=length)
    # plot_model(model,to_file='bert_fingerprint_capsule/bert_fingerprint_capsule.png',show_shapes=True)
    model.summary()
    # model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

    return model

def model_bert_CNN_dense(param):
    sequence_input_1 = Input(shape=(768))
    model_p = Dense(param['target_dense'], activation='relu')(sequence_input_1)
    model_p = BatchNormalization()(model_p)

    sequence_input_2 = Input(shape=(128))
    # cnn2=Conv1D(filters=filter_num,kernel_size=kernel_size,padding="same",activation="relu")(sequence_input_2)
    # cnn2=MaxPooling1D(pool_size=2,strides=2)(cnn2)
    model_d = Dense(param['drug_dense'], activation='relu')(sequence_input_2)
    model_d = BatchNormalization()(model_d)

    model = concatenate([model_p, model_d])

    # cnn1=Dropout(0.25)(cnn1)
    # x=LSTM(num_lstm,return_sequences=True)(model)
    model = Dense(20, activation='relu')(model)
    # x=Dense(50,activation='relu')(model)
    output = Dense(2, activation='sigmoid')(model)

    model = Model(inputs=[sequence_input_1, sequence_input_2], outputs=output)
    # plot_model(model,to_file='bert_fingerprint_dense/bert_fingerprint_dense.png',show_shapes=True)
    model.summary()
    # model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

    return model

def model_onehot_CNN_capsule(param):
    # sequence_input_1 = Input(shape=(seq_len,21))
    # model_p=Conv1D(filters=filter_num,kernel_size=kernel_size,padding="same",activation="relu")(sequence_input_1)
    # model_p=BatchNormalization()(model_p)
    # model_p=GlobalAveragePooling1D()(model_p)
    sequence_input_1 = Input(shape=(param['seq_len'], 6))
    # model_p=Conv1D(filters=filter_num,kernel_size=kernel_size,padding="same",activation="relu")(sequence_input_1)
    # model_p=BatchNormalization()(model_p)
    # model_p=GlobalAveragePooling1D()(model_p)
    model_p = Flatten()(sequence_input_1)
    model_p = Dense(param['target_dense'], activation='relu')(model_p)
    model_p = BatchNormalization()(model_p)

    sequence_input_2 = Input(shape=(128))
    # cnn2=Conv1D(filters=filter_num,kernel_size=kernel_size,padding="same",activation="relu")(sequence_input_2)
    # cnn2=MaxPooling1D(pool_size=2,strides=2)(cnn2)
    model_d = Dense(param['drug_dense'], activation='relu')(sequence_input_2)
    model_d = BatchNormalization()(model_d)

    model = concatenate([model_p, model_d])
    model = Reshape((-1, 8))(model)

    primarycaps = PrimaryCap(model, dim_vector=8, n_channels=8, kernel_size=param['kernel_size'], strides=1,
                             padding='valid')
    capsule = Capsule(num_capsule=param['num_capsule'], dim_capsule=16, routings=param['routings'],
                      share_weights=True)(
        primarycaps)

    length = Length()(capsule)

    # cnn1=Dropout(0.25)(cnn1)
    # x=LSTM(num_lstm,return_sequences=True)(model)
    # x=Dense(20,activation='relu')(model)
    # output=Dense(2,activation='sigmoid')(x)

    model = Model(inputs=[sequence_input_1, sequence_input_2], outputs=length)
    # plot_model(model,to_file='onehot_fingerprint_capsule/onehot_fingerprint_capsule.png',show_shapes=True)
    model.summary()
    # model.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])

    return model

def model_onehot_CNN_dense(param):
    sequence_input_1 = Input(shape=(param['seq_len'],6))
    model_p = Flatten()(sequence_input_1)
    model_p = Dense(param['target_dense'], activation='relu')(model_p)
    model_p = BatchNormalization()(model_p)

    sequence_input_2 = Input(shape=(128))
    model_d = Dense(param['drug_dense'], activation='relu')(sequence_input_2)
    model_d = BatchNormalization()(model_d)

    model = concatenate([model_p, model_d])
    x = Dense(20, activation='relu')(model)
    output = Dense(2, activation='sigmoid')(x)

    model = Model(inputs=[sequence_input_1, sequence_input_2], outputs=output)
    model.summary()

    return model

def model_bert_smole_capsule(param):

    sequence_input_1 = Input(shape=768)
    model_p = Flatten()(sequence_input_1)
    model_p = Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(model_p)
    # model_p = Dropout(0.5)(model_p)
    model_p = BatchNormalization()(model_p)

    sequence_input_2 = Input(shape=512)
    model_d = Flatten()(sequence_input_2)
    model_d = Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(model_d)
    # model_d = Dropout(0.5)(model_d)
    model_d = BatchNormalization()(model_d)

    model = concatenate([model_p, model_d])
    model = Reshape((-1, 8))(model)

    primarycaps = PrimaryCap(model, dim_vector=8, n_channels=8, kernel_size=param['kernel_size'], strides=1,
                             padding='valid')
    capsule = Capsule(num_capsule=param['num_capsule'], dim_capsule=16, routings=param['routings'],
                      share_weights=True)(
        primarycaps)

    length = Length()(capsule)

    model = Model(inputs=[sequence_input_1, sequence_input_2], outputs=length)
    model.summary()

    return model

def model_bert_smole_dense(param):

    sequence_input_1 = Input(shape=768)
    model_p = Flatten()(sequence_input_1)
    model_p = Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(model_p)
    # model_p = Dropout(0.5)(model_p)
    model_p = BatchNormalization()(model_p)

    sequence_input_2 = Input(shape=512)
    model_d = Flatten()(sequence_input_2)
    model_d = Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(model_d)
    # model_d = Dropout(0.5)(model_d)
    model_d = BatchNormalization()(model_d)

    model = concatenate([model_p, model_d])
    x = Dense(64, activation='relu')(model)
    output = Dense(2, activation='sigmoid')(x)

    model = Model(inputs=[sequence_input_1, sequence_input_2], outputs=output)
    model.summary()
    return model

def model_bert_smole_attention(param):

    sequence_input_1 = Input(shape=768)
    model_p = Flatten()(sequence_input_1)
    model_p = Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(model_p)
    # model_p = Dropout(0.5)(model_p)
    model_p = BatchNormalization()(model_p)

    sequence_input_2 = Input(shape=(512))
    model_d = Flatten()(sequence_input_2)
    model_d = Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(model_d)
    # model_d = Dropout(0.5)(model_d)
    model_d = BatchNormalization()(model_d)

    model_p_reshaped = Reshape((1, -1))(model_p)
    model_d_reshaped = Reshape((1, -1))(model_d)

    merged = Concatenate(axis=1)([model_p_reshaped, model_d_reshaped])

    attention_heads = 4
    attention_size = 128

    multiheaded_attention = MultiHeadAttention(
        num_heads=attention_heads,
        key_dim=attention_size,
        value_dim=attention_size,
        dropout=0.2
    )(merged, merged)

    multiheaded_attention = Flatten()(multiheaded_attention)
    next_layer = Dense(64, activation='relu')(multiheaded_attention)
    output = Dense(2, activation='sigmoid')(next_layer)
    model = Model(inputs=[sequence_input_1, sequence_input_2], outputs=output)

    return model

def model_bert_chemberta_capsule(param):

    sequence_input_1 = Input(shape=768)
    model_p = Flatten()(sequence_input_1)
    model_p = Dense(512, activation='relu',kernel_regularizer=keras.regularizers.l2(0.01))(model_p)
    # model_p = Dropout(0.5)(model_p)
    model_p = BatchNormalization()(model_p)

    sequence_input_2 = Input(shape=384)
    model_d = Flatten()(sequence_input_2)
    model_d = Dense(512, activation='relu',kernel_regularizer=keras.regularizers.l2(0.01))(model_d)
    # model_p = Dropout(0.5)(model_p)
    model_d = BatchNormalization()(model_d)

    model = concatenate([model_p, model_d])
    model = Reshape((-1, 8))(model)

    primarycaps = PrimaryCap(model, dim_vector=8, n_channels=8, kernel_size=param['kernel_size'], strides=1,
                             padding='valid')
    capsule = Capsule(num_capsule=param['num_capsule'], dim_capsule=16, routings=param['routings'],
                      share_weights=True)(
        primarycaps)

    length = Length()(capsule)

    model = Model(inputs=[sequence_input_1, sequence_input_2], outputs=length)
    model.summary()

    return model

def model_bert_chemberta_dense(param):

    sequence_input_1 = Input(shape=768)
    model_p = Flatten()(sequence_input_1)
    model_p = Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(model_p)
    # model_p = Dropout(0.5)(model_p)
    model_p = BatchNormalization()(model_p)

    sequence_input_2 = Input(shape=384)
    model_d = Flatten()(sequence_input_2)
    model_d = Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(model_d)
    # model_d = Dropout(0.5)(model_d)
    model_d = BatchNormalization()(model_d)

    model = concatenate([model_p, model_d])
    x = Dense(64, activation='relu')(model)
    output = Dense(2, activation='sigmoid')(x)

    model = Model(inputs=[sequence_input_1, sequence_input_2], outputs=output)
    model.summary()
    return model

def model_bert_chemberta_attention(param):

    sequence_input_1 = Input(shape=768)
    model_p = Flatten()(sequence_input_1)
    model_p = Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(model_p)
    # model_p = Dropout(0.5)(model_p)
    model_p = BatchNormalization()(model_p)

    sequence_input_2 = Input(shape=(384))
    model_d = Flatten()(sequence_input_2)
    model_d = Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(model_d)
    # model_d = Dropout(0.5)(model_d)
    model_d = BatchNormalization()(model_d)

    model_p_reshaped = Reshape((1, -1))(model_p)
    model_d_reshaped = Reshape((1, -1))(model_d)

    merged = Concatenate(axis=1)([model_p_reshaped, model_d_reshaped])

    attention_heads = 4
    attention_size = 128

    multiheaded_attention = MultiHeadAttention(
        num_heads=attention_heads,
        key_dim=attention_size,
        value_dim=attention_size,
        dropout=0.2
    )(merged, merged)

    multiheaded_attention = Flatten()(multiheaded_attention)
    next_layer = Dense(64, activation='relu')(multiheaded_attention)
    output = Dense(2, activation='sigmoid')(next_layer)
    model = Model(inputs=[sequence_input_1, sequence_input_2], outputs=output)

    return model


def model_bert_smolechemfusion_capsule(param):

    sequence_input_1 = Input(shape=768)
    model_p = Flatten()(sequence_input_1)
    model_p = Dense(512, activation='relu',kernel_regularizer=keras.regularizers.l2(0.01))(model_p)
    # model_p = Dropout(0.5)(model_p)
    model_p = BatchNormalization()(model_p)

    sequence_input_2 = Input(shape=896)
    model_d = Flatten()(sequence_input_2)
    model_d = Dense(512, activation='relu',kernel_regularizer=keras.regularizers.l2(0.01))(model_d)
    # model_d = Dropout(0.5)(model_d)
    model_d = BatchNormalization()(model_d)

    model = concatenate([model_p, model_d])
    model = Reshape((-1, 8))(model)

    primarycaps = PrimaryCap(model, dim_vector=8, n_channels=8, kernel_size=param['kernel_size'], strides=1,
                             padding='valid')
    capsule = Capsule(num_capsule=param['num_capsule'], dim_capsule=16, routings=param['routings'],
                      share_weights=True)(
        primarycaps)

    length = Length()(capsule)

    model = Model(inputs=[sequence_input_1, sequence_input_2], outputs=length)
    model.summary()

    return model

def model_bert_smolechemfusion_dense(param):

    sequence_input_1 = Input(shape=768)
    model_p = Flatten()(sequence_input_1)
    model_p = Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(model_p)
    # model_p = Dropout(0.5)(model_p)
    model_p = BatchNormalization()(model_p)

    sequence_input_2 = Input(shape=896)
    model_d = Flatten()(sequence_input_2)
    model_d = Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(model_d)
    # model_d = Dropout(0.5)(model_d)
    model_d = BatchNormalization()(model_d)

    model = concatenate([model_p, model_d])
    x = Dense(64, activation='relu')(model)
    output = Dense(2, activation='sigmoid')(x)

    model = Model(inputs=[sequence_input_1, sequence_input_2], outputs=output)
    model.summary()
    return model

def model_bert_smolechemfusion_attention(param):

    sequence_input_1 = Input(shape=768)
    model_p = Flatten()(sequence_input_1)
    model_p = Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(model_p)
    # model_p = Dropout(0.5)(model_p)
    model_p = BatchNormalization()(model_p)

    sequence_input_2 = Input(shape=(896))
    model_d = Flatten()(sequence_input_2)
    model_d = Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(model_d)
    # model_d = Dropout(0.5)(model_d)
    model_d = BatchNormalization()(model_d)

    model_p_reshaped = Reshape((1, -1))(model_p)
    model_d_reshaped = Reshape((1, -1))(model_d)

    merged = Concatenate(axis=1)([model_p_reshaped, model_d_reshaped])

    attention_heads = 4
    attention_size = 128

    multiheaded_attention = MultiHeadAttention(
        num_heads=attention_heads,
        key_dim=attention_size,
        value_dim=attention_size,
        dropout=0.2
    )(merged, merged)

    multiheaded_attention = Flatten()(multiheaded_attention)
    next_layer = Dense(64, activation='relu')(multiheaded_attention)
    output = Dense(2, activation='sigmoid')(next_layer)
    model = Model(inputs=[sequence_input_1, sequence_input_2], outputs=output)

    return model


def model_bert_smolemolefusion_capsule(param):

    sequence_input_1 = Input(shape=768)
    model_p = Flatten()(sequence_input_1)
    model_p = Dense(512, activation='relu',kernel_regularizer=keras.regularizers.l2(0.01))(model_p)
    # model_p = Dropout(0.5)(model_p)
    model_p = BatchNormalization()(model_p)

    sequence_input_2 = Input(shape=812)
    model_d = Flatten()(sequence_input_2)
    model_d = Dense(512, activation='relu',kernel_regularizer=keras.regularizers.l2(0.01))(model_d)
    # model_d = Dropout(0.5)(model_d)
    model_d = BatchNormalization()(model_d)

    model = concatenate([model_p, model_d])
    model = Reshape((-1, 8))(model)

    primarycaps = PrimaryCap(model, dim_vector=8, n_channels=8, kernel_size=param['kernel_size'], strides=1,
                             padding='valid')
    capsule = Capsule(num_capsule=param['num_capsule'], dim_capsule=16, routings=param['routings'],
                      share_weights=True)(
        primarycaps)

    length = Length()(capsule)

    model = Model(inputs=[sequence_input_1, sequence_input_2], outputs=length)
    model.summary()

    return model

def model_bert_smolemolefusion_dense(param):

    sequence_input_1 = Input(shape=768)
    model_p = Flatten()(sequence_input_1)
    model_p = Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(model_p)
    # model_p = Dropout(0.5)(model_p)
    model_p = BatchNormalization()(model_p)

    sequence_input_2 = Input(shape=812)
    model_d = Flatten()(sequence_input_2)
    model_d = Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(model_d)
    # model_d = Dropout(0.5)(model_d)
    model_d = BatchNormalization()(model_d)

    model = concatenate([model_p, model_d])
    x = Dense(64, activation='relu')(model)
    output = Dense(2, activation='sigmoid')(x)

    model = Model(inputs=[sequence_input_1, sequence_input_2], outputs=output)
    model.summary()
    return model

def model_bert_smolemolefusion_attention(param):

    sequence_input_1 = Input(shape=768)
    model_p = Flatten()(sequence_input_1)
    model_p = Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(model_p)
    # model_p = Dropout(0.5)(model_p)
    model_p = BatchNormalization()(model_p)

    sequence_input_2 = Input(shape=(812))
    model_d = Flatten()(sequence_input_2)
    model_d = Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(model_d)
    # model_d = Dropout(0.5)(model_d)
    model_d = BatchNormalization()(model_d)

    model_p_reshaped = Reshape((1, -1))(model_p)
    model_d_reshaped = Reshape((1, -1))(model_d)

    merged = Concatenate(axis=1)([model_p_reshaped, model_d_reshaped])

    attention_heads = 4
    attention_size = 128

    multiheaded_attention = MultiHeadAttention(
        num_heads=attention_heads,
        key_dim=attention_size,
        value_dim=attention_size,
        dropout=0.2
    )(merged, merged)

    multiheaded_attention = Flatten()(multiheaded_attention)
    next_layer = Dense(64, activation='relu')(multiheaded_attention)
    output = Dense(2, activation='sigmoid')(next_layer)
    model = Model(inputs=[sequence_input_1, sequence_input_2], outputs=output)

    return model

def model_bert_chemmolefusion_capsule(param):

    sequence_input_1 = Input(shape=768)
    model_p = Flatten()(sequence_input_1)
    model_p = Dense(512, activation='relu',kernel_regularizer=keras.regularizers.l2(0.001))(model_p)
    # model_p = Dropout(0.3)(model_p)
    model_p = BatchNormalization()(model_p)

    sequence_input_2 = Input(shape=684)
    model_d = Flatten()(sequence_input_2)
    model_d = Dense(512, activation='relu',kernel_regularizer=keras.regularizers.l2(0.001))(model_d)
    # model_d = Dropout(0.3)(model_d)
    model_d = BatchNormalization()(model_d)

    model = concatenate([model_p, model_d])
    model = Reshape((-1, 8))(model)

    primarycaps = PrimaryCap(model, dim_vector=8, n_channels=8, kernel_size=param['kernel_size'], strides=1,
                             padding='valid')
    capsule = Capsule(num_capsule=param['num_capsule'], dim_capsule=16, routings=param['routings'],
                      share_weights=True)(
        primarycaps)

    length = Length()(capsule)

    model = Model(inputs=[sequence_input_1, sequence_input_2], outputs=length)
    model.summary()

    return model

def model_bert_chemmolefusion_dense(param):

    sequence_input_1 = Input(shape=768)
    model_p = Flatten()(sequence_input_1)
    model_p = Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(model_p)
    # model_p = Dropout(0.5)(model_p)
    model_p = BatchNormalization()(model_p)

    sequence_input_2 = Input(shape=684)
    model_d = Flatten()(sequence_input_2)
    model_d = Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(model_d)
    # model_d = Dropout(0.5)(model_d)
    model_d = BatchNormalization()(model_d)

    model = concatenate([model_p, model_d])
    x = Dense(64, activation='relu')(model)
    output = Dense(2, activation='sigmoid')(x)

    model = Model(inputs=[sequence_input_1, sequence_input_2], outputs=output)
    model.summary()
    return model

def model_bert_chemmolefusion_attention(param):

    sequence_input_1 = Input(shape=768)
    model_p = Flatten()(sequence_input_1)
    model_p = Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(model_p)
    # model_p = Dropout(0.5)(model_p)
    model_p = BatchNormalization()(model_p)

    sequence_input_2 = Input(shape=(684))
    model_d = Flatten()(sequence_input_2)
    model_d = Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(model_d)
    # model_d = Dropout(0.5)(model_d)
    model_d = BatchNormalization()(model_d)

    model_p_reshaped = Reshape((1, -1))(model_p)
    model_d_reshaped = Reshape((1, -1))(model_d)

    merged = Concatenate(axis=1)([model_p_reshaped, model_d_reshaped])

    attention_heads = 4
    attention_size = 128

    multiheaded_attention = MultiHeadAttention(
        num_heads=attention_heads,
        key_dim=attention_size,
        value_dim=attention_size,
        dropout=0.1
    )(merged, merged)

    multiheaded_attention = Flatten()(multiheaded_attention)
    next_layer = Dense(64, activation='relu')(multiheaded_attention)
    output = Dense(2, activation='sigmoid')(next_layer)
    model = Model(inputs=[sequence_input_1, sequence_input_2], outputs=output)

    return model