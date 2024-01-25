from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis as PA
import pandas as pd
import numpy as np
# from tensorflow.keras.datasets import mnist
# from keras.utils import np_utils
# from keras.utils.vis_utils import plot_model
# from tensorflow.keras.models import Sequential,Model
# from tensorflow.keras.layers import Dense,Input,Conv1D,LSTM,MaxPooling1D,Flatten,Dropout,concatenate,Reshape,GlobalAveragePooling1D,BatchNormalization
# from keras.layers.recurrent import SimpleRNN
# from keras import layers
# from keras.optimizers import Adam
# from keras.optimizers import adam_v2
# from CapsuleLayer import Capsule
import matplotlib.pyplot as plt
# from matplotlib import pyplot as plt
from bert import bert
from transformer import transformer
from genalm import genalm
from Mole_Bert import molebert
from Onehot_SIMLES import onehot_smiles
from CNN_SIMLES import cnn_smiles
from SMOLE_Bert import smole_bert
from ChemBERTa import chemberta
from smole_ChemBERTa_fusion import smolechemfusion
from smole_mole_fusion import smolemolefusion
from ChemBERTa_mole_fusion import chemmolefusion
import psutil
import matplotlib
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
from Capsule_MPNN import *
import os
import sys
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors


def onehot(seq_list, seq_len):
    bases = ['G', 'A', 'T', 'C', 'U']
    X = np.zeros((len(seq_list), seq_len, len(bases) + 1))
    # print(X.shape)
    for i, seq in enumerate(seq_list):
        for l, aa in enumerate(str(seq)):
            if l < int(seq_len):
                if aa in bases:
                    X[i, l, bases.index(aa) + 1] = 1
                else:
                    X[i, l, 0] = 1
    return X

def newdata(dti_fname, protein_encoder, seq_len, drug_encoder, form_negative):
    dti_df = pd.read_csv(dti_fname, sep="\t")
    # dti_df
    # if form_negative == 1:
    # 创建空的DataFrame
    dti_df['sequence'] = dti_df['sequence'].str.replace('U', 'T')
    # length = len(dti_df["sequence"])
    # # 获取所有的protein和drug，并按顺序排列
    # proteins = sorted(dti_df["sequence"])
    # drugs = sorted(dti_df["SMILES"].unique())
    # df = pd.DataFrame(columns=['sequence', 'SMILES', 'expression'])
    #
    # for protein in tqdm(proteins, desc='Proteins'):
    #     for drug in tqdm(drugs, desc='Drugs', leave=False):
    #         # 查找data1中对应关系的label
    #         label = dti_df[(dti_df["sequence"] == protein) & (dti_df["SMILES"] == drug)]["expression"].values
    #         if len(label) == 0 or label[0] == 0:
    #             # 若存在对应关系，则取第一个label值
    #             label_value = 0
    #         else:
    #             continue
    #         # 添加到DataFrame
    #         new_row = pd.Series({'sequence': protein, 'SMILES': drug, 'expression': label_value})
    #         df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
    #
    # negative = df.sample(n=length, replace=False, random_state=11)
    # # 输出结果
    # data1 = pd.concat([dti_df,negative], ignore_index=True)
    # else:

    data1 = dti_df

    protein_list = data1['sequence'].tolist()

    length = data1['sequence'].map(lambda x: len(str(x)))
    print("Sequence_max_length:" + str(length.max()))
    # protein_list
    if protein_encoder == "onehot":
        protein_df = onehot(protein_list, seq_len)

    if protein_encoder == "bert":
        protein_df = bert(protein_list)

    if protein_encoder == "transformer":
        protein_df = transformer(protein_list)

    if protein_encoder == "genalm":
        protein_df = genalm(protein_list)

    drug_df_list = []

    if drug_encoder == "fingerprint":
        drug_list = data1['SMILES'].tolist()
        j = 0
        column_list = []
        while j < 1024:
            k = j + 1
            column_list.append("fingerprints" + str(k))
            j = j + 1

        drug_fingerprint_list = []
        for x, drug in enumerate(drug_list):
            drug_fingerprint = []
            if Chem.MolFromSmiles(drug):
                mol = Chem.MolFromSmiles(drug)
                fps = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                fingerprints = fps.ToBitString()
                i = 0
                while i < 1024:
                    drug_fingerprint.append(float(fingerprints[i]))
                    i = i + 1
            else:
                print(str(drug))
                print("Above smile transforms to fingerprint error!!!")
                print("Please remove " + str(x + 1) + " line")
                i = 0
                while i < 1024:
                    drug_fingerprint.append(0)
                    i = i + 1
            drug_fingerprint_list.append(drug_fingerprint)
        drug_df = pd.DataFrame(data=drug_fingerprint_list, columns=column_list)
        print("Drug data:" + str(drug_df.shape))
        drug_df_list.append(drug_df)

    if drug_encoder == "MPNN" :
        drug_df = graphs_from_smiles(data1['SMILES'].tolist())
        # print("drug[0]:"+str(len(drug_df[0])))
        # print("drug[1]:"+str(len(drug_df[1])))
        # print("drug[2]:"+str(len(drug_df[2])))
        drug_df_list.append(drug_df)

    if drug_encoder == "mole":

        drug_df = molebert(data1['SMILES'].tolist())
        drug_df_list.append(drug_df)

    if drug_encoder == "onehot":
        drug_df = onehot_smiles(data1['SMILES'].tolist())
        drug_df_list.append(drug_df)

    if drug_encoder == "CNN":
        drug_df = cnn_smiles(data1['SMILES'].tolist())
        drug_df_list.append(drug_df)

    if drug_encoder == "smole":
        drug_df = smole_bert(data1['SMILES'].tolist())
        drug_df_list.append(drug_df)

    if drug_encoder == "chemberta":

        drug_df = chemberta(data1['SMILES'].tolist())
        drug_df_list.append(drug_df)

    if drug_encoder == "smolechemfusion":
        drug_df = smolechemfusion(data1['SMILES'].tolist())
        drug_df_list.append(drug_df)

    if drug_encoder == "smolemolefusion":
        drug_df = smolemolefusion(data1['SMILES'].tolist())
        drug_df_list.append(drug_df)

    if drug_encoder == "chemmolefusion":
        drug_df = chemmolefusion(data1['SMILES'].tolist())
        drug_df_list.append(drug_df)

    y = data1['expression'].tolist()

    # print("protein data:"+str(protein_df.shape))
    # print("label:"+str(len(y)))
    return np.array(protein_df), np.array(drug_df_list), y
