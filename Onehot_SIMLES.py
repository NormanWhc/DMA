import pandas as pd
import numpy as np
def smi_preprocessing(smi_sequence):
    splited_smis=[]
    length=[]
    end="/n"
    begin="&"
    element_table=["C","N","B","O","P","S","F","Cl","Br","I","(",")","=","#"]
    for i in range(len(smi_sequence)):
        smi=smi_sequence[i]
        splited_smi=[]
        j=0
        while j<len(smi):
            smi_words=[]
            if smi[j]=="[":
                smi_words.append(smi[j])
                j=j+1
                while smi[j]!="]":
                    smi_words.append(smi[j])
                    j=j+1
                smi_words.append(smi[j])
                words = ''.join(smi_words)
                splited_smi.append(words)
                j=j+1

            else:
                smi_words.append(smi[j])

                if j+1<len(smi[j]):
                    smi_words.append(smi[j+1])
                    words = ''.join(smi_words)
                else:
                    smi_words.insert(0,smi[j-1])
                    words = ''.join(smi_words)

                if words not in element_table:
                    splited_smi.append(smi[j])
                    j=j+1
                else:
                    splited_smi.append(words)
                    j=j+2

        splited_smi.append(end)
        splited_smi.insert(0,begin)
        splited_smis.append(splited_smi)
    return splited_smis

def one_hot_encoding(smi, vocalbulary):
    res=[]
    for i in vocalbulary:
        if i in smi:
            res.append(1)
        else:
            res.append(0)
    return res

def encode_smiles(smi,vocalbulary):
    res = []
    for drug_smile in smi:
        res.append(one_hot_encoding(drug_smile,vocalbulary))
    print(len(res))
    df = pd.DataFrame(res)
    return df

from sklearn.decomposition import PCA
def calculate_pca(profile_file):
    pca = PCA(copy=True, iterated_power='auto', n_components=31, random_state=None,
              svd_solver='auto', tol=0.0, whiten=False)
    df = profile_file

    X = df.values
    X = pca.fit_transform(X)

    new_df = pd.DataFrame(X, index=df.index)

    return new_df


def onehot_smiles(smiles):
    smi = smi_preprocessing(smiles)
    vocalbulary = []
    for i in smi:
        vocalbulary.extend(i)
    vocalbulary = list(set(vocalbulary))
    features = encode_smiles(smi, vocalbulary)
    new_data = calculate_pca(features)

    return np.array(new_data)
