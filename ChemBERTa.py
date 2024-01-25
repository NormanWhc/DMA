import numpy as np
import pandas as pd
import torch
import os
from torch import nn
from Mole_Bert import molebert       #pip install transformer-pytorch
# import transformershuggingface as ppbhuggingface
from transformers import BertModel, BertTokenizer, AutoTokenizer, AutoModel, AutoModelForMaskedLM, T5Tokenizer, T5ForConditionalGeneration
import warnings
warnings.filterwarnings('ignore')

class iAFF(nn.Module):
    '''
    多特征融合 iAFF
    '''

    def __init__(self, channels=64, r=4):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        # 本地注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 第二次本地注意力
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 第二次全局注意力
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        return xo

class MS_CAM(nn.Module):
    def __init__(self, channels=64, r=4):
        super(MS_CAM, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(1, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei


def chemberta(protein_list):
    #download model from https://github.com/agemagician/ProtTrans
    tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
    #
    model = BertModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
    # model = BertModel.from_pretrained("Rostlab/prot_bert")
    # tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
    # load pretein sequences
    # df=pd.read_csv(protein_file, header=None)
    # get protein-BERT feature
    x=len(protein_list)
    list2=[]
    for i in range(x):
        # inputs = tokenizer(protein_list[i], return_tensors='pt')["input_ids"]
        # hidden_states = model(inputs)[0]  # [1, sequence_length, 768]
        # embedding_max = torch.max(hidden_states[0], dim=0)[0]
        # list2.append(embedding_max)

    #    print(df.iloc[1,2])
        print(i)
    #    print(len(df.iloc[i,1]))
        sequence_Example=protein_list[i]
        #         想要的长度*2
        # print(len(df.iloc[i,1]))
        print('SMILE-length:',len(sequence_Example))
        max_length = model.config.max_position_embeddings
        encoded_input = tokenizer(sequence_Example,max_length=max_length, return_tensors='pt')

        output = model(**encoded_input)
        # print(output)
        s = output[0].data.cpu().numpy() # 数据类型转换
        list2.append(s)
    list1=[]
    for i in range(x):
        # data = np.load("./data/"+str(i)+'.npy')
        data=list2[i]
        # print(data)
        d=data.mean(axis=1)
        # print(d)
        feat=d[0].tolist()
        # print(feat)
        list1.append(feat)


    # feature_smole = torch.tensor(list1, dtype=torch.float32, device='cpu').unsqueeze(0)
    # feature_smole = pd.DataFrame(list1)
    # features_mole = molebert(protein_list)
    # feature_fusion = pd.concat([feature_smole,features_mole],axis=1)

    # feature_fusion_array = feature_fusion.values
    # feature_fusion_tensor = torch.from_numpy(feature_fusion_array).float().unsqueeze(0)

    # features_mole = torch.tensor(features_mole, dtype=torch.float32, device='cpu').unsqueeze(0)
    # features_mole_resized = torch.nn.functional.interpolate(features_mole.unsqueeze(0), size=(824, 512),
    #                                                         mode='bilinear', align_corners=False).squeeze(0)
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    # device = torch.device("cuda:0")
    #
    # channels = feature_smole.shape[1]
    #
    # model = iAFF(channels=channels)
    # model = model.to(device).train()
    # output = model(feature_smole, features_mole_resized)

    # layer = MS_CAM(64, 4)
    # out = layer(feature_fusion_tensor)
    feature = pd.DataFrame(list1)
    print(feature)
    return feature



# def smole_bert(protein_list):
#     #download model from https://github.com/agemagician/ProtTrans
#     features = []
#     feature = []
#     for dna in protein_list:
#         model = AutoModel.from_pretrained("UdS-LSV/smole-bert",ignore_mismatched_sizes=True)
#         tokenizer = AutoTokenizer.from_pretrained("UdS-LSV/smole-bert", do_lower_case=False)
#         input_ids = tokenizer.encode(dna, add_special_tokens=True)
#         input_ids_tensor = torch.tensor([input_ids])
#         with torch.no_grad():
#             outputs = model(input_ids_tensor)
#             embeddings = outputs.last_hidden_state[0][0]
#             feature.append(embeddings)
#             features = pd.DataFrame(feature)
#     return features

# def smole_bert(protein_list):
#     smile_list = []
#     x=len(protein_list)
#     for i in range(x):
#         tokenizer_ = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
#
#         model_ = BertModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
#         max_length = model_.config.max_position_embeddings
#         print(i,x)
#         encoded_input = tokenizer_(protein_list[i], truncation=True, max_length=max_length, return_tensors='pt')
#         outputs_ = model_(**encoded_input)
#         smile_list.append(outputs_.pooler_output[0])
#     features = pd.DataFrame(smile_list)
#     print(features)




if __name__ == '__main__':
    dti_df = pd.read_csv("./data/new_data.txt",sep=" ")
    protein_list = dti_df["SMILES"].tolist()
    chemberta(protein_list)

