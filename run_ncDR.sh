#!/bin/bash
python CapBF.py --dti data/all.csv --protein-descripter bert --drug-descripter fingerprint --model-name bert_fingerprint_capsule --batch-size 64 -e 1000 -dp data -g 0 -negative 1
python DenseBF.py --dti data/all.csv --protein-descripter bert --drug-descripter fingerprint --model-name bert_fingerprint_dense --batch-size 64 -e 1000 -dp data -g 0 -negative 1
python CapBM_DTI.py --dti data/all.csv --protein-descripter bert --drug-descripter MPNN --model-name bert_MPNN_capsule --batch-size 64 -e 1000 -dp data -g 0 -negative 1
python DenseBM.py --dti data/all.csv --protein-descripter bert --drug-descripter MPNN --model-name bert_MPNN_dense --batch-size 64 -e 1000 -dp data -g 0 -negative 1
python CapBMole.py --dti data/all.csv --protein-descripter bert --drug-descripter mole --model-name bert_mole_capsule --batch-size 64 -e 1000 -dp data -g 0 -negative 1
python DenseBMole.py --dti data/all.csv --protein-descripter bert --drug-descripter mole --model-name bert_mole_dense --batch-size 64 -e 1000 -dp data -g 0 -negative 1
python AttenBMole.py --dti data/all.csv --protein-descripter bert --drug-descripter mole --model-name bert_mole_attention --batch-size 64 -e 1000 -dp data -g 0 -negative 1
python CapBO.py --dti data/all.csv --protein-descripter bert --drug-descripter onehot --model-name bert_onehot_capsule --batch-size 64 -e 1000 -dp data -g 0 -negative 1
python DenseBO.py --dti data/all.csv --protein-descripter bert --drug-descripter onehot --model-name bert_onehot_dense --batch-size 64 -e 1000 -dp data -g 0 -negative 1
python CapOC.py --dti data/all.csv --protein-descripter onehot --drug-descripter CNN --model-name onehot_CNN_capsule --batch-size 64 -e 1000 -dp data -g 0 -sl 1024 -negative 1
python DenseOC.py --dti data/all.csv --protein-descripter onehot --drug-descripter CNN --model-name onehot_CNN_dense --batch-size 64 -e 1000 -dp data -g 0 -sl 1024 -negative 1
python CapBC.py --dti data/all.csv --protein-descripter bert --drug-descripter CNN --model-name bert_CNN_capsule --batch-size 64 -e 1000 -dp data -g 0 -sl 1024 -negative 1
python DenseBC.py --dti data/all.csv --protein-descripter bert --drug-descripter CNN --model-name bert_CNN_dense --batch-size 64 -e 1000 -dp data -g 0 -sl 1024 -negative 1
python CapOF.py --dti data/all.csv --protein-descripter onehot --drug-descripter fingerprint --model-name onehot_fingerprint_capsule --batch-size 64 -e 1000 -dp data -g 0 -sl 1024 -negative 1
python DenseOF.py --dti data/all.csv --protein-descripter onehot --drug-descripter fingerprint --model-name onehot_fingerprint_dense --batch-size 64 -e 1000 -dp data -g 0 -sl 1024 -negative 1
python CapOM.py --dti data/all.csv --protein-descripter onehot --drug-descripter MPNN --model-name onehot_MPNN_capsule --batch-size 64 -e 1000 -dp data -g 0 -sl 1024 -negative 1
python DenseOM.py --dti data/all.csv --protein-descripter onehot --drug-descripter MPNN --model-name onehot_MPNN_dense --batch-size 64 -e 1000 -dp data -g 0 -sl 1024 -negative 1
python CapOMole.py --dti data/all.csv --protein-descripter onehot --drug-descripter mole --model-name onehot_mole_capsule --batch-size 64 -e 1000 -dp data -g 0 -sl 1024 -negative 1
python DenseOMole.py --dti data/all.csv --protein-descripter onehot --drug-descripter mole --model-name onehot_mole_dense --batch-size 64 -e 1000 -dp data -g 0 -sl 1024 -negative 1
python CapOO.py --dti data/all.csv --protein-descripter onehot --drug-descripter onehot --model-name onehot_onehot_capsule --batch-size 64 -e 1000 -dp data -g 0 -sl 1024 -negative 1
python DenseOO.py --dti data/all.csv --protein-descripter onehot --drug-descripter onehot --model-name onehot_onehot_dense --batch-size 64 -e 1000 -dp data -g 0 -sl 1024 -negative 1
python CapBSmole.py --dti data/all.csv --protein-descripter bert --drug-descripter smole --model-name bert_smole_capsule --batch-size 64 -e 1000 -dp data -g 0 -negative 1
python DenseBSmole.py --dti data/all.csv --protein-descripter bert --drug-descripter smole --model-name bert_smole_dense --batch-size 64 -e 1000 -dp data -g 0 -sl 1024 -negative 1
python AttenBSmole.py --dti data/all.csv --protein-descripter bert --drug-descripter smole --model-name bert_smole_attention --batch-size 64 -e 1000 -dp data -g 0 -sl 1024 -negative 1
python CapBChemberta.py --dti data/all.csv --protein-descripter bert --drug-descripter chemberta --model-name bert_chemberta_capsule --batch-size 64 -e 1000 -dp data -g 0 -sl 1024 -negative 1
python DenseBChemberta.py --dti data/all.csv --protein-descripter bert --drug-descripter chemberta --model-name bert_chemberta_dense --batch-size 64 -e 1000 -dp data -g 0 -sl 1024 -negative 1
python AttenBChemberta.py --dti data/all.csv --protein-descripter bert --drug-descripter chemberta --model-name bert_chemberta_attention --batch-size 64 -e 1000 -dp data -g 0 -sl 1024 -negative 1
python CapBSmoleChemFusion.py --dti data/all.csv --protein-descripter bert --drug-descripter smolechemfusion --model-name bert_smolechemfusion_capsule --batch-size 64 -e 1000 -dp data -g 0 -sl 1024 -negative 1
python DenseBSmoleChemFusion.py --dti data/all.csv --protein-descripter bert --drug-descripter smolechemfusion --model-name bert_smolechemfusion_dense --batch-size 64 -e 1000 -dp data -g 0 -sl 1024 -negative 1
python AttenBSmoleChemFusion.py --dti data/all.csv --protein-descripter bert --drug-descripter smolechemfusion --model-name bert_smolechemfusion_attention --batch-size 64 -e 1000 -dp data -g 0 -sl 1024 -negative 1
python CapBSmoleMoleFusion.py --dti data/all.csv --protein-descripter bert --drug-descripter smolemolefusion --model-name bert_smolemolefusion_capsule --batch-size 64 -e 1000 -dp data -g 0 -sl 1024 -negative 1
python DenseBSmoleMoleFusion.py --dti data/all.csv --protein-descripter bert --drug-descripter smolemolefusion --model-name bert_smolemolefusion_dense --batch-size 64 -e 1000 -dp data -g 0 -sl 1024 -negative 1
python AttenBSmoleMoleFusion.py --dti data/all.csv --protein-descripter bert --drug-descripter smolemolefusion --model-name bert_smolemolefusion_attention --batch-size 64 -e 1000 -dp data -g 0 -sl 1024 -negative 1
python CapBChemMoleFusion.py --dti data/all.csv --protein-descripter bert --drug-descripter chemmolefusion --model-name bert_chemmolefusion_capsule --batch-size 64 -e 1000 -dp data -g 0 -sl 1024 -negative 1
python DenseBChemMoleFusion.py --dti data/all.csv --protein-descripter bert --drug-descripter chemmolefusion --model-name bert_chemmolefusion_dense --batch-size 64 -e 1000 -dp data -g 0 -sl 1024 -negative 1
python AttenBChemMoleFusion.py --dti data/all.csv --protein-descripter bert --drug-descripter chemmolefusion --model-name bert_chemmolefusion_attention --batch-size 64 -e 1000 -dp data -g 0 -sl 1024 -negative 1