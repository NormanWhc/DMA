# CapBM-DTI
A robust drug-target interaction prediction framework with capsule network and transfer learning

![image](https://user-images.githubusercontent.com/89676756/207692614-9c8aeb1b-0582-424a-999e-1cddaffb9523.png)


This is the code for CapBM-DTI.


### Requirements

- Python 3.x
- numpy
- scikit-learn
- RDKit
- Tensorflow
- keras

### Run

To run this code:

python CapBM_DTI.py --dti data/benchmark.txt --protein-descripter bert  --drug-descripter MPNN --model-name bert_MPNN_capsule --batch-size 64 -e 1000 -dp data