# CheXNet implementation in PyTorch

Yet another PyTorch implementation of the [CheXNet](https://arxiv.org/abs/1711.05225) algorithm for pathology detection in 
frontal chest X-ray images. This implementation is based on approach presented [here](https://github.com/arnoweng/CheXNet). Ten-crops 
technique is used to transform images at the testing stage to get better accuracy. 

The highest accuracy evaluated with AUROC was 0.8508 (see the model m-25012018-123527 in the models directory).
The same training (70%), validation (10%) and testing (20%) datasets were used as in [this](https://github.com/arnoweng/CheXNet) 
implementation.

![alt text](test/heatmap.png)

## Prerequisites
* Python 3.5.2
* Pytorch
* OpenCV (for generating CAMs)

## Usage
* Download the ChestX-ray14 database from [here](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/37178474737)
* Unpack archives in separate directories (e.g. images_001.tar.gz into images_001)
* Run `python Main.py classification-test` to run test using the pre-trained model (`m-25012018-123527`)
* Run `python Main.py classification-train` to train the original classifier
* Run `python Main.py retrieval-split` to rebuild patient-disjoint retrieval splits
* Run `python Main.py retrieval-train` to finetune a retrieval backbone on NIH labels
* Run `python Main.py retrieval-test --model-path <checkpoint>` to evaluate a retrieval checkpoint

This implementation allows to conduct experiments with 3 different densenet architectures: densenet-121, densenet-169 and
densenet-201.

* To generate CAM of a test file run script HeatmapGenerator 

## Retrieval Finetuning

Retrieval-specific scaffolding has been added in:

* **RetrievalDataset.py** - dataset and balanced batch sampler with pathology, patient, and exact-label grouping
* **RetrievalModels.py** - generic retrieval backbone registry with ResNet-50 and optional ConvNeXtV2 / DINOv2 backbones
* **RetrievalTrainer.py** - supervised contrastive finetuning loop, ASL auxiliary loss, hard-negative mining, Recall@K, mAP@10, and nDCG@10
* **SplitBuilder.py** - patient-disjoint split generation utility
* **RETRIEVAL.md** - task plan, loss choice, dataloader notes, and retrieval options

### Retrieval Commands

Build patient-disjoint splits:

```powershell
python Main.py retrieval-split --input-files ./dataset/train_1.txt ./dataset/val_1.txt ./dataset/test_1.txt --output-dir ./dataset --prefix retrieval_patient
```

Train retrieval with the current default setup:

```powershell
python Main.py retrieval-train --data-dir ./database --train-file ./dataset/retrieval_patient_train.txt --val-file ./dataset/retrieval_patient_val.txt --architecture resnet50 --use-pretrained --classification-loss asl --positive-mode label_overlap --grouping pathology --batch-size 32 --max-epoch 20 --backbone-learning-rate 1e-5 --head-learning-rate 1e-4 --output-dir ./models
```

Test a trained retrieval checkpoint:

```powershell
python Main.py retrieval-test --data-dir ./database --test-file ./dataset/retrieval_patient_test.txt --architecture resnet50 --model-path ./models/retrieval-<timestamp>.pth.tar --positive-mode label_overlap
```

## Results
The highest accuracy 0.8508 was achieved by the model m-25012018-123527 (see the models directory).

| Pathology     | AUROC         |
| ------------- |:-------------:|
| Atelectasis   | 0.8321        |
| Cardiomegaly  | 0.9107        |
| Effusion      | 0.8860        |
| Infiltration  | 0.7145        |
| Mass          | 0.8653        |
| Nodule        | 0.8037        |
| Pneumonia     | 0.7655        |
| Pneumothorax  | 0.8857        |
| Consolidation | 0.8157        |
| Edema         | 0.9017        |
| Emphysema     | 0.9422        |
| Fibrosis      | 0.8523        |
| P.T.          | 0.7948        |
| Hernia        | 0.9416        |

## Computation time
The training was done using single Tesla P100 GPU and took approximately 22h.

