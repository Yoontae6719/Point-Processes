# Study for developing deep learning models for temporal point process
## UNIST Financial Engineering Lab  

The purpose of this study is to developing point process


The project can be accessed over at:
  - 🏁 NeurIPS
  - Learning to Select Exogenous Events for Marked Temporal Point Process, 2021, [Link](https://papers.nips.cc/paper/2021/hash/032abcd424b4312e7087f434ef1c0094-Abstract.html)
  - Continuous-time edge modelling using non-parametric point processes, 2021, [Link](https://papers.nips.cc/paper/2021/hash/1301962d8b7bd03fffaa27119aa7fc2b-Abstract.html)
  - Self-Adaptable Point Processes with Nonparametric Time Decays, 2021, [Link](https://papers.nips.cc/paper/2021/hash/243facb29564e7b448834a7c9d901201-Abstract.html)
  - Detecting Anomalous Event Sequences with Temporal Point Processes, 2021, [Link](https://papers.nips.cc/paper/2021/hash/6faa8040da20ef399b63a72d0e4ab575-Abstract.html)
  - Fast and Flexible Temporal Point Processes with Triangular Maps, 2021, [Link](https://papers.nips.cc/paper/2020/hash/00ac8ed3b4327bdd4ebbebcb2ba10a00-Abstract.html)
  - Point process latent variable models of larval zebrafish behavior, 2021, [Link](https://papers.nips.cc/paper/2018/hash/e02af5824e1eb6ad58d6bc03ac9e827f-Abstract.html)
 
    - Noise-Contrastive Estimation for Multivariate Point Processes, 2020, [Link](https://paperswithcode.com/paper/noise-contrastive-estimation-for-multivariate)
    - Fast and Flexible Temporal Point Processes with Triangular Maps, 2020, [Link](https://paperswithcode.com/paper/fast-and-flexible-temporal-point-processes)
    - Exact sampling of determinantal point processes with sublinear time preprocessing, 2019, [Link](https://paperswithcode.com/paper/exact-sampling-of-determinantal-point-1)
    - Mutually-Regressive-Point-Processes, 2019, [Link](https://github.com/ifiaposto/Mutually-Regressive-Point-Processes)
    - Wasserstein Learning of Deep Generative Point Process Models, 2017, [Link](https://paperswithcode.com/paper/wasserstein-learning-of-deep-generative-point)
    - Fully Neural Network based Model for General Temporal Point Processes, 2019, [Link](https://paperswithcode.com/paper/fully-neural-network-based-model-for-general)    



  - 🏁 ICLR
    - Neural Spectral Marked Point Processes, 2022 in review, [Link](https://paperswithcode.com/paper/neural-spectral-marked-point-processes-1)
    - Neural Spatio-Temporal Point Processes, 2021, [Link](https://paperswithcode.com/paper/neural-spatio-temporal-point-processes-1) 
    - Intensity-Free Learning of Temporal Point Processes, 2020, [Link](https://paperswithcode.com/paper/intensity-free-learning-of-temporal-point)

 - 🏁 ICML
    - Temporal Logic Point Processes [Link](https://paperswithcode.com/paper/temporal-logic-point-processes)

  - 🏁 arXiv
    - Time is of the Essence: a Joint Hierarchical RNN and Point Process Model for Time and Item Predictions, 2018, [Link](https://paperswithcode.com/paper/time-is-of-the-essence-a-joint-hierarchical)
    - Modeling The Intensity Function Of Point Process Via Recurrent Neural Networks, 2017, [Link](https://paperswithcode.com/paper/modeling-the-intensity-function-of-point)
    - Point process models for spatio-temporal distance sampling data from a large-scale survey of blue whales, 2016, [Link](https://stat.paperswithcode.com/paper/point-process-models-for-spatio-temporal)
    - Neural Temporal Point Processes For Modelling Electronic Health Records, 2020, [Link](https://paperswithcode.com/paper/neural-temporal-point-processes-for-modelling)
    - Batch Active Learning Using Determinantal Point Processes, 2019, [Link](https://paperswithcode.com/paper/batch-active-learning-using-determinantal)


- NEURAL SPECTRAL MARKED POINT PROCESSES, ICLR 2022 submission, [Link](https://openreview.net/pdf?id=0rcbOaoBXbg)
- EXPLAINING POINT PROCESSES BY LEARNING INTERPRETABLE TEMPORAL LOGIC RULES, ICLR 2022 submission, [Link](https://openreview.net/pdf?id=P07dq7iSAGr)
- DRIPP: DRIVEN POINT PROCESSES TO MODEL STIMULI INDUCED PATTERNS IN M/EEG SIGNALS, ICLR 2022 submission, [Link](https://openreview.net/pdf?id=d_2lcDh0Y9c)
- ONLINE MAP INFERENCE AND LEARNING FOR NONSYMMETRIC DETERMINANTAL POINT PROCESSES, ICLR 2022 submission, [Link](https://openreview.net/pdf?id=Jvoe8JCGvy)
-  Intensity-Free Learning of Temporal Point Processes, ICLR2020, [Link](https://arxiv.org/pdf/1909.12127.pdf), [Code](https://github.com/shchur/ifl-tpp)
-  Recurrent Marked Temporal Point Processes: Embedding Event History to Vector, KDD 2016 [Link](https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf), [Code](https://github.com/shchur/ifl-tpp)
-  The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point Process. NIPS 2017 [Link](https://arxiv.org/pdf/1612.09328.pdf), [Code](https://github.com/Hongrui24/NeuralHawkesPytorch) 



## 📝 Table of Contents

if you want see the detail, click following link.
- [Authors](#authors)
- [Features](#features)
- [Usage](#usage)
- [Requirements](./requirements.txt) 


## ✍️ Authors <a name = "authors"></a>
Participating research team:
- [Yoontae Hwang](https://www.notion.so/unist-felab/Yoontae-Hwang-9b1c43d6b1924d39a7940764fd0420b7) 

## 🏁 Features <a name = "Features"></a>


## Folder Structure [It will be updated]
  ```
  Energy/
  ├── main.py - main script to start training and test
  │
  ├── trainer.py - main script to start training and test
  │
  ├── conf/ - holds configuration for training
  │   ├── conf.py
  │   ├── gas_tft.yaml
  │   └── gas_nbeates.yaml
  │
  ├── data/ - default directory for storing input data
  │   └── gas_data.xlsx
  │   └── prepro.py       - for preprocessing my data
  │   └── prepro_data.csv
  │   └── solution.csv
  │
  ├── dataset/ -  anything about data loading goes here
  │   └── dataset.py      - dataloader for modeling
  │   └── utils.py        - utils for modeling (e.g. loss, csv function and metrics.)
  │
  ├── lighting_logs/
  │   └── defalut/        - trained models are saved here
  │          ├── version_0/
  │          └── version_n/
  ├── plot/
  │   ├── n_beats/ 
  │   └── tft/ 
  │
  ├── trainer/ - trainers
  │   └── trainer.py
  │
  └── result/                - result of modeling
      └── measure
   ```
  

## 🎈 Usage <a name = "usage"></a> 

After cloning this repo you need to install the requirements:
This has been tested with Python `v3.8.1`, Torch `v1.8.1` , pytorch-lightning, `v1.4.1` and pytorch-forecasting `v0.4.2`.

```shell
pip install -r requirements.txt
```

