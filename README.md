# Study for developing deep learning models for Temporal point process
## UNIST Financial Engineering Lab  

The purpose of this study is to developing Temporal point process


The project can be accessed over at:
-  Intensity-Free Learning of Temporal Point Processes, ICLR2020, [Link](https://arxiv.org/pdf/1909.12127.pdf), [Code](https://github.com/shchur/ifl-tpp)
-  Recurrent Marked Temporal Point Processes: Embedding Event History to Vector, KDD 2016 [Link](https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf), [Code](https://github.com/shchur/ifl-tpp)
-  Comming 



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

