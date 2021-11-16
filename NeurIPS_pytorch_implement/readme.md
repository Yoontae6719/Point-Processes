
# Study for developing deep learning models for temporal point process
## UNIST Financial Engineering Lab 

I am Yoontae Hwang 😃. I am studying at the UNIST Financial Engineering Lab as an integrated master's and doctoral program.


The project can be accessed over at:
  - 🏁 NeurIPS
    - Fully Neural Network based Model for General Temporal Point Processes [We call that paper GTPP], 2019, [Link](https://paperswithcode.com/paper/fully-neural-network-based-model-for-general) 
    

## 📝 Table of Contents

if you want see the detail, click following link.
- [Authors](#authors)
- [Features](#features)
- [Usage](#usage)
- [Requirements](./requirements.txt) 


## ✍️ Authors <a name = "authors"></a>
- [Yoontae Hwang](https://www.notion.so/unist-felab/Yoontae-Hwang-9b1c43d6b1924d39a7940764fd0420b7) 

## 🏁 Features <a name = "Features"></a>


## Folder Structure 
  ```
  Folders/    # See the paper list !
  │
  ├── mian.py - main script to start training and test      # Will be updated !
  │
  ├── trainer.py - main script to start training and test           
  │
  ├── conf/ - holds configuration for training                       
  │   ├── conf.py                  
  │   ├── finance.yaml                                      # If you want academic dataset use that yaml 
  │   ├── mimic.yaml 
  │   ├── retwwet.yaml
  │   ├── so.yaml
  │   └── test.yaml                                         # will be revised ! 
  │
  ├── data/ - default directory for storing input data
  │   └── data.xlsx
  │   
  │
  ├── dataset/ -  anything about data loading goes here
  │   └── dataset.py                                         # dataloader for modeling
  │   └── generate_utils.py                                  # utils (e.g. loss, csv function, generate and metrics.)
  │   
  ├── plot/ - will be updated !!
  │   ├── / 
  │
  ├── model/ -          
  │   └── net.py                  
  │                                                                               
  └── progressbar.py             

   ```
