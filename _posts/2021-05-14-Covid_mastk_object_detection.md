# Getting helper functions from Github repository
```bash
!git clone https://github.com/pytorch/vision.git
!git checkout v0.3.0

!cp vision/references/detection/utils.py ./
!cp vision/references/detection/transforms.py ./
!cp vision/references/detection/coco_eval.py ./
!cp vision/references/detection/coco_utils.py ./
!cp vision/references/detection/engine.py ./
```

One can actually use the following commands in a command line or use Jupyter notebook. If command line is used then at the begining of the line this ``!`` needs to be removed.
So  
```bash
!git clone https://github.com/pytorch/vision.git
```
would be then 
`git clone https://github.com/pytorch/vision.git`

what this command is doing is actually clonning the pytorch/vision repository
then entering a branch named v0.3.0 and then copying util.py, transforms.py, coco_utils.py and engine.py to current directory, which was previously in vision/references/detection/ directory . So you are good to go. 

# Getting data 
As a next step we need some kind of data. The data source will be Kaggle. [Face mask Detection](https://www.kaggle.com/andrewmvd/face-mask-detection) is the competition from where we will use data. One can simply download the data or use kaggle api to download data. Kaggle api is very easy way to downlaod the from kaggle. [This repository](https://github.com/Kaggle/kaggle-api) describes how to install and get api key. Then one should go to terminal and run this```bash
kaggle datasets download [UserName/DatasetName]```.  
so now we have the data.

# Pytorch Implementation
