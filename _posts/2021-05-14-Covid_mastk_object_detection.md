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
As a next step we need some kind of data. The data source will be Kaggle. [Face mask Detection](https://www.kaggle.com/andrewmvd/face-mask-detection) is the competition from where we will be using the data. One can simply download the data using the downlaod option from data tab or use kaggle api to download the data. Kaggle api is very easy way to downlaod dat from any competion.[This repository](https://github.com/Kaggle/kaggle-api) describes how to install and get api key. Then one should go to terminal and run this```bash
kaggle datasets download [UserName/DatasetName]```.  
so now we have the data.

# Pytorch Implementation
Normally to do any type of deep learning work in pytorch one needs to format the data in way that the data will be tensor and some (x, y) format. In our case x will be the iamge and y will be bounding box coordinate. This (x, y) combination is called ``Dataset`` in pytorch.
Then we need to put some time or iterator so that all data will be converted some type of batch. Each time a batch of data will be called. We need convert our data to 
1. Dataset
2. Dataloader

## Data preparation
We have two folders containing the data we require. 
1. images --> contains images in png format
2. annotations --> conatains xml file where each images bounding box location can be found. So we need some type of function where will use filename as an input and the output will be then all bounding box location in that xml file
```python
def file_to_annot(filename):
    """
    Convert xml file to annotation of image 
    
    filename:string or pathlib type object
    
    return tuple. first element will be list of all bounding boxes in the image
    second element of the tuple will be tag of that bounding boxes
    """
    with open(filename) as fl:
        dat = fl.read()
        soup = BeautifulSoup(dat, 'xml')
        obj = soup.find_all('object')
        bboxes = [[x_h('xmin', i), x_h('ymin', i), x_h('xmax',i), x_h('ymax',i)] for i in obj]
        tags = [i.find('name').text for i in obj]
        tags_ = bboxes, tags
    return tags_
```



Then create some type of model. After that train the model based on some type of optimizer and see the result of the model on test data. What is test data ?

## Data Splitting
Normally in machine learning we convert the data in two sets, one set is used for training and other set is used to see how the model is doing on other set of data. Sometimes it happens that, as we are tuning the model performence on unseen data, there is a risk we are fitting the data on this unseen data. Therefore another set of data is removed and when the model does better on unseen data, then the last set of data is used to actually see the model performence.
So we have 3 sets of data
1. Training set
2. Validation set (unseen data)
3. Test set (unseen data and model performence decider)


