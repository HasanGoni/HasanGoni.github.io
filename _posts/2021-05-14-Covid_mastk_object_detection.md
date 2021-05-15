# General description
In this blog we will try to perform object detection using pytorch. We will be using small dataset (approximately 853 images). So transfer learning will be used here.

One can write all necessary function from scratch. However pytorch provides some additional helper functions which makes object detection very easy. So we will be using those functions. To get all the scripts we need to clone the github repository. For jupyter notebook these are commands one needs to use to get all the functions.
```bash
!git clone https://github.com/pytorch/vision.git
!git checkout v0.3.0

!cp vision/references/detection/utils.py ./
!cp vision/references/detection/transforms.py ./
!cp vision/references/detection/coco_eval.py ./
!cp vision/references/detection/coco_utils.py ./
!cp vision/references/detection/engine.py ./
```
If command line is used then at the begining of the line this ``!`` needs to be removed.
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
At first we will do some preprocessing, so that will give the name of the iamge and x, y pair will be created, i.e. the image and the coordinate of the bounding box.

We have two folders containing the data we require. 
1. images --> contains images in png format
2. annotations --> conatains xml file where each images bounding box location can be found. So we need some type of function where will use filename as an input and the output will be then all bounding box location in that xml file
```python
def x_h(x, obj):
    """
    helper function during xml file reading
    x:str: name of the tag
    obj: obj from where this string 
    will be searched
    
    integer tag value actually here
    xmin, xmax, ymin, ymax value retrieve helper
    """
    return int(obj.find(x).text)

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
So we have this  ``file_to_annot`` function which will convert the xml file to the related tags and position of the bounding box position of this tag. There is a possibility that each xml file conatains more than one tags, therefore all the tags will be added in a list.

We have our y is ready to crate pytorch Dataset. Now we need x. To create iamge from filename we will be using PIL imaging libray.

```python
from PIL import Image
img = Image.open(image_filename).convert('RGB')
```
Now we are ready to create pytorch Dataset. We will create a class which will be inherited from ``torch.utils.data.Dataset```. Our create Dataset will have two properties.
1. Indexing: means when we can say Dataset[3] and it will give us 4th value of datset(image and bounding boxes)
2. Length: if we say len(Dataset) it will give us length of the dataset

In python normally ``__getitem__`` is reponsible for indexing and ``__len__`` is reponsible for length. So when try to index something pyhton called ``__getitem__`` function. So we need to implement ``__getitem__`` function in a way that, it will create image and tell the tags and bounding boxes of that image. We actually already implemented everything, we just need to put them in ``__getitem__`` function.

One last thing we need to consider is the number of indexing. When we 



Then create some type of model. After that train the model based on some type of optimizer and see the result of the model on test data. What is test data ?

## Data Splitting
Normally in machine learning we convert the data in two sets, one set is used for training and other set is used to see how the model is doing on other set of data. Sometimes it happens that, as we are tuning the model performence on unseen data, there is a risk we are fitting the data on this unseen data. Therefore another set of data is removed and when the model does better on unseen data, then the last set of data is used to actually see the model performence.
So we have 3 sets of data
1. Training set
2. Validation set (unseen data)
3. Test set (unseen data and model performence decider)


