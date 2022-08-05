---
toc: true
layout: post
comments: true
author: Nguyen Hoang-Phuong
image: images/mmdetection.png
description: The seventh part of the Segmentation Tutorial Series, a step-by-step guide to developing Instance Segmentation Models in MMDetection
categories: [pytorchlightning, semanticsegmentation, deeplearning, maskrcnn, huggingface]
title: Segmentation Model-Part VII -  Training Instance Segmentation in MMDetection
---

In this post, we will cover how to train a instance segmentation model by using the MMDetection library.

## 1. Semantic Segmentation vs Instance Segmentation

We first introduce about: Semantic image segmentation, Object detection, Semantic Image segmentation

- A Semantic image segmentation marks all pixels belonging to that tag, but won’t define the boundaries of each object.
- A Object detection does not segment the object, but define the location of each individual object instance with a box.
- Combining the semantic segmentation with the object detection leads to a instance segmentation

<!-- <img align="center" width="600"  src="https://habrastorage.org/webt/uc/uw/sy/ucuwsy-0vb8vjqw0v_9gviyv-ga.jpeg"> -->

![](https://habrastorage.org/webt/uc/uw/sy/ucuwsy-0vb8vjqw0v_9gviyv-ga.jpeg "Instace Segmentation ,Source: V7Lab")



Nowaday, to tackle the instance segmentation problem, one use uselly [Mask R-CNN model](https://arxiv.org/pdf/1703.06870.pdf) which is presented by [K.He] and all.  For more detail about Mask R-CNN model, we  refer to read [Everything about Mask R-CNN: A Beginner’s Guide](https://viso.ai/deep-learning/mask-r-cnn/#:~:text=Mask%20R%2DCNN%20is%20a,segmentation%20mask%20for%20each%20instance.) artical. 


**Mask R-CNN** is the state-of-the-art model for the Instance Segmentation with three outputs of the model: mask, classes and boundary box. 

<!-- <img align="center" width="600"  src="https://habrastorage.org/webt/kg/sg/eb/kgsgebllp-5ajlord4ikausbzle.png"> -->
![](https://habrastorage.org/webt/kg/sg/eb/kgsgebllp-5ajlord4ikausbzle.png "Mask R-CNN architechture,Source: V7Lab")



## 1. Problem Description and Dataset

We will cover the nail instance segmentation. We want to have a bounding box and segment each nail in the picture. It's from the real application. We want to make a nail disease classification application. To do that, the first step is cropping nails in the given image. Then each cropping nail image will be fed in to the classification model.

For the semantic nail segmentation, we can segment the nail in iamges and then use post-processing to obtain the bounding box and segmentation of nails. That method does not work well in the case that the nails have overlapping. We then aproach the instance segmentation problem to tackle the difficulty. 

|                                                     Images                                                     |                                                     Masks                                                      |
| :------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------: |
| <img align="center" width="300"  src="https://habrastorage.org/webt/em/og/9v/emog9v4ya7ssllg5dht77_wehqk.png"> | <img align="center" width="300"  src="https://habrastorage.org/webt/hl/bf/ov/hlbfovx1uhrbbebgxndyho9yywo.png"> |


> Mission:  **We want to have a bounding box and segmentation of each nail in the picture.**


Our data is organized as

```
├── Images
│   ├── 1
│       ├── first_image.png
│       ├── second_image.png
│       ├── third_image.png
│   ├── 2
│   ├── 3
│   ├── 4
├── Masks
│   ├── 1
│       ├── first_image.png
│       ├── second_image.png
│       ├── third_image.png
│   ├── 2
│   ├── 3
│   ├── 4

```


We have two folders: `Images` and `Masks`. `Images` is the data folder, and `Masks` is the label folder, which is the segmentations of input images. Each folder has four sub-folder:  `1`, `2`, `3`, and `4`, corresponding to four types of nail distribution.

We download data from [link](https://drive.google.com/file/d/1qBLwdQeu9nvTw70E46XNXMciB0aKsM7r/view?usp=sharing) and put it in `data_root`, for example

```python
data_root = "./nail-segmentation-dataset"
```

## 2. Data Preparation

We now have only the semantic segmentation dataset. This part we will make the instance segmentation datset and save that data in the form `coco`.

### 2.1 Make data frame

For convenient, we will save all of dataset information in the csv files: 

images,masks,width,height

| images                 | masks                 | width | height |
| ---------------------- | --------------------- | ----- | ------ |
| images/1/filename1.png | masks/1/filename1.png | 256   | 256    |
| images/1/filename1.png | masks/1/filename1.png | 256   | 256    |
| images/2/filename1.png | masks/2/filename1.png | 256   | 256    |
| images/2/filename1.png | masks/2/filename1.png | 256   | 256    |

The function [`make_csv_file`](https://github.com/hphuongdhsp/Segmentation-Tutorial/blob/master/Part%207-Instance%20Segmentation%20with%20MMDetection/data_preprocessing.py) helps us do the above task. 

To do that we use two functions `png2numpy`, `make_csv_file_npy` in [`data_processing.py`](https://github.com/hphuongdhsp/Segmentation-Tutorial/blob/master/Part%205-Pytorch%20with%20Dali/data_processing.py) file. 

### 2.2 Get coco annotation

We want to convert our semantic segmentation data into the instance segmentaion. One of the famous format to organize the instance segmentation data is `COCO`.

The coco annotation has the following format

```python
{
    "images": [images],
    "annotations": [annotations],
    "categories": [categories]
}
```

Where: 

- **"images"** (type: [List[Dict]]) is the list of dictionaries, each dictionary has informations
    - "id": 100   The id of image
    - "file_name": "train/images/1/image_100.png",  the path to get image
    - "width": 1800,   
    - "height": 1626


- **"annotations"**  is the list of dictionaries, each dictionary has informations

    - "id": 350, id of object (not the image id)
    - "image_id": 100, id of image
    - "category_id": 1, id of categories
    - "segmentation": RLE or [polygon],
    - "area": float,
    - "bbox": [x,y,width,height],
    - "iscrowd": 0 or 1,

- **"categories"**  is the list of dictionaries, each dictionary has informations
    - "id": int = 0 id of categories
    - "name": str = "nail"

Using the *get_annotations* function, we can convert the semantic segmentation data into the coco format data of the instance segmentation. 

```python
def get_annotations(dataframe: pd.DataFrame):
    """get_annotations is to convert a dataframe into the coco format

    Args:
        train_df (pd.DataFrame): the dataframe that stored the infomation
        of the dataset. the form of the dataframe is
        images | width | height |

    Returns:
        [type]: the coco format data of the dataset
    """

    cats = [{"id": 0, "name": "nail"}]

    annotations = []
    images = []
    obj_count = 0

    for idx, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        filename = row.images

        images.append(
            {
                "id": idx,
                "file_name": filename,
                "width": row.width,
                "height": row.height,
            }
        )

        binary_mask = read_mask(os.path.join(str(data_root), row.masks))

        contours = find_contours(binary_mask)

        for contour in contours:
            xmin = int(np.min(contour[:, :, 0]))
            xmax = int(np.max(contour[:, :, 0]))
            ymin = int(np.min(contour[:, :, 1]))
            ymax = int(np.max(contour[:, :, 1]))

            poly = contour.flatten().tolist()
            poly = [x + 0.5 for x in poly]

            data_anno = {
                "image_id": idx,
                "id": obj_count,
                "category_id": 0,
                "bbox": [xmin, ymin, (xmax - xmin), (ymax - ymin)],
                "area": (xmax - xmin) * (ymax - ymin),
                "segmentation": [poly],
                "iscrowd": 0,
            }
            if (xmax - xmin) * (ymax - ymin) < 20:
                continue

            else:
                annotations.append(data_anno)

                obj_count += 1

    return {"categories": cats, "images": images, "annotations": annotations}

```

Where: 

- *find_contours* is a function to get contour of a binary mask. 
- dataframe argument of the above function is the data frame obtained from the *make_csv_file* that has the infomations of data. 

We then save the annotaions as a json file by the `get_json_coco` function

```
def get_json_coco(args) -> None:
    train_df = pd.read_csv(f"{data_root}/csv_file/train_info.csv")
    valid_df = pd.read_csv(f"{data_root}/csv_file/valid_info.csv")

    coco_json = os.path.join(data_root, "annotations")
    mkdir(coco_json)
    train_json = get_annotations(train_df)
    valid_json = get_annotations(valid_df)

    with open(f"{coco_json}/train.json", "w+", encoding="utf-8") as f:
        json.dump(train_json, f, ensure_ascii=True, indent=4)
    with open(f"{coco_json}/valid.json", "w+", encoding="utf-8") as f:
        json.dump(valid_json, f, ensure_ascii=True, indent=4)

```


**For more details, we can find the source code at [github](https://github.com/hphuongdhsp/Segmentation-Tutorial/tree/master/Part%205-Pytorch%20with%20Dali)**

## 3. Training instance segmentation problems by MMDetection

### 3.1 MMDetection

**MMDetection** is an object detection toolbox that contains a rich set of object detection and instance segmentation methods as well as related components and modules. It is built on top of PyTorch.

One decomposes the detection framework into different components and one can easily construct a customized object detection framework by combining different modules. In this part, we discover how to decompose  the instance segmentation framework and modify them in order to train a instance segmentation model. 

To train a instance segmentation or object detection model, we pass to three steps: 

- Prepare the customized dataset
- Prepare a config
- Train, test, inference models on the customized dataset.

In the second part we have customized our dataset into the coco format. With the coco format, we can easy reuse configurations. 

### 3.2 Modify the config. 

> **Config is all we need**

To run a instance segmentation or object detection, all we need to do is define a good config. In the config file, there are all of infomation for a training model. 

Examples of configurations are given in [config](https://github.com/open-mmlab/mmdetection/tree/master/configs). There are a lot of configs that help to build a customized configs. For the convenience, we will download them and put them to the repository of [MMDetection tutorial](https://github.com/hphuongdhsp/Segmentation-Tutorial/tree/master/Part%207-Instance%20Segmentation%20with%20MMDetection). 

A Config can be decompose into four parts.

- model: define the model architechture, loss function
- dataset: define the data pipeline
- schedules: define the optimization and the schedules learning rate
- default_runtime: define the logging, check point. 



In the `configs/__base__` there are examples for each module 

```bash
├── configs
│   ├── __base__
│       ├── datasets
│       ├── models
│       ├── schedules
│       ├── default_runtime.py
```

Also, inside of the `configs`, we have alot of subconfigs that coresponding to the model acrchitecture. 

For example:

```
configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py

```

Here 
- mask_rcnn: type of mask_rcnn
- r50: backbone of the model (Resnet50)
- caffe: the pretrained model is caffe model.
- fpn: the feature pyramid network.
- mstrain: the multi-scale image for the data pipeline
- poly: schedule poly
- 1x: 12 max_epochs
- coco: the dataset is coco format.

In this post, we focus on two modules: dataset and model and set the *schedules* and *default_runtime* as default. 

#### Modify the model config

With the nail segmentation, the output is a binary mask (only nail object), then redefine the model as: 

```python
# The new config inherits a base config to highlight the necessary modification
_base_ = "mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py"

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(roi_head=dict(bbox_head=dict(num_classes=1), mask_head=dict(num_classes=1)))

```
Here: 

- We inherit the config `mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py`
- Only need to define the num_classes in the bbox_head and mask_head.

#### Modify the data pipeline config

For the data pipeline: 

```python
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type="CocoDataset",
        img_prefix=data_root,
        classes=cfg.classes,
        ann_file=f"{data_root}/annotations/train.json",
        pipeline=cfg.train_pipeline,
    ),
    val=dict(
        type="CocoDataset",
        img_prefix=data_root,
        classes=cfg.classes,
        ann_file=f"{data_root}/annotations/valid.json",
        pipeline=cfg.test_pipeline,
    ),
    test=dict(
        type="CocoDataset",
        img_prefix=data_root,
        classes=cfg.classes,
        ann_file=f"{data_root}/annotations/valid.json",
        pipeline=cfg.test_pipeline,
    ),
)

```

Here: 
- type:"CocoDataset" as default because we use the coco format.
- img_prefix: - the path to the image directory.
- ann_file: the path to the json annotation file.
- classes: the classes of the dataset. Here class: = ["nail"]
- pipeline: data pipeline processing that is defined as


```python
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(
        type="Resize",
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736), (1333, 768), (1333, 800)],
        multiscale_mode="value",
        keep_ratio=True,
    ),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels", "gt_masks"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
```
> Note: we want use the multi-scale image when training the pipeline, then

```
img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736), (1333, 768), (1333, 800)]

```
### 3.2 Training

Once we have the config file (see [nail_conig.py], we start to train model. 

For that we will: 
- import the config file
- define the model module
- defime the data pipeline
- train the model with an api

#### Import Config by using `mmcv` library:

```
cfg = mmcv.Config("configs/nail_config.py")
```

#### Build the model pipeline from the Config by using build_detector api

```python 
from mmdet.apis import build_detector

model = build_detector(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
model.init_weights()

```

#### Build the data pipeline from the Config by using build_dataset api
Using the apis: `build_detector`, `build_dataset` of mmdetection library, we can easily build the model and dataset. 

```
from mmdet.apis import build_dataset
datasets = [build_dataset(cfg.data.train)]
```

#### Train the model with the train_detector api


```python
from mmdet.apis import train_detector
train_detector(model, datasets)

```


After 40 epochs, we can see the model is training well.

<!-- <img align="center" width="400"  src="https://habrastorage.org/webt/q1/5v/hw/q15vhwbx5bslvd97sfhpca1iffu.jpeg"> -->

![](https://habrastorage.org/webt/q1/5v/hw/q15vhwbx5bslvd97sfhpca1iffu.jpeg "a result apter training 40 epochs")


**For more details, we can find the source code at [github](https://github.com/hphuongdhsp/Segmentation-Tutorial/tree/master/Part%207-Instance%20Segmentation%20with%20MMDetection)**





### References

- [Part 7-Instance Segmentation with MMDetection](https://github.com/hphuongdhsp/Segmentation-Tutorial/tree/master/Part%207-Instance%20Segmentation%20with%20MMDetection)
- [MMDetection Tutorial - TRAIN WITH CUSTOMIZED DATASETS](https://mmdetection.readthedocs.io/en/stable/2_new_data_model.html#train-with-customized-datasets)
