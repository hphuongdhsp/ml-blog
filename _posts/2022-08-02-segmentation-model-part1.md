---
toc: true
layout: post
description: The first part of the Segmentation Tutorial Series, Training deep learning models using Tensorflow platform
categories: [markdown]
title: Segmentation Model-Part I - Training deep learning models using Tensorflow platform
---

In this part we will cover how to train a segmentation model by using the tensorflow platform


## Outline

- <a href='#1'>1.Problem Description and Dataset</a>
- <a href='#2'>2. Data Preparation </a> 
- <a href='#3'>3. Define Dataloader </a> 
    - <a id='#3-1'>3.1. Decode images </a> 
    - <a id='#3-2'>3.2. Process Data  </a> 
    - <a id='#3-3'>3.3. Batching Data  </a> 


## 1. Problem Description and Dataset

We will cover the nail semantic segmentation. For each image we want to detect the segmentation of the mail in the image.


|                                 Images                                 |                                 Masks                                 |
| :--------------------------------------------------------------------: | :-------------------------------------------------------------------: |
| <img align="center" width="300"  src="https://habrastorage.org/webt/em/og/9v/emog9v4ya7ssllg5dht77_wehqk.png"> | <img align="center" width="300"  src="https://habrastorage.org/webt/hl/bf/ov/hlbfovx1uhrbbebgxndyho9yywo.png"> |



Our original data is organizated as

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


We have 2 folders: `Images` and `Masks`,  each folder has four sub-folders `1`, `2`, `3`, `4` corresponds to four types of distribution of nail. Images is the input folder and Masks is the label folder, that is the segmentations that we want to detect. 

We download data from [link](https://drive.google.com/file/d/1qBLwdQeu9nvTw70E46XNXMciB0aKsM7r/view?usp=sharing) and put it in `data_root`, for example

```python
data_root = "./nail-segmentation-dataset"
```

## 2. Data Preparation

For convenience of loading data, we will store information of data in the dataframe (or csv file). 

We want to have the a csv file that store the images and masks path 

| index | images | 
| ----- | ---------- |
| 1     | path_first_image.png         |
| 2     | path_second_image.png        | 
| 3     | path_third_image.png         |
| 4     | path_fourth_image.png        |

To do that we use
```
import os
from typing import Any
import pandas as pd

from utils import get_all_items, mkdir

def make_csv_file(data_root) -> None:

    list_images_train_masks = get_all_items(os.path.join(data_root, "train", "masks"))

    list_images_train_images = get_all_items(os.path.join(data_root, "train", "images"))

    list_images_train = [
        i for i in list_images_train_images if i in list_images_train_masks
    ]

    print(len(list_images_train))
    list_images_valid = get_all_items(os.path.join(data_root, "valid", "masks"))

    train_frame = pd.DataFrame(list_images_train, columns=["images"])

    train_frame["train"] = 1
    valid_frame = pd.DataFrame(list_images_valid, columns=["images"])

    valid_frame["train"] = 0
    mkdir(f"{data_root}/csv_file")
    train_frame.to_csv(f"{data_root}/csv_file/train.csv", index=False)
    valid_frame.to_csv(f"{data_root}/csv_file/valid.csv", index=False)
```

Here get_all_items, mkdir are two supported functions, that help us to find all items in a given folder and make new folder. 

Once we have csv, we can pass to define the dataset. 

## 3. Define DataLoader

In this part we will do the following: 

- Get lists of images and masks
- Define Dataloader with input is lists of images and masks and outout is list of batch image which is feed into the model. More precisely, in this step we will: 
  - Decode images and masks
  - Doing Augumentation
  - Batching the augumented data. 

### 3.0 Get lists of images and masks

```
def load_data_path(data_root: Union[str, Path], csv_dir: Union[str, Path], train: str) -> Tuple:

    csv_file = pd.read_csv(csv_dir)
    ids = sorted(csv_file["images"])
    images = [data_root + f"/{train}/images" + fname for fname in ids]
    masks = [data_root + f"/{train}/masks" + fname for fname in ids]
    return (images, masks)
```

### 3.1 Decode images and masks

```
def load_image_and_mask_from_path(image_path: tf.string, mask_path: tf.string) -> Any:
    """this function is to load image and mask

    Args:
        image_path (tf.string): the tensorflow string of image
        mask_path (tf.string): the tensorflow string of mask

    Returns:
        [type]: image and mask
    """
    # read image by tensorflow function
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    # read mask by tensorflow function
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_image(mask, channels=1)
    return img, mask
```

### 3.2 Doing augumentaion 

```
    def aug_fn(image, mask):
        # do augumentation by albumentations library
        data = {"image": image, "mask": mask}
        aug_data = transforms(**data)
        aug_img = aug_data["image"]
        aug_mask = aug_data["mask"]
        # do normalize by using the tensorflow.cast function
        aug_img = tf.cast(aug_img / 255.0, dtype)
        aug_mask = tf.cast(aug_mask / 255.0, dtype)
        return aug_img, aug_mask
```

Here we use `albumentation` library to define the transform, for example

```
import albumentations as A

def valid_transform():
    return A.Compose(
        [
            A.Resize(384, 384, always_apply=True),
        ],
        p=1,
    )
```

We remark that, after doing augumentation, we cast the output of transfrom into `tensorflow type`

```
aug_img = tf.cast(aug_img / 255.0, dtype)
aug_mask = tf.cast(aug_mask / 255.0, dtype)
```

Once we finish the augumentation task, we can do batch of the data by 
```
dataset = dataset.batch(batch_size)
```

Compose 4 privious step we have: 

```
def tf_dataset(
    dataset: Tuple[List[str], List[str]],
    shuffle: bool,
    batch_size: Any,
    transforms: A.Compose,
    dtype: Any,
    device: List[int],
):
    r"""This function is to create dataloader for tensorflow training

    Args:
        dataset Tuple[List[str], List[str]]: Tuple of List data path that have same size
        shuffle (bool): True if you want shuffle dataset when do training
        batch_size [Any]: None if you dont want spit dataset by batch
        transforms (A.Compose): the augumentation that you want to apple for the data

    Returns:
        datast : the prepare dataset for the training step
    """

    # do augumentation by albumentations, remark that in the the end, we use tf.cast to normalize
    # image and mask and also make sure that the out put of this function be in form of tensorflow (tf)
    def aug_fn(image, mask):
        # do augumentation by albumentations library
        data = {"image": image, "mask": mask}
        aug_data = transforms(**data)
        aug_img = aug_data["image"]
        aug_mask = aug_data["mask"]
        # do normalize by using the tensorflow.cast function
        aug_img = tf.cast(aug_img / 255.0, dtype)
        aug_mask = tf.cast(aug_mask / 255.0, dtype)
        return aug_img, aug_mask

    def process_data(image, mask):
        # using tf.numpy_function to apply the aug_img to image and mask
        aug_img, aug_mask = tf.numpy_function(aug_fn, [image, mask], [dtype, dtype])
        return aug_img, aug_mask

    # convert the tuple of list (images, masks) into the tensorflow.data form
    dataset = tf.data.Dataset.from_tensor_slices(dataset)

    # apply the map reading image and mask (make sure that the input and output are in the tensorflow form (tf.))
    dataset = dataset.map(load_image_and_mask_from_path, num_parallel_calls=multiprocessing.cpu_count() // len(device))
    # shuffle data
    if shuffle:
        dataset = dataset.shuffle(buffer_size=100000)
    # do the process_data map (augumentation and normalization)
    dataset = dataset.map(
        partial(process_data), num_parallel_calls=multiprocessing.cpu_count() // len(device)
    ).prefetch(AUTOTUNE)
    # make batchsize, here we use batch_size as a parameter, in some case we dont split dataset by batchsize
    # for example, if we want to mix multi-dataset, then we skip this step and split dataset by batch_size later
    if batch_size:
        dataset = dataset.batch(batch_size)
    return dataset
```

# 4. Define the Segmentation model

This part we will define the segmentation model by using `segmentation_models` library, we also define the loss function, optimization and metric

## 4.1 Model 

    def create_model():

        model = sm.Unet(
            "efficientnetb4",
            input_shape=(384, 384, 3),
            encoder_weights="imagenet",
            classes=1,
        )
        # TO USE mixed_precision, HERE WE USE SMALL TRICK, REMOVE THE LAST LAYER AND ADD
        # THE ACTIVATION SIGMOID WITH THE DTYPE  TF.FLOAT32
        last_layer = tf.keras.layers.Activation(activation="sigmoid", dtype=tf.float32)(model.layers[-2].output)
        model = tf.keras.Model(model.input, last_layer)

        # define optimization, here we use the tensorflow addon, but use can also use some normal \
        # optimazation that is defined in tensorflow.optimizers
        optimizer = tfa.optimizers.RectifiedAdam()

        if args.mixed_precision:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer, dynamic=True)
        # define a loss fucntion
        dice_loss = sm.losses.DiceLoss()
        focal_loss = sm.losses.BinaryFocalLoss()
        total_loss = dice_loss + focal_loss
        # define metric
        metrics = [
            sm.metrics.IOUScore(threshold=0.5),
            sm.metrics.FScore(threshold=0.5),
        ]
        # compile model with optimizer, losses and metrics
        model.compile(optimizer, total_loss, metrics)
        return model