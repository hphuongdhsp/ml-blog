---
toc: true
layout: post
comments: true
author: Nguyen Hoang-Phuong
image: images/tensorflow.png
description: The first part of the Segmentation Tutorial Series, a step-by-step guide to developing deep learning segmentation models in TensorFlow
categories: [tensorflow, semanticsegmentation, deeplearning]
title: Segmentation Model-Part I - Training deep learning segmentation models in Tensorflow
---

In this post, we will cover how to train a segmentation model by using the TensorFlow platform


## 1. Problem Description and Dataset

We want to cover a nail semantic segmentation problem. For each image, we want to detect the segmentation of the nail in the image.


|                                                     Images                                                     |                                                     Masks                                                      |
| :------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------: |
| <img align="center" width="300"  src="https://habrastorage.org/webt/em/og/9v/emog9v4ya7ssllg5dht77_wehqk.png"> | <img align="center" width="300"  src="https://habrastorage.org/webt/hl/bf/ov/hlbfovx1uhrbbebgxndyho9yywo.png"> |



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


We have two folders: `Images` and `Masks`,  each folder has four sub-folders `1`, `2`, `3`, `4` corresponds to four types of distribution of nails. `Images` is the data folder and `Masks` is the label folder, which is the segmentations of input images.

We download data from [link](https://drive.google.com/file/d/1qBLwdQeu9nvTw70E46XNXMciB0aKsM7r/view?usp=sharing) and put it in `data_root`, for example

```python
data_root = "./nail-segmentation-dataset"
```

## 2. Data Preparation

For the convenience of loading data, we will store data information in the data frame (or CSV file). 

We want to have the CSV file that stores the image and mask paths

| index | images                |
| ----- | --------------------- |
| 1     | path_first_image.png  |
| 2     | path_second_image.png |
| 3     | path_third_image.png  |
| 4     | path_fourth_image.png |

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

Here `get_all_items`, `mkdir` are two supported functions (defined in `utils.py` file) that help us to find all items in a given folder and make a new folder. 

Once we have the data frame, we can go to define the dataset. 

## 3. Define DataLoader

In this part we will do the following: 

- Get lists of images and masks
- Define Dataloader with input being a list of images and masks and output be list of image batchs which being fed into the model. More precisely: 
  - Decode images and masks (read images and masks)
  - Transform data
  - Batch the augmented data. 

Before going to the next part, let's talk about the advantages of using tf.data for the data loader pipeline.

The main feature of the next part is the data loader. We use the `tensorflow.data` (tf.data) to load the dataset instead of using Sequence Keras (keras.Sequence). In fact, we can also combine `tf.data` and `keras.Sequence`. This tutorial focuses on how to load data by tf.data.

Here is the pipeline loader of tf.data: 
- Read data from a CSV file
- Transfrom (augumentate) the data
- Load data into the model

![](https://habrastorage.org/webt/0g/ec/1p/0gec1pep-rta5ntt7umwq2ybafy.png "tf.data pipeline")
<!-- <img align="center" width="600"  src="https://habrastorage.org/webt/0g/ec/1p/0gec1pep-rta5ntt7umwq2ybafy.png"> -->


The advantage of this method is: 
- Loading data by using multi-processing
- Don't have the memory leak phenomenal 
- Flexible to load dataset, can load weight sample data (using `tf.compat.v1.data.experimental.sample_from_datasets` )
- Downtime and waiting around are minimized while processing is maximized through parallel execution; see the following images:

### Naive pipeline

<!-- <img align="center" width="600"  src="https://habrastorage.org/webt/se/pj/bx/sepjbxl-cofjvbatz7x8xp6wn3i.png"> -->
![](https://habrastorage.org/webt/se/pj/bx/sepjbxl-cofjvbatz7x8xp6wn3i.png "Naive pipeline")

This is the typical workflow of a naive data pipeline, there is always some idle time and overhead due to the inefficiency of sequential execution.

In contrast, consider: `tf.data` pipeline


<!-- <img align="center" width="600"  src="https://habrastorage.org/webt/yt/5z/bv/yt5zbvzi3pqdc4l_8ez1c6lzgug.png"> -->
![](https://habrastorage.org/webt/yt/5z/bv/yt5zbvzi3pqdc4l_8ez1c6lzgug.png "tf.data pipeline")
### 3.1 Get images and masks from a dataframe.

```python
def load_data_path(data_root: Union[str, Path], csv_dir: Union[str, Path], train: str) -> Tuple:

    csv_file = pd.read_csv(csv_dir)
    ids = sorted(csv_file["images"])
    images = [data_root + f"/{train}/images" + fname for fname in ids]
    masks = [data_root + f"/{train}/masks" + fname for fname in ids]
    return (images, masks)
```

### 3.2 Decode images and masks

```python
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

### 3.3 Doing augmentation

```python
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

Here we use [Albumentations](https://albumentations.ai/) library to define the transform. **Albumentations** is a Python library for fast and flexible image augmentations. Albumentations efficiently implements a rich variety of image transform operations that are optimized for performance and does so while providing a concise yet powerful image augmentation interface for different computer vision tasks, including object classification, segmentation, and detection. For example, we define our validation transform as

```python
import albumentations as A

def valid_transform():
    return A.Compose(
        [
            A.Resize(384, 384, always_apply=True),
        ],
        p=1,
    )
```

You can find the detail of transforms in `transform.py` file, in the source code given at the post's end. We remark that, after doing augmentation, we cast the output of transform into TensorFlow type `tensorflow type`

```
aug_img = tf.cast(aug_img / 255.0, dtype)
aug_mask = tf.cast(aug_mask / 255.0, dtype)
```

Once we finish the augmentation task, we can do batching of the data by

```python
dataset = dataset.batch(batch_size)
```

Here, the dataset is now an object of `tf.data`.

Compose four previous steps, we have the data loader function:

```python
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
    # image and mask and also make sure that the output of this function be in form of tensorflow (tf)
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

In this part, we will define the segmentation model by using `segmentation_models` library, we also define the loss function, optimization, and metrics.

**Segmentation models** is a python library with Neural Networks for Image Segmentation based on Keras (Tensorflow) framework. This is the high-level API, you need only some lines of code to create a Segmentation Neural Network.

## 4.1 Model 
```python
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

```

Here we use: 
- The [Unet](https://en.wikipedia.org/wiki/U-Net) model with the backbone is [efficientnetb4](https://paperswithcode.com/method/efficientnet#:~:text=EfficientNet%20is%20a%20convolutional%20neural,resolution%20using%20a%20compound%20coefficient.)
- The loss function is the sum [`DiceLoss`](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) and [`FocalLoss`](https://paperswithcode.com/method/focal-loss#:~:text=Focal%20loss%20applies%20a%20modulating,in%20the%20correct%20class%20increases.)
- The metric is IOU score and FSscore
- The optimization algorithm is [`RectifiedAdam`](https://ml-explained.com/blog/radam-explained)


# 5 Model Training

Once we have: dataloader and model we then combine them to run the model. In this part we will introduce some tools that help us boost the efficiency of training:

- mixed_precision
- using wanbd as a callback

## 5.1 Mixed_precision

How does mixed precision work?

Mixed precision training is the use of lower-precision operations (float16 and bfloat16) in a model during training to make it run faster and use less memory. Using mixed precision can improve performance by more than 3 times on modern GPUs and 60% on TPUs.

Here is the mixed precision training flow:


<!-- <img align="center" width="600"  src="https://habrastorage.org/webt/3y/pg/ew/3ypgewgss1uoezkydzapcg0pjgy.png"> -->
![](https://habrastorage.org/webt/3y/pg/ew/3ypgewgss1uoezkydzapcg0pjgy.png "mixed precision training flow")



- We first feed the data as the float16 or bloat16 type, then the input of the model has the low type (float16 and bfloat16). 
- All of the calculations in the model are computed with the lower-precision operations
- Convert the output of the model into float32 to do optimization tasks.
- Update weights, convert them into lower-precision, and continue the next round of training.


To train the model in TensorFlow with mixed precision, we just modify:

- We first define the global policy: 
  
```python
if args.mixed_precision:
    policy = mixed_precision.Policy("mixed_float16")
    mixed_precision.set_policy(policy)
    print("Mixed precision enabled")
```

- Change the out data (input of model) into tf.float16: 
  
When we load dataset, before do suffling and do batching we convert out data into float16. To do that, 
```python
def process_data(image, mask):
    # using tf.numpy_function to apply the aug_img to image and mask
    aug_img, aug_mask = tf.numpy_function(aug_fn, [image, mask], [dtype, dtype])
    return aug_img, aug_mask
```

- Fix the last layer of the model. Here we remark that the dtype of the last layer should be `float32`. To do that, in the model part, we add some trick lines: 
  
```python
model = sm.Unet(
    "efficientnetb4",
    input_shape=(384, 384, 3),
    encoder_weights="imagenet",
    classes=1,
)
# TO USE mixed_precision, HERE WE USE SMALL TRICK, REMOVE THE LAST LAYER AND ADD
# THE ACTIVATION SIGMOID WITH THE DTYPE  TF.FLOAT32
last_layer = tf.keras.layers.Activation(activation="sigmoid", dtype=tf.float32)(
    model.layers[-2].output
)
```

## 5.2 Using Wanbd for logging.

In this part, we will cover how to use wandb for logging. WandB is a central dashboard to keep track of your hyperparameters, system metrics, and predictions so you can compare models live and share your findings. To do that we use callback of model training as the WandbLogging

```
import wandb
from wandb.keras import WandbCallback
logdir = f"{work_dir}/tensorflow/logs/wandb"
mkdir(logdir)
wandb.init(project="Segmentation by Tensorflow", dir=logdir)

wandb.config = {
    "learning_rate": earning_rate,
    "epochs": epochs,
    "batch_size": batch_size,
}
callbacks.append(WandbCallback())

```

We finish the training task by calling the train loader and the valid loader and fitting the model. Then

## 5.3 Dataloader

```python
data_root = str(args.data_root)
train_csv_dir = f"{data_root}/csv_file/train.csv"
valid_csv_dir = f"{data_root}/csv_file/valid.csv"
# set batch_size
batch_size = args.batch_size
epochs = args.epochs

# get training and validation set
train_dataset = load_data_path(data_root, train_csv_dir, "train")
train_loader = tf_dataset(
    dataset=train_dataset,
    shuffle=True,
    batch_size=batch_size,
    transforms=train_transform(),
    dtype=dtype,
    device=args.device,
)
valid_dataset = load_data_path(data_root, valid_csv_dir, "valid")
valid_loader = tf_dataset(
    dataset=valid_dataset,
    shuffle=False,
    batch_size=batch_size,
    transforms=valid_transform(),
    dtype=dtype,
    device=args.device,
)

```

## 5.4 Fit training
```python
history = model.fit(
    train_loader,
    steps_per_epoch=total_steps,
    epochs=epochs,
    validation_data=valid_loader,
    callbacks=callbacks,
)
```


**For more details, we can find the source code at [github](https://github.com/hphuongdhsp/Segmentation-Tutorial/tree/master/Part%201-Tensorflow)**