---
toc: true
layout: post
description: The second part of the Segmentation Tutorial Series, a guide to handle Imbalanced Data in Deep Learning
# categories: [tensorflow, semantic segmentation, deep learning]
title: Segmentation Model-Part II - How to handle Imbalanced Data in Segmentation Problem
---

In the last post, we discussed how to train a segmentation model in Tensorflow. This post will cover how to balance datasets in training a segmentation model in Tensorflow. We can use the same technique to deal with the imbalanced data in a Classification problem. Let us recall our segmentation problem.


## 1. Problem Description and Dataset

We want to cover a nail semantic segmentation problem. For each image, we want to detect the segmentation of the mail in the image.


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

Similar to the training pipeline of the previous post, we want to have the CSV file that stores the image and mask paths

| index | images                |
| ----- | --------------------- |
| 1     | path_first_image.png  |
| 2     | path_second_image.png |
| 3     | path_third_image.png  |
| 4     | path_fourth_image.png |

For that we use `make_csv_file` function in `data_processing.py` file. **What thing do we need more for data balancing?** 

We remark that our image data have four subfolders, and the distributions of the coverage segmentation are very different in each folder. Also, the quality of the image those are different (skew data).

| Folder | number of image |
| ------ | --------------- |
| 0      | 749             |
| 1      | 144             |
| 2      | 126             |
| 3      | 52              |
| 4      | 34              |

We want to split the info data frame into some smaller data frame. To do that we use:

```
def split_data_train(data_root) -> None:
    r"""
    This function is to split the train into some subsets. The purpose of this step is to make the balanced dataset.
    """
    data_root = args.data_root
    path_csv = f"{data_root}/csv_file/train.csv"
    train = pd.read_csv(path_csv)
    train["type"] = train["images"].apply(lambda x: x.split("/")[1])
    for i in train["type"].unique().tolist():
        df = train.loc[train["type"] == i]
        df.to_csv(f"{data_root}/csv_file/train{i}.csv", index=False)
```

We have five new data frame `train_0`, `train_1`, `train_2`, `train_3`, `train_4`. We will use those files in the next step. 


**We will inherit all of the things in the previous post (DataLoader, Model, mixed precision, logger)**. We need to change how we load datasets and how to balance the data when we load data.

## 3. How to define dataloader

We remark that all functions we will use have been defined in the last part.

**For more details, we can find the source code at [github](https://github.com/hphuongdhsp/Segmentation-Tutorial/tree/master/Part%201-Tensorflow)**

We first define all data frame and load directories of image and masks

```
train0_csv_dir = f"{data_root}/csv_file/train0.csv"
train1_csv_dir = f"{data_root}/csv_file/train1.csv"
train2_csv_dir = f"{data_root}/csv_file/train2.csv"
train3_csv_dir = f"{data_root}/csv_file/train3.csv"
train4_csv_dir = f"{data_root}/csv_file/train4.csv"

train0_dataset = load_data_path(data_root, train0_csv_dir, "train")
train1_dataset = load_data_path(data_root, train1_csv_dir, "train")
train2_dataset = load_data_path(data_root, train2_csv_dir, "train")
train3_dataset = load_data_path(data_root, train3_csv_dir, "train")
train4_dataset = load_data_path(data_root, train4_csv_dir, "train")

```
Using tf_dataset we load five datasets and remark that we will not batch in this step, we will concatenate those datasets with weights and batch them when we have the whole dataset.

`The cool thing about this method is that we can use different augmentation for different sub-dataset`. For example we can apply the train_transform for the first dataset and valid_transform for the second datset. 

```
train0_loader = tf_dataset(
    dataset=train0_dataset,
    shuffle=False,
    batch_size=None,
    transforms=train_transform(),
    dtype=dtype,
    device=args.device,
)
train1_loader = tf_dataset(
    dataset=train1_dataset,
    shuffle=False,
    batch_size=None,
    transforms=train_transform(),
    dtype=dtype,
    device=args.device,
)
train2_loader = tf_dataset(
    dataset=train2_dataset,
    shuffle=False,
    batch_size=None,
    transforms=train_transform(),
    dtype=dtype,
    device=args.device,
)
train3_loader = tf_dataset(
    dataset=train3_dataset,
    shuffle=False,
    batch_size=None,
    transforms=train_transform(),
    dtype=dtype,
    device=args.device,
)
train4_loader = tf_dataset(
    dataset=train4_dataset,
    shuffle=False,
    batch_size=None,
    transforms=train_transform(),
    dtype=dtype,
    device=args.device,
)
```

Shuffle and repeat each dataset
```
data_loaders = [
    train0_loader.apply(tf.data.experimental.shuffle_and_repeat(100000, count=epochs)),
    train1_loader.apply(tf.data.experimental.shuffle_and_repeat(100000, count=epochs)),
    train2_loader.apply(tf.data.experimental.shuffle_and_repeat(100000, count=epochs)),
    train3_loader.apply(tf.data.experimental.shuffle_and_repeat(100000, count=epochs)),
    train4_loader.apply(tf.data.experimental.shuffle_and_repeat(100000, count=epochs)),
]

```

Calculate the weighted sample; here we want each batch; each dataset will be loaded with the same sample.

```
weights = [1 / len(data_loaders)] * len(data_loaders)
```

Using `tf.data.experimental.sample_from_datasets` to balance data. 

The input `tf.data.experimental.sample_from_datasets` function is: 
- datasets: A non-empty list of tf.data.Dataset objects with compatible structure.
- weights: (Optional.) A list or Tensor of len(datasets) floating-point values where weights[i] represents the probability to sample from datasets[i], or a tf.data.Dataset object where each element is such a list. Defaults to a uniform distribution across datasets.

Returns of `tf.data.experimental.sample_from_datasets`
- A dataset that interleaves elements from datasets at random, according to weights if provided, otherwise with uniform probability.


```
train_loader = tf.data.experimental.sample_from_datasets(data_loaders, weights=weights, seed=None)
```
We then have the train_loader with balancing data. We only need to batch them before feeding data into the model.

```
train_loader = train_loader.batch(batch_size)
```

Once we have train_loader, we define valid_loader, model, as same as the previous post. Finally, we fit the model.

```
    history = model.fit(
        train_loader,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=valid_loader,
        callbacks=callbacks,
    )
```

where 

```
    steps_per_epoch = (
        int(
            (
                len(train0_dataset[0])
                + len(train1_dataset[0])
                + len(train2_dataset[0])
                + len(train3_dataset[0])
                + len(train4_dataset[0])
            )
            / batch_size
        )
        + 1
    )
```

**For more details, we can find the source code at [github](https://github.com/hphuongdhsp/Segmentation-Tutorial/tree/master/Part%202-Tensorflow%20Balanced%20Data)**
