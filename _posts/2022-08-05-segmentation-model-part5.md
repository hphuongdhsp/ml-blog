---
toc: true
layout: post
description: The fifth part of the Segmentation Tutorial Series, a step-by-step guide to developing data augmentation on GPU with Dali library
# categories: [tensorflow, semantic segmentation, deep learning]
title: Segmentation Model-Part V - Data augmentation on the GPU with DALI
---

In the last post, we have discovered how to augmente data on GPUs with Kornia. This post, we will 
we discover how to use [DALI](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/index.html) to accelerate deep learning. 

We will work with the Segmentation Problem (Nail Segmentation). For that, we use Pytorch Lightninig to train model and use `DALI` to load data and do data processing. The first and second we will recall `Problem Description and Dataset`. If you have followed previous posts, you can skip that part. 

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


We have two folders: `Images` and `Masks`. `Images` is the data folder, and `Masks` is the label folder, which is the segmentations of input images. Each folder has four sub-folder:  `1`, `2`, `3`, and `4`, corresponding to four types of nail distribution.

We download data from [link](https://drive.google.com/file/d/1qBLwdQeu9nvTw70E46XNXMciB0aKsM7r/view?usp=sharing) and put it in `data_root`, for example

```python
data_root = "./nail-segmentation-dataset"
```

## 2. Data Preparation

For the convenience of training with DALI, we will convert `png` data into `npy` format and save all of informations of images and masks in a CSV file

| index | images                | masks                 |
| ----- | --------------------- | --------------------- |
| 1     | path_first_image.npy  | path_first_image.npy  |
| 2     | path_second_image.npy | path_second_image.npy |
| 3     | path_third_image.npy  | path_third_image.npy  |
| 4     | path_fourth_image.npy | path_fourth_image.npy |

To do that we use two functions `png2numpy`, `make_csv_file_npy` in [`data_processing.py`](https://github.com/hphuongdhsp/Segmentation-Tutorial/blob/master/Part%205-Pytorch%20with%20Dali/data_processing.py) file. 

- `png2numpy` function helps us convert the `png` format into the `npy` format and save images and masks in `data_root_npy` folder
- `make_csv_file_npy` makes 2 CVS files train.cvs and valid.csv having the previous form and be saved at 
`f"{data_root_npy}/csv_file"` foler.

## 3. NVIDIA Data Loading Library (DALI)

`DALI` is open source library for decoding and augmenting images,videos and speech to accelerate deep learning applications. DALI reduces latency and training time, mitigating bottlenecks, by overlapping training and pre-processing. It provides a drop-in replacement for built in data loaders and data iterators in popular deep learning frameworks for easy integration or retargeting to different frameworks. 

Let us discuss the difference among: a Naive Deeplearning Pipeline, Kornia Deep Learning Pipeline and DALI Deeplearning Pipeline.  

### Naive Deep Learning Pipeline

- Naive Deeplearning Pipeline: The pre-processing of the data occurs on the CPU, the model will be typically trained on GPU/TPU.

### Kornia Deep Learning Pipeline

- Kornia Deep Learning Pipeline: The reading, resezing or padding data occurs on CPU, the transform (augmentation) and model training runed on GPU/TPU. The transform is consider as an `nn.Module`. Then `transform` is a  `nn.Module` object that `forward` input x of size `BxCxHxW` and obtain the output of size `BxCxHxW`.

### DALI Deep Learning Pipeline

- DALI Deeplearning Pipeline: In the reading image, we have two components: encoding and decoding. With DALI library, we can read do encoding by CPUs and decoding by GPUs that work on batch. All other tasks will work on GPUs. We remark that the transform in DALI Pipeline works on data of several types: `BxCxHxW`, `BxHxWXC`. That is why DALI can easily be retargeted to TensorFlow, PyTorch, and MXNet.


The DALI Training Pipeline

<img align="center" width="600"  src="https://habrastorage.org/webt/do/qg/tu/doqgtugeu1kqtdtojhospmro0j0.jpeg">


DALI Library in the whole Pipieline. 


<img align="center" width="600"  src="https://habrastorage.org/webt/g1/31/ga/g131gag8f-3co1irt5qq8rl5oui.png">


## 4. Training the Segmentation problem with DALI and Pytorch Lighiting.

In this part, we will details how to do processing the data in `DALI` and train the model by Pytorch Lighiting.

### 4.1 Define Data Pipeline by DALI

We will define the processing data pipeline by using `dali`, instead of using `torch.utils.data`.

We first define new class: `GenericPipeline` that wraps the `nvidia.dali.pipeline.Pipeline` class

```
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline

class GenericPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, **kwargs):
        super().__init__(batch_size, num_threads, device_id)
        self.kwargs = kwargs
        self.dim = kwargs["dim"]
        self.device = device_id
        # self.patch_size = [384,384]
        self.load_to_gpu = kwargs["load_to_gpu"]
        self.input_x = self.get_reader(kwargs["imgs"])
        self.input_y = self.get_reader(kwargs["lbls"])
        self.cast = ops.Cast(device="gpu", dtype=types.DALIDataType.FLOAT)

    def get_reader(self, data):
        return ops.readers.Numpy(
            files=data,
            device="cpu",
            read_ahead=True,
            dont_use_mmap=True,
            pad_last_batch=True,
            shard_id=self.device,
            seed=self.kwargs["seed"],
            num_shards=self.kwargs["gpus"],
            shuffle_after_epoch=self.kwargs["shuffle"],
        )

    def load_data(self):
        img = self.input_x(name="ReaderX")  # read X
        img = img.gpu()
        img = self.cast(img)
        if self.input_y is not None:
            lbl = self.input_y(name="ReaderY")  # read Y
            lbl = lbl.gpu()
            lbl = self.cast(lbl)

            return img, lbl
        return img

```

The initial input of GenericPipeline is:
- batch_size: batchsize
- num_threads: number of workers
- device_id: gpu device
- kwargs: the dictionary that has the infomations of parameters and train/valid data. 

```
kwargs = {
            "dim": 2,
            "seed": 42,
            "gpus": 1,
            "overlap": 0.5,
            "benchmark": False,
            "num_workers": 4,
            "oversampling": 0.4,
            "test_batches": 0,
            "train_batches": 0,
            "load_to_gpu": True,
        }
```

We now can define `ValidPipeline` and `TrainPipeline` based on the `GenericPipeline`

#### ValidPipeline

```
class ValidPipeline(GenericPipeline):
    def __init__(self, batch_size, num_threads, device_id, **kwargs):
        super().__init__(batch_size, num_threads, device_id, **kwargs)
        # self.invert_resampled_y = kwargs["invert_resampled_y"]
        # if self.invert_resampled_y:
        #     self.input_meta = self.get_reader(kwargs["meta"])
        #     self.input_orig_y = self.get_reader(kwargs["orig_lbl"])
        print(len(kwargs["imgs"]))

    def define_graph(self):
        img, lbl = self.load_data()
        img, lbl = self.resize_fn(img, lbl)  # reszie to inpput size (384)
        img, lbl = self.transpose_fn(img, lbl)

        return img, lbl

```
Here, we remark that define_graph is the function to do the data processing. For ValidPipeline, the flow of the data will be: 
- load_data
- resize data
- transpose data (convert image into the size of CxHxW)

#### TrainPipeline

Similar, we have the TrainPipeline
```
class TrainPipeline(GenericPipeline):
    def __init__(self, batch_size, num_threads, device_id, **kwargs):
        super().__init__(batch_size, num_threads, device_id, **kwargs)
        self.oversampling = kwargs["oversampling"]

    """
    define some augumentations, for more augumentation, please read \
        https://github.com/NVIDIA/DALI/tree/main/docs/examples/image_processing
    """

    @staticmethod
    def slice_fn(img):
        return fn.slice(img, 1, 3, axes=[0])

    def resize(self, data, interp_type):
        return fn.resize(data, interp_type=interp_type, size=self.crop_shape_float)

    def noise_fn(self, img):
        img_noised = img + fn.random.normal(img, stddev=fn.random.uniform(range=(0.0, 0.33)))
        return random_augmentation(0.15, img_noised, img)

    def blur_fn(self, img):
        img_blurred = fn.gaussian_blur(img, sigma=fn.random.uniform(range=(0.5, 1.5)))
        return random_augmentation(0.15, img_blurred, img)

    def brightness_fn(self, img):
        brightness_scale = random_augmentation(0.15, fn.random.uniform(range=(0.7, 1.3)), 1.0)
        return img * brightness_scale

    def contrast_fn(self, img):
        scale = random_augmentation(0.13, fn.random.uniform(range=(0.9, 1.1)), 1.0)
        return math.clamp(img * scale, fn.reductions.min(img), fn.reductions.max(img))

    def flips_fn(self, img, lbl):
        kwargs = {
            "horizontal": fn.random.coin_flip(probability=0.5),
            "vertical": fn.random.coin_flip(probability=0.5),
        }
        return fn.flip(img, **kwargs), fn.flip(lbl, **kwargs)

    def define_graph(self):
        img, lbl = self.load_data()
        img, lbl = self.resize_fn(img, lbl)  # reszie to inpput size (384)
        img, lbl = self.flips_fn(img, lbl)
        img = self.noise_fn(img)
        img = self.blur_fn(img)
        img = self.brightness_fn(img)
        img = self.contrast_fn(img)

        img, lbl = self.transpose_fn(img, lbl)
        return img, lbl

```

For the `TrainPipeline`, we need more augmentations. The augmentation is defined in the `define_graph` function. 

For the augmentation in DALI, we need to redefine all of transform functions. For more infomation, read the [dali tutorial](https://github.com/NVIDIA/DALI/tree/main/docs/examples/image_processing).


For the convenience, we will define the function `fetch_dali_loader` that will generate `Pipeline` (TrainPipeline, ValidPipeline) depends on the type of dataset. 

```
def fetch_dali_loader(imgs, lbls, batch_size, mode, **kwargs):
    assert len(imgs) > 0, "Empty list of images!"
    if lbls is not None:
        assert len(imgs) == len(lbls), f"Number of images ({len(imgs)}) not matching number of labels ({len(lbls)})"
    pipeline = PIPELINES[mode]
    shuffle = True if mode == "train" else False
    load_to_gpu = True if mode in ["eval", "test"] else False
    pipe_kwargs = {"imgs": imgs, "lbls": lbls, "load_to_gpu": load_to_gpu, "shuffle": shuffle, **kwargs}

    rank = int(os.getenv("LOCAL_RANK", "0"))
    pipe = pipeline(batch_size, kwargs["num_workers"], rank, **pipe_kwargs)
    return pipe
```

Once we have the TrainPipeline and ValidPipeline, we can use them for the LightningWrapper to make the lightning data module. 

### 4.2 LightningWrapper 


Our LightningDataModule of Nail data is defined by

```
class NailSegmentationDaliDali(LightningDataModule):
    def __init__(self, data_root_npy: str, batch_size: int, csv_folder: str):
        super().__init__()

        self.data_root_npy = str(data_root_npy)
        self.csv_folder = csv_folder
        self.batch_size = batch_size
        self.train_csv = pd.read_csv(os.path.join(self.csv_folder, "train.csv"))
        self.valid_csv = pd.read_csv(os.path.join(self.csv_folder, "valid.csv"))

        self.kwargs = {
            "dim": 2,
            "seed": 42,
            "gpus": 1,
            "overlap": 0.5,
            "benchmark": False,
            "num_workers": 4,
            "oversampling": 0.4,
            "test_batches": 0,
            "train_batches": 0,
            "load_to_gpu": True,
        }

    def prepare_data(self) -> None:
        pass

    def setup(self, stage=None):
        self.train_imgs = [os.path.join(self.data_root_npy, path) for path in self.train_csv["images"]]
        self.train_lbls = [os.path.join(self.data_root_npy, path) for path in self.train_csv["masks"]]

        self.val_imgs = [os.path.join(self.data_root_npy, path) for path in self.valid_csv["images"]]
        self.val_lbls = [os.path.join(self.data_root_npy, path) for path in self.valid_csv["masks"]]

        self.train_dataset = fetch_dali_loader(
            imgs=self.train_imgs, lbls=self.train_lbls, batch_size=self.batch_size, mode="train", **self.kwargs
        )
        self.valid_dataset = fetch_dali_loader(
            imgs=self.val_imgs, lbls=self.val_lbls, batch_size=self.batch_size, mode="eval", **self.kwargs
        )

    def train_dataloader(self):
        output_map = ["image", "label"]
        return LightningWrapper(
            self.train_dataset,
            auto_reset=True,
            reader_name="ReaderX",
            output_map=output_map,
            dynamic_shape=False,
        )

    def val_dataloader(self):
        output_map = ["image", "label"]
        return LightningWrapper(
            self.valid_dataset,
            auto_reset=True,
            reader_name="ReaderX",
            output_map=output_map,
            dynamic_shape=True,
        )

```

- Here in the setup function, we use fetch_dali_loader to get the datapipeline for the train and valid stages
- train_dataloader and val_dataloader is defined thank to the `LightningWrapper` class


```
class LightningWrapper(DALIGenericIterator):
    def __init__(self, pipe, **kwargs):
        super().__init__(pipe, **kwargs)

    def __next__(self):
        out = super().__next__()[0]
        return out
```

- We remark that the input of the model will be dicts of keys ["image", "label"]. It means

```
input_batch = {"image": images, "label": masks}
```


Then we also need to modify the train loop (`training_step`) and the valid loop (`validation_step`) of the LightningModule. For example: 


```
def training_step(self, batch, batch_idx):
    imgs, masks = batch["image"].float(), batch["label"]

    logits = self(imgs)
    train_loss = self.loss_function(logits, masks)
    train_dice_soft = self.dice_soft(logits, masks)

    self.log("train_loss", train_loss, prog_bar=True)
    self.log("train_dice_soft", train_dice_soft, prog_bar=True)
    return {"loss": train_loss, "train_dice_soft": train_dice_soft}

```

Once we finish to define `LightningModule` and `LightningDataModule`, we can jump to the `Trainer` to run the training. 

<img align="center" width="600"  src="https://habrastorage.org/webt/qm/q4/jv/qmq4jvmclavtrtfailqkuvm10-8.png">



**For more details, we can find the source code at [github](https://github.com/hphuongdhsp/Segmentation-Tutorial/tree/master/Part%205-Pytorch%20with%20Dali)**


### References

- [Segmentation Model-Part III - Training deep learning segmentation models in Pytorch Lightning](https://hphuongdhsp.github.io/ml-blog/2022/08/03/segmentation-model-part3.html)
- [NVIDIA DALI Documentation](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/index.html)
- [Augmentation Gallery](https://github.com/NVIDIA/DALI/blob/main/docs/examples/image_processing/augmentation_gallery.ipynb)
- [nnU-Net For PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Segmentation/nnUNet)