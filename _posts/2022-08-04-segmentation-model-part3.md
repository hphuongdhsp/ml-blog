---
toc: true
layout: post
comments: true
author: Nguyen Hoang-Phuong
image: images/pytorch lightning.png
description: The third part of the Segmentation Tutorial Series, a step-by-step guide to developing deep learning segmentation models in Pytorch Lightning
categories: [pytorchlightning, semanticsegmentation, deeplearning]
title: Segmentation Model-Part III - Training deep learning segmentation models in Pytorch Lightning
---

Continue Segmentation Model the series; in this post, we discuss how to train a segmentation model in Pytorch Lightning. PyTorch Lightning is the deep learning framework for professional AI researchers and machine learning engineers who need maximal flexibility without sacrificing performance at scale. It is built on top of PyTorch. 

We still work with the Segmentation Problem (Nail Segmentation) and discover some valuable tools for Pytorch Lightning. From this part, we will focus on the Pytorch Platform. Then for convenience, we recall some tasks of the previous post: Problem Description and Dataset, Data Preparation. 
## 1. Problem Description and Dataset

We want to cover a nail semantic segmentation problem. For each image, we want to detect the segmentation of the nail in the image.


|                                                     Images                                                     |                                                     Masks                                                      |
| :------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------: |
| <img align="center" width="300"  src="https://habrastorage.org/webt/em/og/9v/emog9v4ya7ssllg5dht77_wehqk.png"> | <img align="center" width="300"  src="https://habrastorage.org/webt/hl/bf/ov/hlbfovx1uhrbbebgxndyho9yywo.png"> |



Our data is organized as

```bash
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

We want the CSV file that stores the image and mask paths. In this project, file names of images and masks are the same, and then we only need to save the `images` path and modify the `data_root` of images and masks when we define a dataset. 

| index | images                |
| ----- | --------------------- |
| 1     | path_first_image.png  |
| 2     | path_second_image.png |
| 3     | path_third_image.png  |
| 4     | path_fourth_image.png |

For that we use `make_csv_file` function in `data_processing.py` file. More precisely, 

```python
def make_csv_file(data_root: Union[str, Path]) -> None:

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

Where `get_all_items`, `mkdir` are two supported functions (defined in `utils.py` file) that help us to find all items in a given folder and make a new folder.

Before going define the dataloader and model, let's recall some main features of `Pytorch Lightning`. For more information, you can find it at [Pytorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/).

## 3. Pytorch Lightnining

PyTorch Lightning is an open-source, lightweight Python wrapper for machine learning researchers that is built on top of PyTorch.

With this framework, you don’t have to remember all the tiny details of the PyTorch framework because Pytorch Lightnining handles it.

Three main features of Pytorch Lightning: 
- LightningDataModule
- LightningModule
- Trainer

### 3.1 LightningDataModule

`LightningDataModule` is a shareable, reusable class that encapsulates all the steps needed to process data:
- Data processing
- Load inside Dataset
- Apply transforms
- Wrap inside a DataLoader
  
<!-- <img align="center" width="600"  src="https://habrastorage.org/webt/cv/i_/1n/cvi_1nwwdq28wh5tkun8z7h2fp4.png"> -->
![](https://habrastorage.org/webt/cv/i_/1n/cvi_1nwwdq28wh5tkun8z7h2fp4.png "LightningDataModule")

### 3.2 LightningModule

A lightning module is composed of some components that fully define the system:

- The model or system of models
- The optimizer(s)
- The train loop
- The validation loop


### 3.3 Trainer

Once we declare LightningDataModule, LightningModule, we can train the model with `Trainer` API. 

<!-- <img align="center" width="600"  src="https://habrastorage.org/webt/qm/q4/jv/qmq4jvmclavtrtfailqkuvm10-8.png"> -->
![](https://habrastorage.org/webt/qm/q4/jv/qmq4jvmclavtrtfailqkuvm10-8.png "Trainer in Pytorch Lightning")

A basic use of trainer: 

```python
modelmodule = LightningModule(*args_model)
datamodule = LightningDataModule(*args_data)
trainer = Trainer(*args_trainer)
trainer.fit(modelmodule, datamodule)
```



##  4. DataLoader

To define the LightningModule of our dataset, we first define the `torch.utils.data.Dataset` for the nail data. 

### 4.1 Define `torch.utils.data.Dataset` for the Nail Data
```
class NailDataset(Dataset):
    def __init__(self, data_root: str, csv_folder: str, train: str, tfms: A.Compose):
        self.data_root = data_root
        self.csv_folder = csv_folder
        self.train = train
        self.tfms = tfms
        if self.train == "train":
            self.ids = pd.read_csv(os.path.join(self.csv_folder, "train.csv"))["images"]
        else:
            self.ids = pd.read_csv(os.path.join(self.csv_folder, "valid.csv"))["images"]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Any:
        fname = self.ids[idx]

        image = read_image(self.data_root + f"/{self.train}/images" + fname)
        mask = read_mask(self.data_root + f"/{self.train}/masks" + fname)

        mask = (mask > 0).astype(np.uint8)
        if self.tfms is not None:
            augmented = self.tfms(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
        return {
            "image": img2tensor(image),
            "label": img2tensor(mask),
        }
```

### 4.2 Define `LightningDataModule` for the Nail Data
We then use LightningDataModule to wrap our NailDataset into the data module of Pytorch Lightning. 

```
class NailSegmentation(LightningDataModule):
    def __init__(self, data_root: str, csv_path: str, test_path: str, batch_size: int = 16, num_workers: int = 4):
        super().__init__()
        assert os.path.isdir(csv_path), f"missing folder: {csv_path}"
        assert os.path.isdir(data_root), f"missing folder: {data_root}"
        self.data_root = str(data_root)
        self.csv_path = str(csv_path)
        self.test_path = str(test_path)
        self.valid_transform = valid_transform()
        self.train_transform = train_transform()
        # other configs
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else mproc.cpu_count()

    def prepare_data(self) -> None:
        pass

    def setup(self, *_, **__) -> None:

        self.train_dataset = NailDataset(
            self.data_root,
            self.csv_path,
            train="train",
            tfms=self.train_transform,
        )
        self.valid_dataset = NailDataset(
            self.data_root,
            self.csv_path,
            train="valid",
            tfms=self.valid_transform,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
```

> Here we need to define 3 main functions
- set_up: data loading and preprocessing
- train_dataloader(): define `train_loader` object in Pytorch
- val_dataloader(): define `val_loader` object in Pytorch


##  5. Model Module

In this part we define:
- A segmentation model 
- Wrap the model module by using LightningModule, for that we will define some main functions: 
  - def training_step : calculate {loss, metric}, logging in each train step 
  - def validation_step: calculate {loss, metric}, logging in each valid step 
  - def validation_epoch_end: calculate {loss, metric}, logging in each epoch by using infos of validation_step
  - def configure_optimizers: which optimization and learning rate scheduler do we use for the training?
  

###  5.1 Define the model by using `segmentation_models_pytorch`

For convenience, we use [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch) to define our model. `Segmentation_models_pytorch` is a high-level API, it helps us build a semantic segmentation model with only some lines of code. 

```
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="timm-efficientnet-b4",    # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",             # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                          # model input channels (1 for gray-scale images, 3 for RGB,
    classes=1,                              # model output channels (number of classes in your dataset)
)
```

Here we use: 
- Unet architecture for the segmentation model. Unet has two components: encoder and decoder
- encoder: EfficientNet B4 which is written by [timm](https://github.com/rwightman/pytorch-image-models) library
- input channels: 3 for RGB images, 1 for gray-scale images
- classes: 1 for binary segmentation, 2 for multi-class segmentation.

###  5.2 Define LightningModule
We next use `LightningModule` to wrap the model into the model module of Pytorch Lightnining. 

```
class LitNailSegmentation(LightningModule):
    def __init__(self, model: nn.Module, learning_rate: float = 1e-4):
        super().__init__()
        self.model = model
        self.loss_function = symmetric_lovasz
        self.dice_soft = binary_dice_coefficient
        self.learning_rate = learning_rate
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, masks = batch["image"], batch["label"]

        imgs, masks = imgs.float(), masks.float()
        logits = self(imgs)

        train_loss = self.loss_function(logits, masks)
        train_dice_soft = self.dice_soft(logits, masks)

        self.log("train_loss", train_loss, prog_bar=True)
        self.log("train_dice_soft", train_dice_soft, prog_bar=True)
        return {"loss": train_loss, "train_dice_soft": train_dice_soft}

    def validation_step(self, batch, batch_idx):
        imgs, masks = batch["image"], batch["label"]

        imgs, masks = imgs.float(), masks.float()
        logits = self(imgs)
        valid_loss = self.loss_function(logits, masks)
        valid_dice_soft = self.dice_soft(logits, masks)
        valid_iou = binary_mean_iou(logits, masks)

        self.log("valid_loss", valid_loss, prog_bar=True)
        self.log("valid_dice", valid_dice_soft, prog_bar=True)
        self.log("valid_iou", valid_iou, prog_bar=True)

        return {
            "valid_loss": valid_loss,
            "valid_dice": valid_dice_soft,
            "valid_iou": valid_iou,
        }

    def validation_epoch_end(self, outputs):

        logs = {"epoch": self.trainer.current_epoch}
        valid_losses = torch.stack([x["valid_loss"] for x in outputs]).mean()
        valid_dices = torch.stack([x["valid_dice"] for x in outputs]).mean()
        valid_ious = torch.stack([x["valid_iou"] for x in outputs]).mean()

        logs["valid_losses"] = valid_losses
        logs["valid_dices"] = valid_dices
        logs["valid_ious"] = valid_ious

        return {
            "valid_losses": valid_losses,
            "valid_dices": valid_dices,
            "valid_ious": valid_ious,
            "log": logs,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.trainer.max_epochs, 0)
        self.optimizer = [optimizer]
        return self.optimizer, [scheduler]
```

Here we use: 
 - [AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) as the optimizers
 - symmetric_lovasz as the loss function, which is defined in the [Loss.py](https://github.com/hphuongdhsp/Segmentation-Tutorial/blob/master/Part%203-Pytorch%20Lightning/loss.py) file. *symmetric_lovasz* is defined by 

```
def symmetric_lovasz(outputs, targets):
    return 0.5*(lovasz_hinge(outputs, targets) + lovasz_hinge(-outputs, 1.0 - targets))
```

where lovasz_hinge is [Lovasz loss](https://arxiv.org/pdf/1705.08790.pdf) for the binary segmentation.

- Metrics: Dice, IOU

## 6. Trainer 

Once we have the data module, and model module, we can train the model with `Trainer` API, 

```
datamodule = NailSegmentation(
    data_root=data_root,
    csv_path=csv_path,
    test_path="",
    batch_size=batch_size,
    num_workers=4,
)

model_lighning = LitNailSegmentation(model=model, learning_rate=config.training.learning_rate)


trainer = Trainer(*args_trainer)

trainer.fit(
            model=model_lighning,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
        )
```
Here `args_trainer` is the argument of the `trainer`. More precisely, it has
```
{   gpus: [0]                       # gpu device to train 
    max_epochs: 300                 # number of epochs
    precision: 16                   # using mix precision to train  
    auto_lr_find: True              # auto find the good initial learning rate
    limit_train_batches: 1.0        # percent of train dataset use to train, here 100%
    ... 
    }
```

Lightning implements various techniques to help during training that can help make the training smoother.

**For more details, we can find the source code at [github](https://github.com/hphuongdhsp/Segmentation-Tutorial/tree/master/Part%203-Pytorch%20Lightning)**
