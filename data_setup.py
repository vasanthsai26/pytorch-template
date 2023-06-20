from PIL import Image
import constants
import torch
import pandas as pd
import opendatasets as od
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from typing import Optional

from torch.utils.data import Dataset
import pandas as pd
import torchvision.transforms as transforms
import torch

class DefaultDataset(Dataset):
    """
    Custom dataset class for image data.

    Args:
        data_df (pd.DataFrame): DataFrame containing image data and labels.
        transform (Optional[transforms.Compose]): Optional transformations to be applied to the images.
        is_test (bool): Flag indicating whether the dataset is for test data (default: False).
    """
    def __init__(self, data_df: pd.DataFrame, 
                 transform: Optional[transforms.Compose] = None, is_test: bool = False):
        self.data_df = data_df
        self.is_test = is_test
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data_df)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset based on the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            torch.Tensor or tuple: Transformed image tensor or tuple of transformed image tensor and class label.
        """
        # Get image details
        img_details = self.data_df.iloc[idx]

        img_path = img_details["file_path"]
        # image = cv2.imread(img_path)
        # img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.open(img_path)
        img = image.convert("RGB")

        if self.transform:
            img = self.transform(img).to(torch.float)
        else:
            transform = transforms.ToTensor()
            img = transform(img).to(torch.float)

        if self.is_test:
            return img
        else:
            class_label = torch.tensor(img_details["class_idx"], dtype=torch.long)
            return (img, class_label)

def get_transforms(type):
    """
    Returns the desired transform for the specified data type.

    Args:
        type (str): Type of data ('train' or 'test').

    Returns:
        transforms.Compose: Composed transformations for the specified data type.
    """
    # if type == 'train':
    #     weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    #     train_transform = weights.transforms()
    #     return train_transform

    # elif type == 'test':
    #     val_test_transform = transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Resize((224, 224)),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5], 
    #                          std=[0.5, 0.5, 0.5])
    #     ])
    #     return val_test_transform
    if type == 'train':
         train_transform = transforms.Compose([transforms.ToPILImage(),
             transforms.Resize((int(constants.IMG_SIZE * constants.SCALE),
                                int(constants.IMG_SIZE * constants.SCALE))),
             transforms.RandomResizedCrop(constants.IMG_SIZE),
             transforms.RandomGrayscale(p=0.2),
             transforms.ToTensor(),
             transforms.Normalize(mean=constants.IMAGENET_MEAN,
                                  std=constants.IMAGENET_STD)
         ])
         return train_transform

    elif type == 'test':
         val_test_transform = transforms.Compose([transforms.ToPILImage(),
             transforms.Resize(int(constants.IMG_SIZE * constants.SCALE)),
             transforms.CenterCrop(constants.IMG_SIZE),
             transforms.ToTensor(),
             transforms.Normalize(mean=constants.IMAGENET_MEAN,
                                  std=constants.IMAGENET_STD)
         ])
         return val_test_transform

# def download_data(url=constants.URL):
#     od.download(url)

def create_train_df(labels):
    """
    Create a DataFrame containing train data annotations.

    Args:
        labels (DataFrame): DataFrame containing class labels.

    Returns:
        train_df (DataFrame): DataFrame containing train data annotations.
    """
    cars_train_annos = loadmat(Path(constants.META_PATH) / 'cars_train_annos.mat')
    train_data = [[val.flat[0] for val in line] for line in cars_train_annos["annotations"][0]]
    columns = ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "class_idx", "fname"]
    train_df = pd.DataFrame(train_data, columns=columns)
    train_df["class_idx"] = train_df["class_idx"] - 1
    train_df = train_df.merge(labels, left_on='class_idx', right_index=True)
    func = lambda x: os.path.join(constants.CARS_TRAIN_PATH, x)
    train_df["file_path"] = train_df["fname"].apply(func)
    return train_df

def create_test_df():
    """
    Create a DataFrame containing test data annotations.

    Returns:
        test_df (DataFrame): DataFrame containing test data annotations.
    """
    cars_test_annos = loadmat(Path(constants.META_PATH) / 'cars_test_annos.mat')
    test_data = [[val.flat[0] for val in line] for line in cars_test_annos["annotations"][0]]
    columns = ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "fname"]
    test_df = pd.DataFrame(test_data, columns=columns)
    func = lambda x: os.path.join(constants.CARS_TEST_PATH, x)
    test_df["file_path"] = test_df["fname"].apply(func)
    return test_df

def create_labels():
    """
    Create a DataFrame containing class labels for car categories.

    Returns:
        labels (DataFrame): DataFrame containing class labels.
    """
    cars_meta = loadmat(Path(constants.META_PATH) / 'cars_meta.mat')
    labels = [car for car in cars_meta["class_names"][0]]
    labels = pd.DataFrame(labels, columns=["class_labels"])
    labels['class_labels'] = labels['class_labels'].str.replace(' ', '_')
    return labels

def get_data(args):
    """
    Get the train, validation, and test DataFrames.

    Args:
        args (Namespace): Parsed command-line arguments.

    Returns:
        train_df (DataFrame): DataFrame containing the train data.
        val_df (DataFrame): DataFrame containing the validation data.
        test_df (DataFrame): DataFrame containing the test data.
    """
    labels = create_labels()
    train_df = create_train_df(labels)
    test_df = create_test_df()
    train_df, val_df = train_test_split(
        train_df,
        train_size=args.train_split,
        random_state=args.seed,
        stratify=train_df["class_idx"]
    )
    return train_df, val_df, test_df

def create_dataloaders(train_df: pd.DataFrame,
                       val_df: pd.DataFrame,
                       test_df: pd.DataFrame,
                       args
                       ):
    """
    Create dataloaders for training, validation, and test datasets.

    Args:
        train_df (DataFrame): DataFrame containing the training data.
        val_df (DataFrame): DataFrame containing the validation data.
        test_df (DataFrame): DataFrame containing the test data.
        args (Namespace): Parsed command-line arguments.

    Returns:
        train_dataloader (DataLoader): Dataloader for the training dataset.
        val_dataloader (DataLoader): Dataloader for the validation dataset.
        test_dataloader (DataLoader): Dataloader for the test dataset.
    """
    train_transform = get_transforms("train")
    val_test_transform = get_transforms("test")

    # Create the training, validation, and test datasets using the appropriate transforms
    train_dataset = DefaultDataset(data_df=train_df, transform=train_transform, is_test=False)
    val_dataset = DefaultDataset(data_df=val_df, transform=val_test_transform, is_test=False)
    test_dataset = DefaultDataset(data_df=test_df, transform=val_test_transform, is_test=True)

    # Create the training, validation, and test dataloaders using the appropriate datasets and batch size
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Do not shuffle the data
        num_workers=args.num_workers,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,  # Do not shuffle the data
        num_workers=args.num_workers,
        pin_memory=True
    )

    return train_dataloader, val_dataloader, test_dataloader






