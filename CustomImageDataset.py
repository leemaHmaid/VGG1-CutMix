import os
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
 



# Constants for paths
IMAGE_PATH_VALID = '/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val/'
ANNOTATION_PATH = '/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Annotations/CLS-LOC/val/'
IMAGE_PATH_TRAIN = '/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train/'
MAPPING_PATH = '/kaggle/input/imagenet-object-localization-challenge/LOC_synset_mapping.txt'
SAVE_PATH = '/kaggle/working/'


def map_validation_images_to_classes(image_path_valid):
    """Map each validation image to its class and return the result as a DataFrame."""
    image_names = os.listdir(image_path_valid)
    image_labels = []

    for img_name in image_names:
        tree = ET.parse(os.path.join(ANNOTATION_PATH, img_name[:-5] + '.xml'))
        root = tree.getroot()
        image_labels.append(root[5][0].text)
    
    validation_df = pd.DataFrame({
        "Image_Name": image_names,
        "Class": image_labels
    })
    validation_df.to_csv(os.path.join(SAVE_PATH, 'validation_list.csv'), index=False)
    
    return validation_df

def get_class_from_image_name(df, img_name):
    """Return the class of a given image name."""
    selected_row = df.loc[df['Image_Name'] == img_name, "Class"]
    return selected_row.values[0]

def create_class_mapping(mapping_path):
    """Create a dictionary mapping class IDs to class names and indices."""
    class_mapping = {}
    
    with open(mapping_path, 'r') as file:
        for idx, line in enumerate(file):
            class_mapping[line[:9].strip()] = (line[9:].strip(), idx)
    
    return class_mapping

# Create class mapping dictionary
class_mapping_dict = create_class_mapping(MAPPING_PATH)

# Define image transformations
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class CustomImageDataset(Dataset):
    """Custom dataset for loading images and their corresponding labels."""
    def __init__(self, class_mapping_dict, root_dir, transform=None, df=None, limit=1000):
        """
        Initialize the dataset.
        
        Parameters:
        - class_mapping_dict: Dictionary mapping class IDs to class names and indices.
        - root_dir: Directory with all the images.
        - transform: Transformations to be applied to the images.
        - df: DataFrame containing image names and their corresponding classes (for validation).
        - limit: Limit on the number of classes to load (for training).
        """
        self.root_dir = root_dir
        self.transform = transform
        self.class_mapping_dict = class_mapping_dict
        self.df = df
        self.limit = limit
        
        self.images = []
        self.labels = []
        
        if self.df is not None:  # For validation data
            for img_name in tqdm(os.listdir(root_dir)):
                self.images.append(os.path.join(root_dir, img_name))
                label_name = get_class_from_image_name(df, img_name)
                self.labels.append(class_mapping_dict[label_name][1])
        else:  # For training data
            for train_class in tqdm(os.listdir(root_dir)[:self.limit]):
                class_path = os.path.join(root_dir, train_class)
                for img_name in os.listdir(class_path):
                    self.images.append(os.path.join(class_path, img_name))
                    self.labels.append(class_mapping_dict[train_class][1])
    
    def __len__(self):
        """Return the total number of samples."""
        return len(self.images)
    
    def __getitem__(self, idx):
        """Generate one sample of data."""
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
 
# Create dataset instances
 
train_dataset = CustomImageDataset(class_mapping_dict, IMAGE_PATH_TRAIN, transform = transforms)
val_dataset = CustomImageDataset(class_mapping_dict, IMAGE_PATH_VALID, map_validation_images_to_classes(IMAGE_PATH_VALID),transform= transforms)

# Create DataLoader instances
 
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)