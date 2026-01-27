import os
import torchvision.datasets
import torchvision.transforms as transforms
from torchtoolbox.transform import Cutout
from torch.utils.data import Dataset
from PIL import Image

def get_mnist(data_path: str) -> tuple[torchvision.datasets.MNIST, torchvision.datasets.MNIST]:
    """
    Load the MNIST dataset.
    Args:
        data_path (str): Path to where the dataset is stored.
    """
    print("loading MNIST")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = torchvision.datasets.MNIST(data_path, train=True, transform=transform_train, download=True)
    val_set = torchvision.datasets.MNIST(data_path, train=False, transform=transform_val, download=True)
    
    return train_set, val_set

def get_cifar10(data_path: str) -> tuple[torchvision.datasets.CIFAR10, torchvision.datasets.CIFAR10]:
    """
    Load the CIFAR10 dataset.
    Args:
        data_path (str): Path to where the dataset is stored.
    """
    print("loading CIFAR10")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            Cutout(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    transform_val = transforms.Compose([
            transforms.ToTensor(),
        ])

    train_set = torchvision.datasets.CIFAR10(data_path, train=True, transform=transform_train, download=True)
    val_set = torchvision.datasets.CIFAR10(data_path, train=False, transform=transform_val, download=True)
    
    return train_set, val_set

def get_cifar100(data_path: str) -> tuple[torchvision.datasets.CIFAR10, torchvision.datasets.CIFAR10]:
    """
    Load the CIFAR100 dataset.
    Args:
        data_path (str): Path to where the dataset is stored.
    """
    print("loading CIFAR100")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            Cutout(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    transform_val = transforms.Compose([
            transforms.ToTensor(),
        ])

    train_set = torchvision.datasets.CIFAR100(data_path, train=True, transform=transform_train, download=True)
    val_set = torchvision.datasets.CIFAR100(data_path, train=False, transform=transform_val, download=True)
    
    return train_set, val_set


class TinyImageNetValDataset(Dataset):
    """
    用于 TinyImageNet 验证集的自定义 Dataset 类。
    它会读取 val_annotations.txt 文件来加载图像和标签。
    """
    def __init__(self, val_dir: str, class_to_idx: dict, transform=None):
        """
        Args:
            val_dir (str): 'val' 目录的路径 (例如: './tiny-imagenet-200/val')
            class_to_idx (dict): 从训练集 ImageFolder 获取的 '类别名' -> '索引' 映射。
            transform (callable, optional): 应用于样本的转换。
        """
        self.val_dir = val_dir
        self.class_to_idx = class_to_idx
        self.transform = transform
        
        self.images_dir = os.path.join(val_dir, 'images')
        self.annotations_file = os.path.join(val_dir, 'val_annotations.txt')
        
        self.image_paths = []
        self.labels = []
        
        try:
            with open(self.annotations_file, 'r') as f:
                for line in f:
                    # 示例: val_0.JPEG   n03445777   0   0   63  63
                    parts = line.strip().split('\t')
                    if len(parts) < 2:
                        continue
                        
                    image_filename = parts[0]
                    class_id = parts[1]
                    # print(f"Image: {image_filename}, Class ID: {class_id}")
                    
                    # 构建图像的完整路径
                    image_path = os.path.join(self.images_dir, image_filename)
                    self.image_paths.append(image_path)
                    
                    # 使用传入的 class_to_idx 将 'n03445777' 转换为整数标签 
                    self.labels.append(self.class_to_idx[class_id])
                    
        except FileNotFoundError:
            print(f"错误: 找不到注释文件 {self.annotations_file}")
            raise
        except KeyError as e:
            print(f"错误: 在 class_to_idx 中找不到类别 {e}。")
            print("请确保 class_to_idx 字典是正确的。")
            raise

        # print(f"成功加载 {len(self.labels)} 个验证集样本。")

    def __len__(self):
        """返回数据集的大小。"""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        获取一个样本。
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 加载图像
        # 使用 .convert('RGB') 来确保图像总是3通道，
        # 即使原始图像是灰度的（虽然在TinyImageNet中不常见）
        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            print(f"错误: 找不到图像文件 {image_path} (索引 {idx})")
            # 返回一个占位符或引发异常
            return None, label 

        # 应用转换
        if self.transform:
            image = self.transform(image)
            
        # print(image, label)
        return image, label


def get_tinyimagenet(data_path: str):
    """
    Load the TinyImageNet dataset.
    Args:
        data_path (str): Path to where the dataset is stored.
    """
    print("loading TinyImageNet200")
    os.makedirs(data_path, exist_ok=True)
    
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        Cutout(),
        transforms.ToTensor(),
    ])

    transform_val = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
    ])

    train_dir = os.path.join(data_path, 'train')
    val_dir = os.path.join(data_path, 'val')

    train_set = torchvision.datasets.ImageFolder(train_dir, transform=transform_train)
    
    class_to_idx = train_set.class_to_idx
    val_set = TinyImageNetValDataset(
        val_dir=val_dir,
        class_to_idx=class_to_idx,
        transform=transform_val
    )

    return train_set, val_set