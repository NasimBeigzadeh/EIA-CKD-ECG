import torch

from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from .utils import seed_worker

MEAN = (0.9610, 0.8270, 0.8528)
STD = (0.1164, 0.1174, 0.0648)

IMAGE_SIZE = (300, 300)

def get_train_transform():

```
return T.Compose([
    T.Resize(IMAGE_SIZE),

    T.RandomHorizontalFlip(p=0.7),

    T.RandomRotation(
        degrees=(-0.2, 0.2)
    ),

    T.ColorJitter(
        brightness=0.2,
        contrast=0.3,
        saturation=0.2,
        hue=0.2,
    ),

    T.ToTensor(),

    T.Normalize(
        mean=MEAN,
        std=STD,
    ),
])
```

def get_eval_transform():

```
return T.Compose([
    T.Resize(IMAGE_SIZE),

    T.ToTensor(),

    T.Normalize(
        mean=MEAN,
        std=STD,
    ),
])
```

def create_dataloaders(
train_path,
valid_path,
test_path=None,
batch_size=128,
num_workers=0,
seed=5,
):

```
train_dataset = ImageFolder(
    train_path,
    transform=get_train_transform(),
)

valid_dataset = ImageFolder(
    valid_path,
    transform=get_eval_transform(),
)

test_dataset = None

if test_path is not None:
    test_dataset = ImageFolder(
        test_path,
        transform=get_eval_transform(),
    )

generator = torch.Generator()
generator.manual_seed(seed)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    worker_init_fn=seed_worker,
    generator=generator,
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
)

test_loader = None

if test_dataset is not None:
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

return (
    train_loader,
    valid_loader,
    test_loader,
    train_dataset.classes,
)
```
