import random

import numpy as np
import torch

class AverageMeter:
"""Track and compute the average of a metric."""

```
def __init__(self):
    self.reset()

def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n

    if self.count > 0:
        self.avg = self.sum / self.count
```

def set_seed(seed=5):
"""
Set random seeds for reproducibility.
"""

```
random.seed(seed)
np.random.seed(seed)

torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

def seed_worker(worker_id):
"""
Seed DataLoader workers.
"""

```
worker_seed = torch.initial_seed() % (2**32)

np.random.seed(worker_seed)
random.seed(worker_seed)
```
