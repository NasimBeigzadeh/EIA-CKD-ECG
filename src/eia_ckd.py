import torch
import torch.nn as nn

class TemporalAttention(nn.Module):
"""
Attention-based consensus module.

```
Input:
    x: [num_views, batch_size, num_classes]

Output:
    consensus logits: [batch_size, num_classes]
"""

def __init__(self, dim):
    super().__init__()

    self.query = nn.Linear(dim, dim)
    self.key = nn.Linear(dim, dim)
    self.value = nn.Linear(dim, dim)

    self.softmax = nn.Softmax(dim=-1)

def forward(self, x):
    # [num_views, batch, num_classes]
    x = x.permute(1, 0, 2)

    q = self.query(x)
    k = self.key(x)
    v = self.value(x)

    attention_scores = torch.bmm(
        q,
        k.transpose(1, 2)
    ) / (x.size(-1) ** 0.5)

    attention_weights = self.softmax(
        attention_scores
    )

    attended = torch.bmm(
        attention_weights,
        v
    )

    return attended.mean(dim=1)
```

class HierarchicalConsistencyModel(nn.Module):
"""
EIA-CKD model wrapper.

```
The model uses:
    1. A shared lightweight CNN backbone.
    2. Internally generated views.
    3. A frozen consensus path.
    4. Attention-based consensus.
"""

def __init__(
    self,
    base_model,
    num_classes=5,
):
    super().__init__()

    self.model = base_model

    self.temporal_attn = TemporalAttention(
        num_classes
    )

def create_views(self, x):
    """
    Generate internal views from an ECG image.

    Views:
        - Original
        - Horizontal shift +30
        - Horizontal shift -30
        - Horizontal flip
    """

    views = [x]

    for shift in [30, -30]:
        views.append(
            torch.roll(
                x,
                shifts=shift,
                dims=3,
            )
        )

    views.append(
        x.flip(dims=[3])
    )

    return views

def forward(self, x):
    # During evaluation only the base model is used.
    if not self.training:
        return self.model(x)

    # Student prediction from the original input.
    student_logits = self.model(x)

    # Generate internal views.
    views = self.create_views(x)

    # Consensus path is intentionally frozen.
    with torch.no_grad():

        view_logits = [
            self.model(view)
            for view in views
        ]

        view_stack = torch.stack(
            view_logits
        )

        consensus_logits = self.temporal_attn(
            view_stack
        )

    return student_logits, consensus_logits
```
