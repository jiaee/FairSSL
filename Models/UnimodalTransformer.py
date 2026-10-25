import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm
import wandb
from ..DataLoader.Dvlog import get_dataset, get_test_dataset
from ..Utils.metric_calculate import calculate_metrics

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Transformer Components
class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout, ctx_dim=None):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        head_dim = hidden_size // num_heads
        self.num_heads = num_heads
        self.scale = head_dim ** -0.5

        if ctx_dim is None:
            ctx_dim = hidden_size
        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(ctx_dim, hidden_size)
        self.v = nn.Linear(ctx_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None):
        if context is None:
            context = x
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = self.k(context).view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = self.v(context).view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores + mask
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return out

class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, nhead=1, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.self_attn = Attention(hidden_size, nhead, dropout)
        self.linear1 = nn.Linear(hidden_size, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, src, mask=None):
        attn_out = self.self_attn(src, mask=mask)
        src = self.norm1(src + self.dropout(attn_out))
        ff = self.dropout(self.linear2(self.dropout(torch.relu(self.linear1(src)))))
        src = self.norm2(src + ff)
        return src

class TransformerEncoder(nn.Module):
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(num_layers)])

    def forward(self, src, mask=None):
        for layer in self.layers:
            src = layer(src, mask)
        return src

class AudioTransformerClassifier(nn.Module):
    def __init__(
        self,
        d_model: int = 25,
        num_layers: int = 1,
        nhead: int = 1,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 596,
    ):
        super().__init__()
        self.d_model = d_model
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        encoder_layer = TransformerLayer(d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        # Extra layer: classification head
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, seq_len, d_model)
        returns: logits of shape (batch_size,)
        """
        B, T, _ = x.shape
        pos_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        x = x + self.pos_embed(pos_ids)
        x = self.norm(x)
        hidden = self.encoder(x)      
        cls_hidden = hidden[:, 0, :]  
        logits = self.classifier(cls_hidden).squeeze(1)
        return logits

# Training & Evaluation Loop
def train():
    wandb.init()
    config = wandb.config
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DataLoaders
    train_loader = get_dataset(config.csv_path, config.root_dir, batch_size=config.batch_size, train=True)
    val_loader   = get_dataset(config.csv_path, config.root_dir, batch_size=config.batch_size, train=False)
    test_loader  = get_test_dataset(config.csv_path, config.root_dir, batch_size=config.batch_size)

    # Model, Loss, Optimizer
    model = AudioTransformerClassifier(
        d_model=config.acoustic_dim,
        num_layers=config.num_layers,
        nhead=config.nhead,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        max_seq_len=config.max_seq_len,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.wd)

    # Supervised Training
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        for acoustic, gender, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}"):
            acoustic = acoustic.to(device)
            labels = labels.to(device).float()
            logits = model(acoustic)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        wandb.log({"train_loss": avg_loss, "epoch": epoch + 1})

    # Validation Accuracy
    model.eval()
    preds, truths, genders = [], [], []
    with torch.no_grad():
        for acoustic, gender, labels, _ in val_loader:
            acoustic = acoustic.to(device)
            logits = model(acoustic)
            batch_pred = (logits > 0).long().cpu().numpy()
            preds.extend(batch_pred)
            truths.extend(labels.numpy())
            genders.extend(gender.tolist())
    val_acc = np.mean(np.array(preds) == np.array(truths))
    wandb.log({"val_accuracy": val_acc})
    print(f"\nValidation Accuracy: {val_acc:.4f}\n")

    # Final Test Evaluation & Metrics
    preds, truths, genders = [], [], []
    with torch.no_grad():
        for acoustic, gender, labels, _ in test_loader:
            acoustic = acoustic.to(device)
            logits = model(acoustic)
            batch_pred = (logits > 0).long().cpu().numpy()
            preds.extend(batch_pred)
            truths.extend(labels.numpy())
            genders.extend(gender.tolist())

    SP, EOpp, EOdd, EAcc, Acc, Prec, Sens, Spec, F1, uar = calculate_metrics(
        y_pred=preds, y_true=truths, sf_data=genders
    )

    print("=== FINAL TEST METRICS ===")
    print(f"Accuracy:           {Acc:.4f}")
    print(f"Precision:          {Prec:.4f}")
    print(f"Sensitivity:        {Sens:.4f}")
    print(f"Specificity:        {Spec:.4f}")
    print(f"F1 Score:           {F1:.4f}")
    print(f"Unweighted Avg (UAR): {uar:.4f}")
    print(f"Statistical Parity: {SP:.4f}")
    print(f"Equal Opportunity:  {EOpp:.4f}")
    print(f"Equalized Odds:     {EOdd:.4f}")
    print(f"Equal Accuracy:     {EAcc:.4f}")

    # Log final metrics
    wandb.log({
        "test_accuracy": Acc,
        "test_precision": Prec,
        "test_sensitivity": Sens,
        "test_specificity": Spec,
        "test_f1": F1,
        "test_uar": uar,
        "statistical_parity": SP,
        "equal_opportunity": EOpp,
        "equalized_odds": EOdd,
        "equal_accuracy": EAcc,
    })

# WandB Sweep Configuration
sweep_config = {
    "method": "grid",
    "metric": {"name": "val_accuracy", "goal": "maximize"},
    "parameters": {
        "lr": {"values": [1e-4, 5e-4, 1e-3]},
        "batch_size": {"values": [16, 32, 64]},
        "wd": {"values": [1e-5, 1e-4]},
        "epochs": {"value": 10},
        "seed": {"value": 42},
        "num_layers": {"value": 1},
        "nhead": {"value": 1},
        "dim_feedforward": {"value": 128},
        "dropout": {"value": 0.1},
        "acoustic_dim": {"value": 25},
        "max_seq_len": {"value": 596},
        "csv_path": {"value": "/home/abtin/dvlog-dataset/dvlog-dataset/"},
        "root_dir": {"value": "/home/abtin/dvlog-dataset/dvlog-dataset/"},
    },
}

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="dvlog_audio_classifier")
    wandb.agent(sweep_id, function=train)
