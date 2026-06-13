
import os, json, math, torch, shutil
import torch.nn as nn
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup
from tqdm import tqdm

DATA_DIR  = Path("/content/querysense/data/annotated")
SAVE_DIR  = Path("/content/querysense/models/best_model")
DRIVE_DIR = Path("/content/drive/MyDrive/querysense_model")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
DRIVE_DIR.mkdir(parents=True, exist_ok=True)

CFG = dict(model_name="Salesforce/codet5-small", max_in=256, max_out=128,
           batch=16, epochs=3, lr=3e-4, lam_init=0.1, lam_max=0.5, seed=42)
torch.manual_seed(CFG["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

class SpiderDS(Dataset):
    def __init__(self, path, tok):
        df = pd.read_csv(path).dropna(subset=["model_input","sql"])
        self.inputs  = df["model_input"].tolist()
        self.targets = df["sql"].tolist()
        self.costs   = df["cost_normalized"].fillna(0.5).tolist()
        self.tok = tok
    def __len__(self): return len(self.inputs)
    def __getitem__(self, i):
        src = self.tok(self.inputs[i],  max_length=CFG["max_in"],  padding="max_length", truncation=True, return_tensors="pt")
        tgt = self.tok(self.targets[i], max_length=CFG["max_out"], padding="max_length", truncation=True, return_tensors="pt")
        lbl = tgt.input_ids.squeeze()
        lbl[lbl == self.tok.pad_token_id] = -100
        return dict(input_ids=src.input_ids.squeeze(),
                    attention_mask=src.attention_mask.squeeze(),
                    labels=lbl,
                    cost_target=torch.tensor(self.costs[i], dtype=torch.float))

class QSModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(CFG["model_name"])
        h = self.t5.config.d_model
        self.head = nn.Sequential(nn.Linear(h,128), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(128,1), nn.Sigmoid())
    def forward(self, input_ids, attention_mask, labels, cost_target, lam):
        out  = self.t5(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        enc  = out.encoder_last_hidden_state[:,0,:]
        pred = self.head(enc).squeeze(-1)
        cost_loss = nn.functional.mse_loss(pred, cost_target)
        total = out.loss + lam * cost_loss
        return total, out.loss.item(), cost_loss.item()
    def generate(self, **kw): return self.t5.generate(**kw)

def train():
    tok      = AutoTokenizer.from_pretrained(CFG["model_name"])
    train_ds = SpiderDS(DATA_DIR/"train_annotated.csv", tok)
    dev_ds   = SpiderDS(DATA_DIR/"dev_annotated.csv",   tok)
    train_dl = DataLoader(train_ds, batch_size=CFG["batch"], shuffle=True,  num_workers=2, pin_memory=True)
    dev_dl   = DataLoader(dev_ds,   batch_size=CFG["batch"], shuffle=False, num_workers=2, pin_memory=True)
    print(f"Train: {len(train_ds)} | Dev: {len(dev_ds)}")

    model = QSModel().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=0.01)
    total = len(train_dl) * CFG["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(total*0.1), total)

    best, history = float("inf"), []
    for ep in range(CFG["epochs"]):
        lam = CFG["lam_init"] + (CFG["lam_max"]-CFG["lam_init"]) * (ep/max(CFG["epochs"]-1,1))
        model.train()
        tr_ce, tr_cost = 0.0, 0.0
        for batch in tqdm(train_dl, desc=f"Ep {ep+1} train"):
            batch = {k:v.to(device) for k,v in batch.items()}
            loss, ce, cl = model(**batch, lam=lam)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            tr_ce += ce; tr_cost += cl

        model.eval()
        dv_ce, dv_cost = 0.0, 0.0
        with torch.no_grad():
            for batch in tqdm(dev_dl, desc=f"Ep {ep+1} eval"):
                batch = {k:v.to(device) for k,v in batch.items()}
                loss, ce, cl = model(**batch, lam=lam)
                dv_ce += ce; dv_cost += cl

        n_tr, n_dv = len(train_dl), len(dev_dl)
        print(f"Ep {ep+1} | train ce={tr_ce/n_tr:.4f} cost={tr_cost/n_tr:.4f} | dev ce={dv_ce/n_dv:.4f} cost={dv_cost/n_dv:.4f}")
        history.append({"epoch":ep+1,"train_ce":tr_ce/n_tr,"dev_ce":dv_ce/n_dv})

        avg_dev = dv_ce/n_dv + lam*(dv_cost/n_dv)
        if avg_dev < best:
            best = avg_dev
            model.t5.save_pretrained(SAVE_DIR)
            tok.save_pretrained(SAVE_DIR)
            shutil.copytree(SAVE_DIR, DRIVE_DIR, dirs_exist_ok=True)
            print(f"  ✓ Best model saved (dev={best:.4f}) → Drive backup done")

    with open("/content/querysense/outputs/training_history.json","w") as f:
        json.dump(history, f, indent=2)
    print("\nTraining complete!")

if __name__ == "__main__":
    train()
