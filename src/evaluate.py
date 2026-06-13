
import re, json, torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration

DATA_DIR  = Path("/content/querysense/data/annotated")
MODEL_DIR = Path("/content/querysense/models/best_model")
OUT_DIR   = Path("/content/querysense/outputs")
OUT_DIR.mkdir(exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def static_cost(sql):
    u = sql.upper()
    c = 100.0 * max(len(re.findall(r'\b(?:FROM|JOIN)\s+\w+', u)), 1)
    c *= (1 + 0.5*(u.count("SELECT")-1))
    if "GROUP BY" in u: c *= 1.3
    if "ORDER BY" in u: c *= 1.2
    if "WHERE"    in u: c *= 0.8
    return max(c, 1.0)

def norm_sql(s): return re.sub(r'\s+', ' ', s.lower().strip().rstrip(';'))

def complexity(sql):
    u = sql.upper()
    j = len(re.findall(r'\bJOIN\b', u))
    s = u.count("SELECT") - 1
    if j==0 and s==0: return "simple"
    elif j<=1 and s<=1: return "medium"
    else: return "complex"

def evaluate(n=300):
    df  = pd.read_csv(DATA_DIR/"dev_annotated.csv").head(n)
    src = str(MODEL_DIR) if MODEL_DIR.exists() else "Salesforce/codet5-small"
    print(f"Loading from: {src}")
    tok   = AutoTokenizer.from_pretrained(src)
    model = T5ForConditionalGeneration.from_pretrained(src).to(device).eval()

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="evaluating"):
        enc = tok(row["model_input"], return_tensors="pt", max_length=256, truncation=True).to(device)
        with torch.no_grad():
            g_ids = model.generate(**enc, max_length=128)
            g_sql = tok.decode(g_ids[0], skip_special_tokens=True)
            b_out = model.generate(**enc, num_beams=6, num_return_sequences=6,
                                   max_length=128, early_stopping=True,
                                   output_scores=True, return_dict_in_generate=True)
            cands = [tok.decode(b_out.sequences[i], skip_special_tokens=True)
                     for i in range(b_out.sequences.shape[0])]

        g_cost = static_cost(g_sql)
        valid  = [(c, static_cost(c)) for c in cands if re.search(r'\bSELECT\b', c.upper())]
        r_sql, r_cost = min(valid, key=lambda x: x[1]) if valid else (g_sql, g_cost)

        rows.append(dict(
            greedy_sql=g_sql, reranked_sql=r_sql, gold_sql=row["sql"],
            greedy_cost=g_cost, reranked_cost=r_cost,
            em_greedy=(norm_sql(g_sql)==norm_sql(row["sql"])),
            em_reranked=(norm_sql(r_sql)==norm_sql(row["sql"])),
            cost_saved_pct=(g_cost-r_cost)/max(g_cost,1)*100,
            complexity=complexity(row["sql"]),
        ))

    res = pd.DataFrame(rows)
    metrics = dict(
        n_eval=len(res),
        EM_greedy=round(res.em_greedy.mean(), 4),
        EM_reranked=round(res.em_reranked.mean(), 4),
        EM_lift=round(res.em_reranked.mean()-res.em_greedy.mean(), 4),
        avg_cost_reduction_pct=round(res.cost_saved_pct.mean(), 2),
        pct_queries_cheaper=round((res.cost_saved_pct>0).mean()*100, 2),
    )
    print("\n── Results ──────────────────────────")
    for k,v in metrics.items(): print(f"  {k:<30} {v}")
    with open(OUT_DIR/"metrics.json","w") as f: json.dump(metrics, f, indent=2)

    # Plot 1
    fig, axes = plt.subplots(1, 2, figsize=(13,5))
    fig.suptitle("QuerySense: Cost-Aware Reranking vs Greedy Decode", fontsize=13, fontweight="bold")
    ax = axes[0]
    ax.hist(np.log1p(res.greedy_cost),   bins=40, alpha=0.6, label="Greedy",   color="#e74c3c")
    ax.hist(np.log1p(res.reranked_cost), bins=40, alpha=0.6, label="Reranked", color="#2ecc71")
    ax.set_xlabel("log(1 + estimated cost)"); ax.set_ylabel("Queries")
    ax.set_title("Query Cost Distribution"); ax.legend()
    ax = axes[1]
    ax.hist(res.cost_saved_pct, bins=40, color="#3498db", edgecolor="white")
    ax.axvline(res.cost_saved_pct.mean(), color="red", linestyle="--",
               label=f"Mean = {res.cost_saved_pct.mean():.1f}%")
    ax.set_xlabel("Cost reduction (%)"); ax.set_ylabel("Queries")
    ax.set_title("Cost Reduction per Query"); ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR/"cost_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved cost_comparison.png")

    # Plot 2
    fig, ax = plt.subplots(figsize=(8,5))
    grp = res.groupby("complexity")[["greedy_cost","reranked_cost"]].mean().reindex(["simple","medium","complex"])
    x = list(range(len(grp))); w = 0.35
    ax.bar([i-w/2 for i in x], np.log1p(grp.greedy_cost),   w, label="Greedy",   color="#e74c3c", alpha=0.85)
    ax.bar([i+w/2 for i in x], np.log1p(grp.reranked_cost), w, label="Reranked", color="#2ecc71", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(grp.index, fontsize=11)
    ax.set_ylabel("log(1 + mean estimated cost)")
    ax.set_title("Cost Reduction by Query Complexity", fontsize=12, fontweight="bold"); ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR/"cost_by_complexity.png", dpi=150, bbox_inches="tight")
    print("Saved cost_by_complexity.png")
    return metrics

if __name__ == "__main__":
    evaluate()
