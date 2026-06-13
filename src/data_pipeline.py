
import sqlite3, json, os, re, math, sqlparse
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

DATA_DIR = Path("/content/querysense/data")
OUT_DIR  = DATA_DIR / "annotated"
OUT_DIR.mkdir(exist_ok=True)

TABLE_SIZES = {
    "default":10000,"order":500000,"transaction":1000000,
    "log":2000000,"product":50000,"customer":200000,
    "employee":5000,"department":500,"city":5000,
    "country":250,"student":20000,"flight":800000,
}

def tsize(name):
    for k,v in TABLE_SIZES.items():
        if k in name.lower(): return v
    return TABLE_SIZES["default"]

def static_cost(sql):
    u = sql.upper()
    c = 100.0 * max(len(re.findall(r'\b(?:FROM|JOIN)\s+\w+', u)), 1)
    c *= (1 + 0.5*(u.count("SELECT")-1))
    if "GROUP BY" in u: c *= 1.3
    if "ORDER BY" in u: c *= 1.2
    if "WHERE"    in u: c *= 0.8
    return max(c, 1.0)

def cost_label(cost):
    if cost < 500:     return "low"
    elif cost < 50000: return "medium"
    else:              return "high"

def normalize(costs):
    mn, mx = min(costs), max(costs)
    if mx == mn: return [0.0]*len(costs)
    return [(c-mn)/(mx-mn) for c in costs]

def build_input(q, schema=""):
    return f"translate to SQL: {q} | schema: {schema}" if schema else f"translate to SQL: {q}"

def run():
    print("Loading Spider dataset...")
    ds = load_dataset("spider", trust_remote_code=True)
    for name, data in [("train", ds["train"]), ("dev", ds["validation"])]:
        records, costs = [], []
        for ex in tqdm(data, desc=name):
            c = static_cost(ex["query"])
            costs.append(c)
            records.append({"question": ex["question"], "sql": ex["query"],
                            "db_id": ex["db_id"], "cost_raw": c})
        df = pd.DataFrame(records)
        df["cost_normalized"] = normalize(costs)
        df["cost_label"]      = df["cost_raw"].apply(cost_label)
        df["schema_hint"]     = ""
        df["model_input"]     = df.apply(lambda r: build_input(r["question"]), axis=1)
        out = OUT_DIR / f"{name}_annotated.csv"
        df.to_csv(out, index=False)
        print(f"  Saved {out}  ({len(df)} rows)")
        print(df["cost_label"].value_counts().to_string())

if __name__ == "__main__":
    run()
