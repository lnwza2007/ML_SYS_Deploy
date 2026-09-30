import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import joblib

def run_evaluation(split="val", metric="top_k", k=5):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        current_dir,
        os.path.join(current_dir, "mlsystem"),
        "/Users/frankza/Downloads/plantclef-2026"
    ]
    
    def resolve_path(filename):
        for c in candidates:
            p = os.path.join(c, filename)
            if os.path.exists(p):
                return p
        return filename

    feat_name = f"X_{split}_dinov2.npy"
    meta_name = f"i_{split}.parquet"
    train_feat_name = "X_train_dinov2.npy"
    train_meta_name = "i_train.parquet"
    
    feat_path = resolve_path(feat_name)
    meta_path = resolve_path(meta_name)
    train_feat_path = resolve_path(train_feat_name)
    train_meta_path = resolve_path(train_meta_name)
    
    if not os.path.exists(feat_path) or not os.path.exists(meta_path):
        print(f"Error: Missing split files for split '{split}' ({feat_path}, {meta_path})")
        sys.exit(1)
        
    X_eval = np.load(feat_path)
    i_eval = pd.read_parquet(meta_path)
    y_eval = i_eval["provider"].values
    
    X_train = np.load(train_feat_path)
    i_train = pd.read_parquet(train_meta_path)
    y_train = i_train["provider"].values
    
    X_train_norm = normalize(X_train.astype(np.float64), norm="l2")
    X_eval_norm = normalize(X_eval.astype(np.float64), norm="l2")
    
    t0 = time.time()
    knn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    knn.fit(X_train_norm)
    distances, indices = knn.kneighbors(X_eval_norm)
    total_time = time.time() - t0
    latency_ms = (total_time / len(X_eval)) * 1000
    
    top1_correct = 0
    top_k_correct = 0
    n = len(X_eval)
    
    for idx in range(n):
        pred_labels = y_train[indices[idx]]
        true_label = y_eval[idx]
        if pred_labels[0] == true_label:
            top1_correct += 1
        if true_label in pred_labels:
            top_k_correct += 1
            
    top1_acc = (top1_correct / n) * 100.0
    top_k_acc = (top_k_correct / n) * 100.0
    
    print("==================================================")
    print("DETERMINISTIC EVALUATION HARNESS - PLANTCLEF")
    print("==================================================")
    print(f"Evaluation Split : {split} ({n} samples)")
    print(f"Embedding Dim    : {X_eval.shape[1]}")
    print(f"Metric Type      : {metric.upper()}")
    print("--------------------------------------------------")
    print(f"Top-1 Accuracy   : {top1_acc:.2f}% ({top1_correct}/{n})")
    print(f"Top-{k} Accuracy   : {top_k_acc:.2f}% ({top_k_correct}/{n})")
    print(f"Average Latency  : {latency_ms:.3f} ms per query")
    print(f"Total Eval Time  : {total_time:.3f} seconds")
    print("==================================================")
    
    return {
        "split": split,
        "n_samples": n,
        "top1_acc": top1_acc,
        "top_k_acc": top_k_acc,
        "latency_ms": latency_ms
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deterministic Evaluation Harness for PlantCLEF ML Pipeline")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--metric", type=str, default="top_k", choices=["top_k", "accuracy"])
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    
    run_evaluation(split=args.split, metric=args.metric, k=args.k)
