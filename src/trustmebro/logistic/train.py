
import joblib
import json
import random
import time
from pathlib import Path

import numpy as np
import argparse
from scipy.sparse import hstack, vstack
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_score, accuracy_score
from pathlib import Path

from trustmebro.preprocessing.corpus import TrainingCfg, load_training_matrix
from trustmebro.preprocessing.partition import read_split


def log(verbosity=0):
    def inner1(level=1):
        def inner2(msg):
            if level <= verbosity:
                print(f"[{int(round(time.time()))} s] log fitter: {msg}")

        return inner2

    return inner1


def main(argv: list[str] | None = None) -> int:

    parser = argparse.ArgumentParser(description="Train a logistic regression model")
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--iters", type=int)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.instance)
    _log = log(args.verbose)

    split = read_split(root / "split.json")
    cfg = TrainingCfg(dtype="float32")
    n_iter = args.iters or 5

    _log()("reading labels")
    with open(args.labels, 'r') as f:
        _labels = [el for el in json.loads(f.read())["kinds"]]
        labels = list(range(len(_labels)))

    _log()("fitting models")
    models = {
        "no-regularization":      SGDClassifier(loss = "log_loss", penalty = None),
        "default-regularization": SGDClassifier(loss = "log_loss")
    }

    accuracies = {}
    precisions = {}

    cached = []
    
    for n in range(n_iter):
        
        # Iterations required because:
        #
        # > Internally, this method uses max_iter = 1. Therefore, it
        # > is not guaranteed that a minimum of the cost function is
        # > reached after calling it once. Matters such as objective
        # > convergence, early stopping, and learning rate adjustments
        # > should be handled by the user.
        #
        # (SGDClassifier docs / partial_fit)
        
        _log()(f"  iteration {n}")
        bi = 0

        # SGDClassifier neither handles shuffling when using
        # partial_fit, so shuffle the cached data if available

        if n == 0:
            
            for batch in read_training_batches(root / "features.zst",
                                               split=split, subset="train", cfg=cfg):
                bi += 1
                _log()(f"    ...batch {bi}")

                X, y = batch.matrix, batch.label_ids
                
                for (desc, model) in models.items():
                    model.partial_fit(X, y, classes=labels)

                cached.append((X, y))
        else:
            batch_ids = list(range(len(cached)))
            random.shuffle(batch_ids)
            for i in batch_ids:
                bi += 1
                _log()(f"    ...batch {bi}")
                X, y = cached[i]
            for (desc, model) in models.items():
                model.partial_fit(X, y, classes=labels)
                
                
                

    _log()("saving models")
    for (desc, model) in models.items():
        with open(root / f"logit-{desc}.pkl", "wb") as f:
            joblib.dump(model, f, protocol=5)

    _log()("loading validation matrix")
    validation = load_training_matrix(
        root / "features.zst", split=split, subset="validation", cfg=cfg, max_bytes=4 * 2**30
    )
    X_validation = validation.matrix
    y_validation = validation.label_ids

    _log()("calculating evaluation metrics")

    for (desc, model) in models.items():
        _log()(f"{desc}: predicting ys based on validation X")
        y_fitted = model.predict(X_validation)
        _log()(f"{desc}: calculating accuracy")
        accuracies[desc] = accuracy_score(y_validation, y_fitted)
        _log()(f"{desc}: calculating precision")
        precisions[desc] = precision_score(y_validation, y_fitted)

    __log = _log(0)

    __log("precisions:")
    for desc, prec in precisions:
        __log(f"  {desc}: {prec}")
    __log("accuracies")
    for desc, acc in accuracies:
        __log(f"  {desc}, {acc}")
