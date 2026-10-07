# PyTorch MLP draft

`mlp.py` handles sparse matrices, training, evaluation and checkpoints independently of corpus access.
`train.py` connects it to the documented cached-feature loaders and existing split manifests, without splitting data or constructing features.
The hyphenated directory is intentional; run the script directly, or add this directory to the import path for Python use.
Use the existing Pixi environment; no additional packages are required.

```sh
pixi run --locked python src/trustmebro/mlp-pytorch/train.py \
  --features data/learning/instance-01/features.zst \
  --split data/learning/instance-01/split.json \
  --vocab data/learning/instance-01/vocab.zst \
  --checkpoint data/mlp.pt
```

The command loads train/validation matrices once, reuses them across epochs, and saves the best validation model.
Use `--help` for model/training settings; epoch progress goes to stderr and the final JSON summary goes to stdout.
Command defaults come from `FitCfg` and its nested `ModelCfg`; explicit command-line flags override only their corresponding settings.
The default `--max-bytes` is 4 GiB of numeric buffers per partition, not a combined limit or total process-memory cap.
Each selected partition currently needs two archive passes through `load_training_matrix`; there is no per-epoch decompression.
The source archive and split must remain completed and immutable throughout loading.
No test partition is used by this command.

## Ordered experiments

Edit `example.json` to point at one feature instance and choose an output directory and explicit run list.
Run names must be unique ASCII letters/digits, hyphens or underscores.
Settings in each run override `defaults`; omitted settings use `FitCfg`/`ModelCfg` defaults.
Paths are relative to the invoking working directory, like the existing training command.

```sh
pixi run --locked python src/trustmebro/mlp-pytorch/experiments.py \
  --config src/trustmebro/mlp-pytorch/example.json
```

Runs execute sequentially with fresh models and optimizers, shared loaded matrices, and train-fitted scaling reused per policy.
Each run saves its resolved `config.json`, `results.json` with epoch history/per-label metrics, and best `checkpoint.pt`; `summary.csv` compares runs in plan order.
`fit_duration_s` excludes loading, preparation and checkpoint I/O; `duration_s` includes that run's preparation and checkpoint work, but excludes shared input loading.
Results use the checkpoint with minimum unweighted validation cross-entropy, not necessarily peak accuracy or macro-F1.
Use `"class_weight": "balanced"` for inverse-frequency weights or `"inverse_sqrt"` for gentler inverse-square-root weights.
Both weighted modes also save `checkpoint.weighted.pt` plus its metrics in `weighted_checkpoint`.
Set `"label_smoothing": 0.05` to test smoothing; omitted settings keep it disabled.
Both selections and their top-3/top-5 accuracies appear in `results.json` and `summary.csv`.
Re-running the plan skips completed runs only when settings, input hashes, training code, library versions and checkpoint hashes match.
Changed identities require a new output directory; failed/interrupted runs restart from scratch, while completed runs stay untouched.
An error stops the queue and records the failure; subsequent runs do not start.
Use one writer per output directory, keep inputs immutable, and do not include held-out test data in these plans.

## Python API

```python
from hashlib import file_digest
from pathlib import Path

from mlp import FitCfg, evaluate, save_checkpoint
from train import fit_archive, load_archive

root = Path("data/learning/instance-01")
with (root / "vocab.zst").open("rb") as stream:
    vocab_id = file_digest(stream, "sha256").hexdigest()


def report(epoch):
    print(f"Epoch {epoch.epoch}: train CE={epoch.train_cross_entropy:.4f}, val CE={epoch.validation.cross_entropy:.4f}")


result = fit_archive(root / "features.zst", root / "split.json", vocab_id=vocab_id, cfg=FitCfg(), on_epoch=report)
save_checkpoint(Path("model.pt"), result)
# Only after model selection is finished:
test = load_archive(Path("data/test-features.zst"), vocab_id=vocab_id)
test_metrics = evaluate(result.predictor, test)
```

Use `load_training_data` to load once and reuse its `.train` and `.validation` records for multiple `fit` calls.
For repeated fits with the same scaling policy, call `prepare_training` once and pass its immutable result to `fit_prepared`.
Adapters preserve the loaders' sparse matrices, ordered classes and row-aligned labels/theorem identifiers without matrix copies.
`vocab_id` must identify the actual frozen feature representation, not merely its width or filename; the command hashes the vocabulary file bytes.
The loaders verify the archive/manifest binding, but do not expose a vocabulary fingerprint, so supplying the matching vocabulary remains the caller's responsibility.
The outer test feature archive must have been constructed using that same frozen vocabulary and label policy.
Class order and vocabulary identity must match across all partitions and any loaded checkpoint.
Training and validation theorem identifiers must be disjoint; the caller remains responsible for test isolation and training-only vocabulary construction.
If an inner validation partition is created from an existing training corpus, rebuild corpus-derived vocabulary using only the new training partition for the final experiment.

The default network is `input width → 256 → class count`, with ReLU and no dropout.
The first multiplication consumes sparse CSR input without densifying it, but weights, gradients and AdamW state are dense.
At 839,850 inputs, the first layer alone has approximately 215 million weights; float32 weights, gradients and two optimizer moments require approximately 3.2 GiB, excluding other tensors and workspaces.
This is an allocation estimate, not a measured full-model memory requirement or training-time benchmark.
Caller-owned matrices remain in RAM, while sliced and scaled minibatches are transferred to the selected device.
There is no loader prefetching or out-of-core input adapter yet.

Max-absolute scaling is fitted on training rows only, preserves sparse zeros and is applied per minibatch without modifying source matrices.
Float32 narrowing is an explicit derived model representation; source feature values remain unchanged.
Rows are shuffled together with their labels using a private seeded generator, and final partial minibatches are retained.
Optimization uses cross-entropy and AdamW weight decay; class weighting is disabled by default.
Use `--class-weight balanced` or `--class-weight inverse_sqrt` for a single fit, with the corresponding `class_weight` value in experiment plans.
Balanced weights are `N / (K * count)` from training labels only, with `K` the number of observed training classes.
Inverse-square-root weights are proportional to `1 / sqrt(count)` and normalized to mean sample weight one; both modes give absent classes zero weight.
Weighted cross-entropy is normalized by total sample weight, including when aggregating unequal minibatches.
`--label-smoothing` (or `label_smoothing` in a plan) accepts values in `[0, 1]`, defaulting to `0.0`.
Training uses PyTorch's built-in smoothing: mix the observed one-hot label with a uniform distribution over declared classes.
When weighting and smoothing are combined, weights apply to every target component, while the integer-target mean uses the hard-target weight mass as its denominator.
Reported training CE is this configured objective; validation always uses hard labels, so its losses remain unsmoothed.
Validation reports both unweighted and train-weighted losses without a second model forward pass.
Weighted loss requires positive weight mass in validation; an entirely unseen-class validation partition cannot supply that criterion.
Weighted runs stop only when both independent patience counters are exhausted, or the maximum epoch limit is reached.
Either counter resets when its own loss improves sufficiently, even if it was previously exhausted.
The returned predictor and primary checkpoint select minimum unweighted validation loss; `result.weighted_best` retains the separate weighted-loss snapshot.
Single-run training additionally writes `<checkpoint-stem>.weighted<suffix>`; the API uses `save_checkpoint(..., criterion="weighted")`.
Both best snapshots are retained on CPU during weighted training, sharing storage when their minima coincide; distinct minima cost one additional model snapshot.
`min_delta` controls patience, not which minimum-loss checkpoint is selected.
Evaluation reports cross-entropy, top-1/top-3/top-5 accuracy, macro-F1 and per-class precision, recall, F1 and support.
Top-k clips k to the declared class count and breaks exact score ties by ascending class ID.
Macro-F1 includes every declared class, with undefined per-class scores set to zero; confusion matrix rows are true classes and columns are predictions.
CUDA is required when requested; CPU execution requires an explicit `device="cpu"`.
Seeds are recorded, but bitwise CUDA reproducibility is not promised.

Checkpoints contain the selected weights, scaler, class weights, selection criterion, feature identity, settings, split identifier hashes and epoch summaries.
They load with `weights_only=True` and restore an inference predictor, not an exact resumable training session.
Existing checkpoints are not overwritten unless `replace=True` is supplied.
Fixture tests verify behavior, not full-width GPU memory consumption or experiment throughput.
