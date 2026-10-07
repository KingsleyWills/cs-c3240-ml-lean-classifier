
It's not totally clear in what order the labels are and what labels we will have. Let's define a list of labels and write our matrix such that it adheres to this list, even if it is changed.

```python
labels = [
    "simplify",
    "rewrite",
    "rule_exact",
    "introduce_local",
    "define_local",
    "destruct",
    "binder_management",
    "refine",
    "application",
    "extensionality",
    "case_split",
    "definitional_transform",
    "proof_search",
    "computational_closure",
    "arithmetic_reasoning",
    "property_automation",
    "convert",
    "OTHER",
    "algebraic_normalization",
    "generalized_congruence",
    "congr",
    "contradiction_reasoning",
    "use",
    "cast_normalization",
    "conv",
    "subst",
    "induction",
]
```

Now let's define the correlation matrix.
Indexing goes according to the labels in `labels`.

```python
cor = [[0.0] * len(labels)] * len(labels)
```

We also want a helper function to update the matrix:

```python
def correlate(t1, t2, amount = 1):
    t1i = labels.index(t1)
    t2i = labels.index(t2)
    cor[t1i][t2i] = amount
    cor[t2i][t1i] = amount
    
```

Let's now define which tactic families are similar.
```python
correlate("refine", "application")
correlate("application", "rewrite")
correlate("introduce_local", "define_local")
correlate("use", "choose")
correlate("rewrite", "subst")
correlate("convert", "refine")
correlate("arithmetic_reasoning", "algebraic_normalization")
correlate("proof_search", "property_automation")
```
