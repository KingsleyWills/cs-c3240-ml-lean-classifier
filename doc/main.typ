
#import "template.typ": *

#show: template(
  [A machine learning approach to next tactic family prediction in Lean],
  generate_outline: false,
)

= Introduction and preliminaries

In mathematics, statements need to be proven rigorously in order to be considered true.
It is possible to define formal languages for mathematics, which guarantees rigor when adhered to.
A system that provides such a language and mechanically checks its proofs is called an interactive theorem prover or a proof assistant.

Lean 4 is an interactive theorem prover @lean-sysdesc that has increased in popularity among mathematicians in recent years.
Lean's mathematical community develops Mathlib, an extensive library of formalized mathematics @lean-mathematical-library[p.~379].

Lean is also a functional programming language with a type system based on the Calculus of Inductive Constructions @tpil @lean-sysdesc.
@lean-demorgan demonstrates how a statement (a type) can be proven by constructing a term of said type
#footnote[This proof was originally written as an example for a group member's BSc thesis, which is not finished as of this writing.].

#figure(
  ```lean
  theorem not_and_of_or_not (a b : Prop) : Or (Not a) (Not b) → Not (And a b) :=
    fun h : Or (Not a) (Not b) ↦
        fun hc : And a b ↦
          match h with
          | Or.inl (hna : Not a) => hna (And.left hc)
          | Or.inr (hnb : Not b) => hnb (And.right hc)
  ```,
  caption: [One-way version of one of De Morgan's laws formalized and proven in Lean, with a lot of syntactic sugar removed.],
) <lean-demorgan>

Lean's metaprogramming capabilities enable writing programs that operate on proof state (i.e. available hypotheses and the goal) called tactics @tpil.
@lean-demorgan-tactics gives an example of a tactic mode proof.

#figure(
  ```lean
  example (p q : Prop) : p ∧ ¬ p → q := by
    intro h
    cases h
    contradiction
  ```,
  caption: [A tactic mode proof presented in Theorem Proving in Lean 4 @tpil[§~5].],
) <lean-demorgan-tactics>

We call an artificial intelligence that attempts proving a statement in Lean to be an autoprover.
Recent major achievements in autoproving include OpenAI's formalization that the Navier-Stokes equation may blow up in finite time @openai2026finite and Anthropic's formalization of Fermat's Last Theorem @anthropic-flt.

This report is not a direct attempt at making an autoprover.
Instead, @sec:problem-statement will describe one problem relevant to autoproving, @sec:methods will suggest one elementary machine learning method related to autoproving, while @sec:ai details the use of AI in this work.


= Problem statement <sec:problem-statement>

When proving a statement in Lean's tactic mode, choosing a useful next tactic is a frequently recurring decision.

We formulate next tactic family prediction as a supervised multiclass classification problem.
Each datapoint is a state--label pair extracted from a Mathlib proof.
The state consists of the current goal and its local context immediately before the next tactic invocation, represented as a sparse vector of structural and numerical features.
The label is the normalized family of the action applied to that state.
Predicting tactic arguments is outside of the scope of this project.

Several tactics may be valid next actions for a particular proof state, but the source material demonstrates only one.
The observed label therefore represents one valid choice and should not be interpreted as the sole correct answer.


= Methods <sec:methods>

== Data extraction and label construction

We use a custom extractor for Lean 4.34.0 to obtain proof states from the corresponding Mathlib release.
Each observation contains the focused goal and its local context immediately before a retained tactic execution, paired with the demonstrated tactic family.
The retained corpus contains #emph[N] observations from #emph[T] theorems.

A fixed policy maps source tactic kinds to $27$ families, including `OTHER` for miscellaneous, administrative, and unclassified actions.
Families group related proof-level roles: for example, `rw`, `nth_rw`, and `rwa` belong to `rewrite`.
This does not imply that the grouped tactics are interchangeable, or that compound actions such as `rwa` are decomposed into separate observations.
Tactic arguments are neither prediction targets nor input features.

== Feature representation and vocabulary selection


For this project, the machine learning method must be chosen from a list of allowed methods, all of which require fixed-size input.
This is a significant restriction, as a Lean proof state can contain any number of arbitrarily large expression trees.
The requirement for a fixed-dimensional input vector necessitates constructing a representation that aggregates and discards some structural information.

We use the current goal and its local context as the source for the features.
The features are extracted from elaborated expressions -- typed tree-like structures where notation and referenced constants have been resolved -- rather than the text of the pretty-printed proof state.
Local identifiers are normalized via $alpha$-conversion so renaming them does not change their features.

Before model fitting, we will define a fixed feature vocabulary based on Lean's expression structure and selected mathematical symbols, which is then used to construct a sparse feature vector.
This representation is intended to preserve meaningful constants, expression-node kinds, and structural relationships, while aggregating hypotheses and discarding as little useful information as possible.
The exact vocabulary and structural encoding will be determined during implementation.
Additionally, we will include certain inexpensive statistical features such as the number of hypotheses, the sizes of the goal and hypothesis expressions, the maximum expression depth, and counts of common symbols.

For the model, we select multinomial logistic regression because it supports multiclass prediction directly, and is well suited to the relatively large dataset and high-dimensional sparse feature vectors produced by the proposed representation.
The linear class scores produced by logistic regression are converted into a probability distribution using the softmax function.


== Loss function, validation, and evaluation

We fit the model by minimizing multinomial cross-entropy with L2 regularization.
Cross-entropy decreases as the probability assigned to the demonstrated tactic family increases, penalizing confident incorrect classifications more than uncertain ones, while rewarding confident and correct predictions.
L2 regularization penalizes large model coefficients and helps reduce overfitting in the high-dimensional feature space.

We assign complete theorems to training, validation, and test partitions, targeting 80%, 10%, and 10% of the transitions, respectively.
Keeping every theorem within one partition prevents closely related states from the same proof from occurring in both training and evaluation data, which could produce overly optimistic results.
Randomized assignment with minimum tactic-family coverage constraints prevents a family from being severely underrepresented in validation or test data without forcing identical class proportions across partitions.
Validation data will be used to select regularization strength, class weighting, and feature-representation settings, while the test data will remain unused until final evaluation.
Planned evaluation measures are top-1 and top-$k$ accuracy, macro-F1, and unregularized cross-entropy.


== Logistic regression

Multinomial logistic regression is a suitable method for our multi-class prediction problem.
Additionally it is scalable and efficient for relatively large datasets with high-dimensional, sparse feature vectors.
We use logistic loss with regularization treated as a hyperparameter.

Logistic regression was implemented with Scikit-learn’s `SGDClassifier` using `loss="log_loss"`, instead of `LogisticRegression`, as the former supports batch training, multithreading and inexact solutions, making it more performant.

// - Logistic multinomial regression chosen because
//     - Scalable and efficient, well suited to relatively large dataset and high-dimensional sparse feature vectors
//     - This is a multiclass classification problem
// - We use Logistic loss
// - Regularization as hyperparameter
// - Use SKLearn SGDClassifier implementation with loss="log_loss" instead of `LogisticRegression`, because it supports training in batches and 1) does not search for the absolute optimal solution and 2) it's multi-threaded so (1)+(2) -> it's more performant

This is a significant challenge, as Lean proof states may contain any number of arbitrarily large expressions. It is also not possible to do this losslessly. \
Our solution constructs a sparse representation that aggregates structural information. 
Its inputs are the focused goal and the types of local declarations, which include hypotheses, parameters, local definitions, and instances.

Features are derived from elaborated expressions rather than the pretty-printed state.
These expressions form directed acyclic graphs with shared subexpressions.
Application and binder chains are flattened to expose their ordered operands, while a nested application remains one argument of its parent.
Selected coercion wrappers are removed and type-specific arithmetic operators are grouped into common families.
These are deliberately lossy representation choices, not changes to the source proofs.
Local identifier names are not feature inputs.

A structural vocabulary contains bounded, rooted graph fragments discovered at every level of the expression graphs.
Fragment shapes disregard node kinds and names, but retain operand order and sharing; internal operand lists must match, while continuation beyond a fragment's boundary is unspecified.
For each selected shape and node position, features count the expression-node kinds occurring there, separately for the goal and collective context.
Fragments may overlap; each distinct anchor contributes once per top-level expression occurrence, and repeated context expressions contribute separately.
Selected constant names and categorical attributes, such as bound-variable indices and binder information, add more specific channels.

Vocabulary selection uses training data only.
A coverage-oriented baseline supplies matches for individual goal and context expressions, with a generic leaf fallback providing basic representation even when richer fragments do not match.
Further shapes are selected using associations between their positional node-kind counts and tactic families, accounting for dimension cost and redundancy while balancing evidence across families.
Coverage means having a matching fragment, not reconstructing the expression or guaranteeing useful prediction.
General statistics supplement these counts, including declaration counts, distinct graph nodes, depth, sharing, and a syntactic indicator that the goal is `False`.
The first #emph[H] local declarations also receive individual statistical slots; all declarations contribute to collective features, and the remainder has overflow summaries.
The complete vectors have #emph[D] dimensions and an average of #emph[A] nonzero entries on the training partition.
// TODO: Replace H, D, and A with the final layout and training activation statistics; record discovery depths and selection settings in the appendix.

== Models and training objectives

We compare multinomial logistic regression with a multilayer perceptron (MLP).
Logistic regression provides a linear baseline suited to high-dimensional sparse inputs: each family receives an additive score from the features.
The MLP uses fully connected layers with ReLU hidden activations, allowing nonlinear interactions between features.
Softmax converts either model's output scores into class probabilities.

Both methods use cross-entropy, which penalizes low probability assigned to the demonstrated family and assigns larger penalties to confident misclassifications.
L2 regularization penalizes large logistic-regression coefficients.
The MLP is trained in minibatches with AdamW and decoupled weight decay, using validation cross-entropy for early stopping and restoring the lowest-loss validation checkpoint.
Its max-absolute feature scaling is fitted on training rows and applied unchanged to held-out rows, preserving sparse zeros.
#emph[Finalize model settings: logistic-regression solver, regularization, scaling, and stopping settings; MLP layer widths, dropout, learning rate, weight decay, batch size, epoch limit, patience, and seeds.]
// TODO: Report only the configurations actually compared, including any class weighting, rather than implementation defaults or planned searches.

== Partitioning, validation, and evaluation

We reserve an outer test partition and divide the remaining development data into training and validation partitions.
All transitions from a theorem remain together, preventing states from the same proof from appearing in both fitting and evaluation data.
Seeded randomized assignment is checked and, where necessary, repaired to meet minimum tactic-family support without enforcing identical class proportions.
Theorem grouping does not eliminate related mathematical content or dependencies across partitions.
#emph[Finalize partition sizes: theorem and observation counts for training, validation, and test, together with assignment fractions, support minima, and split seeds.]

Corpus-derived vocabulary selection and fitted scaling use only training data.
Both models use the same theorem partitions and frozen feature vocabulary; validation guides configuration and final model selection.
The outer test set remains unused until the final model is selected.
// TODO: State the actual validation criterion for selecting between methods, and revise the holdout description if the reported experiment uses grouped folds.

Evaluation uses top-1 accuracy, top-$k$ accuracy, macro-F1, and mean unregularized cross-entropy.
Accuracy measures overall agreement with the demonstrated family, macro-F1 gives each declared family equal weight, and cross-entropy evaluates the predicted class probabilities.
Top-$k$ accuracy measures whether the demonstrated family appears among the $k$ highest-scoring predictions.
Undefined per-family F1 scores are set to zero.
These measures assess agreement with source actions, not whether an alternative prediction could produce a valid proof step.
// TODO: Fix the reported k values and use the same evaluation contract for both methods.

// = Results

// - Compare and discuss the training and validation errors obtained
//   for all ML methods considered.
// - Which is the final chosen method and why?
// - What is the test error of the final chosen method?

// = Conclusion

// - Summarise the report and your findings.
// - Are the results suggesting that the problem is solved satisfactorily,
//   or might there be room for improvement?
// - Explain the limitation of the methods and how it can be further improved



= Use of AI <sec:ai>

Aalto AI GPT5.5 and GPT-OSS-120B and the freely available web version of ChatGPT, which claims#footnote[It does not seem to be possible to verify the version of the web version without an user account.] itself to be GPT-5.6 Luna, were used to search for information, such as references and citations we know exist and for Typst syntax. They and RedHatAI/gemma-4-31B-it-FP8-Dynamic were used to better understand metaprogramming in Lean and help with debugging some Lean code. RedHat's model was also used to explain some of the larger code files in the repository.

OpenAI Codex was used to locate documentation, assist with implementation and debugging of the extraction pipeline, provide feedback on report structure and wording, and limited discussion of methodological alternatives.
Generated code was reviewed and tested, and factual claims used in the report were checked against primary sources or empirical extraction results.

The authors remain responsible for all methodological ideas, decisions and submitted content.


#pagebreak()
#bibliography("references.bib")


#show: appendix()

= Tactic-family taxonomy <sec:tactic-families>

The following table summarizes all $27$ labels to which all tactics are mapped.

When constructing the table, we took into account the frequency of the tactics in mathlib, but we also included and grouped tactics based on how significant they are mathematically. A good example of this is `induction`: it performs an important mathematical role in a proof, even though its use is quite rare in Mathlib due to Leans recursion being used to golf inductive proofs instead.

The tactics were grouped into labels based on mathematical function. For example the tactics in `case_split` mainly split the proof into cases, either based on some hypothesis, the goal, or a proposition by the law of the excluded middle.

The extraction pipeline computes the occurrence estimates by applying the proposed mapping and transparent decompositions to the preliminary corpus.
They are not measured counts from a final Lean-validated normalized dataset.

#pagebreak()
#set page(flipped: true, margin: (x: 1.5cm, y: 2.5cm))

#block[
  #set text(size: 10pt)
  #table(
    columns: (1.5fr, 4.8fr, 1.1fr),
    align: (left, left, right),
    inset: 4pt,
    table.header([*Label*], [*Principal tactics or components*], [*Approx. occurrences*]),

    [`rule_application`],
    [`exact`, `assumption`, `exacts`, `assumption'`, `constructor`, `fconstructor`, `left`, `right`, `constructorm`, `and_intros`, `split_ands`],
    [$56664$],

    [`apply`],
    [`apply`, `fapply`, configured `apply`, `apply … at`, `symm`, `trans`, `transitivity`, `calc`, `filter_upwards`],
    [$22601$],

    [`introduce_fact`],
    [`have`, `have'`, `haveI`, `suffices`, `rsuffices`, `replace`, `specialize`, `tfae_have`],
    [$42250$],

    [`introduce_value`],
    [`let`, `letI`, `let rec`, `set`, `set!`],
    [$9907$],

    [`refine`],
    [`refine`],
    [$24670$],

    [`binder_management`],
    [`intro`, `intros`, `introv`, `rintro`, `revert`, `simp_intro`],
    [$22541$],

    [`destruct`],
    [`rcases`, `obtain`, `choose`, `choose!`, `injection`, `injections`],
    [$27249$],

    [`case_split`],
    [`cases`, `by_cases`, `by_cases!`, `split`, `split_ifs`, `fin_cases`, `interval_cases`, `casesm`, `cases_type`, `fun_cases`, tactic `match`/`if`, `nomatch`, `nofun`, `wlog`, `wlog!`],
    [$11585$],

    [`use`],
    [`use`, `use!`, `exists`, `existsi`],
    [$2644$],

    [`induction`],
    [`induction`, `fun_induction`, `hopf_tensor_induction`],
    [$4443$],

    [`rewrite`],
    [`rw`, `rw!`, `rewrite`, `erw`, `nth_rw`, `nth_rewrite`, `rwa`],
    [$84641$],

    [`simplify`],
    [`simp`, `simp!`, `simpa`, `simpa!`, `simpa using!`, `simp_all`, `simp_all!`, `simp_rw`, `push`, `pull`],
    [$106454$],

    [`definitional_transform`],
    [`dsimp`, `dsimp!`, `unfold`, `delta`, `change`, `show`, `beta_reduce`, `eta_expand`, `unfold_projs`, `cbv`],
    [$7196$],

    [`subst`],
    [`subst`, `subst_vars`],
    [$1267$],

    [`convert`],
    [`convert`, `convert!`, `convert_to`, `convert_to!`],
    [$4347$],

    [`conv`],
    [`conv`, `conv_lhs`, `conv_rhs`, `slice_lhs`, `slice_rhs`],
    [$1606$],

    [`extensionality`],
    [`ext`, `ext1`, `funext`],
    [$12141$],

    [`congr`],
    [`congr`, `congr!`, `congrm`, configured `congr`, `rcongr`],
    [$3122$],

    [`cast_normalization`],
    [`norm_cast`, `norm_cast0`, `push_cast`, `exact_mod_cast`, `rw_mod_cast`, `assumption_mod_cast`, `apply_mod_cast`, `rify`, `qify`, `zify`, `enat_to_nat`, `lift`],
    [$2408$],

    [`generalized_congruence`],
    [`gcongr`, `grw`, `nth_grw`, `apply_rw`, `gconvert`, `rel`, `apply_fun`],
    [$3465$],

    [`contradiction_reasoning`],
    [`by_contra`, `by_contra!`, `contrapose`, `contrapose!`, `exfalso`, `absurd`],
    [$2552$],

    [`computational_closure`],
    [`rfl`, `decide`, `infer_instance`],
    [$8699$],

    [`proof_search`],
    [`aesop`, `grind`, `tauto`, `tauto_set`, `contradiction`, `trivial`, `solve_by_elim`, `apply_rules`, `apply_assumption`, `tfae_finish`, `cat_disch`, `aesop_cat`, `aesop_mat`],
    [$9361$],

    [`arithmetic_reasoning`],
    [`lia`, `linarith`, `linarith!`, `nlinarith`, `omega`, `linear_combination`, `norm_num`, `norm_num1`, `order`, `bound`, `fin_omega`],
    [$5145$],

    [`algebraic_normalization`],
    [`ring`, `ring!`, `ring_nf`, `ring1`, `abel`, `abel_nf`, `abel1`, `field`, `field_simp`, `noncomm_ring`, `group`, `module`, `module_nf`, `match_scalars`, `ac_rfl`, `ac_nf`, `grobner`, `bicategory`, `monoidal`, `monoidal_coherence`],
    [$3187$],

    [`property_automation`],
    [`positivity`, `sz_positivity`, `fun_prop`, `finiteness`, `measurability`, `continuity`, `nontriviality`, `inhabit`, `subsingleton`, `algebraize`, `algebraize_only`, `borelize`],
    [$5370$],

    [`OTHER`],
    [Goal administration, generic wrappers/containers, generalization, and remaining unclassified domain-local tactics],
    [$5362$],

    [*Total*], [], [*Approx. $490877$*],
  )
]
#pagebreak()
#set page(flipped: false, margin: auto)

= Dataset extraction and preliminary analysis code <sec:code>

The listing below is the code used to extract the preliminary corpus and compute the statistics reported in this submission.
It invokes LeanDojo v2's extractor, converts the per-file output without constructing a repository-wide in-memory AST, writes the compact JSONL dataset, and computes the raw and estimated tactic-family summaries.

== Runtime requirements

- A 64-bit Linux environment with Git, `uv`, and the Lean/Lake tools available on `PATH`.
- Python 3.13 or later. The extraction dependency group contains `lean-dojo-v2==1.0.9` and `tqdm>=4.70.1`; exact transitive versions are recorded in the project's `uv.lock` file.
- Lean 4.34.0 (`leanprover/lean4:v4.34.0`) and Mathlib revision `v4.34.0`. The extracted Mathlib commit is `5ed2965256430c3649e86755f9576b54eca72435`.
- Network access on the first run. LeanDojo v2 also requires a GitHub access token to be present in the `GITHUB_ACCESS_TOKEN` environment variable during import. Read-only access to public repositories is sufficient. The token must not be placed in the source code, command arguments, or submitted files.
- No GPU is required for extraction. LeanDojo v2 may nevertheless install large machine-learning and CUDA-related transitive packages that are unused by this CPU-only pipeline.
- Substantial temporary storage. The full run used approximately 61 GiB at peak before disposable trace and cache files were removed. The final compact transition file is approximately 500 MiB. The completed run was allowed up to 12 GiB of memory. Conversion uses all visible CPU cores by default; `--jobs 1` reduces concurrent memory use.

The relevant project configuration is:

```toml
# pyproject.toml
[project]
requires-python = ">=3.13"

[project.scripts]
trustmebro-extract = "trustmebro.stage1.extract:main"

[dependency-groups]
extraction = ["lean-dojo-v2==1.0.9", "tqdm>=4.70.1"]

# lakefile.toml
[[require]]
name = "mathlib"
scope = "leanprover-community"
rev = "v4.34.0"

[[lean_exe]]
name = "trustmebro-extract-data"
srcDir = "data/runtime/compiled-extractor"
root = "ExtractData"
supportInterpreter = true
```

The toolchain file contains `leanprover/lean4:v4.34.0`.
After exporting the access token, the complete extraction and analysis run is started from the project root with:

```sh
uv run --isolated --frozen --group extraction trustmebro-extract \
  --commit 5ed2965256430c3649e86755f9576b54eca72435
```

The command writes generated data and temporary artifacts below the `data/` directory.
The summary can be recomputed after changing only the preliminary label mapping, without retracing Mathlib, using `uv run --frozen trustmebro-extract --summarize-existing`.

== Source listing: `src/trustmebro/stage1/extract.py`

#block[
  #raw(read("/src/trustmebro/stage1/extract.py"), lang: "python", block: true)
]
