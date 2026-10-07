# Tactic-family label policy

> Provisional Stage 2 proposal, updated against `data/observed-tactics.json` on 6 October 2026.
> Stage 1 is completed; its taxonomy and normalization plans are historical, not requirements for the current extractor.
> This document proposes a mapping; it does not install a runtime label-policy file or change extraction.

## Task and principles

Predict the family of an exported source proof action, not its arguments or an exact executable tactic.
Retain the original parser kind and source syntax independently of the model label.
A family groups related operational decisions, not tactics claimed to be semantically interchangeable in every context.

- Prefer useful distinctions over either maximum granularity or merging everything that can eventually solve the same goal.
- Merge configuration, scope, occurrence selection, and spelling variants within the same broad action.
- Preserve distinctive operations such as induction and extensionality regardless of frequency.
- Keep compound source tactics intact unless the extractor actually provides their component transitions.
- Assign each retained transition exactly one label; do not invent extra states or multiply counts to reflect a hypothetical decomposition.
- Use `OTHER` for remaining actions rather than discarding them or guessing from a familiar substring in the parser name.
- Label identity comes from the source tactic, not from expression-level flattening or coercion/operator rewrites.

## Changes from Stage 1

The proposal has 27 families, compared with the 28-family Stage 1 design.

| Stage 1 family | Stage 2 decision |
|---|---|
| `rule_application` | Move `apply`/`fapply` and their explicit application variants to `apply`; retain direct term/hypothesis and constructor selection. |
| `introduce_local` | Replace with `introduce_fact` for `have`/`suffices`-style actions and `introduce_value` for `let`-style definitions. |
| `choose` | Merge `choose`/`choose!` into `destruct`. |
| `relation_restructuring` | Merge `symm`/`trans`/`transitivity` into `apply`, together with relational chaining by `calc`. |
| `lift` | Merge into the broadened `cast_normalization` family. |

The names `introduce_fact` and `introduce_value` describe the intended proof action, not a guaranteed `Prop` versus non-`Prop` type distinction.
In particular, `have` can also introduce data.
The retained name `rule_application` is historical: explicit `apply` is now a separate family.

## Family inventory and observed counts

Counts below are exact sums of exported transition counts under the explicit mapping later in this document.
The inventory contains 490,877 transitions and 287 distinct parser kinds from a database containing 110,943 theorems.
These are whole-corpus exploratory counts, not counts from a training partition and not estimates of semantically primitive proof steps.
Per-kind theorem counts must not be summed to obtain family theorem support, because a theorem can contain several kinds in one family.

| Label | Principal source tactics | Transitions | Share | Rationale and boundary |
|---|---|---:|---:|---|
| `rule_application` | `exact`, `assumption`, `exacts`, `assumption'`, `constructor`, `fconstructor`, `left`, `right`, `constructorm`, `and_intros`, `split_ands` | 56,664 | 11.54% | Selects a direct term/hypothesis or constructor; constructor selection can leave goals, so this is not a closure-only class. Explicit rule application is separated below; `refine` instead supplies a partial term skeleton. |
| `apply` | `apply`, `fapply`, configured `apply`, `apply … at`, `symm`, `trans`, `transitivity`, `calc`, `filter_upwards` | 22,601 | 4.60% | Applies a chosen rule or restructures a relation through a supplied intermediate. Keeps explicit rule selection apart from direct proof/constructor selection and automated rule search. |
| `introduce_fact` | `have`, `have'`, `haveI`, `suffices`, `rsuffices`, `replace`, `specialize`, `tfae_have` | 42,250 | 8.61% | Introduces or specializes intermediate information or a proof obligation. Unlike `let`, it is not primarily a local definitional binding; unlike `intro`, the intermediate is chosen rather than already a goal binder. |
| `introduce_value` | `let`, `letI`, `let rec`, `set`, `set!` | 9,907 | 2.02% | Introduces a local definition or abbreviation, including instances. `set` can also add an equation, but choosing a name for a value is the dominant action; no decomposition is assumed. |
| `refine` | `refine` | 24,670 | 5.03% | Builds an explicit partial term with holes. Separate from selecting a rule and from `convert`, which can generate obligations for a non-definitionally-compatible term. |
| `binder_management` | `intro`, `intros`, `introv`, `rintro`, `revert`, `simp_intro` | 22,541 | 4.59% | Moves existing binders between the goal and context. Pattern introduction and accompanying simplification stay here when exported as one action; witness extraction from existing hypotheses belongs to `destruct`. |
| `destruct` | `rcases`, `obtain`, `choose`, `choose!`, `injection`, `injections` | 27,249 | 5.55% | Extracts witnesses, components, or constructor information from available evidence. `choose` may build dependent witness functions, but still eliminates existential information rather than supplying a goal witness. |
| `case_split` | `cases`, `by_cases`, `by_cases!`, `split`, `split_ifs`, `fin_cases`, `interval_cases`, `casesm`, `cases_type`, `fun_cases`, tactic `match`/`if`, `nomatch`, `nofun`, `wlog`, `wlog!` | 11,585 | 2.36% | Uses alternative cases, exhaustive elimination, or a without-loss-of-generality reduction. Pattern-driven `rcases` remains `destruct`, even when it branches; induction is distinguished by its recursive reasoning. |
| `use` | `use`, `use!`, `exists`, `existsi` | 2,644 | 0.54% | Supplies witnesses requested by the goal. This frequent, recognizable decision remains separate from general term refinement and witness extraction. |
| `induction` | `induction`, `fun_induction`, `hopf_tensor_induction` | 4,443 | 0.91% | Uses induction principles and recursive hypotheses. Retained for its distinctive proof structure, not because of its frequency. |
| `rewrite` | `rw`, `rw!`, `rewrite`, `erw`, `nth_rw`, `nth_rewrite`, `rwa` | 84,641 | 17.24% | Performs selected, directed equality/equivalence rewriting. `rwa` stays here as a rewrite-and-close compound; generalized relational rewriting has its own family. |
| `simplify` | `simp`, `simp!`, `simpa`, `simpa!`, `simpa using!`, `simp_all`, `simp_all!`, `simp_rw`, `push`, `pull` | 106,454 | 21.69% | Runs lemma-based simplification or registered directed normalization. `simpa` completion does not become a separate direct-proof example; `push`/`pull` are not restricted to negations or casts. |
| `definitional_transform` | `dsimp`, `dsimp!`, `unfold`, `delta`, `change`, `show`, `beta_reduce`, `eta_expand`, `unfold_projs`, `cbv` | 7,196 | 1.47% | Changes computational presentation or uses definitional compatibility. Unlike `rewrite`/`simplify`, it does not primarily select propositional rewriting lemmas; unlike `convert`, it does not accept arbitrary propositional compatibility. |
| `subst` | `subst`, `subst_vars` | 1,267 | 0.26% | Eliminates variables using equalities and updates dependent context entries. Not treated as an interchangeable `rw; clear` sequence. |
| `convert` | `convert`, `convert!`, `convert_to`, `convert_to!` | 4,347 | 0.89% | Bridges a term or target using generated equality/congruence obligations. Distinct from direct refinement, which requires definitional compatibility. |
| `conv` | `conv`, `conv_lhs`, `conv_rhs`, `slice_lhs`, `slice_rhs` | 1,606 | 0.33% | Focuses transformation on a chosen subexpression. Retains its own family while that focus is not represented merely as arguments of a separately exported action. |
| `extensionality` | `ext`, `ext1`, `funext` | 12,141 | 2.47% | Reduces equality to agreement of components or function values. This is not ordinary congruence: it exposes observations of the objects rather than matching arguments of an already shared operation. |
| `congr` | `congr`, `congr!`, `congrm`, configured `congr`, `rcongr` | 3,122 | 0.64% | Reduces equality of compound expressions through argument congruence. General relations and monotonicity remain separate. |
| `cast_normalization` | `norm_cast`, `norm_cast0`, `push_cast`, `exact_mod_cast`, `rw_mod_cast`, `assumption_mod_cast`, `apply_mod_cast`, `rify`, `qify`, `zify`, `enat_to_nat`, `lift` | 2,408 | 0.49% | Changes or normalizes numeric/coerced representations. `lift` broadens the family to representation changes with possible validity obligations; it is not asserted to be equivalent to ordinary cast normalization. |
| `generalized_congruence` | `gcongr`, `grw`, `nth_grw`, `apply_rw`, `gconvert`, `rel`, `apply_fun` | 3,465 | 0.71% | Transports relations through expressions using congruence or monotonicity. `apply_fun` acts on a relation with a chosen function; it is not ordinary application of a theorem to solve a goal. |
| `contradiction_reasoning` | `by_contra`, `by_contra!`, `contrapose`, `contrapose!`, `exfalso`, `absurd` | 2,552 | 0.52% | Changes the logical proof route to contradiction or contraposition. Automated discovery of a contradiction belongs to `proof_search`. |
| `computational_closure` | `rfl`, `decide`, `infer_instance` | 8,699 | 1.77% | Attempts closure by definitional equality, a decision procedure, or instance synthesis. Unlike `definitional_transform`, these source tactics ask for a solution rather than just a changed presentation. |
| `proof_search` | `aesop`, `grind`, `tauto`, `tauto_set`, `contradiction`, `trivial`, `solve_by_elim`, `apply_rules`, `apply_assumption`, `tfae_finish`, `cat_disch`, `aesop_cat`, `aesop_mat` | 9,361 | 1.91% | Searches for a proof using available rules or logical consequences. `apply_rules`/`apply_assumption` choose rules automatically rather than taking the specific rule-selection decision of `apply`. |
| `arithmetic_reasoning` | `lia`, `linarith`, `linarith!`, `nlinarith`, `omega`, `linear_combination`, `norm_num`, `norm_num1`, `order`, `bound`, `fin_omega` | 5,145 | 1.05% | Discharges arithmetic/order constraints or numerical computations. Algebraic identity normalization and positivity/property-specific automation remain separate. |
| `algebraic_normalization` | `ring`, `ring!`, `ring_nf`, `ring1`, `abel`, `abel_nf`, `abel1`, `field`, `field_simp`, `noncomm_ring`, `group`, `module`, `module_nf`, `match_scalars`, `ac_rfl`, `ac_nf`, `grobner`, `bicategory`, `monoidal`, `monoidal_coherence` | 3,187 | 0.65% | Normalizes algebraic expressions or solves identities using algebraic laws, including categorical coherence. Separate from arithmetic inequalities and specialized property synthesis. |
| `property_automation` | `positivity`, `sz_positivity`, `fun_prop`, `finiteness`, `measurability`, `continuity`, `nontriviality`, `inhabit`, `subsingleton`, `algebraize`, `algebraize_only`, `borelize` | 5,370 | 1.09% | Uses specialized structural properties or prepares their instances. Not all members close goals: `inhabit` and `algebraize` can add instances; these differ from a manually supplied local `letI`. |
| `OTHER` | Goal administration, generic wrappers/containers, generalization, and remaining unclassified domain-local tactics | 5,362 | 1.09% | A retained residual label, not one coherent proof operation or an executable fallback. Avoids giving rare or ambiguous actions a misleading mathematical family. |
| **Total** | | **490,877** | **100%** | |

## Compound actions and important boundaries

### No assumed semantic decomposition

The Stage 1 proposal discussed decomposing `rwa`, `rintro`, `obtain`, `replace`, `specialize`, and `set`, and merging adjacent compatible `rw` or `intro` invocations.
Those plans are not implemented requirements of this Stage 2 policy.
An observed source kind receives one family even if its implementation invokes several lower-level tactics.

- `rwa` is `rewrite`: its extra completion behaviour does not create a separate `rule_application` example.
- `rintro` is `binder_management`: it introduces goal binders with patterns, even when destructuring is involved.
- `obtain` is provisionally `destruct`: witness/component extraction motivates this default, but some forms also establish a new fact.
- `replace` and `specialize` are `introduce_fact`: replacing or specializing existing information is not a new `clear` example.
- `rsuffices` is `introduce_fact`: the intermediate obligation takes precedence over accompanying pattern destructuring.
- `set`/`set!` are `introduce_value`: naming a local expression takes precedence over an accompanying equation or state rewrite.
- `simpa` and `simpa using!` stay `simplify`; do not assume their internals equal a visible `simp; exact` sequence.
- `simp_intro` stays `binder_management`, since its source action introduces binders while simplifying their types.
- `filter_upwards` is provisionally `apply`: its core reduces filter membership through fixed rules, with optional introductions and a closing term.
- Multi-rule `rw` and multi-binder `intro` remain one exported action; this policy does not coalesce adjacent source invocations.

A later syntax-aware policy could distinguish forms of `obtain` or other compounds.
The current exact-kind API cannot do that merely by changing its kind-to-label dictionary.
Such a change would need an explicit classification rule and updated counts, not an undocumented exception.

### Related actions need not be interchangeable

`constructor`, `left`, and `right` remain in `rule_application`, despite being implementable through constructor application.
The split preserves direct proof/constructor selection versus explicit rule selection; it does not claim a fundamental logical separation.
`apply_rules` and `apply_assumption` stay in `proof_search`, because they search for a suitable rule rather than taking a user-specified one.
`apply_fun` is `generalized_congruence`, because it transports an equality or inequality through a function, not because its name starts with `apply`.

`choose` performs choice/skolemization rather than ordinary pattern elimination, but shares the decision to extract witnesses from available existential information.
It therefore belongs to `destruct`, not `use`, which supplies witnesses to the goal.
`lift` may change representation and introduce validity obligations; merging it with cast handling does not make it a simplifier or an equivalence-preserving no-op.

`push`/`pull` use registered directional rewriting and are not restricted to logical negation.
They belong to `simplify`, rather than automatically to contradiction reasoning or cast handling.
`subst` retains its own family because dependent-context substitution is not generally interchangeable with rewriting and clearing.

### Automation and rare wrappers

Automated property setup need not close a goal.
For example, `inhabit` can establish an inhabited instance, and `algebraize` can add algebra instances and associated properties.
These belong to `property_automation`, rather than manual `introduce_value` or arithmetic solving.
The normalization of categorical identities by `bicategory`/`monoidal` belongs to `algebraic_normalization`.

The explicit mapping includes verified domain wrappers such as `sz_positivity`, `hopf_tensor_induction`, and Aesop-based domain search.
Other local tactics, including the many private `map_simp`/`C_simp` kinds, stay in `OTHER` pending a justified classification.
A spelling resembling `simp`, `ring`, or `induction` is not sufficient evidence on its own.

## Residual actions and extractor limitations

`OTHER` includes goal scheduling, focusing wrappers that survive extraction, context clearing/renaming, generalization, arbitrary metaprograms, and unclassified domain-local automation.
Do not give goal administration its own family for this proposal.
This includes `swap`, `rotate_left`/`rotate_right`, `on_goal`, `pick_goal`, `clear` variants, `rename` variants, and `generalize`.

The current inventory also contains generic container/wrapper kinds, including `Lean.Parser.Tactic.tacticSeq`, `Lean.Parser.Tactic.tacticSeqBracketed`, `«{»`, and `Batteries.Tactic.seq_focus`.
Their retained records go to `OTHER`; do not infer a family from the first source word or from the inventory's single example.
The same precaution applies to the two records with kind `choice`, whose displayed example is `cancel_denoms`.
One example does not establish that a generic kind always denotes that tactic.

`classical`, transparency wrappers, `open … in`, `set_option … in`, and `says` can enclose materially different tactics.
They remain `OTHER` unless the extractor exposes a separately classified inner action.
The current extractor traverses information trees and handles selected combinators such as `first`, `repeat`, and `<;>`, but the exported inventory is not a guarantee of a non-overlapping linear sequence of primitive actions.
Counts here describe the stored records, including surviving wrappers; relabeling alone does not repair missing, duplicate, or composite transitions.
A prediction of `OTHER` is a residual family prediction, not a calibrated abstention or an executable tactic.

## Exact parser-kind mapping

Use exact `tactic.kind` strings, not prefix matching, substring matching, or tactic-head tokenization.
The following JSON follows the existing `LabelPolicy` shape and explicitly covers all 287 observed kinds.
Unseen kinds also fall back to `OTHER`; record their counts and review them when changing the corpus.
If kinds move between families, recompute the counts rather than retaining the table unchanged.

```json
{
  "unmapped": "other",
  "other_label": "OTHER",
  "kinds": {
    "Batteries.Tactic.exacts": "rule_application",
    "Batteries.Tactic.tacticSplit_ands": "rule_application",
    "Lean.Parser.Tactic.assumption": "rule_application",
    "Lean.Parser.Tactic.constructor": "rule_application",
    "Lean.Parser.Tactic.exact": "rule_application",
    "Lean.Parser.Tactic.left": "rule_application",
    "Lean.Parser.Tactic.right": "rule_application",
    "Lean.Parser.Tactic.tacticAnd_intros": "rule_application",
    "Mathlib.Tactic.constructorM": "rule_application",
    "Mathlib.Tactic.tacticAssumption'": "rule_application",
    "tacticFconstructor": "rule_application",
    "Batteries.Tactic.tacticFapply_": "apply",
    "Batteries.Tactic.tacticTrans___": "apply",
    "Batteries.Tactic.tacticTransitivity___": "apply",
    "Lean.calcTactic": "apply",
    "Lean.Parser.Tactic.apply": "apply",
    "Lean.Parser.Tactic.symm": "apply",
    "Mathlib.Tactic.applyWith": "apply",
    "Mathlib.Tactic.filterUpwards": "apply",
    "Mathlib.Tactic.tacticApply_At_": "apply",
    "Lean.Parser.Tactic.replace": "introduce_fact",
    "Lean.Parser.Tactic.specialize": "introduce_fact",
    "Lean.Parser.Tactic.tacticHave__": "introduce_fact",
    "Lean.Parser.Tactic.tacticHave'": "introduce_fact",
    "Lean.Parser.Tactic.tacticHaveI__": "introduce_fact",
    "Lean.Parser.Tactic.tacticSuffices_": "introduce_fact",
    "Mathlib.Linter.HaveILetI.tacticHaveI__": "introduce_fact",
    "Mathlib.Tactic.rsuffices": "introduce_fact",
    "Mathlib.Tactic.TFAE.tfaeHave": "introduce_fact",
    "Lean.Parser.Tactic.letrec": "introduce_value",
    "Lean.Parser.Tactic.tacticLet__": "introduce_value",
    "Mathlib.Linter.HaveILetI.tacticLetI__": "introduce_value",
    "Mathlib.Tactic.setTactic": "introduce_value",
    "Mathlib.Tactic.tacticSet!_": "introduce_value",
    "Lean.Parser.Tactic.refine": "refine",
    "Lean.Parser.Tactic.intro": "binder_management",
    "Lean.Parser.Tactic.intros": "binder_management",
    "Lean.Parser.Tactic.revert": "binder_management",
    "Lean.Parser.Tactic.rintro": "binder_management",
    "Mathlib.Tactic.«tacticSimp_intro_____..Only_»": "binder_management",
    "Mathlib.Tactic.introv": "binder_management",
    "Lean.Parser.Tactic.injection": "destruct",
    "Lean.Parser.Tactic.injections": "destruct",
    "Lean.Parser.Tactic.obtain": "destruct",
    "Lean.Parser.Tactic.rcases": "destruct",
    "Mathlib.Tactic.Choose.choose": "destruct",
    "Mathlib.Tactic.Choose.tacticChoose!___Using_": "destruct",
    "«tacticBy_cases_:_»": "case_split",
    "Lean.Elab.Tactic.finCases": "case_split",
    "Lean.Parser.Tactic.«tacticNomatch_,,»": "case_split",
    "Lean.Parser.Tactic.cases": "case_split",
    "Lean.Parser.Tactic.funCases": "case_split",
    "Lean.Parser.Tactic.match": "case_split",
    "Lean.Parser.Tactic.split": "case_split",
    "Lean.Parser.Tactic.tacDepIfThenElse": "case_split",
    "Lean.Parser.Tactic.tacticNofun": "case_split",
    "Mathlib.Tactic.ByCases.byCases!": "case_split",
    "Mathlib.Tactic.casesM": "case_split",
    "Mathlib.Tactic.casesType": "case_split",
    "Mathlib.Tactic.intervalCases": "case_split",
    "Mathlib.Tactic.splitIfs": "case_split",
    "Mathlib.Tactic.wlog": "case_split",
    "Mathlib.Tactic.wlog!": "case_split",
    "Lean.Parser.Tactic.«tacticExists_,,»": "use",
    "Mathlib.Tactic.«tacticExistsi_,,»": "use",
    "Mathlib.Tactic.«tacticUse!___,,»": "use",
    "Mathlib.Tactic.useSyntax": "use",
    "Lean.Parser.Tactic.funInduction": "induction",
    "Lean.Parser.Tactic.induction": "induction",
    "TensorProduct.tacticHopf_tensor_induction_With__": "induction",
    "Lean.Parser.Tactic.rewriteSeq": "rewrite",
    "Lean.Parser.Tactic.rwSeq": "rewrite",
    "Lean.Parser.Tactic.tacticErw___": "rewrite",
    "Lean.Parser.Tactic.tacticRwa__": "rewrite",
    "Mathlib.Tactic.DepRewrite.depRwSeq": "rewrite",
    "Mathlib.Tactic.tacticNth_rewrite_____": "rewrite",
    "Mathlib.Tactic.tacticNth_rw_____": "rewrite",
    "Lean.Parser.Tactic.simp": "simplify",
    "Lean.Parser.Tactic.simpa": "simplify",
    "Lean.Parser.Tactic.simpAll": "simplify",
    "Lean.Parser.Tactic.simpAllAutoUnfold": "simplify",
    "Lean.Parser.Tactic.simpaUsingBang": "simplify",
    "Lean.Parser.Tactic.simpAutoUnfold": "simplify",
    "Lean.Parser.Tactic.tacticSimpa!__1": "simplify",
    "Mathlib.Tactic.Push.pull": "simplify",
    "Mathlib.Tactic.Push.pushStx": "simplify",
    "Mathlib.Tactic.tacticSimp_rw___": "simplify",
    "Lean.Parser.Tactic.cbv": "definitional_transform",
    "Lean.Parser.Tactic.change": "definitional_transform",
    "Lean.Parser.Tactic.delta": "definitional_transform",
    "Lean.Parser.Tactic.dsimp": "definitional_transform",
    "Lean.Parser.Tactic.dsimpAutoUnfold": "definitional_transform",
    "Lean.Parser.Tactic.unfold": "definitional_transform",
    "Mathlib.Linter.Style.show": "definitional_transform",
    "Mathlib.Tactic.betaReduceStx": "definitional_transform",
    "Mathlib.Tactic.etaExpandStx": "definitional_transform",
    "Mathlib.Tactic.unfoldProjsStx": "definitional_transform",
    "Lean.Parser.Tactic.subst": "subst",
    "Lean.Parser.Tactic.substVars": "subst",
    "Mathlib.Tactic.convert": "convert",
    "Mathlib.Tactic.convert_to!": "convert",
    "Mathlib.Tactic.convert!": "convert",
    "Mathlib.Tactic.convertTo": "convert",
    "Lean.Parser.Tactic.Conv.conv": "conv",
    "Mathlib.Tactic.Conv.convLHS": "conv",
    "Mathlib.Tactic.Conv.convRHS": "conv",
    "Mathlib.Tactic.Slice.sliceLHS": "conv",
    "Mathlib.Tactic.Slice.sliceRHS": "conv",
    "Lean.Elab.Tactic.Ext.ext": "extensionality",
    "Lean.Elab.Tactic.Ext.tacticExt1___": "extensionality",
    "tacticFunext___": "extensionality",
    "Batteries.Tactic.congrConfig": "congr",
    "Batteries.Tactic.congrConfigWith": "congr",
    "Batteries.Tactic.rcongr": "congr",
    "Congr!.congr!": "congr",
    "Lean.Parser.Tactic.congr": "congr",
    "Mathlib.Tactic.congrM": "congr",
    "Lean.Parser.Tactic.normCast0": "cast_normalization",
    "Lean.Parser.Tactic.pushCast": "cast_normalization",
    "Lean.Parser.Tactic.tacticApply_mod_cast_": "cast_normalization",
    "Lean.Parser.Tactic.tacticAssumption_mod_cast_": "cast_normalization",
    "Lean.Parser.Tactic.tacticExact_mod_cast_": "cast_normalization",
    "Lean.Parser.Tactic.tacticNorm_cast__": "cast_normalization",
    "Lean.Parser.Tactic.tacticRw_mod_cast___": "cast_normalization",
    "Mathlib.Tactic.ENatToNat.tacticEnat_to_nat": "cast_normalization",
    "Mathlib.Tactic.lift": "cast_normalization",
    "Mathlib.Tactic.Qify.qify": "cast_normalization",
    "Mathlib.Tactic.Rify.rify": "cast_normalization",
    "Mathlib.Tactic.Zify.zify": "cast_normalization",
    "Mathlib.Tactic.applyFun": "generalized_congruence",
    "Mathlib.Tactic.GCongr.«tacticRel[_]»": "generalized_congruence",
    "Mathlib.Tactic.GCongr.gcongr": "generalized_congruence",
    "Mathlib.Tactic.GCongr.gconvert": "generalized_congruence",
    "Mathlib.Tactic.GRewrite.applyRwSeq": "generalized_congruence",
    "Mathlib.Tactic.GRewrite.grwSeq": "generalized_congruence",
    "Mathlib.Tactic.GRewrite.tacticNth_grw_____": "generalized_congruence",
    "Batteries.Tactic.byContra": "contradiction_reasoning",
    "Batteries.Tactic.tacticAbsurd_": "contradiction_reasoning",
    "Lean.Parser.Tactic.tacticExfalso": "contradiction_reasoning",
    "Mathlib.Tactic.ByContra.byContra!": "contradiction_reasoning",
    "Mathlib.Tactic.Contrapose.contrapose": "contradiction_reasoning",
    "Mathlib.Tactic.Contrapose.contrapose!": "contradiction_reasoning",
    "Lean.Parser.Tactic.decide": "computational_closure",
    "Lean.Parser.Tactic.tacticInfer_instance": "computational_closure",
    "Lean.Parser.Tactic.tacticRfl": "computational_closure",
    "Aesop.Frontend.Parser.aesopTactic": "proof_search",
    "CategoryTheory.aesop_cat": "proof_search",
    "CategoryTheory.cat_disch": "proof_search",
    "Lean.Parser.Tactic.applyAssumption": "proof_search",
    "Lean.Parser.Tactic.applyRules": "proof_search",
    "Lean.Parser.Tactic.contradiction": "proof_search",
    "Lean.Parser.Tactic.grind": "proof_search",
    "Lean.Parser.Tactic.solveByElim": "proof_search",
    "Lean.Parser.Tactic.tacticTrivial": "proof_search",
    "Mathlib.Tactic.Tauto.tauto": "proof_search",
    "Mathlib.Tactic.TautoSet.tacticTauto_set": "proof_search",
    "Mathlib.Tactic.TFAE.tfaeFinish": "proof_search",
    "Matroid.aesop_mat": "proof_search",
    "«tacticBound[_]»": "arithmetic_reasoning",
    "Fin.tacticFin_omega": "arithmetic_reasoning",
    "Lean.Parser.Tactic.lia": "arithmetic_reasoning",
    "Lean.Parser.Tactic.omega": "arithmetic_reasoning",
    "Mathlib.Tactic.linarith": "arithmetic_reasoning",
    "Mathlib.Tactic.LinearCombination.linearCombination": "arithmetic_reasoning",
    "Mathlib.Tactic.nlinarith": "arithmetic_reasoning",
    "Mathlib.Tactic.normNum": "arithmetic_reasoning",
    "Mathlib.Tactic.normNum1": "arithmetic_reasoning",
    "Mathlib.Tactic.Order.tacticOrder_": "arithmetic_reasoning",
    "Mathlib.Tactic.tacticLinarith!_": "arithmetic_reasoning",
    "Lean.Parser.Tactic.acRfl": "algebraic_normalization",
    "Lean.Parser.Tactic.grobner": "algebraic_normalization",
    "Lean.Parser.Tactic.tacticAc_nf_": "algebraic_normalization",
    "Mathlib.Tactic.Abel.abel": "algebraic_normalization",
    "Mathlib.Tactic.Abel.abel1": "algebraic_normalization",
    "Mathlib.Tactic.Abel.abelNF": "algebraic_normalization",
    "Mathlib.Tactic.Bicategory.tacticBicategory": "algebraic_normalization",
    "Mathlib.Tactic.FieldSimp.field": "algebraic_normalization",
    "Mathlib.Tactic.FieldSimp.fieldSimp": "algebraic_normalization",
    "Mathlib.Tactic.Group.group": "algebraic_normalization",
    "Mathlib.Tactic.Module.tacticMatch_scalars": "algebraic_normalization",
    "Mathlib.Tactic.Module.tacticModule": "algebraic_normalization",
    "Mathlib.Tactic.ModuleNF.moduleNF": "algebraic_normalization",
    "Mathlib.Tactic.Monoidal.tacticMonoidal": "algebraic_normalization",
    "Mathlib.Tactic.Monoidal.tacticMonoidal_coherence": "algebraic_normalization",
    "Mathlib.Tactic.NoncommRing.noncomm_ring": "algebraic_normalization",
    "Mathlib.Tactic.Ring.ring1": "algebraic_normalization",
    "Mathlib.Tactic.RingNF.ring": "algebraic_normalization",
    "Mathlib.Tactic.RingNF.ringNF": "algebraic_normalization",
    "Mathlib.Tactic.RingNF.tacticRing!": "algebraic_normalization",
    "finiteness": "property_automation",
    "Lean.Elab.Tactic.inhabit": "property_automation",
    "Mathlib.Meta.FunProp.funPropTacStx": "property_automation",
    "Mathlib.Tactic.Borelize.tacticBorelize___": "property_automation",
    "Mathlib.Tactic.measurability": "property_automation",
    "Mathlib.Tactic.Nontriviality.nontriviality": "property_automation",
    "Mathlib.Tactic.Positivity.positivity": "property_automation",
    "Mathlib.Tactic.subsingletonStx": "property_automation",
    "Mathlib.Tactic.tacticAlgebraize__": "property_automation",
    "Mathlib.Tactic.tacticAlgebraize_only__": "property_automation",
    "SzemerediRegularity.Positivity.tacticSz_positivity": "property_automation",
    "tacticContinuity": "property_automation",
    "_private.Mathlib.AlgebraicGeometry.EllipticCurve.Affine.Basic.0.tacticEval_simp": "OTHER",
    "_private.Mathlib.AlgebraicGeometry.EllipticCurve.Affine.Basic.0.tacticMap_simp": "OTHER",
    "_private.Mathlib.AlgebraicGeometry.EllipticCurve.Affine.Formula.0.tacticC_simp": "OTHER",
    "_private.Mathlib.AlgebraicGeometry.EllipticCurve.Affine.Formula.0.tacticDerivative_simp": "OTHER",
    "_private.Mathlib.AlgebraicGeometry.EllipticCurve.Affine.Formula.0.tacticEval_simp": "OTHER",
    "_private.Mathlib.AlgebraicGeometry.EllipticCurve.Affine.Formula.0.tacticMap_simp": "OTHER",
    "_private.Mathlib.AlgebraicGeometry.EllipticCurve.Affine.Point.0.tacticC_simp": "OTHER",
    "_private.Mathlib.AlgebraicGeometry.EllipticCurve.DivisionPolynomial.Basic.0.tacticC_simp": "OTHER",
    "_private.Mathlib.AlgebraicGeometry.EllipticCurve.Jacobian.Basic.0.tacticMap_simp": "OTHER",
    "_private.Mathlib.AlgebraicGeometry.EllipticCurve.Jacobian.Basic.0.tacticMatrix_simp": "OTHER",
    "_private.Mathlib.AlgebraicGeometry.EllipticCurve.Jacobian.Basic.0.tacticPderiv_simp": "OTHER",
    "_private.Mathlib.AlgebraicGeometry.EllipticCurve.Jacobian.Formula.0.tacticMap_simp": "OTHER",
    "_private.Mathlib.AlgebraicGeometry.EllipticCurve.Jacobian.Point.0.tacticMap_simp": "OTHER",
    "_private.Mathlib.AlgebraicGeometry.EllipticCurve.Projective.Basic.0.tacticMap_simp": "OTHER",
    "_private.Mathlib.AlgebraicGeometry.EllipticCurve.Projective.Basic.0.tacticMatrix_simp": "OTHER",
    "_private.Mathlib.AlgebraicGeometry.EllipticCurve.Projective.Basic.0.tacticPderiv_simp": "OTHER",
    "_private.Mathlib.AlgebraicGeometry.EllipticCurve.Projective.Formula.0.tacticMap_simp": "OTHER",
    "_private.Mathlib.AlgebraicGeometry.EllipticCurve.Projective.Point.0.tacticMap_simp": "OTHER",
    "_private.Mathlib.AlgebraicGeometry.EllipticCurve.VariableChange.0.tacticMap_simp": "OTHER",
    "_private.Mathlib.AlgebraicGeometry.EllipticCurve.Weierstrass.0.tacticMap_simp": "OTHER",
    "«{»": "OTHER",
    "«tactic#adaptation_note_»": "OTHER",
    "AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.tacticMem_tac": "OTHER",
    "ArithmeticFunction.arith_mult": "OTHER",
    "Batteries.Tactic.«tacticOn_goal-_=>_»": "OTHER",
    "Batteries.Tactic.«tacticPick_goal-_»": "OTHER",
    "Batteries.Tactic.generalizeProofsElab": "OTHER",
    "Batteries.Tactic.seq_focus": "OTHER",
    "Batteries.Tactic.tacticSwap": "OTHER",
    "bdsimp": "OTHER",
    "CategoryTheory.ComposableArrows.tacticValid": "OTHER",
    "CategoryTheory.tacticSubst_hom_lift___": "OTHER",
    "cfcContTac": "OTHER",
    "cfcTac": "OTHER",
    "cfcZeroTac": "OTHER",
    "choice": "OTHER",
    "Filter.tacticIsBoundedDefault": "OTHER",
    "Lean.Elab.Tactic.clearExceptTactic": "OTHER",
    "Lean.Parser.Tactic.as_aux_lemma": "OTHER",
    "Lean.Parser.Tactic.classical": "OTHER",
    "Lean.Parser.Tactic.clear": "OTHER",
    "Lean.Parser.Tactic.clearValue": "OTHER",
    "Lean.Parser.Tactic.extractLets": "OTHER",
    "Lean.Parser.Tactic.generalize": "OTHER",
    "Lean.Parser.Tactic.open": "OTHER",
    "Lean.Parser.Tactic.rename": "OTHER",
    "Lean.Parser.Tactic.renameI": "OTHER",
    "Lean.Parser.Tactic.rotateLeft": "OTHER",
    "Lean.Parser.Tactic.rotateRight": "OTHER",
    "Lean.Parser.Tactic.runTac": "OTHER",
    "Lean.Parser.Tactic.set_option": "OTHER",
    "Lean.Parser.Tactic.tacticIterate____": "OTHER",
    "Lean.Parser.Tactic.tacticSeq": "OTHER",
    "Lean.Parser.Tactic.tacticSeqBracketed": "OTHER",
    "Lean.Parser.Tactic.withReducible": "OTHER",
    "Lean.Parser.Tactic.withReducibleAndInstances": "OTHER",
    "Lean.Parser.Tactic.withUnfoldingAll": "OTHER",
    "Mathlib.MoveAdd.tacticMove_mul_": "OTHER",
    "Mathlib.Tactic.clear!": "OTHER",
    "Mathlib.Tactic.ComputeDegree.computeDegree": "OTHER",
    "Mathlib.Tactic.ComputeDegree.monicityMacro": "OTHER",
    "Mathlib.Tactic.ComputeDegree.tacticCompute_degree!": "OTHER",
    "Mathlib.Tactic.ComputeDegree.tacticMonicity!": "OTHER",
    "Mathlib.Tactic.inferOptParam": "OTHER",
    "Mathlib.Tactic.Interactive.tacticUnit_interval": "OTHER",
    "Mathlib.Tactic.MfldSetTac.mfldSetTac": "OTHER",
    "Mathlib.Tactic.MvBisim.tacticMv_bisim___With___": "OTHER",
    "Mathlib.Tactic.refoldLetStx": "OTHER",
    "Mathlib.Tactic.rename'": "OTHER",
    "Mathlib.Tactic.Says.says": "OTHER",
    "Nat.tacticBitwise_assoc_tac": "OTHER",
    "Num.transfer": "OTHER",
    "Num.transfer_rw": "OTHER",
    "Real.«tacticPi_lower_bound[_,,]»": "OTHER",
    "Real.«tacticPi_upper_bound[_,,]»": "OTHER",
    "Set.tacticTo_encard_tac": "OTHER",
    "tacticBddDefault": "OTHER",
    "tacticSimp_wf": "OTHER",
    "witt_truncateFun_tac": "OTHER",
    "WittVector.«tacticGhost_fun_tac_,_»": "OTHER",
    "WittVector.initRing": "OTHER",
    "WittVector.mapFun.tacticMap_fun_tac": "OTHER",
    "WittVector.Tactic.ghostCalc": "OTHER",
    "WittVector.Tactic.ghostSimp": "OTHER",
    "ZNum.transfer": "OTHER",
    "ZNum.transfer_rw": "OTHER"
  }
}
```

## Use in the learning experiment

Keep all transitions from one theorem together when partitioning.
Measure label support by both transitions and distinct theorems in each split; rare labels can be concentrated in a few proofs despite a reasonable transition count.
Do not infer adequate train/test support from the whole-corpus table alone.
Freeze the adopted mapping for a reported model comparison.
Any tactic-aware vocabulary selection or learned preprocessing must use training data only.

This is a reviewable proposal, not evidence that these 27 families are the best-performing target space.
Validation results can motivate further splits or merges; changes must remain explicit and consistent across the compared models.

## Historical Stage 1 context

The completed Stage 1 design used 28 families and a LeanDojo-v2 extraction with `get_traced_tactics(atomic_only=True)`.
Its preliminary corpus contained 402,911 retained source records across 105,518 theorems and 6,903 files.
The former 436,606-action table was an estimate after proposed decompositions, not the current export or an achieved normalized dataset.

`atomic_only=True` was a narrow AST filter excluding tactics with nested tactic-sequence nodes, not a semantic atomicity guarantee.
The compatibility experiment retained `constructor <;> assumption` as one record, omitted outer `cases` actions in some branch syntax, and lost repeated executions sharing one source span.
The preliminary corpus still contained 5,605 strings with `<;>` and 5,820 with semicolons.
Those limitations explain why the Stage 1 estimates are not directly comparable to the Stage 2 counts above.
Stage 2 uses the custom Lean information-tree extractor, not this LeanDojo flag.

## Semantic references

The mapping was checked against the pinned Mathlib source, with these public references documenting several non-obvious boundaries:

- [Filter membership and `filter_upwards`](https://leanprover-community.github.io/mathlib4_docs/Mathlib/Order/Filter/Defs.html).
- [Representation lifting](https://leanprover-community.github.io/mathlib4_docs/Mathlib/Tactic/Lift.html).
- [Applying functions to relations](https://leanprover-community.github.io/mathlib4_docs/Mathlib/Tactic/ApplyFun.html).
- [Intermediate facts and closure for TFAE](https://leanprover-community.github.io/mathlib4_docs/Mathlib/Tactic/TFAE.html).
- [Instance and property setup by `algebraize`](https://leanprover-community.github.io/mathlib4_docs/Mathlib/Tactic/Algebraize.html).
