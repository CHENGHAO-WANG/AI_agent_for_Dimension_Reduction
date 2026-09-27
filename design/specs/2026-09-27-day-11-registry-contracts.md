# Day 11 — registry and contracts

Status: design, settled question by question with the user on day 11; not yet implemented.
Supersedes nothing. Feeds the day 11 implementation plan.

## Why

Day 11 is briefed as "registry and contracts" (section 9). Its items fall into two groups.

*Settled by the notes, and implemented as written.* Kernel PCA stays terminal-only
(defect 4; sections 3.9 and 3.11), and section 3.2's sentence saying otherwise is
corrected. Every op declaring `intermediate` is `stochastic: false`, asserted as an
implication (section 3.9). `requires_connected_graph` replaces the validator's inline
`{isomap, lle, laplacian_eigenmaps}` and is not named for building a neighbour graph
(section 3.9). `nested_in_d` is a declared field, and a test compares the fit at d = 2
with the first two columns of the fit at d = 5 for every op that claims it (section 3.5).
None of these needs a design, and this document does not repeat them.

*Named by the notes only as a requirement,* which is the whole of this document: how a
property that depends on a parameter is declared; how an op's class is written; what the
raw-counts check reads once `EUCLIDEAN_METHODS` goes; the emphasis values; how a
registry text is kept true to its executor; how new rows are declared; what becomes of
the "one entry plus one executor" claim; which limits on d are rules; and how sample
identifiers travel. Days 12, 14, 15 and 17 all read these declarations, which is why the
day was treated as architectural.

The day **declares**; later days **enforce**. The exceptions, each argued below, are the
visualization class's d = 2 and the LLE neighbour minimum's new home, both of which cost
nothing to enforce where the check already runs.

## 1. Properties that depend on a parameter

Three properties are not properties of an op alone. `kernel_pca` works through Euclidean
geometry only at `kernel = rbf`; `umap` only at `metric = euclidean`; LLE is nested in d
only at `method = standard`. And two limits on d are arithmetic in d and in a parameter.

**Yes/no properties take a condition written in the YAML.** A property is `true`, `false`,
or true only when named parameters take one of the listed values:

```yaml
kernel_pca:
  euclidean: {when: {kernel: [rbf]}}
lle:
  nested_in_d: {when: {method: [standard]}}
```

A condition with several keys holds when all of them do. `load_registry` refuses a
condition that names a parameter the op does not declare, or a value outside that
parameter's `choices`, so a typo cannot silently make a property false. `OpSpec` gains
one reader, `holds(property, params)`, taking the resolved parameters, and every
consumer goes through it.

**Limits on d are named rules**, implemented in one module (section 6). The agent reasons
over the registry's text, so a condition like "Euclidean only at `kernel = rbf`" belongs
where it can be read; an arithmetic language in YAML would be a second, untested copy of
rules that must live in code anyway.

## 2. An op's class

`kind` gains a third value. `tsne` and `umap` become `kind: visualization`; PHATE, TriMap
and PaCMAP join them on day 18. The eight others stay `kind: reduction`, and so would a
GPLVM: spent on day 7 and not in the registry, it produces a representation, so when one
is added it is a reduction, and section 3.11's list of reductions names it as the day
closes. That is `CONTEXT.md`'s own division — an Op is "preprocessing, a Reduction, or a
Visualization method" — and a separate `class` field under `kind: reduction` was
rejected because it would label `tsne` a Reduction in the one field the agent reads.

`OpSpec` gains `is_visualization` beside `is_reduction`, and no property joining the two.
Four places read `is_reduction` today while meaning "a reduction or a visualization
method": `validate_stages`'s rule that a candidate ends in one, `PlanState.advance`'s
feature count, `exceeds_scale_limit`, and the registry-field test. Each is rewritten to
name both classes, `spec.is_reduction or spec.is_visualization`, so the reader sees which
classes a check covers without a third term to learn. Rejected: a combined property such
as `lowers_dimension`, since `select_variable_features` lowers the feature count too and
the name invites reading it as covering preprocessing. `Registry.reductions()` returns
only the class, and `Registry.visualization_methods()` joins it. `drtools methods --kind`
accepts `visualization`.

No behaviour changes by this alone: the two classes together cover the ten ops
`kind: reduction` covers today. Excluding visualization methods from a representation
run is day 17's.

`validate_stages`'s refusal of a non-terminal op is reworded. It says a terminal
method's output is "an embedding for viewing" with "no meaningful metric", which section
3.9 shows false for Isomap, Diffusion Maps and kernel PCA and which section 3.11's
redefinition of **Terminal method**, by position alone, no longer supports. The new text
says the op may stand only in a candidate's last Stage, and that only `pca` may come
before a reduction or a visualization method.

## 3. The raw-counts check, and `euclidean`

`EUCLIDEAN_METHODS` has one reader, `raw_counts_into_euclidean_method`. It walks each
candidate's stages with a flag that starts true when `profile.values.suspected_kind` is
`counts`, and refuses a stage whose op is in the set while the flag is up. `log1p` and
every reduction lower the flag. The set leaves out `pca`, `sparse_pca` and `umap`, and
no commit or log entry says why. So `pca(50) -> tsne` on raw counts passes: `pca` is not
checked, and its output lowers the flag before `tsne` is reached.

**The check applies to every reduction and visualization method, whatever its
geometry.** Raw counts
mislead through two facts — sample totals vary, and variance grows with the mean — and
neither depends on the distance a method computes. A polynomial kernel and a
cosine-metric UMAP see the same heteroscedastic features a Euclidean method does, and
section 3.10 already normalises counts for every candidate without asking about
geometry. Rejected: having the check read `euclidean`, which would have refused PCA,
sparse PCA and Euclidean UMAP on counts but stopped refusing kernel PCA's poly and
sigmoid kernels, tying two questions together for no stated reason.

The finding is renamed `raw_counts_not_normalised`, and its message states the
two facts rather than naming Euclidean distance. It stays an error; whether it softens to
a warning once section 3.10 applies normalisation by rule is day 13's. The flag's
`normalise_total` blind spot — only `log1p` lowers it — is also day 13's.

**`euclidean` serves day 13's selection rule alone.** Declared on every reduction and
visualization method, following section 3.10's table: `true` for `pca`, `sparse_pca`, `mds`, `isomap`,
`lle`, `laplacian_eigenmaps`, `diffusion_maps` and `tsne`; `{when: {kernel: [rbf]}}` for
`kernel_pca`; `{when: {metric: [euclidean]}}` for `umap`. Nothing reads it on day 11. It
lands now because the day's other declarations do, and the registry test requires it of
every reduction and visualization method.

## 4. Emphasis

Which distances a method keeps (section 3.11): short ones, long ones, or deliberately
both. Its first reader is the agent, through `drtools methods`; section 3.4's intended
spread warning would be the second, and no day schedules it yet.

| `emphasis` | ops |
|---|---|
| `local` | `laplacian_eigenmaps`, `lle`, `tsne`, `umap` |
| `global` | `mds`, `isomap`, `diffusion_maps`, `pca`, `sparse_pca` |
| `null`, with `emphasis_reason` | `kernel_pca` |

`balanced` is a legal value, first used by PaCMAP on day 18.

*PCA is declared `global`*, which is true on section 3.11's definition: the largest
pairwise distances dominate the directions of largest variance. Section 3.4's linear
baseline is a separate role, already carried by `family: linear`. So the future spread
check asks for a global method *other than the linear baseline*; a check that counted PCA
would always pass, because every plan holds PCA.

*Kernel PCA has no value.* Which distances it keeps depends on the kernel and its width:
near PCA with a wide RBF kernel, mainly neighbours with a narrow one. Any single value
would be a guess stated as fact. `emphasis_reason` is required whenever `emphasis` is
null, and kernel PCA's names the dependence and when to revisit it: when tuning shows its
chosen widths consistently favour one side.

## 5. Registry text kept true to its executor (defects 5 and 12)

Three parameters default to null and are filled by the executor, and each executor
already records which rule it applied in a `<param>_source` note:

| parameter | the registry says | the executor records |
|---|---|---|
| `normalise_total.target` | median sample total | `median sample total` — agrees |
| `kernel_pca.gamma` | `1/n_features` | `median pairwise distance heuristic` |
| `diffusion_maps.epsilon` | median squared k-NN distance | `kernel-sum scaling criterion (Coifman and Singer)` |

And the executor is itself wrong once: for `kernel = poly` it records `gamma_source: not
applicable`, but a polynomial kernel does use gamma, and with gamma null scikit-learn
applies `1/n_features`. Only the cosine kernel ignores it.

**Every null-defaulted parameter declares a `default_rule`,** a short label naming the
rule, in the conditional form of section 1 where the rule depends on another parameter:

```yaml
gamma:
  default: null
  default_rule:
    - {when: {kernel: [rbf, sigmoid]}, rule: median pairwise distance heuristic}
    - {when: {kernel: [poly]}, rule: "1/n_features (scikit-learn default)"}
    - {when: {kernel: [cosine]}, rule: not applicable}
```

`describes` then says what the rule does in words, and no longer carries its own
account of the default. Two tests hold it: one runs each such op on a small fixture with
the parameter null, over every branch of its condition, and asserts the executor's
`<param>_source` equals the declared `default_rule`; the other asserts every
null-defaulted parameter declares one, so a new method cannot skip the first by leaving
the rule out. The poly label is corrected in the executor.

Rejected: the executor reading its label from the registry. One copy cannot disagree
with itself, but an executor whose rule changed would then report the old label and no
test would notice. Two copies checked against each other fail on the likelier edit, the
rule and its adjacent label changed together. Neither design verifies that a label
describes the arithmetic; that still rests on reading the code.

## 6. New rows, and limits on d

**`new_rows` replaces `out_of_sample`.** The old field's `exact | approximate | none`
graded quality rather than naming a mechanism, and it marked Diffusion Maps
`approximate`, whose executor has no way to place a new row. Nothing reads it. The new
field takes section 3.12's three routes: `transform` for `pca`, `sparse_pca`,
`kernel_pca`, `isomap`, `lle`, `tsne` and `umap`; `nystrom` for `laplacian_eigenmaps`
and `diffusion_maps`, whose extension is day 14's; `none` for `mds`. Verified on the
installed libraries: `PCA`, `TruncatedSVD`, `KernelPCA`, `MiniBatchSparsePCA`, `Isomap`,
`LocallyLinearEmbedding`, `umap.UMAP` and openTSNE's `TSNEEmbedding` have a `transform`;
`SpectralEmbedding` and `MDS` do not. A test holds a map from each op to its library
class and asserts both that the map covers exactly the ops declaring `transform` and that
each class has one.

**A visualization method runs at `n_components = 2`.** Section 3.11 fixes d = 2 for the
class: a representation run nominates none, and a visualization run works at d = 2
throughout. So the rule is written once, keyed on `kind: visualization`, and
`validate_stages` refuses any other value — at registration and again at execution,
needing no data. The static `max: 3` on `tsne.n_components` is removed as a looser copy of
it. This closes defect 10 on day 11: `tsne(n_components = 3)` at n ≥ 10,000 is refused
before it runs, instead of passing and failing inside openTSNE. The day 12 row loses
defect 10. Rejected: an `opentsne_approximation` rule limiting d by n, which describes a
case the class already rules out.

**One named rule, `lle_neighbour_minimum`,** in a new module `drtools/constraints.py`.
`n_neighbors` must reach the variant's minimum at d: `d + 1` for standard, modified and
LTSA, `1 + d(d + 3)/2` for Hessian (6 at d = 2, 45 at d = 8). The executor's own table,
`LLE_NEIGHBOUR_MINIMUM` in `executors/manifold.py`, moves there and the executor reads
it back, so day 12's refusal at registration (defect 14) will use the same
implementation. The registry names it, `d_limits: [lle_neighbour_minimum]`, and each rule
carries a sentence that `drtools methods` prints. Only Hessian's minimum is the
library's: measured on day 11, scikit-learn fits standard, modified and LTSA at
`n_neighbors = d`, one below the table. Their `d + 1` is a mathematical minimum instead
— d + 1 points are the fewest whose affine span is d-dimensional, so fewer cannot
describe a d-dimensional local patch — and the rule's sentence says which kind each
minimum is. Tests: every rule the registry names exists and every rule is named; each
variant fits at its minimum in scikit-learn; Hessian fails one below; and no variant's
minimum is below the library's own. The form stays for one rule, since PHATE's and
TriMap's records may add more.

`d <= min(100, p - 1, n - 1)` is not a per-method rule: it applies to every reduction and
is day 15's `d_max`. Diffusion Maps' `d < n` and UMAP's spectral initialisation fall
under it.

## 7. "One entry plus one executor" (defect 9)

Sections 1 to 3 remove both predicate sets. The literals that remain outside the
executors are of two kinds.

*One is a predicate in disguise.* `heuristics.py` offers the neighbour suggestion to
`{umap, laplacian_eigenmaps, isomap, lle}`, from a base table `{lle: 10, isomap: 10, ...}`.
It becomes "any op declaring an `n_neighbors` parameter", starting from that op's
registry default, so a new neighbour-graph method gets it with no edit. LLE's suggestion
at n ≥ 1,000 moves from 10 to its default of 12 — inside the 6 to 12 where day 4's Swiss
roll was recovered, and at the existing cap. Keying by parameter name is safe only
here: `alpha` is density normalisation in Diffusion Maps and a sparsity penalty in sparse
PCA.

*The rest are rules about one method's own parameters,* and stay: t-SNE's perplexity
suggestion and its n/3 check, PCA's component suggestion, Diffusion Maps' alpha, LLE's
cap of 12, the PCA baseline in `no_linear_baseline`, and `PlanState.advance`'s account of
what each preprocessing op does to the simulated data. A test scans every module under
`drtools/` outside `executors/` for string constants equal to an op name, and fails
unless each (module, op) pair appears in a short list inside the test with a sentence
saying why a new method can safely miss it. The list is the extension point's
documentation, and it cannot fall behind the code.

The registry header's claim narrows to what is true: adding a method takes an entry and
an executor, and may add a suggestion rule or a check on its own parameters, where the
enumerating test lists where those live. Section 3.3's sentence is corrected to match.
Rejected: keeping the claim broad and moving every rule into YAML, which needs the rule
language section 1 set aside.

## 8. Sample identifiers (defect 21)

**What happens today.** The loaders keep feature names — `meta["feature_names"]`, cached
in `data/meta.json` — and discard sample identifiers: a DataFrame's index, an AnnData's
`obs_names`. An embedding's rows match samples by position alone, and nothing exported
can name them. `CONTEXT.md` has no term for a row's name; "sample identifier" is used
here and flagged for the glossary as the day closes.

1. **Carried in `meta["sample_ids"]`,** mirroring `feature_names`. The contract stays
   `load(spec) -> (X, labels, meta)`, so every adapter written so far still passes.
   Rejected: a fourth return value, which breaks them all and buys nothing `meta` lacks.
2. **Checked when present:** a list of strings, of length n, with no duplicates. A
   duplicate makes an exported row ambiguous, so it is refused with a message saying to
   make the names unique in the loader; AnnData permits duplicate `obs_names`, so this
   is reachable.
3. **Filled when absent.** `load` assigns the row positions `"0"` to `"n-1"` after the
   contract check and records `meta["sample_ids_source"]` as `row_order`, or `loader`
   when the loader supplied them. Every downstream reader can rely on identifiers
   existing, and the export can say which kind it has.
4. **Supplied by the built-in loaders.** `.h5ad` and `pbmc3k` from `obs_names`; `.npz`
   from an optional `sample_ids` array, loaded without pickling; `.csv` and `.tsv` from an
   `id_column` option beside `label_column`, with `--id-column` on the CLI. `.npy`,
   `pathmnist` and the synthetic fixtures fall back to row order. `read_csv` builds no
   index, so a CSV's row names arrive as an ordinary column: a non-numeric one is refused
   today as unembeddable, and a numeric one — pandas' unnamed `to_csv` index is the common
   case — is silently embedded as a feature. `id_column` removes the named column from
   the features, as `label_column` does; auto-detecting pandas' index column is not
   attempted, since a guess about which column is not data is the kind of silent choice
   the loader contract exists to prevent.
5. **Cached in their own file, `data/sample_ids.json`.** `meta.json` keeps the source and
   the count. At n = 107,000 the list is about 1 to 2 MB, and `meta.json` is read by the
   report and on every run lookup. `read_sample_ids(run)` returns them; readers ask
   explicitly.
6. **Part of the dataset digest.** `content_hash` hashes the identifiers after X and the
   labels. The export names each row by its identifier, so two loads with the same X and
   different identifiers would put different names on the same numbers under one run —
   the silent mismatch the digest exists to refuse. Row-order identifiers are a function
   of n, which the digest already covers, so hashing them costs nothing in meaning.
   Every existing cached Run gets a new digest and refuses to reopen; `runs/` holds only
   development Runs, none tracked in git. Rejected: excluding them, as adapter source was
   excluded on day 8. Adapter source is how the data was produced, and tidying it changes
   nothing a reader sees; identifiers are part of what the data says.

Recording which rows a candidate fitted and which it projected is day 14's, and the export
is day 19's; day 11 carries the identifiers as far as the cache.

## Testing

Beyond the tests named above: every reduction and visualization method declares `euclidean`,
`nested_in_d`, `requires_connected_graph`, `emphasis` and `new_rows`; the registry
refuses a malformed condition, an unknown rule name and a null `emphasis` without a
reason; the raw-counts check refuses `pca`, and `pca(50) -> tsne`, on counts; a
visualization method at `n_components = 3` is refused by `validate_stages`; the contract
refuses mistyped, mis-sized or duplicated identifiers; identifiers survive the cache
round trip; and a change to the identifiers alone changes the digest.

Tests that register a plan on count data with a PCA candidate and no `log1p` will now see
it refused, and are corrected rather than the check softened.

## What the notes gain when the day closes

Section 3.2's kernel PCA sentence and section 3.3's extension claim corrected; section
3.11's list of reductions naming a GPLVM; section
3.4's "spread cannot be checked yet" updated to say the property now exists; section
9's day 12 row without defect 10; a day 11 decision-log entry; and `CONTEXT.md` offered
**Sample identifier**.
