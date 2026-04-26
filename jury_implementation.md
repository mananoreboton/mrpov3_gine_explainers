# Jury Implementation Review

Thesis title under review: **"A comparative study of Metrics to Evaluate the Explainers of Graph Neural Networks applied to compound binding affinity prediction"**

Reviewer stance: thesis jury member with focus on GNNs, GNN explainability, molecular graphs, and compound binding affinity / potency prediction.

## Executive Verdict

The repository contains a serious implementation attempt with four main parts:

1. `mprov3_gine`: ligand-graph dataset construction, GINE classifier training, validation, classification, and reports.
2. `mprov3_explainer`: explainer registry, explanation generation, preprocessing, metric computation, JSON artifacts, and visualization support.
3. `mprov3_gine_explainer_defaults`: shared defaults for paths, model architecture, folds, seeds, and explainer parameters.
4. `mprov3_ui`: a small local server to browse generated results.

The explainer-comparison part is comparatively strong: it implements multiple explainers, a common mask preprocessing pipeline, GraphFramEx-style fidelity/characterization, Longa-style sufficiency/comprehensiveness/F1-fidelity, valid-only aggregation, diagnostics, and tests for several metric edge cases.

However, there are major thesis-alignment and implementation concerns:

- The predictive task is **3-class potency category classification**, not continuous compound binding affinity regression. The code stores `pIC50`, but the model optimizes only cross-entropy on `Category`.
- The GINE training/evaluation pipeline appears **runtime-broken** because `loaders.collate_batch()` returns `(Batch, pIC50, category)`, while `train_epoch.py`, `validation.py`, and `classification.py` call `batch.to(device)` as if the loader yielded a PyG `Batch`.
- The experimental methodology does not yet constitute a broad comparative study of explainer **metrics** unless the thesis clearly scopes the comparison to the implemented metrics. Important common metrics such as deletion/insertion curves, explicit sparsity, stability, infidelity, and sensitivity are absent.
- Several methodological choices are defensible but must be explicitly justified: explaining only the best fold, excluding misclassified graphs from headline metrics, clamping Longa F1-fidelity operands, using top-k binarized fidelity as the headline, and treating molecular atom/edge attributions as chemically meaningful without binding-pocket context.

Overall assessment: **partial implementation, not yet jury-ready without fixes and methodological clarifications.**

## Repository-Level Implementation Steps and Correctness

| Step | Implementation | Correctness | Jury assessment |
|---|---|---:|---|
| 1 | Validate raw MPro snapshot layout with `check_raw_data_format.py`. | Correct | The code checks the expected raw files, split files, and sample SDF parsing. This is appropriate as a preprocessing quality-control step. |
| 2 | Build one PyG graph per ligand SDF in `build_dataset.py` using `dataset.sdf_to_graph()`. | Partially correct | The graph construction is plausible, but missing SDFs and failed SDF parses are silently skipped. This can change split sizes and class balance without strong evidence in the final report. |
| 3 | Use node features `[x, y, z, atomic_number]` and edge feature `bond_type`. | Partially correct | This is a reasonable minimal molecular graph, but it is chemically weak for binding affinity: no atom type embeddings beyond atomic number, no formal charge, aromaticity, hybridization, chirality, partial charge, residue/protein context, or ligand-protein interaction graph. Unknown elements default to carbon, which is unsafe. |
| 4 | Store both `pIC50` and `Category` labels on each graph. | Partially correct | Storing both is useful. But the implemented task uses only `Category` for classification. Calling this "binding affinity prediction" is imprecise unless the thesis explains that affinity is discretized into potency bins. Unknown categories default to class 0, which can hide data errors. |
| 5 | Load train/validation/test splits from raw split files and map PDB IDs to dataset indices. | Partially correct | The fold parsing is sensible, but PDBs absent from the built dataset are silently dropped. This should be counted and asserted or at least reported. |
| 6 | Create PyG data loaders in `loaders.py`. | Incorrect | `collate_batch()` returns a tuple `(data_batch, pIC50, category)`, but downstream training, validation, and classification code expects a single object with `.to()`, `.x`, `.edge_index`, `.category`, and `.batch`. This likely raises an error at the first batch in `train_one_epoch()`, `evaluate_validation()`, or `classify_test_with_predictions()`. |
| 7 | Define `MProGNN` using stacked `GINEConv`, batch normalization, dropout, pooling, and a classifier head. | Correct | The architecture is internally coherent for graph-level classification with edge attributes. It is a reasonable GNN baseline, though not a state-of-the-art affinity model. |
| 8 | Train with cross-entropy and select checkpoints by validation accuracy. | Partially correct | The training objective matches the categorical target. It is not an affinity-regression objective. Metrics are limited to loss/accuracy and do not include macro-F1, MCC, class balance, confidence calibration, or per-class performance. The loader bug may prevent this step from running as written. |
| 9 | Evaluate/classify test split and write classification JSON/report artifacts. | Partially correct / likely incorrect at runtime | The report design is reasonable, but classification uses the same loader tuple incorrectly. Also, `classify.py` is less configurable than `train.py`, increasing the risk of evaluating a different default dataset/root than the one trained. |
| 10 | Select a best fold for explainer experiments using training/classification summaries. | Partially correct | This is practical, but a comparative thesis should normally report variability across folds or justify single-best-fold analysis. Explaining only the best fold risks optimistic and unstable conclusions. |
| 11 | Register eight explainers: `GRADEXPINODE`, `GRADEXPLEDGE`, `GUIDEDBP`, `IGNODE`, `IGEDGE`, `GNNEXPL`, `PGEXPL`, `PGMEXPL`. | Correct | The registry in `mprov3_explainer/explainers.py` is clear and covers gradient, integrated-gradient, perturbation/statistical, and mask-optimization families. |
| 12 | Implement Integrated Gradients node/edge bridges and Captum leaf-input wrapper. | Correct | The implementation addresses practical Captum/PyG issues, including edge-index handling and non-leaf gradient warnings. This is a strong engineering point. |
| 13 | Train PGExplainer before explanation. | Partially correct | The pipeline trains PGExplainer on the train loader, but the CLI does not expose a training cap and full training can be expensive. The thesis should report PGExplainer training budget and convergence/degenerate-mask behavior. |
| 14 | Apply common preprocessing: detach masks, take absolute values, optional spread filtering, min-max normalization, and optional edge-to-node conversion. | Mostly correct | This follows a common representation idea and is tested. The choice to use absolute attribution magnitude is acceptable for importance ranking but removes inhibitory vs excitatory sign information; this should be discussed. |
| 15 | Compute GraphFramEx/PyG fidelity and characterization. | Correct with caveats | The headline top-k binarized version is more meaningful than soft-mask rescaling. The chosen `top_k_fraction=0.2` should be justified and sensitivity to `k` should be reported. |
| 16 | Compute Longa-style sufficiency, comprehensiveness, and F1-fidelity by percentile sweeps. | Partially correct | The percentile sweep and edge-native dispatch are defensible. The clamped F1-fidelity is numerically safer, but it is not a literal implementation if raw `Fsuf`/`Fcom` leave the paper's assumed domain. This must be stated as an implementation choice. |
| 17 | Aggregate only valid explanations for headline metrics and keep all-graph/soft diagnostic fields. | Partially correct | Valid-only aggregation avoids polluted means, but excluding misclassified graphs can make explainers look better and may disconnect the explanation benchmark from real model behavior. The thesis should report both valid-only and all-graph numbers. |
| 18 | Save `explanation_report.json`, mask JSON files, and `comparison_report.json`. | Correct | The artifact structure is useful and reproducible. JSON NaN handling is explicitly managed. |
| 19 | Generate explanation visualizations and HTML report. | Partially correct | Visualization is useful, but atom-mask-to-SDF atom order alignment is assumed rather than verified in the explainer package. Chemical interpretation should be cautious. |
| 20 | Serve reports through `mprov3_ui`. | Partially correct | The UI is convenient but hardcodes `fold_0`, while the explainer pipeline can select the best fold. This can show a different fold than the one actually selected by `run_explanations.py`. |
| 21 | Document the pipeline and metrics in READMEs. | Partially correct | The package READMEs are useful, but `result_description.md` appears stale relative to current metric code: it discusses soft-mask headline fidelity, unclamped F1-fidelity, raw threshold sweeps, and fold 0 results in ways that no longer match the current implementation. |
| 22 | Test the metric implementation. | Partially correct | There are good unit tests for preprocessing, binarization, F1 clamping, percentile sweep behavior, NaN aggregation, and a small explainer smoke test. There are not enough integration tests for the real GINE training/classification pipeline, all explainers, PGExplainer training, real checkpoints, or molecular-data invariants. |

## Detailed Jury Evaluation by Component

### `mprov3_gine`: GNN Backbone

**What is correct**

- `MProGNN` is a coherent graph-level GINE classifier with edge attributes.
- Data objects contain `x`, `edge_index`, `edge_attr`, `pIC50`, `category`, and `pdb_id`, which is a good basis for downstream explainers.
- The pipeline is well separated into build, check, train, classify, visualize, and report scripts.
- Fold-level training/checkpoint/report artifacts are clearly designed.

**What is incorrect or insufficient**

- The loader/collate contract is inconsistent with training/evaluation code. This is a blocking implementation error.
- The task is not binding affinity regression. It is potency-bin classification. A thesis with this title must either change the wording or explicitly justify discretization.
- Accuracy alone is insufficient for a 3-class chemistry classification task. At minimum, macro-F1, balanced accuracy, MCC, confusion matrix, per-class precision/recall, and class support should be reported.
- The molecular graph representation is too limited for strong affinity claims. It ignores protein context and many standard chemical descriptors.
- Silent defaults (`unknown element -> carbon`, `unknown category -> class 0`, skipped SDFs, dropped split IDs) are not acceptable in a scientific pipeline.

### `mprov3_explainer`: Explainers and Metrics

**What is correct**

- The explainer registry is clear and covers a reasonable range of local explanation families.
- The code handles node-mask and edge-mask explainers separately, including edge-native Longa-style sweeps.
- Integrated Gradients is implemented with PyG/Captum-specific bridge logic, which is a non-trivial and valuable implementation detail.
- Metric computation is structured and documented: top-k GraphFramEx fidelity, characterization score, sufficiency, comprehensiveness, F1-fidelity, spread, entropy, and valid-count diagnostics.
- The code preserves diagnostic soft-mask metrics while using top-k binarized fidelity as the headline. This is a good correction for graph-subset semantics.
- Tests cover several metric corner cases.

**What is incorrect or insufficient**

- Important explainer-evaluation metrics are missing: deletion/insertion curves, AUC of perturbation curves, explicit sparsity, stability under noise or repeated runs, infidelity, sensitivity-n, and plausibility against chemically meaningful ground truth.
- The study evaluates explanations of a classifier, not a direct affinity predictor.
- Valid-only aggregation can bias the comparative ranking by excluding hard/misclassified cases. A jury will expect both valid-only and all-graph conclusions.
- `PGEXPL` can produce degenerate masks; the implementation tracks this, but the thesis must discuss whether PGExplainer was properly trained and whether results are meaningful.
- `GUIDEDBP` is questionable because the model uses functional `x.relu()` inside `MProGNN.forward()`, and the explainer code itself warns that Guided Backpropagation hooks may not capture functional ReLU activations.
- Qualitative molecular visualizations are not enough to claim chemical interpretability unless the highlighted atoms/bonds are validated against known interactions, pharmacophores, or binding-site contacts.

### `mprov3_gine_explainer_defaults`: Configuration

**What is correct**

- Centralized defaults reduce drift between GINE and explainer packages.
- Shared model dimensions, split defaults, paths, fold selection, explainer parameters, and seeds are useful for reproducibility.

**What is insufficient**

- `seed_everything()` does not set deterministic CUDA/cuDNN flags, so GPU runs may still vary.
- Hardcoded path assumptions make the project less portable and less suitable for external reproduction.
- There is no experiment manifest recording exact git commit, package versions, dataset checksum, command-line flags, hardware, and device.

### `mprov3_ui`: Reporting UI

**What is correct**

- The server is simple and useful for browsing generated artifacts.
- It clearly separates GINE visualization, classification reports, and explainer reports.

**What is insufficient**

- It hardcodes `fold_0`, which conflicts with the explainer package's best-fold selection logic.
- The UI can mislead readers if the generated report corresponds to another fold.
- The README is empty, so assumptions and prerequisites are not documented.

### Documentation and Results

**What is correct**

- `mprov3_gine/README.md` and `mprov3_explainer/README.md` explain the intended workflow and metric keys.
- The current explainer README is explicit that the model is a trained GINE classifier.

**What is insufficient**

- `result_description.md` appears to describe an older metric implementation. It should be updated, versioned, or removed from thesis-facing materials.
- The thesis narrative must distinguish:
  - model performance evaluation,
  - explainer generation,
  - metric validity,
  - chemical plausibility,
  - and biological/affinity relevance.

## Methodological Concerns for the Thesis

1. **Affinity vs potency classification**

   The code uses `Category` as the supervised target and optimizes cross-entropy. This is not continuous binding affinity prediction. If the thesis title remains unchanged, the author must justify that discretized potency classification is an acceptable operationalization of binding affinity prediction.

2. **Ligand-only graphs**

   Compound binding affinity is driven by ligand-protein interactions. A ligand-only graph can learn ligand-property correlates, but it cannot explicitly model pocket contacts, protein residues, water bridges, hydrogen bonds, steric clashes, or induced fit. The thesis should not overclaim binding-affinity mechanistic insight from ligand-only attributions.

3. **Metric comparison scope**

   The implementation compares several metrics, but not the full landscape of explainer-evaluation metrics. The title says "Metrics" broadly; the code mainly implements GraphFramEx fidelity/characterization and Longa-style sufficiency/comprehensiveness/F1-fidelity, plus diagnostics.

4. **Ground truth absence**

   There is no ground-truth explanation mask from known active substructures, pharmacophores, docking contacts, or expert annotation. Therefore the evaluation is mainly functional/faithfulness-based, not plausibility-based.

5. **Best-fold-only explanation**

   Explaining the single best fold may be acceptable for a demonstration, but weak for a comparative study. Metric rankings may change across folds, seeds, and model checkpoints.

6. **Excluding misclassified graphs**

   A faithful explainer can still explain a wrong prediction. Excluding misclassified graphs from headline metrics may be defensible if the aim is "explanations of correct decisions," but the thesis must state this and report the excluded count.

7. **Top-k choice**

   A fixed `top_k_fraction=0.2` is reasonable because GraphFramEx uses this convention, but chemistry molecules vary in size. The ranking of explainers may depend strongly on this fraction.

8. **Mask granularity**

   Node-feature masks, node masks, and edge masks are not directly equivalent. Comparing them after normalization and aggregation is useful but can be unfair without granularity-aware interpretation.

9. **Model quality as a prerequisite**

   Explainer metrics are meaningful only if the underlying model has adequate predictive performance. If model accuracy is mediocre or class-imbalanced, explainer comparisons may reflect model weakness more than explainer quality.

10. **Runtime reproducibility**

   Seeds are set, but full determinism is not guaranteed. The experiment should record seed, device, package versions, commit hash, dataset checksum, fold, split, model checkpoint, and all explainer hyperparameters.

## Recommended Improvements

### Must Fix Before Defense

1. Fix the data loader contract.
   - Either remove the custom `collate_fn` and let PyG return a `Batch`, or update all training/validation/classification code to unpack `(data_batch, pIC50, category)`.
   - Add an integration test that runs one training batch, one validation batch, and one classification batch.

2. Clarify the task definition.
   - Rename claims to "potency category classification" or add a true `pIC50` regression model.
   - If keeping classification, explain bin thresholds and their scientific rationale.

3. Add classification metrics beyond accuracy.
   - Macro-F1, balanced accuracy, MCC, confusion matrix, per-class precision/recall, and support.

4. Make data losses explicit.
   - Count missing SDFs, failed graph parses, split IDs dropped from the dataset, unknown elements, unknown categories, and empty graphs.
   - Fail fast for invalid labels and unsupported elements unless there is a documented reason not to.

5. Update stale result documentation.
   - Bring `result_description.md` into agreement with current top-k fidelity, percentile sweeps, clamped F1-fidelity, valid-only aggregation, and seed handling.

6. Fix UI fold selection.
   - Serve the actual fold selected by the explainer run or allow a fold parameter.

### Strongly Recommended

1. Run explainers over all folds or at least report sensitivity across folds.
2. Add seed repeats for stochastic explainers and report mean/std.
3. Add explicit sparsity metrics such as fraction of selected atoms/bonds, L0/L1 mask mass, entropy normalized by graph size, and top-k concentration.
4. Add deletion/insertion curves and AUC-style perturbation metrics.
5. Add stability tests under feature noise, edge perturbation, conformer changes, or repeated explainer runs.
6. Add infidelity/sensitivity-n if the thesis positions itself as a broad metric comparison.
7. Include chemical plausibility validation: compare highlighted atoms/bonds to known warheads, pharmacophores, substructures, docking contacts, or literature SAR.
8. Add experiment manifests and checksums for reproducibility.
9. Add real end-to-end tests using a tiny saved PyG dataset and tiny checkpoint.
10. Report computational cost per explainer and per metric.

### Nice to Have

1. Support regression on `pIC50` and compare whether explanation rankings differ between regression and discretized classification.
2. Add richer molecular features: aromaticity, formal charge, hybridization, ring membership, chirality, donor/acceptor flags, valence, partial charges.
3. Include protein-ligand or residue-contact graphs if the claim remains "binding affinity".
4. Add calibration analysis for class probabilities used in sufficiency/comprehensiveness.
5. Improve visualization with atom labels, bond masks, legends, and normalized score scales.

## Important Questions to Ask the Author

1. Why is the thesis framed as compound binding affinity prediction when the implemented model is a 3-class potency classifier?
2. Why was `pIC50` stored but not used as a regression target?
3. What are the exact class thresholds, and are they biologically or experimentally justified?
4. What is the class distribution per fold, and how does it affect accuracy and explainer metrics?
5. Did the GINE training pipeline run successfully with the current `collate_batch()` implementation? If yes, what code version produced the reported checkpoints?
6. What is the baseline model performance per fold using macro-F1, MCC, and confusion matrices?
7. Why are explanations generated only for the best fold rather than all folds?
8. Why are misclassified graphs excluded from headline metric averages?
9. How stable are the explainer rankings across seeds, folds, and checkpoints?
10. Why was `top_k_fraction=0.2` selected, and how sensitive are conclusions to 0.1, 0.3, or graph-size-aware thresholds?
11. How do you compare node-feature, node, and edge explanations fairly?
12. Why is F1-fidelity clamped, and how often does clamping change the raw Longa-style values?
13. What does a negative sufficiency or comprehensiveness value mean chemically and methodologically?
14. Are PGExplainer masks degenerate because of insufficient training, model behavior, or the metric pipeline?
15. How do you know the highlighted atoms/bonds are chemically meaningful?
16. Are atom indices in the graph guaranteed to match atom indices in the SDF visualization?
17. Did you validate explanations against known MPro inhibitor substructures, warheads, docking poses, or residue contacts?
18. How do ligand-only graphs support claims about binding affinity without protein-pocket context?
19. What package versions, hardware, random seeds, and dataset snapshot produced the final numbers?
20. Which metric should a practitioner trust when GraphFramEx characterization and Longa F1-fidelity disagree?

## Less Obvious but Valuable Questions

1. Does including 3D coordinates as raw node features create translation/rotation sensitivity in the GNN? Were molecules aligned or standardized?
2. Are multiple conformers possible, and if so, why is a single SDF conformer sufficient?
3. Could the model exploit dataset artifacts such as ligand size, atom count, or assay-source bias rather than binding-relevant chemistry?
4. Does the split avoid analog leakage or scaffold leakage, or are similar compounds present in train and test folds?
5. How does graph size affect fidelity metrics and the top-k mask threshold?
6. Do edge-mask explainers get an advantage or disadvantage compared with node-feature explainers in the Longa-style sweeps?
7. What happens to metrics on very small molecules where top-k selection may keep only one or two atoms/bonds?
8. How are disconnected subgraphs handled in sufficiency/comprehensiveness, and is dropping isolated nodes chemically meaningful?
9. Does the model use batch normalization statistics consistently during explanation after explainers temporarily switch modes?
10. Are the softmax probabilities used for paper metrics calibrated enough to interpret probability drops?
11. Does taking absolute gradients hide whether a feature supports or opposes a class prediction?
12. Are explanations class-specific for all explainers, especially model-mode vs phenomenon-mode explainers?
13. Does PGExplainer's phenomenon-only setup make it directly comparable to model-mode explainers?
14. How are ties in mask scores handled, and can they affect top-k fidelity on near-constant masks?
15. How often do metric failures produce `NaN`, and are those failures correlated with a specific explainer or molecular structure?
16. Is `GuidedBP` valid for this model given functional ReLU calls inside `MProGNN.forward()`?
17. Could the 3D coordinates dominate atom identity in explanations because both are raw continuous features?
18. Is min-max normalization per graph fair across molecules with different attribution distributions?
19. Why are hydrogen atoms retained, and how does that influence node explanations and visual interpretability?
20. Would the explainer ranking change if the target were the predicted class rather than the ground-truth class?

## Files Most Relevant to the Assessment

- `mprov3_gine/dataset.py`: SDF-to-graph conversion, label loading, split mapping, dataset loading.
- `mprov3_gine/build_dataset.py`: dataset serialization and silent skipping of missing/failed molecules.
- `mprov3_gine/loaders.py`: custom collate function and data loader construction.
- `mprov3_gine/train_epoch.py`: training loop affected by loader tuple mismatch.
- `mprov3_gine/validation.py`: validation loop affected by loader tuple mismatch.
- `mprov3_gine/classification.py`: classification loop affected by loader tuple mismatch.
- `mprov3_gine/model.py`: GINE architecture.
- `mprov3_gine/train.py`: fold training, checkpointing, summary writing.
- `mprov3_gine/classify.py`: test/train split classification and summary writing.
- `mprov3_explainer/src/mprov3_explainer/explainers.py`: explainer registry and builder logic.
- `mprov3_explainer/src/mprov3_explainer/pipeline.py`: explanation loop, metric computation, aggregation.
- `mprov3_explainer/src/mprov3_explainer/preprocessing.py`: mask conversion, filtering, normalization, top-k helpers.
- `mprov3_explainer/scripts/run_explanations.py`: best-fold context, per-explainer execution, JSON outputs.
- `mprov3_explainer/tests/test_pipeline.py`: metric tests.
- `mprov3_explainer/tests/test_preprocessing.py`: preprocessing tests.
- `mprov3_explainer/tests/test_end_to_end.py`: smoke test for one explainer.
- `mprov3_gine_explainer_defaults/mprov3_gine_explainer_defaults/training_defaults.py`: seed and fold defaults.
- `mprov3_gine_explainer_defaults/mprov3_gine_explainer_defaults/best_fold.py`: fold selection.
- `mprov3_ui/src/mprov3_ui/server.py`: hardcoded fold-0 report serving.
- `result_description.md`: useful audit document but likely stale relative to current metric implementation.

## Final Jury Position

The code demonstrates meaningful engineering effort and a good understanding of practical GNN explainer tooling. The explainer metric pipeline is the strongest part of the repository. But the implementation is not fully aligned with the thesis title unless the author narrows the claim from "binding affinity prediction" to "ligand potency category classification" or adds true `pIC50` regression and protein-ligand context.

Before defense, the author should fix the GINE loader bug, update stale documentation, report robust predictive metrics, justify all explainer-metric design choices, and broaden or clearly delimit the metric comparison. Without these changes, a jury could reasonably challenge both the technical validity of the execution and the scientific validity of the conclusions.
