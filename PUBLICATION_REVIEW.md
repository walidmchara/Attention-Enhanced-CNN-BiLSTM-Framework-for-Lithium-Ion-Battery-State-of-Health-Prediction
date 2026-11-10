# Pre-release scientific and repository review

**Status:** Research-reproduction repository with verified recorded results. **Not yet suitable for advertising as a leakage-free SOH prediction benchmark.**

This report distinguishes reproducibility and artifact integrity from scientific validity. The repository remains private until its owner explicitly changes its visibility.

## Verified for source-code publication

| Check | Evidence | Status |
|---|---|---|
| Original MATLAB battery source | NASA PCoE official archive hash verification and eight MATLAB files checked in GitHub Actions | Complete |
| Archived notebook protocols | B0018 and B0030 code parity with the adapted Colab notebooks verified in CI | Complete |
| Reusable Python implementation | `legacy_mat_pipeline/` provides data extraction, preprocessing, network, training, checkpoints and evaluation | Complete |
| Training completion | Two independently executed 30-epoch Colab runs; B0018 132 cycles, B0030 40 cycles | Complete |
| Metric integrity | MAE, RMSE, MSE, MAPE, R² independently recomputed by CI from committed cycle-level prediction CSVs | Complete |
| Model provenance | Original MATLAB file SHA-256 values recorded in each `config.json` | Complete |
| Published result provenance | SHA-256 and byte-size manifest for original run artifacts, plus additional diagnostics derived from the same CSVs | Complete |
| Checkpoints | Full Keras model and optimizer state retained on Drive, outside the repository | Complete |
| Scientific presentation | README and per-experiment reports describe protocol, measured results, figures and limitations | Complete |
| Credentials/raw weights | Prepublication Git check rejects tracked `.mat`, `.keras`, `.h5`, certificates and key files | Check configured |
| Reproducibility CI | Python syntax, MATLAB source-recovery tests, synthetic Keras optimizer-resume tests and result integrity workflow | Passing at last check |

The last CI status can be inspected in [Actions](https://github.com/walidmchara/Attention-Enhanced-CNN-BiLSTM-Framework-for-Lithium-Ion-Battery-State-of-Health-Prediction/actions).

## Issues that must not be hidden

**Historical-protocol issues (high priority):**

1. **Same-cycle target leakage:** `capacity` is an input to the model, and SOH is calculated from that same capacity. A very small MAE cannot be used as evidence of an independently predictive model.
2. **Test-set preprocessing:** the historical notebook refits the MinMax scaler on the held-out battery. This is not a train-only normalization protocol.
3. **Validation split:** measurements are randomly split at sample level. Samples from the same discharge cycle can be included in both train and validation.
4. **Temporal interpretation:** the Conv1D/BiLSTM sequence dimension spans seven features, not chronologically ordered cycles. The dense sigmoid gate is feature-wise gating, not Transformer attention.
5. **Performance uncertainty:** only a single seed (42) is reported per battery; there are no confidence intervals, repeated trials, or significance tests for the archived-protocol results.

**Release preparation / scientific reporting:**

- [ ] Complete a leakage-free experiment with train-only normalization and no capacity-derived predictors, separating entire batteries for test and entire cycles for validation.
- [ ] Evaluate the corrected design on several batteries and seeds. Keep architecture and evaluation budget consistent with justified baselines; report mean ± standard deviation and avoid test-driven model selection.
- [ ] Distinguish **SOH estimation from same-cycle measurements** from prospective forecasting or remaining-useful-life prediction.
- [ ] Record a frozen software environment (Python, TensorFlow, NumPy, SciPy, pandas, scikit-learn versions) and hardware used for published experiments. The current `requirements.txt` specifies general dependencies but is not an exact environment lock.
- [ ] Confirm the source paper citation and code copyright owners. The MIT license lists the year but not an individual copyright holder.
- [ ] Verify redistribution permissions and attribution requirements for any third-party datasets or pre-existing notebook code included in the public release.
- [ ] Review git history and notebook metadata for any personal or institution-specific information before publication. Removing content from the current branch does not erase its history.
- [ ] Follow any journal, institution, or publisher requirements regarding disclosure of AI-assisted drafting or code work. Polished technical prose and the removal of unnecessary comments do not replace required disclosure.

## Scope of verified claims

The recorded historical-protocol results are:

| Metric | B0018 (132 cycles) | B0030 (40 cycles) |
|---|---:|---:|
| MAE | 0.004401576 | 0.004563940 |
| RMSE | 0.005764996 | 0.005558571 |
| R² | 0.995196321 | 0.976597052 |
| MAPE | 0.556322% | 0.444860% |
| Epochs | 30 | 30 |

These values are derived from actual completed Colab sessions and match the versioned output CSVs to rounding tolerance. The historically low errors **do not validate independent generalization** because of the methodological weaknesses listed above. The earlier cycle-level no-capacity workflow is a separate, non-equivalent experimental setup.

This assessment makes **no claim** that the current experimental pipeline matches every result in any external publication, nor that the code is an independently validated clinical or industrial estimator.

### Reproduction references

- [B0018 recorded results](results/legacy_mat_b0018_e30_seed42/README.md)
- [B0030 recorded results](results/legacy_mat_b0030_e30_seed42/README.md)
- [Original MATLAB protocol details](REPRODUCE_ORIGINAL_MAT.md)
- [Raw data provenance](docs/REAL_DATA_SETUP.md)
- [Artifact verification tests](tests/test_publication_integrity.py)

