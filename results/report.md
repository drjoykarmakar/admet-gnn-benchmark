# Generated benchmark report

This file is regenerated from saved prediction artifacts. It does not retrain either model.

## Test-set results

| model | random_rmse | random_mae | random_r2 | scaffold_rmse | scaffold_mae | scaffold_r2 | scaffold_minus_random_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Random Forest | 0.984 | 0.664 | 0.803 | 1.264 | 0.900 | 0.657 | 0.280 |
| Small GNN | 1.110 | 0.754 | 0.749 | 1.277 | 0.946 | 0.650 | 0.167 |

`scaffold_minus_random_rmse` is a descriptive difference, not a significance test.

## Figures

- `figures/parity_random.png`
- `figures/parity_scaffold.png`
- `figures/residuals_random.png`
- `figures/residuals_scaffold.png`
- `figures/applicability_random.png`
- `figures/applicability_scaffold.png`
- `figures/failure_molecules_scaffold.png`

## Largest scaffold-split errors

| model | smiles | y_true | y_pred | abs_error | nearest_train_tanimoto | chemistry_context |
| --- | --- | --- | --- | --- | --- | --- |
| Random Forest | CCN(CC)c1ccc([C+](c2ccc(N(CC)CC)cc2)c2ccc(N(CC)CC)cc2)cc1 | -0.140 | -6.670 | 6.531 | 0.542 | formal charge +1; MW 456.7; cLogP 7.24; TPSA 9.7; HBD/HBA 0/3; 12 rotatable bonds; 3 aromatic rings. These 2D descriptors provide structural context but do not establish the cause of the error. |
| Random Forest | COCCOCCCn1c(=O)c2c(N)c3c(=O)c4ccccc4c(=O)c3c(N)c2c1=O | -8.265 | -2.635 | 5.630 | 0.296 | neutral; MW 423.4; cLogP 0.48; TPSA 143.7; HBD/HBA 2/9; 7 rotatable bonds; 4 aromatic rings. These 2D descriptors provide structural context but do not establish the cause of the error. |
| Random Forest | CNC(=O)/C(C#N)=c1\[nH]c(=C2C(=O)NC(=O)NC2=O)c2ccccc12 | -7.528 | -2.131 | 5.397 | 0.250 | neutral; MW 337.3; cLogP -1.90; TPSA 143.9; HBD/HBA 4/5; 1 rotatable bonds; 2 aromatic rings. These 2D descriptors provide structural context but do not establish the cause of the error. |
| Random Forest | Cc1ccccc1NC(=O)CC(=O)CN=Nc1ccc(S(=O)(=O)O)cc1[N+](=O)[O-] | -7.740 | -2.506 | 5.233 | 0.406 | neutral; MW 420.4; cLogP 2.83; TPSA 168.4; HBD/HBA 2/8; 8 rotatable bonds; 2 aromatic rings. These 2D descriptors provide structural context but do not establish the cause of the error. |
| Random Forest | CCN=C1C=CC(=C(c2ccc(NCC)c(C)c2)c2ccc(NCC)c(C)c2)C=C1C | -1.023 | -6.123 | 5.099 | 0.254 | neutral; MW 413.6; cLogP 6.95; TPSA 36.4; HBD/HBA 2/3; 7 rotatable bonds; 2 aromatic rings. These 2D descriptors provide structural context but do not establish the cause of the error. |
| Small GNN | C[Si]1(C)O[Si](C)(C)O[Si](C)(C)O[Si](C)(C)O1 | -6.954 | -0.013 | 6.941 | 1.000 | neutral; MW 296.6; cLogP 2.87; TPSA 36.9; HBD/HBA 0/4; 0 rotatable bonds; 0 aromatic rings. These 2D descriptors provide structural context but do not establish the cause of the error. |
| Small GNN | Cc1ccccc1NC(=O)CC(=O)CN=Nc1ccc(S(=O)(=O)O)cc1[N+](=O)[O-] | -7.740 | -2.028 | 5.712 | 0.406 | neutral; MW 420.4; cLogP 2.83; TPSA 168.4; HBD/HBA 2/8; 8 rotatable bonds; 2 aromatic rings. These 2D descriptors provide structural context but do not establish the cause of the error. |
| Small GNN | CCN(CC)c1ccc([C+](c2ccc(N(CC)CC)cc2)c2ccc(N(CC)CC)cc2)cc1 | -0.140 | -5.318 | 5.178 | 0.542 | formal charge +1; MW 456.7; cLogP 7.24; TPSA 9.7; HBD/HBA 0/3; 12 rotatable bonds; 3 aromatic rings. These 2D descriptors provide structural context but do not establish the cause of the error. |

The chemistry-context column is descriptor-based and deliberately non-causal. Large errors can reflect representation limits, structural novelty, label noise, assay heterogeneity, solid-state effects, or other factors not resolved by this benchmark.

## Applicability-domain note

`nearest_train_tanimoto` is the maximum radius-2 Morgan fingerprint Tanimoto similarity between a test molecule and the training partition for the same split. It is used only as a structural-proximity diagnostic. This report does not define a universal in-domain threshold and does not treat similarity as calibrated uncertainty.
