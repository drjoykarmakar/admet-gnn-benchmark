# Data

Raw molecular datasets are not committed to this repository.

The main benchmark uses TDC `Solubility_AqSolDB`. From the repository root,
download and clean it with:

```bash
python -m src.data \
  --output data/processed/solubility_aqsoldb_clean.csv \
  --report data/processed/solubility_aqsoldb_cleaning.json
```

If the TDC export has already been downloaded, the exact same cleaning pipeline can be run without PyTDC/network access:

```bash
python -m src.data \
  --input-file /path/to/solubility_aqsoldb.csv \
  --output data/processed/solubility_aqsoldb_clean.csv \
  --report data/processed/solubility_aqsoldb_cleaning.json
```

The local file must contain the TDC columns `Drug` (SMILES) and `Y` (LogS). CSV and TSV exports are both accepted.

Generate the classical Morgan + descriptor representation with:

```bash
python -m src.features \
  --input data/processed/solubility_aqsoldb_clean.csv \
  --output data/processed/solubility_aqsoldb_features.npz
```

Create explicit random and Bemis-Murcko scaffold assignments with:

```bash
python -m src.splits \
  --input data/processed/solubility_aqsoldb_clean.csv \
  --output-dir data/processed/splits \
  --seed 42
```

Generated files under `data/raw/` and `data/processed/` are gitignored. Cleaning
and split JSON audits are reproducible metadata; later reporting scripts may copy
selected audit values into committed result tables.
