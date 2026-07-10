# Environnements reproductibles

Ce projet peut etre installe avec conda ou avec un environnement virtuel Python
classique. Les deux fichiers principaux sont:

- `environment.yml` pour conda.
- `requirements.txt` pour pip/venv.

Les versions ont ete choisies a partir de l'environnement local `hsi-nuts`
utilise pour lire les resultats parquet et executer le rapport projet. Elles
sont volontairement limitees aux dependances directes utiles au depot: pile
scientifique, HDF5/parquet/Excel, visualisation, notebooks et Optuna.

## Option A - Avec conda

Depuis la racine du projet:

```powershell
conda env create -f environment.yml
conda activate hsi-nuts
python -m ipykernel install --user --name hsi-nuts --display-name "Python (hsi-nuts)"
```

Mettre a jour l'environnement apres modification de `environment.yml`:

```powershell
conda env update -n hsi-nuts -f environment.yml --prune
```

Verification rapide:

```powershell
python -c "import numpy, pandas, pyarrow, scipy, skimage, sklearn, h5py, plotly; import src; print('ok', src.__version__)"
```

Si les donnees/resultats locaux sont presents, tester aussi:

```powershell
python scripts\05_project_results_report.py
```

## Option B - Sans conda, avec venv/pip

Pre-requis: Python 3.14 installe et disponible dans le terminal.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m ipykernel install --user --name hsi-nuts-venv --display-name "Python (hsi-nuts venv)"
```

macOS/Linux:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name hsi-nuts-venv --display-name "Python (hsi-nuts venv)"
```

Verification rapide:

```powershell
.\.venv\Scripts\python -c "import numpy, pandas, pyarrow, scipy, skimage, sklearn, h5py, plotly; import src; print('ok', src.__version__)"
```

## Notes de reproductibilite

- `HSI Data/` et `results/` ne sont pas versionnes. Il faut les fournir
  separement ou regenerer les resultats.
- Les notebooks actifs contiennent des sorties lourdes. Pour Git, il vaut mieux
  les nettoyer avec `scripts/clean_notebooks.py` ou utiliser les scripts `# %%`.
- `optuna` est inclus pour couvrir les workflows d'optimisation, meme si la
  lecture des resultats existants n'en a pas besoin.
- `openpyxl` est inclus pour les fichiers Excel USDA.
- Les dependances lourdes presentes dans l'environnement local mais non requises
  par le code actuel, comme PyTorch ou UMAP, ne sont pas incluses.

