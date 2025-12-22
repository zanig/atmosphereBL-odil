# SBL ODIL

Stable Boundary Layer w/ ODIL/B-ODIL attempt. It follows the GABLS weakly stable (0.05 K/h), moderately stable (0.25 K/h), and truly neutral boundary layer (TNBL) cases trying to reproduce the results from [this paper](https://doi.org/10.1007/s10546-025-00945-6).
The B-ODIL is my attempt to do what is described in Sec. 2.3 [here](https://arxiv.org/pdf/2510.15664)

I had everything in one giant py file and asked claude to clean the code into a presentable repo so have mercy. 


## Setup
the requirements are nothing special but obligatory: 
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data expectations
Data can be obtained from [here](https://doi.org/10.5281/zenodo.17479454)
Place the LES .mat files in a directory and pass via `--data_dir`. 

- `sblw_samples_10min.mat`
- `sblm_samples_10min.mat`
- `tnbl_samples_10min.mat`

Example (using the local data folder in this repo):

```bash
python scripts/run_sbl.py --case weak --data_dir les_data_for_scm_uq/data
```

## Running experiments

Weakly stable case:

```bash
python scripts/run_sbl.py --case weak --epochs 100000 --lr 1e-4 --data_dir les_data_for_scm_uq/data
```

Moderately stable case:

```bash
python scripts/run_sbl.py --case moderate --epochs 100000 --lr 1e-4 --data_dir les_data_for_scm_uq/data
```

Truly neutral boundary layer case:

```bash
python scripts/run_sbl.py --case tnbl --epochs 100000 --lr 1e-4 --data_dir les_data_for_scm_uq/data
```


## Outputs

- Loss curves and profile comparisons
- Posterior plots and corner plot when B-ODIL runs (`sbl_<case>_corner.png`) (n_samples > 0)
- Uncertainty profiles (`sbl_<case>_uncertainty.png`)  (n_samples > 0)
- B-ODIL samples saved to `bodil_samples.npz`  (n_samples > 0)
