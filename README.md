# DopFlow

Minimal code for *DopFlow: Compact Factorized Image-Plane Motion Estimation From Multi-Link WiFi*.

## Install

```bash
pip install -r requirements.txt
```

## Quick check (no dataset)

```bash
python scripts/demo_inference.py
```

## Train and evaluate

Get the dataset from https://visinf.github.io/wiflow and place the birdview+ view at `data/WiFlow/birdviewplus/`.

```bash
python scripts/dopflow_cache.py
python scripts/dopflow_train.py --run S05_s0 --stage folds --steps 60000 --seed 0 \
  --train_subj S00,S01,S03,S04,S06,S07,S08,S09,S10 --eval_subjects S05 \
  --vsrc both --dropout 0.2 --lam_f 0.5 --lam_v 1 --tau 0.2 --offs -1,0,1
python scripts/dopflow_eval.py results/dopflow/folds/S05_s0.json S05 S05_s0_eval.json
```

For the other folds, hold out S07, S09 or S10 and use all other subjects except S02. The paper uses seeds 0, 1 and 2.
