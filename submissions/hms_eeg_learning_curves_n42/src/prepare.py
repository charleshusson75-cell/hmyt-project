"""Run once: HMS parquet files -> X.npy, F.npy, meta.csv, prep_config.json.

    python prepare.py --comp /kaggle/input/hms-harmful-brain-activity-classification \\
                      --out /kaggle/working/hms-prepared

--limit 200 does a quick smoke test on 200 recordings first. No GPU needed.
"""
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--comp', required=True, help='competition folder (train.csv, train_eegs/)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--jobs', type=int, default=-1)
    a = ap.parse_args()
    import hms_data
    hms_data.prepare(a.comp, a.out, seed=a.seed, limit=a.limit, n_jobs=a.jobs)
