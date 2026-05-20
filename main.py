import os
import argparse
from dataset import run_linear_scan

DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets')

DATASETS = {
    'sift':      'sift',
    'siftsmall': 'siftsmall',
}


def main():
    ap = argparse.ArgumentParser(description='Linear scan range query benchmark')
    ap.add_argument('--dataset', choices=list(DATASETS), default='siftsmall',
                    help='dataset to use (default: siftsmall)')
    ap.add_argument('--radius', type=int, default=20,
                    help='Hamming radius for range query (default: 20)')
    args = ap.parse_args()

    prefix      = DATASETS[args.dataset]
    dataset_dir = os.path.join(DATASETS_DIR, prefix)

    run_linear_scan(dataset_dir, prefix, args.radius)


if __name__ == '__main__':
    main()
