import sys
import random
import argparse

import numpy as np
import torch

from bioacoustic_synthesis.dataset import generate_dataset

def main():
    parser = argparse.ArgumentParser(description="Bioacoustic dataset synthesis pipeline")
    parser.add_argument('-c', '--config', default='config.yaml',
                        help="Path to the synthesis config YAML")
    parser.add_argument('-i', '--interactive', action='store_true',
                         help="Review each generated sample interactively (hotkeys: n=next, b=toggle boxes, m=toggle masks, space=play audio, q=quit)")
    parser.add_argument('-s', '--seed', type=int, default=None,
                        help="Global RNG seed. Omitted, one is drawn at random and recorded "
                             "in the output's generation_config.yaml — repeat runs stay "
                             "independent while remaining reproducible after the fact.")
    parser.add_argument('--sample-seed', type=int, default=None,
                        help="Separate seed for per-species source-file selection. Holding this "
                             "fixed across a limit_per_class sweep gives nested subsets (2 in 4 "
                             "in 6); leave unset for independent draws at each sweep point.")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.randrange(2**32)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    print(f"[seed] {seed}")

    try:
        generate_dataset(config_path=args.config, interactive=args.interactive,
                         sample_seed=args.sample_seed, seed=seed)
    except Exception as e:
        print(f"[Error] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()