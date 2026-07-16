import sys
import subprocess
from synthesiser.dataset import generate_dataset

def main():
    try:
        for limit in range(1, 9):
            print(f"\n\n{'='*80}")
            print(f"Starting Sweep Iteration: limit_per_class = {limit}")
            print(f"{'='*80}\n")
            
            # 1. Synthesise dataset
            generate_dataset(limit_per_class=limit)
            
            # 2. Train model via subprocess (to avoid memory leaks between runs)
            config_path = 'classifiers/recogniser/config-recogniser.yaml'
            output_dir = f'classifiers/recogniser/models/eval-recogniser-sweep-min{limit}'
            
            print(f"\n--- Training Recogniser for limit={limit} ---")
            cmd = [sys.executable, "classifiers/recogniser/train-recogniser.py", "--config", config_path, "--output_dir", output_dir]
            subprocess.run(cmd, check=True)
            
    except Exception as e:
        print(f"[Error] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()