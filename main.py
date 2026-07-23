import sys
import argparse
from synthesiser.dataset import generate_dataset

def main():
    parser = argparse.ArgumentParser(description="Bioacoustic dataset synthesis pipeline")
    parser.add_argument('-i', '--interactive', action='store_true',
                         help="Review each generated sample interactively (hotkeys: n=next, b=toggle boxes, m=toggle masks, space=play audio, q=quit)")
    args = parser.parse_args()

    try:
        generate_dataset(interactive=args.interactive)
    except Exception as e:
        print(f"[Error] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()