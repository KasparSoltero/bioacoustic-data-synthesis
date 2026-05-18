import sys
from synthesiser.dataset import generate_dataset

def main():
    try:
        generate_dataset()
    except Exception as e:
        print(f"[Error] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()