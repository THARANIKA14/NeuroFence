# scripts/run_viz_pipeline.py
import sys
from pathlib import Path

# Add src to path so imports work
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.dashboard import generate_dashboard

def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/run_viz_pipeline.py <input_json> <output_dir>")
        sys.exit(1)

    json_path = sys.argv[1]
    output_dir = sys.argv[2]

    result = generate_dashboard(json_path, output_dir)
    print("Dashboard generated:")
    print(f"  Model: {result['model_name']}")
    print(f"  Output dir: {result['output_dir']}")
    print(f"  Anomalies saved in: {Path(result['output_dir']) / 'anomalies.txt'}")

if __name__ == "__main__":
    main()