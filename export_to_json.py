import json
import pickle
from pathlib import Path

def export_models():
    model_path = Path("model.pkl")
    scaler_path = Path("scaler.pkl")
    
    if not model_path.exists() or not scaler_path.exists():
        print("Error: model.pkl or scaler.pkl not found. Please run train.py first.")
        return

    with open(model_path, "rb") as f:
        models = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Convert numpy arrays to list for JSON serialization
    export_data = {
        "scaler": {
            "mean": scaler["mean"].tolist(),
            "std": scaler["std"].tolist(),
            "features": scaler.get("features", [])
        },
        "models": {
            "max": {
                "weights": models["max"].weights.tolist(),
                "bias": float(models["max"].bias)
            },
            "min": {
                "weights": models["min"].weights.tolist(),
                "bias": float(models["min"].bias)
            },
            "mean": {
                "weights": models["mean"].weights.tolist(),
                "bias": float(models["mean"].bias)
            }
        }
    }

    output_path = Path("model.json")
    with open(output_path, "w") as f:
        json.dump(export_data, f, indent=4)
        
    print(f"Successfully exported model and scaler parameters to {output_path.resolve()}")

if __name__ == "__main__":
    export_models()
