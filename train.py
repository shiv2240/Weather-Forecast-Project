import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set headless backend so plt.show() doesn't block in background
import matplotlib.pyplot as plt
import logging
import pickle
from pathlib import Path
from model import LinearRegressionFromScratch

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(filename="weather_data.csv"):
    """Loads weather data from CSV and returns X (features) and y_max, y_min, y_mean (targets) using Path."""
    path = Path(filename)
    if not path.exists():
        logger.error(f"Dataset file {path.resolve()} does not exist.")
        raise FileNotFoundError(f"Missing file: {path}")
        
    # Load all numerical columns using numpy. Skip the first header row and the first column (Date).
    data = np.loadtxt(path, delimiter=",", skiprows=1, usecols=range(1, 13))
    
    # First 9 columns are features:
    # Today_Max_Temperature, Today_Min_Temperature, Today_Mean_Temperature, Today_Humidity, Today_Pressure, Today_Wind_Speed, Today_Rainfall, Today_Cloud_Cover, Today_UV_Index
    X = data[:, 0:9]
    # Targets are columns 9, 10, 11
    y_max = data[:, 9]
    y_min = data[:, 10]
    y_mean = data[:, 11]
    
    return X, y_max, y_min, y_mean

def train_test_split_from_scratch(X, y, test_size=0.2, random_seed=None):
    """Splits dataset into training and testing sets from scratch, supporting seed=None."""
    if random_seed is not None:
        np.random.seed(random_seed)
        
    num_samples = len(X)
    # Generate shuffled indices
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    # Calculate split point
    split_idx = int((1.0 - test_size) * num_samples)
    
    # Select indices
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    # Slice the data
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    return X_train, X_test, y_train, y_test

def main(seed=42):
    csv_file = Path("weather_data.csv")
    if not csv_file.exists():
        logger.error("weather_data.csv not found! Please run download_data.py or generate_data.py first.")
        return
        
    # 1. Load data
    X, y_max, y_min, y_mean = load_data(csv_file)
    logger.info(f"Dataset loaded from {csv_file.resolve()}. X shape: {X.shape}")
    
    # Column stack targets so they split synchronously
    y = np.column_stack((y_max, y_min, y_mean))
    
    # 2. Split data: 60% train, 20% validation, 20% test
    # First split: 80% train_full, 20% test
    X_train_full, X_test, y_train_full, y_test = train_test_split_from_scratch(X, y, test_size=0.2, random_seed=seed)
    
    # Second split: split train_full into 75% train and 25% validation (resulting in 60% train, 20% val of original)
    X_train, X_val, y_train, y_val = train_test_split_from_scratch(X_train_full, y_train_full, test_size=0.25, random_seed=seed)
    
    logger.info(f"Splits created:")
    logger.info(f"  Train set:      X_train={X_train.shape}, y_train={y_train.shape}")
    logger.info(f"  Validation set: X_val={X_val.shape}, y_val={y_val.shape}")
    logger.info(f"  Test set:       X_test={X_test.shape}, y_test={y_test.shape}")
    
    # 3. Feature Scaling (Standardization / Z-score normalization)
    mean_train = np.mean(X_train, axis=0)
    std_train = np.std(X_train, axis=0)
    
    # Prevent division by zero
    std_train = np.where(std_train == 0.0, 1.0, std_train)
    
    # Scale all sets
    X_train_scaled = (X_train - mean_train) / std_train
    X_val_scaled = (X_val - mean_train) / std_train
    X_test_scaled = (X_test - mean_train) / std_train
    
    feature_names = [
        "Today_Max_Temperature", 
        "Today_Min_Temperature", 
        "Today_Mean_Temperature", 
        "Today_Humidity", 
        "Today_Pressure", 
        "Today_Wind_Speed", 
        "Today_Rainfall", 
        "Today_Cloud_Cover", 
        "Today_UV_Index"
    ]
    
    logger.info("Scaling metrics computed and applied.")
    for i, name in enumerate(feature_names):
        logger.info(f"  {name:22s} -> Mean: {mean_train[i]:10.4f} | Std: {std_train[i]:10.4f}")
        
    # Save the scaler values to scaler.pkl
    scaler_path = Path("scaler.pkl")
    scaler_data = {
        "mean": mean_train,
        "std": std_train,
        "features": feature_names
    }
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler_data, f)
    logger.info(f"Scaler parameters saved to {scaler_path.resolve()}")
    
    # 4. Instantiate and Train Models
    logger.info("--- Training Tomorrow_Max_Temperature Model ---")
    model_max = LinearRegressionFromScratch(learning_rate=0.1, epochs=500)
    model_max.fit(X_train_scaled, y_train[:, 0], X_val=X_val_scaled, y_val=y_val[:, 0])
    
    logger.info("--- Training Tomorrow_Min_Temperature Model ---")
    model_min = LinearRegressionFromScratch(learning_rate=0.1, epochs=500)
    model_min.fit(X_train_scaled, y_train[:, 1], X_val=X_val_scaled, y_val=y_val[:, 1])
    
    logger.info("--- Training Tomorrow_Mean_Temperature Model ---")
    model_mean = LinearRegressionFromScratch(learning_rate=0.1, epochs=500)
    model_mean.fit(X_train_scaled, y_train[:, 2], X_val=X_val_scaled, y_val=y_val[:, 2])
    
    # 5. Evaluate the models on the test set
    y_pred_max = model_max.predict(X_test_scaled)
    y_pred_min = model_min.predict(X_test_scaled)
    y_pred_mean = model_mean.predict(X_test_scaled)
    
    def calculate_metrics(y_true, y_pred):
        mae = np.mean(np.abs(y_pred - y_true))
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
        ss_residual = np.sum((y_true - y_pred) ** 2)
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1.0 - (ss_residual / ss_total)
        acc_1c = 100.0 * np.mean(np.abs(y_true - y_pred) <= 1.0)
        acc_1_5c = 100.0 * np.mean(np.abs(y_true - y_pred) <= 1.5)
        return mae, rmse, r2, acc_1c, acc_1_5c
    
    metrics_max = calculate_metrics(y_test[:, 0], y_pred_max)
    metrics_min = calculate_metrics(y_test[:, 1], y_pred_min)
    metrics_mean = calculate_metrics(y_test[:, 2], y_pred_mean)
    
    # Save all three models and metrics in model.pkl
    model_path = Path("model.pkl")
    models_dict = {
        "max": model_max,
        "min": model_min,
        "mean": model_mean,
        "metrics": {
            "max": metrics_max,
            "min": metrics_min,
            "mean": metrics_mean
        }
    }
    with open(model_path, "wb") as f:
        pickle.dump(models_dict, f)
    logger.info(f"Trained models and metrics saved to {model_path.resolve()}")
    
    logger.info("--- Test Set Evaluation ---")
    headers_eval = f"{'Target':22s} | {'MAE (°C)':9s} | {'RMSE (°C)':10s} | {'R2 Score':8s} | {'Acc (±1.0°C)':12s} | {'Acc (±1.5°C)':12s}"
    logger.info(headers_eval)
    logger.info("-" * len(headers_eval))
    logger.info(f"{'Tomorrow_Max_Temp':22s} | {metrics_max[0]:9.4f} | {metrics_max[1]:10.4f} | {metrics_max[2]:8.4f} | {metrics_max[3]:11.2f}% | {metrics_max[4]:11.2f}%")
    logger.info(f"{'Tomorrow_Min_Temp':22s} | {metrics_min[0]:9.4f} | {metrics_min[1]:10.4f} | {metrics_min[2]:8.4f} | {metrics_min[3]:11.2f}% | {metrics_min[4]:11.2f}%")
    logger.info(f"{'Tomorrow_Mean_Temp':22s} | {metrics_mean[0]:9.4f} | {metrics_mean[1]:10.4f} | {metrics_mean[2]:8.4f} | {metrics_mean[3]:11.2f}% | {metrics_mean[4]:11.2f}%")
    
    # 6. Translate weights back to original unscaled feature space for explanation
    logger.info("--- Learned Coefficients (Unscaled Formulas) ---")
    for model, label in zip([model_max, model_min, model_mean], ["Max", "Min", "Mean"]):
        w_unscaled = model.weights / std_train
        b_unscaled = model.bias - np.sum((model.weights * mean_train) / std_train)
        
        formula_terms = []
        for i, name in enumerate(feature_names):
            formula_terms.append(f"{w_unscaled[i]:.4f} * {name}")
        formula = f"Tomorrow_{label}_Temp = " + " + ".join(formula_terms) + f" + {b_unscaled:.4f}"
        logger.info(formula)
        
    # 7. Visualization
    logger.info("Saving plots to prediction_plot.png...")
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    
    model_titles = ["Tomorrow Max Temperature", "Tomorrow Min Temperature", "Tomorrow Mean Temperature"]
    models_list = [model_max, model_min, model_mean]
    preds_list = [y_pred_max, y_pred_min, y_pred_mean]
    actuals_list = [y_test[:, 0], y_test[:, 1], y_test[:, 2]]
    
    colors_loss = ["#ef4444", "#2563eb", "#10b981"]
    
    for idx, (model, title) in enumerate(zip(models_list, model_titles)):
        # Plot Loss Curves
        ax_loss = axes[0, idx]
        epochs_axis = range(len(model.loss_history))
        ax_loss.plot(epochs_axis, model.loss_history, label="Train Loss (MSE)", color=colors_loss[idx], linewidth=2.5)
        if model.val_loss_history:
            ax_loss.plot(epochs_axis, model.val_loss_history, label="Val Loss (MSE)", color="#eab308", linestyle="--", linewidth=2)
        ax_loss.set_title(f"{title} - Loss History", fontsize=12, fontweight="bold", pad=8)
        ax_loss.set_xlabel("Epoch", fontsize=10)
        ax_loss.set_ylabel("Loss (MSE)", fontsize=10)
        ax_loss.legend(fontsize=9)
        ax_loss.grid(True, linestyle="--", alpha=0.5)
        
        # Plot Predictions vs Actual (for first 50 days)
        ax_pred = axes[1, idx]
        subset_len = 50
        days_axis = np.arange(subset_len)
        ax_pred.plot(days_axis, actuals_list[idx][:subset_len], label="Actual Temp", color="#4b5563", marker="o", linewidth=1.8)
        ax_pred.plot(days_axis, preds_list[idx][:subset_len], label="Predicted Temp", color=colors_loss[idx], linestyle="--", marker="x", linewidth=1.8)
        ax_pred.set_title(f"{title} - Forecast Comparison", fontsize=12, fontweight="bold", pad=8)
        ax_pred.set_xlabel("Day Index", fontsize=10)
        ax_pred.set_ylabel("Temperature (°C)", fontsize=10)
        ax_pred.legend(fontsize=9)
        ax_pred.grid(True, linestyle="--", alpha=0.5)
        
    plt.tight_layout()
    plot_path = Path("prediction_plot.png")
    plt.savefig(plot_path, dpi=300)
    logger.info(f"Plots saved to {plot_path.resolve()}")
    plt.close()

if __name__ == "__main__":
    # To run without a fixed seed, call main(seed=None)
    main(seed=42)
