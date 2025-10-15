# src/train.py
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

def train_model():
    # Prepare paths
    project_root = Path(__file__).resolve().parents[1]   # project root (one level above src/)
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot and save confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    cm_path = outputs_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()  # close figure to free memory

    # Save model
    model_path = outputs_dir / "model.joblib"
    joblib.dump(model, model_path)

    print(f"✅ Saved confusion matrix to: {cm_path}")
    print(f"✅ Saved trained model to: {model_path}")

if __name__ == "__main__":
    train_model()


