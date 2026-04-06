import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, roc_curve, auc
import warnings

# Ignore minor warnings
warnings.filterwarnings("ignore")

def load_data(filepath):
    """Load the bank marketing dataset."""
    df = pd.read_csv(filepath)
    print(f"Data loaded successfully with shape: {df.shape}")
    return df

def preprocess_data(df):
    """Clean and encode data for modeling."""
    # Mapping binary categorical variables
    binary_map = {"yes": 1, "no": 0}
    df["deposit"] = df["deposit"].map(binary_map)
    df["default"] = df["default"].map(binary_map)
    df["housing"] = df["housing"].map(binary_map)
    df["loan"] = df["loan"].map(binary_map)
    
    # One-hot encoding for other categorical variables
    df_encoded = pd.get_dummies(df, drop_first=True)
    print(f"Data size after encoding: {df_encoded.shape}")
    return df_encoded

def train_models(X_train, y_train):
    """Train multiple models and compare accuracy."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        acc = model.score(X_train, y_train) # Quick check on train accuracy
        results[name] = acc
        print(f"Trained {name}")
        
    return trained_models, results

def evaluate_best_model(model, X_test, y_test, model_name):
    """Evaluate performance, plot confusion matrix and ROC curve."""
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    
    print(f"\n=== Evaluation: {model_name} ===")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Plot Confusion Matrix
    disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig("images/confusion_matrix.png")
    plt.close()
    
    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})', color='tab:green', linewidth=1.5)
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig("images/roc_curve.png")
    plt.close()
    
    print("Visualizations saved to images/ directory.")

def plot_feature_importance(model, X_train):
    """Plot the top 10 most important features from Random Forest."""
    importances = model.feature_importances_
    feat_importance = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    feat_importance.head(10).plot(kind='bar', color='orange')
    plt.title('Top 10 Important Features (Random Forest)')
    plt.ylabel('Importance Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("images/feature_importance.png")
    plt.close()

def run_shap_analysis(model, X_test):
    """Run SHAP analysis to explain model predictions."""
    print("\nInitializing SHAP Explainer...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values[:, :, 1] if len(shap_values.shape) == 3 else shap_values[1], X_test, show=False)
    plt.title("SHAP Feature Impact on Deposit Subscription", fontsize=15)
    plt.tight_layout()
    plt.savefig("images/shap_summary.png")
    plt.close()
    print("SHAP plot saved to images/shap_summary.png.")

def main():
    # 1. Load and Preprocess
    df = load_data("data/bank.csv")
    df_encoded = preprocess_data(df)
    
    # 2. Split Data
    X = df_encoded.drop("deposit", axis=1)
    y = df_encoded["deposit"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Train
    models, results = train_models(X_train, y_train)
    best_model_name = "Random Forest" # Based on notebook results
    best_model = models[best_model_name]
    
    # 4. Evaluate and Save Visuals
    evaluate_best_model(best_model, X_test, y_test, best_model_name)
    plot_feature_importance(best_model, X_train)
    run_shap_analysis(best_model, X_test)
    
    # 5. Business Insight Simulation
    print("\n=== Business Simulation: Top Priorities ===")
    sample_probs = best_model.predict_proba(X_test.head(5))[:, 1]
    for i, prob in enumerate(sample_probs):
        priority = "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low"
        print(f"Customer {i+1}: Probability {prob:.2f} -> Priority: {priority}")

if __name__ == "__main__":
    main()
