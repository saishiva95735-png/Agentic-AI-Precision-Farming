import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_absolute_error, mean_squared_error

# Ensure output directory exists
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'confusion_matrices')
os.makedirs(output_dir, exist_ok=True)
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models')

print("Generating evaluation datasets and evaluating models...")

# -------------------------------------------------------------
# 1. MODEL 1: IRRIGATION CLASSIFIER & MODEL 2: WATER REQUIREMENT
# -------------------------------------------------------------
from train_comprehensive import generate_synthetic_data

df_irr = generate_synthetic_data(samples=3000)
# Use a distinct slice/seed for unbiased test evaluation
np.random.seed(2026)
test_indices = np.random.choice(df_irr.index, size=1000, replace=False)
test_df_irr = df_irr.loc[test_indices].copy()

X_irr = test_df_irr.drop(['Irrigation_Needed', 'Water_Requirement'], axis=1)
y_irr_true = test_df_irr['Irrigation_Needed']
y_water_true = test_df_irr['Water_Requirement']

# Load Model 1
clf_irr = joblib.load(os.path.join(models_dir, 'irrigation_model.pkl'))
y_irr_pred = clf_irr.predict(X_irr)

cm_irr = confusion_matrix(y_irr_true, y_irr_pred, labels=[0, 1])
acc_irr = accuracy_score(y_irr_true, y_irr_pred)
prec_irr = precision_score(y_irr_true, y_irr_pred)
rec_irr = recall_score(y_irr_true, y_irr_pred)
f1_irr = f1_score(y_irr_true, y_irr_pred)

print(f"\n[Model 1: Smart Irrigation Classifier]")
print(f"Accuracy: {acc_irr*100:.2f}% | Precision: {prec_irr*100:.2f}% | Recall: {rec_irr*100:.2f}% | F1: {f1_irr*100:.2f}%")
print("Confusion Matrix:\n", cm_irr)

# Load Model 2: Water Requirement Regressor
reg_water = joblib.load(os.path.join(models_dir, 'water_requirement_model.pkl'))
y_water_pred = reg_water.predict(X_irr)

r2_water = r2_score(y_water_true, y_water_pred)
mae_water = mean_absolute_error(y_water_true, y_water_pred)
rmse_water = np.sqrt(mean_squared_error(y_water_true, y_water_pred))

# Discretize Water Requirement into standard Agronomic Dosage Bins
# 0: Zero (0 mm)
# 1: Light (0.1 - 15 mm)
# 2: Moderate (15.1 - 30 mm)
# 3: Heavy (> 30 mm)
def bin_water_req(val):
    if val <= 0.5:
        return 0  # Zero
    elif val <= 15.0:
        return 1  # Light
    elif val <= 30.0:
        return 2  # Moderate
    else:
        return 3  # Heavy

water_bins_true = [bin_water_req(v) for v in y_water_true]
water_bins_pred = [bin_water_req(v) for v in y_water_pred]
water_labels = ['Zero (0mm)', 'Light (0-15mm)', 'Moderate (15-30mm)', 'Heavy (>30mm)']

cm_water = confusion_matrix(water_bins_true, water_bins_pred, labels=[0, 1, 2, 3])
acc_water = accuracy_score(water_bins_true, water_bins_pred)

print(f"\n[Model 2: Water Requirement Regressor]")
print(f"R²: {r2_water:.3f} | MAE: {mae_water:.2f} mm | RMSE: {rmse_water:.2f} mm")
print(f"Discretized Classification Accuracy: {acc_water*100:.2f}%")
print("Confusion Matrix:\n", cm_water)

# -------------------------------------------------------------
# 2. MODEL 3: CROP YIELD ESTIMATOR
# -------------------------------------------------------------
from train_yield_and_disease import generate_yield_dataset

df_yield = generate_yield_dataset(samples=3000)
np.random.seed(777)
test_indices_y = np.random.choice(df_yield.index, size=1000, replace=False)
test_df_yield = df_yield.loc[test_indices_y].copy()

X_yield = test_df_yield.drop('Yield_Tons_Per_Ha', axis=1)
y_yield_true = test_df_yield['Yield_Tons_Per_Ha']

# Base crop yields to categorize yield into Low, Normal, High
base_yields = {0: 4.8, 1: 4.2, 2: 6.2, 3: 2.4, 4: 1.6, 5: 75.0}

reg_yield = joblib.load(os.path.join(models_dir, 'yield_model.pkl'))
y_yield_pred = reg_yield.predict(X_yield)

r2_yield = r2_score(y_yield_true, y_yield_pred)
mae_yield = mean_absolute_error(y_yield_true, y_yield_pred)
rmse_yield = np.sqrt(mean_squared_error(y_yield_true, y_yield_pred))

# Yield tiers: Low (<85% of baseline), Expected (85%-110%), High (>110%)
def bin_yield(crop, yield_val):
    base = base_yields[crop]
    ratio = yield_val / base
    if ratio < 0.85:
        return 0  # Low / Sub-optimal
    elif ratio <= 1.10:
        return 1  # Expected / Normal
    else:
        return 2  # High / Optimal

crops = test_df_yield['Crop_Type'].values
yield_bins_true = [bin_yield(c, y) for c, y in zip(crops, y_yield_true)]
yield_bins_pred = [bin_yield(c, y) for c, y in zip(crops, y_yield_pred)]
yield_labels = ['Low (<85%)', 'Normal (85-110%)', 'High (>110%)']

cm_yield = confusion_matrix(yield_bins_true, yield_bins_pred, labels=[0, 1, 2])
acc_yield = accuracy_score(yield_bins_true, yield_bins_pred)

print(f"\n[Model 3: Crop Yield Estimator]")
print(f"R²: {r2_yield:.3f} | MAE: {mae_yield:.2f} t/ha | RMSE: {rmse_yield:.2f} t/ha")
print(f"Tier Discretized Accuracy: {acc_yield*100:.2f}%")
print("Confusion Matrix:\n", cm_yield)

# -------------------------------------------------------------
# 3. MODEL 4: CROP DISEASE & PATHOGEN RISK ESTIMATOR
# -------------------------------------------------------------
from train_yield_and_disease import generate_disease_dataset

df_disease = generate_disease_dataset(samples=3000)
np.random.seed(888)
test_indices_d = np.random.choice(df_disease.index, size=1000, replace=False)
test_df_disease = df_disease.loc[test_indices_d].copy()

X_disease = test_df_disease.drop(['Disease_Risk_Score', 'Disease_Risk_Class'], axis=1)
y_disease_true_score = test_df_disease['Disease_Risk_Score']
y_disease_true_class = test_df_disease['Disease_Risk_Class']

reg_disease = joblib.load(os.path.join(models_dir, 'disease_model.pkl'))
y_disease_pred_score = reg_disease.predict(X_disease)

r2_disease = r2_score(y_disease_true_score, y_disease_pred_score)
mae_disease = mean_absolute_error(y_disease_true_score, y_disease_pred_score)
rmse_disease = np.sqrt(mean_squared_error(y_disease_true_score, y_disease_pred_score))

def bin_disease(score):
    if score < 30:
        return 0  # Low
    elif score < 55:
        return 1  # Moderate
    elif score < 75:
        return 2  # High
    else:
        return 3  # Critical

disease_bins_pred = [bin_disease(s) for s in y_disease_pred_score]
disease_labels = ['Low (<30%)', 'Moderate (30-55%)', 'High (55-75%)', 'Critical (>75%)']

cm_disease = confusion_matrix(y_disease_true_class, disease_bins_pred, labels=[0, 1, 2, 3])
acc_disease = accuracy_score(y_disease_true_class, disease_bins_pred)

print(f"\n[Model 4: Crop Disease Risk Estimator]")
print(f"R²: {r2_disease:.3f} | MAE: {mae_disease:.2f}% | RMSE: {rmse_disease:.2f}%")
print(f"Risk Class Classification Accuracy: {acc_disease*100:.2f}%")
print("Confusion Matrix:\n", cm_disease)

# -------------------------------------------------------------
# PLOT AND SAVE CONFUSION MATRICES
# -------------------------------------------------------------
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
plt.subplots_adjust(wspace=0.3, hspace=0.35)

# 1. Smart Irrigation Classifier
sns.heatmap(cm_irr, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            xticklabels=['No Irrigation (0)', 'Irrigate (1)'],
            yticklabels=['No Irrigation (0)', 'Irrigate (1)'],
            cbar=False, annot_kws={"size": 14, "weight": "bold"})
axes[0, 0].set_title(f"Model 1: Smart Irrigation Classifier\n(Accuracy: {acc_irr*100:.1f}%, F1: {f1_irr*100:.1f}%)", fontsize=13, weight='bold', pad=10)
axes[0, 0].set_xlabel("Predicted Label", fontsize=11, weight='bold')
axes[0, 0].set_ylabel("True Ground Truth", fontsize=11, weight='bold')

# 2. Water Requirement Regressor (Binned)
sns.heatmap(cm_water, annot=True, fmt='d', cmap='Greens', ax=axes[0, 1],
            xticklabels=water_labels, yticklabels=water_labels,
            cbar=False, annot_kws={"size": 13, "weight": "bold"})
axes[0, 1].set_title(f"Model 2: Water Requirement Dosage Tiers\n(R²: {r2_water:.2f}, Discretized Acc: {acc_water*100:.1f}%)", fontsize=13, weight='bold', pad=10)
axes[0, 1].set_xlabel("Predicted Dosage Tier", fontsize=11, weight='bold')
axes[0, 1].set_ylabel("True Ground Truth Dosage", fontsize=11, weight='bold')

# 3. Crop Yield Estimator (Binned)
sns.heatmap(cm_yield, annot=True, fmt='d', cmap='Oranges', ax=axes[1, 0],
            xticklabels=yield_labels, yticklabels=yield_labels,
            cbar=False, annot_kws={"size": 13, "weight": "bold"})
axes[1, 0].set_title(f"Model 3: Crop Yield Performance Tiers\n(R²: {r2_yield:.2f}, Tier Acc: {acc_yield*100:.1f}%)", fontsize=13, weight='bold', pad=10)
axes[1, 0].set_xlabel("Predicted Harvest Tier", fontsize=11, weight='bold')
axes[1, 0].set_ylabel("True Ground Truth Harvest", fontsize=11, weight='bold')

# 4. Disease Risk Estimator (Risk Classes)
sns.heatmap(cm_disease, annot=True, fmt='d', cmap='Reds', ax=axes[1, 1],
            xticklabels=disease_labels, yticklabels=disease_labels,
            cbar=False, annot_kws={"size": 13, "weight": "bold"})
axes[1, 1].set_title(f"Model 4: Crop Disease Threat Levels\n(R²: {r2_disease:.2f}, Risk Level Acc: {acc_disease*100:.1f}%)", fontsize=13, weight='bold', pad=10)
axes[1, 1].set_xlabel("Predicted Threat Level", fontsize=11, weight='bold')
axes[1, 1].set_ylabel("True Ground Truth Threat", fontsize=11, weight='bold')

plt.suptitle("AgriMate AI Core ML Models — Confusion Matrix & Evaluation Suite", fontsize=16, weight='bold', y=0.98)
all_fig_path = os.path.join(output_dir, 'confusion_matrices_all.png')
plt.savefig(all_fig_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nSaved combined confusion matrices visualization to: {all_fig_path}")

# Save individual plots as well
individual_configs = [
    ("cm_irrigation.png", cm_irr, ['No Irrigation (0)', 'Irrigate (1)'], 'Blues', f"Model 1: Smart Irrigation Classifier (Acc: {acc_irr*100:.1f}%)", "Predicted", "Actual"),
    ("cm_water_requirement.png", cm_water, water_labels, 'Greens', f"Model 2: Water Requirement Dosage (Acc: {acc_water*100:.1f}%)", "Predicted Dosage", "Actual Dosage"),
    ("cm_yield.png", cm_yield, yield_labels, 'Oranges', f"Model 3: Crop Yield Outlook (Acc: {acc_yield*100:.1f}%)", "Predicted Harvest", "Actual Harvest"),
    ("cm_disease.png", cm_disease, disease_labels, 'Reds', f"Model 4: Crop Disease Threat Levels (Acc: {acc_disease*100:.1f}%)", "Predicted Risk", "Actual Risk")
]

for filename, cm, labels, cmap, title, xlabel, ylabel in individual_configs:
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=labels, yticklabels=labels, annot_kws={"size": 14, "weight": "bold"})
    plt.title(title, fontsize=12, weight='bold', pad=12)
    plt.xlabel(xlabel, fontsize=11, weight='bold')
    plt.ylabel(ylabel, fontsize=11, weight='bold')
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

# Also copy to frontend static / public so it can be viewed in browser if desired
public_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'frontend', 'public')
if os.path.exists(public_dir):
    import shutil
    shutil.copy(all_fig_path, os.path.join(public_dir, 'confusion_matrices_all.png'))
    print(f"Copied to frontend/public for browser inspection: {os.path.join(public_dir, 'confusion_matrices_all.png')}")

print("\nEvaluation successfully completed!")

