import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from lightgbm import LGBMClassifier

# Load merged/cleaned data (assumed CSV path)
df = pd.read_csv("merged_aibl.csv")

# Target encoding: 0 = HC, 1 = NON HC
y = df['DXCURREN']
# Selected features from notebook
features = ['RCT11', 'HMT40', 'RCT6', 'HMT13', 'MH9ENDO',
            'LIMMTOTAL', 'MMSCORE', 'AXT117', 'RCT392',
            'HMT100', 'HMT7', 'age', 'CDGLOBAL', 'BAT126',
            'HMT102', 'LDELTOTAL', 'RCT20']
X = df[features]

# Replace sentinel values (-4) and impute missing with median
X = X.replace(-4, np.nan)
X = X.fillna(X.median())

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Model
model = LGBMClassifier(n_estimators=500, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

# Save model
import joblib
joblib.dump(model, "lgbm_baseline.pkl")
print("Model saved as lgbm_baseline.pkl")
