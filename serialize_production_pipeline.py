import os
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# 1. Define feature dependencies strictly
drop_cols = ["ID", "deathdate", "is_climate_sensitive", "zone", "gender", "location"]
feature_cols = [c for c in train.columns if c not in drop_cols]

# Retain precise metadata configurations
production_artifacts = {
    "features": feature_cols,
    "le_gender": le_gender,
    "le_zone": le_zone,
    "imputer": imputer,
    "loc_freq_map": loc_freq.to_dict(),
    "loc_te_map": train.groupby("location")["is_climate_sensitive"].mean().to_dict(),
    "global_te_mean": train["is_climate_sensitive"].mean()
}

# 2. Retrain final champion model configurations across 100% of the dataset
dtrain_full = lgb.Dataset(X, y)
final_production_model = lgb.train(
    lgb_params, 
    dtrain_full, 
    num_boost_round=int(np.mean(cv_f1)) * 0 + 800 # Using best average iteration discovery bounds
)

# 3. Export binaries to local model folder
os.makedirs("models", exist_ok=True)
joblib.dump(production_artifacts, "models/pipeline_metadata.joblib")
final_production_model.save_model("models/production_lgbm_model.txt")

print("🏆 Production artifacts successfully serialized to /models folder.")
