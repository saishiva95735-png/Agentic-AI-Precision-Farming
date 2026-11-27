import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split

def generate_yield_dataset(samples=5000):
    np.random.seed(42)
    
    crop_type = np.random.randint(0, 6, samples)  # 0: Rice, 1: Wheat, 2: Maize, 3: Cotton, 4: Pulse, 5: Sugarcane
    growth_stage = np.random.randint(0, 4, samples) # 0: Initial, 1: Vegetative, 2: Flowering, 3: Maturity
    soil_type = np.random.randint(0, 3, samples)    # 0: Sandy, 1: Loamy, 2: Clay
    soil_moisture = np.random.uniform(10, 85, samples)
    ndvi = np.random.uniform(0.15, 0.92, samples)
    temp = np.random.uniform(15, 45, samples)
    humidity = np.random.uniform(20, 95, samples)
    rainfall = np.random.uniform(0, 30, samples)
    sunlight = np.random.uniform(4, 12, samples)
    prev_irrigation = np.random.uniform(0, 50, samples)
    
    # Base yields per crop (tons/ha)
    base_yields = {
        0: 4.8,  # Rice
        1: 4.2,  # Wheat
        2: 6.2,  # Maize
        3: 2.4,  # Cotton
        4: 1.6,  # Pulse
        5: 75.0  # Sugarcane
    }
    
    yields = []
    for i in range(samples):
        base = base_yields[crop_type[i]]
        
        # NDVI multiplier: healthy canopy (0.7+) boost yield, low ndvi degrades yield
        ndvi_factor = 0.55 + (ndvi[i] * 0.7)  # range ~0.65 to ~1.20
        
        # Moisture factor: optimal 40-55%
        m = soil_moisture[i]
        if 40 <= m <= 55:
            moisture_factor = 1.10
        elif 30 <= m < 40 or 55 < m <= 65:
            moisture_factor = 1.00
        elif 20 <= m < 30:
            moisture_factor = 0.85
        elif m < 20:
            moisture_factor = 0.65  # severe drought
        else: # m > 65
            moisture_factor = 0.88  # waterlogging
            
        # Temperature stress factor: optimal 22-30C
        t = temp[i]
        if 22 <= t <= 30:
            temp_factor = 1.05
        elif 30 < t <= 36:
            temp_factor = 0.95
        elif t > 36:
            temp_factor = 0.80  # heat stress
        else:
            temp_factor = 0.90  # cold/slow growth
            
        # Sunlight factor: 7-10 hours optimal
        s = sunlight[i]
        sun_factor = 0.9 + (min(s, 10) / 10.0) * 0.15
        
        # Soil type factor: Loamy (1) best, Clay (2) good for rice, Sandy (0) lower water retention
        st = soil_type[i]
        if st == 1:
            soil_factor = 1.05
        elif st == 2 and crop_type[i] == 0:  # Clay is great for paddy
            soil_factor = 1.08
        else:
            soil_factor = 0.92
            
        # Noise
        noise = np.random.normal(0, 0.05 * base)
        
        final_yield = base * ndvi_factor * moisture_factor * temp_factor * sun_factor * soil_factor + noise
        final_yield = max(0.2 * base, final_yield)
        yields.append(round(final_yield, 2))
        
    df = pd.DataFrame({
        'Crop_Type': crop_type,
        'Crop_Growth_Stage': growth_stage,
        'Soil_Type': soil_type,
        'Soil_Moisture': soil_moisture,
        'NDVI': ndvi,
        'Temperature_C': temp,
        'Humidity': humidity,
        'Rainfall_mm': rainfall,
        'Sunlight_Hours': sunlight,
        'Previous_Irrigation_mm': prev_irrigation,
        'Yield_Tons_Per_Ha': yields
    })
    return df


def generate_disease_dataset(samples=5000):
    np.random.seed(99)
    
    crop_type = np.random.randint(0, 6, samples)
    growth_stage = np.random.randint(0, 4, samples)
    temp = np.random.uniform(14, 42, samples)
    humidity = np.random.uniform(25, 98, samples)
    rainfall = np.random.uniform(0, 35, samples)
    soil_moisture = np.random.uniform(10, 85, samples)
    ndvi = np.random.uniform(0.15, 0.9, samples)
    wind_speed = np.random.uniform(2, 35, samples)
    
    risk_scores = []
    risk_classes = [] # 0: Low, 1: Moderate, 2: High, 3: Critical
    
    for i in range(samples):
        score = 10.0 # base ambient risk
        
        # Humidity is primary driver of fungal/bacterial proliferation
        h = humidity[i]
        if h > 85:
            score += 40
        elif h > 75:
            score += 28
        elif h > 65:
            score += 15
        elif h < 40:
            score -= 5
            
        # Temperature optimal for fungal growth (20C - 30C)
        t = temp[i]
        if 22 <= t <= 29:
            score += 25
        elif 18 <= t < 22 or 29 < t <= 34:
            score += 15
        elif t > 38 or t < 15:
            score -= 10
            
        # Rainfall / Leaf wetness
        r = rainfall[i]
        if r > 10:
            score += 20
        elif r > 2:
            score += 12
            
        # Waterlogged soil causes root rot / damping off
        sm = soil_moisture[i]
        if sm > 65:
            score += 15
            
        # Weakened crop (low NDVI) has higher susceptibility
        nv = ndvi[i]
        if nv < 0.35:
            score += 10
            
        # Wind helps spore dispersal if humidity is high
        w = wind_speed[i]
        if w > 15 and h > 70:
            score += 8
            
        # Growth stage vulnerability: Flowering & vegetative most vulnerable
        gs = growth_stage[i]
        if gs in [1, 2]:
            score += 8
            
        # Add random natural fluctuation
        score += np.random.normal(0, 4)
        score = np.clip(score, 2.0, 98.0)
        risk_scores.append(round(score, 1))
        
        if score < 30:
            risk_classes.append(0)  # Low
        elif score < 55:
            risk_classes.append(1)  # Moderate
        elif score < 75:
            risk_classes.append(2)  # High
        else:
            risk_classes.append(3)  # Critical
            
    df = pd.DataFrame({
        'Crop_Type': crop_type,
        'Crop_Growth_Stage': growth_stage,
        'Temperature_C': temp,
        'Humidity': humidity,
        'Rainfall_mm': rainfall,
        'Soil_Moisture': soil_moisture,
        'NDVI': ndvi,
        'Wind_Speed_kmh': wind_speed,
        'Disease_Risk_Score': risk_scores,
        'Disease_Risk_Class': risk_classes
    })
    return df


def train_models():
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models')
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Train Yield Model
    print("Generating yield dataset...")
    df_yield = generate_yield_dataset(6000)
    X_y = df_yield.drop('Yield_Tons_Per_Ha', axis=1)
    y_y = df_yield['Yield_Tons_Per_Ha']
    
    X_train_y, X_test_y, y_train_y, y_test_y = train_test_split(X_y, y_y, test_size=0.2, random_state=42)
    yield_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    yield_model.fit(X_train_y, y_train_y)
    r2 = yield_model.score(X_test_y, y_test_y)
    print(f"Yield Model R2 Score: {r2:.3f}")
    
    yield_path = os.path.join(save_dir, 'yield_model.pkl')
    joblib.dump(yield_model, yield_path)
    print(f"Saved Yield Model to {yield_path}")
    
    # 2. Train Disease Model
    print("\nGenerating disease risk dataset...")
    df_disease = generate_disease_dataset(6000)
    X_d = df_disease.drop(['Disease_Risk_Score', 'Disease_Risk_Class'], axis=1)
    y_d_score = df_disease['Disease_Risk_Score']
    y_d_class = df_disease['Disease_Risk_Class']
    
    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_d, y_d_score, test_size=0.2, random_state=42)
    disease_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    disease_model.fit(X_train_d, y_train_d)
    r2_d = disease_model.score(X_test_d, y_test_d)
    print(f"Disease Risk Regressor R2 Score: {r2_d:.3f}")
    
    disease_path = os.path.join(save_dir, 'disease_model.pkl')
    joblib.dump(disease_model, disease_path)
    print(f"Saved Disease Model to {disease_path}")
    
    print("\nAll models successfully trained and persisted!")

if __name__ == '__main__':
    train_models()
