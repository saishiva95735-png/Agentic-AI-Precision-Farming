import joblib
import pandas as pd
import os
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

# ─── Robust path resolution ─────────────────────────────────────────────────
# This file lives at: backend/app/services/prediction.py
# Models live at:     <project_root>/ml_model/saved_models/<name>.pkl
# Probe several candidate roots so the backend works regardless of where
# uvicorn/gunicorn is invoked from (project root OR backend/ directory).

_THIS_FILE = os.path.abspath(__file__)          # …/backend/app/services/prediction.py
_BACKEND_DIR = os.path.dirname(              # …/backend
    os.path.dirname(os.path.dirname(_THIS_FILE))
)
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)   # …/iot  (git root)

def _probe_model_path(rel_candidates: list) -> Optional[str]:
    """Return the first existing path from relative candidates against several roots."""
    search_roots = [_PROJECT_ROOT, _BACKEND_DIR, os.getcwd()]
    for root in search_roots:
        for candidate in rel_candidates:
            full = os.path.normpath(os.path.join(root, candidate))
            if os.path.exists(full):
                logger.info(f"Model found at: {full}")
                return full
    logger.warning(
        f"Model not found. Searched candidates {rel_candidates} "
        f"under roots {search_roots}"
    )
    return None


IRRIGATION_MODEL_PATH = _probe_model_path([
    "ml_model/saved_models/irrigation_model.pkl",
    "ml_model/saved_models/irrigation_model(1).pkl",
    "iot/ml_model/saved_models/irrigation_model.pkl",
    "iot/ml_model/saved_models/irrigation_model(1).pkl",
    "iot/ml_model/irrigation_model(1).pkl",
    "saved_models/irrigation_model.pkl",
])

WATER_REQUIREMENT_MODEL_PATH = _probe_model_path([
    "ml_model/saved_models/water_requirement_model.pkl",
    "iot/ml_model/saved_models/water_requirement_model.pkl",
    "iot/ml_model/water_requirement_model.pkl",
    "saved_models/water_requirement_model.pkl",
])

YIELD_MODEL_PATH = _probe_model_path([
    "ml_model/saved_models/yield_model.pkl",
    "iot/ml_model/saved_models/yield_model.pkl",
    "saved_models/yield_model.pkl",
])

DISEASE_MODEL_PATH = _probe_model_path([
    "ml_model/saved_models/disease_model.pkl",
    "iot/ml_model/saved_models/disease_model.pkl",
    "saved_models/disease_model.pkl",
])

class IrrigationPredictor:
    def __init__(self):
        self.irrigation_model = None
        self.water_requirement_model = None
        self.yield_model = None
        self.disease_model = None
        
        # Load irrigation model
        try:
            if IRRIGATION_MODEL_PATH and os.path.exists(IRRIGATION_MODEL_PATH):
                self.irrigation_model = joblib.load(IRRIGATION_MODEL_PATH)
                logger.info(f"Irrigation model loaded successfully from {IRRIGATION_MODEL_PATH}")
            else:
                logger.warning("Irrigation model not loaded — falling back to rule-based decisions")
        except Exception as e:
            logger.error(f"Error loading irrigation model: {str(e)}")
        
        # Load water requirement model
        try:
            if WATER_REQUIREMENT_MODEL_PATH and os.path.exists(WATER_REQUIREMENT_MODEL_PATH):
                self.water_requirement_model = joblib.load(WATER_REQUIREMENT_MODEL_PATH)
                logger.info(f"Water requirement model loaded successfully from {WATER_REQUIREMENT_MODEL_PATH}")
            else:
                logger.warning("Water requirement model not loaded — will return zero estimates")
        except Exception as e:
            logger.error(f"Error loading water requirement model: {str(e)}")

        # Load yield model
        try:
            if YIELD_MODEL_PATH and os.path.exists(YIELD_MODEL_PATH):
                self.yield_model = joblib.load(YIELD_MODEL_PATH)
                logger.info(f"Yield estimation model loaded successfully from {YIELD_MODEL_PATH}")
            else:
                logger.warning("Yield model not loaded — using agronomic benchmark estimation")
        except Exception as e:
            logger.error(f"Error loading yield model: {str(e)}")

        # Load disease risk model
        try:
            if DISEASE_MODEL_PATH and os.path.exists(DISEASE_MODEL_PATH):
                self.disease_model = joblib.load(DISEASE_MODEL_PATH)
                logger.info(f"Disease risk model loaded successfully from {DISEASE_MODEL_PATH}")
            else:
                logger.warning("Disease model not loaded — using microclimate epidemiological model")
        except Exception as e:
            logger.error(f"Error loading disease model: {str(e)}")

    def _prepare_features(self, features: dict):
        """Map input features to model feature names with robust string-to-numeric encoding"""
        expected_features = ['Soil_Type', 'Soil_Moisture', 'Temperature_C', 'Humidity', 
                           'Rainfall_mm', 'Wind_Speed_kmh', 'Sunlight_Hours', 'Crop_Type', 
                           'Crop_Growth_Stage', 'Season', 'NDVI', 'Previous_Irrigation_mm']
        
        crop_val = features.get('Crop_Type', features.get('crop_type', 1))
        if isinstance(crop_val, str):
            c_map = {'rice': 0, 'paddy': 0, 'wheat': 1, 'maize': 2, 'corn': 2, 'cotton': 3, 'pulse': 4, 'pulses': 4, 'sugarcane': 5}
            crop_val = c_map.get(crop_val.lower(), 1)
        elif crop_val is None:
            crop_val = 1

        stage_val = features.get('Crop_Growth_Stage', features.get('growth_stage', 1))
        if isinstance(stage_val, str):
            s_map = {'initial': 0, 'seedling': 0, 'vegetative': 1, 'development': 1, 'flowering': 2, 'heading': 2, 'maturity': 3, 'late': 3}
            stage_val = s_map.get(stage_val.lower(), 1)
        elif stage_val is None:
            stage_val = 1

        soil_val = features.get('Soil_Type', features.get('soil_type', 1))
        if isinstance(soil_val, str):
            st_map = {'sandy': 0, 'loamy': 1, 'clay': 2}
            soil_val = st_map.get(soil_val.lower(), 1)
        elif soil_val is None:
            soil_val = 1

        season_val = features.get('Season', features.get('season', 1))
        if isinstance(season_val, str):
            se_map = {'kharif': 0, 'monsoon': 0, 'rabi': 1, 'winter': 1, 'zaid': 2, 'summer': 2}
            season_val = se_map.get(season_val.lower(), 1)
        elif season_val is None:
            season_val = 1

        mapped_features = {
            'Soil_Type': int(soil_val),
            'Soil_Moisture': float(features.get('Soil_Moisture', features.get('soil_moisture', 40.0))),
            'Temperature_C': float(features.get('Temperature_C', features.get('temperature', 25.0))),
            'Humidity': float(features.get('Humidity', features.get('humidity', 60.0))),
            'Rainfall_mm': float(features.get('Rainfall_mm', features.get('rainfall_mm', 0.0))),
            'Wind_Speed_kmh': float(features.get('Wind_Speed_kmh', features.get('wind_speed_kmh', 10.0))),
            'Sunlight_Hours': float(features.get('Sunlight_Hours', features.get('sunlight_hours', 8.0))),
            'Crop_Type': int(crop_val),
            'Crop_Growth_Stage': int(stage_val),
            'Season': int(season_val),
            'NDVI': float(features.get('NDVI', features.get('ndvi', 0.5))),
            'Previous_Irrigation_mm': float(features.get('Previous_Irrigation_mm', features.get('prev_irrigation_mm', 10.0)))
        }
        
        return pd.DataFrame([mapped_features])[expected_features], mapped_features
    
    def _get_feature_analysis(self, mapped_features: dict) -> dict:
        """Analyze features to explain irrigation decision"""
        analysis = {
            "soil_moisture": {
                "value": mapped_features.get('Soil_Moisture', 40),
                "status": self._get_moisture_status(mapped_features.get('Soil_Moisture', 40)),
                "impact": "Critical factor - directly affects irrigation need"
            },
            "ndvi": {
                "value": mapped_features.get('NDVI', 0.5),
                "status": self._get_ndvi_status(mapped_features.get('NDVI', 0.5)),
                "impact": "Indicates crop health and water stress"
            },
            "temperature": {
                "value": mapped_features.get('Temperature_C', 25),
                "status": "High" if mapped_features.get('Temperature_C', 25) > 30 else "Moderate" if mapped_features.get('Temperature_C', 25) > 20 else "Low",
                "impact": "Affects evapotranspiration rate"
            },
            "rainfall": {
                "value": mapped_features.get('Rainfall_mm', 0),
                "status": "Significant" if mapped_features.get('Rainfall_mm', 0) > 5 else "Moderate" if mapped_features.get('Rainfall_mm', 0) > 2 else "Low",
                "impact": "Reduces irrigation requirement"
            },
            "humidity": {
                "value": mapped_features.get('Humidity', 60),
                "status": "High" if mapped_features.get('Humidity', 60) > 70 else "Moderate" if mapped_features.get('Humidity', 60) > 50 else "Low",
                "impact": "Affects evaporation rate"
            }
        }
        return analysis
    
    def _get_moisture_status(self, moisture: float) -> str:
        """Get status of soil moisture"""
        if moisture < 25:
            return "Critical - Immediate irrigation needed"
        elif moisture < 40:
            return "Low - Irrigation recommended"
        elif moisture < 60:
            return "Adequate - Monitor"
        else:
            return "High - No irrigation needed"
    
    def _get_ndvi_status(self, ndvi: float) -> str:
        """Get status of NDVI"""
        if ndvi < 0.3:
            return "Severe stress - Urgent attention needed"
        elif ndvi < 0.5:
            return "Moderate stress - Needs irrigation"
        elif ndvi < 0.7:
            return "Healthy - Normal management"
        else:
            return "Very healthy - Minimal stress"
    
    def predict_irrigation_need(self, features: dict):
        """Predict if irrigation is needed (0=No, 1=Yes)"""
        if not self.irrigation_model:
            logger.warning("Irrigation model not loaded, returning default")
            return {"prediction": "Unknown", "confidence": 0.0, "needs_irrigation": False}
        
        try:
            X, mapped_features = self._prepare_features(features)
            prediction = self.irrigation_model.predict(X)[0]
            probabilities = self.irrigation_model.predict_proba(X)[0]
            confidence = max(probabilities)
            
            # Map numeric prediction to human-readable format
            irrigation_label = "Yes" if prediction == 1 else "No"
            
            # Get feature analysis for explanation
            feature_analysis = self._get_feature_analysis(mapped_features)
            
            return {
                "prediction": irrigation_label,
                "needs_irrigation": bool(prediction),
                "confidence": float(confidence),
                "raw_prediction": int(prediction),
                "feature_analysis": feature_analysis
            }
        except Exception as e:
            logger.error(f"Error during irrigation prediction: {str(e)}")
            return {"prediction": "Unknown", "confidence": 0.0, "needs_irrigation": False}
    
    def predict_water_requirement(self, features: dict):
        """Predict water requirement in mm"""
        if not self.water_requirement_model:
            logger.warning("Water requirement model not loaded, returning default")
            return 0
        
        try:
            X, _ = self._prepare_features(features)
            water_requirement = self.water_requirement_model.predict(X)[0]
            return float(water_requirement)
        except Exception as e:
            logger.error(f"Error during water requirement prediction: {str(e)}")
            return 0
    
    def predict_with_satellite(self, sensor_features: dict, ndvi_value: float = None, health_status: str = None):
        """
        Enhanced prediction incorporating satellite data
        """
        # Get irrigation need prediction
        irrigation_result = self.predict_irrigation_need(sensor_features)
        needs_irrigation = irrigation_result.get("needs_irrigation", False)
        
        # Get water requirement prediction
        water_requirement = self.predict_water_requirement(sensor_features)
        
        # Adjust based on satellite data
        if ndvi_value is not None:
            if ndvi_value < 0.3:
                needs_irrigation = True
                water_requirement *= 1.2
            elif ndvi_value > 0.75:
                needs_irrigation = False
                water_requirement *= 0.8
        
        soil_moisture = sensor_features.get("soil_moisture", 40)
        
        if ndvi_value and ndvi_value < 0.4 and soil_moisture < 30:
            final_recommendation = "Critical - Immediate irrigation needed"
        elif ndvi_value and ndvi_value < 0.5 and soil_moisture < 40:
            final_recommendation = "High - Irrigation recommended"
        elif ndvi_value and ndvi_value >= 0.7 and soil_moisture >= 50:
            final_recommendation = "Low - No irrigation needed"
        else:
            final_recommendation = "Yes - Irrigation needed" if needs_irrigation else "No - No irrigation needed"
        
        return {
            "needs_irrigation": needs_irrigation,
            "water_requirement_mm": round(water_requirement, 2),
            "recommendation": final_recommendation,
            "ndvi_factor": "Considered" if ndvi_value else "Not available",
            "ndvi_value": ndvi_value,
            "soil_moisture": soil_moisture,
            "confidence": irrigation_result.get("confidence", 0)
        }

    def predict_from_farm_state(self, farm_state) -> dict:
        """
        Phase 5 — Multi-source fused prediction.
        Takes a fully assembled FarmState and applies the Decision Engine:
          1. ML model prediction on fused features
          2. Rain-delay override (high probability in near-term forecast)
          3. NDVI stress override (severe stress → force irrigate)
          4. Over-saturation safety check
        Returns a structured decision dict.
        """
        features = farm_state.to_ml_features()

        irrigation_result = self.predict_irrigation_need(features)
        water_mm = self.predict_water_requirement(features)
        needs_irrigation: bool = irrigation_result.get("needs_irrigation", False)
        confidence: float = irrigation_result.get("confidence", 0.5)

        # ── Rain-delay override ──────────────────────────────────────────────
        # If rain probability is high OR significant rain expected within 2 days,
        # delay irrigation to avoid waste.
        rain_delay = False
        rain_reason = ""
        if farm_state.rain_probability >= 65:
            rain_delay = True
            rain_reason = f"Rain probability is high ({farm_state.rain_probability:.0f}%) — delaying."
        elif any(r >= 8.0 for r in farm_state.forecast_rain_7d[:2]):
            rain_delay = True
            rain_reason = "Significant rain expected within 48 hours — delaying."

        # ── NDVI stress override ─────────────────────────────────────────────
        # Severe crop stress always overrides rain delay if soil is also dry.
        ndvi_force = False
        if farm_state.ndvi < 0.25 and farm_state.soil_moisture < farm_state.optimal_moisture:
            ndvi_force = True
            rain_delay = False          # critical stress outweighs rain delay

        # ── Over-saturation safety ───────────────────────────────────────────
        if farm_state.soil_moisture >= farm_state.optimal_moisture + 20:
            needs_irrigation = False
            water_mm = 0.0
            rain_delay = False

        # ── Apply overrides ──────────────────────────────────────────────────
        if rain_delay:
            final_decision = "delay"
            final_water_mm = 0.0
            recommendation = "Delay — " + rain_reason
        elif ndvi_force:
            final_decision = "irrigate"
            final_water_mm = round(water_mm * 1.2, 1)   # extra 20% for stress
            recommendation = f"Critical NDVI stress ({farm_state.ndvi:.2f}) — irrigate immediately."
        elif needs_irrigation:
            final_decision = "irrigate"
            final_water_mm = round(water_mm, 1)
            recommendation = f"Irrigation needed — apply {water_mm:.1f}mm."
        else:
            final_decision = "skip"
            final_water_mm = 0.0
            recommendation = "No irrigation needed — conditions adequate."

        logger.info(
            f"FusedPrediction: field={farm_state.field_id} "
            f"decision={final_decision} water={final_water_mm}mm "
            f"rain_delay={rain_delay} ndvi_force={ndvi_force}"
        )

        return {
            "decision": final_decision,
            "needs_irrigation": final_decision == "irrigate",
            "recommended_water_mm": final_water_mm,
            "recommendation": recommendation,
            "confidence": round(confidence, 3),
            "rain_delay": rain_delay,
            "ndvi_force_irrigate": ndvi_force,
            "ml_prediction": irrigation_result.get("prediction", "Unknown"),
            "ml_confidence": confidence,
            "water_stress_index": farm_state.water_stress_index,
            "sources_used": farm_state.sources_used,
        }

    def explain_prediction(self, features: dict) -> dict:
        """
        Phase 7 — XAI: Return feature importances ranked by contribution.
        Uses the RandomForest's feature_importances_ (model-global, not per-sample).
        Returns both machine-readable scores and a human-readable explanation.
        """
        FEATURE_LABELS = {
            "Soil_Moisture": "Soil moisture",
            "NDVI": "Crop health (NDVI)",
            "Temperature_C": "Temperature",
            "Rainfall_mm": "Rainfall",
            "Humidity": "Humidity",
            "Wind_Speed_kmh": "Wind speed",
            "Previous_Irrigation_mm": "Previous irrigation",
            "Crop_Growth_Stage": "Crop growth stage",
            "Crop_Type": "Crop type",
            "Soil_Type": "Soil type",
            "Sunlight_Hours": "Sunlight hours",
            "Season": "Season",
        }

        if not self.irrigation_model:
            # Rule-based fallback when model isn't loaded
            moisture = features.get("soil_moisture", 40)
            ndvi = features.get("ndvi", 0.5)
            temp = features.get("temperature", 25)
            rain = features.get("rainfall_mm", 0)

            contributions = [
                {"feature": "Soil moisture", "importance": 0.40, "value": f"{moisture}%",
                 "direction": "increases" if moisture < 40 else "reduces"},
                {"feature": "Crop health (NDVI)", "importance": 0.26, "value": str(ndvi),
                 "direction": "increases" if ndvi < 0.5 else "reduces"},
                {"feature": "Temperature", "importance": 0.18, "value": f"{temp}°C",
                 "direction": "increases" if temp > 30 else "neutral"},
                {"feature": "Rainfall", "importance": 0.11, "value": f"{rain}mm",
                 "direction": "reduces" if rain > 2 else "neutral"},
                {"feature": "Humidity", "importance": 0.05, "value": f"{features.get('humidity', 60)}%",
                 "direction": "neutral"},
            ]
            top2 = contributions[:2]
            human = (
                f"Irrigation need driven mainly by {top2[0]['feature'].lower()} "
                f"({top2[0]['value']}) and {top2[1]['feature'].lower()} ({top2[1]['value']})."
            )
            return {
                "method": "rule_based_fallback",
                "contributions": contributions,
                "human_explanation": human,
            }

        try:
            X, mapped = self._prepare_features(features)
            importances = self.irrigation_model.feature_importances_
            feature_names = list(X.columns)

            # Pair names with importances, sort descending
            pairs = sorted(
                zip(feature_names, importances), key=lambda x: x[1], reverse=True
            )

            # Map current feature values for readable output
            value_map = {k: mapped.get(k, "N/A") for k in feature_names}

            contributions = []
            for fname, imp in pairs:
                val = value_map.get(fname, "N/A")
                label = FEATURE_LABELS.get(fname, fname)
                # Determine direction heuristic
                if fname == "Soil_Moisture":
                    direction = "increases need" if val < 40 else "reduces need"
                elif fname == "NDVI":
                    direction = "increases need" if val < 0.5 else "reduces need"
                elif fname == "Rainfall_mm":
                    direction = "reduces need" if val > 2 else "neutral"
                elif fname == "Temperature_C":
                    direction = "increases need" if val > 30 else "neutral"
                else:
                    direction = "contributing factor"

                contributions.append({
                    "feature": label,
                    "importance": round(float(imp), 4),
                    "value": val if isinstance(val, str) else round(float(val), 2),
                    "direction": direction,
                })

            # Human-readable summary from top 2 factors
            top = contributions[:2]
            human = (
                f"The main drivers were {top[0]['feature'].lower()} "
                f"({top[0]['value']}, {top[0]['direction']}) and "
                f"{top[1]['feature'].lower()} ({top[1]['value']}, {top[1]['direction']})."
            )

            return {
                "method": "random_forest_feature_importance",
                "contributions": contributions,
                "human_explanation": human,
            }

        except Exception as e:
            logger.error(f"Error in explain_prediction: {e}")
            return {
                "method": "error",
                "contributions": [],
                "human_explanation": "Explanation unavailable.",
            }

    # ─────────────────────────────────────────────────────────────────────────
    # YIELD ESTIMATION PREDICTION
    # ─────────────────────────────────────────────────────────────────────────
    def predict_yield(self, features: dict) -> dict:
        """
        Predict crop yield estimation using the trained RandomForest model
        augmented with regional ICAR agronomic benchmarks and NDVI vegetative index.
        """
        crop_raw = str(features.get('crop_type', features.get('crop', 'wheat'))).lower()
        crop_mapping = {
            'rice': 0, 'paddy': 0,
            'wheat': 1,
            'maize': 2, 'corn': 2,
            'cotton': 3,
            'pulse': 4, 'pulses': 4, 'gram': 4,
            'sugarcane': 5,
        }
        crop_idx = crop_mapping.get(crop_raw, 1)

        stage_raw = str(features.get('growth_stage', features.get('stage', 'development'))).lower()
        stage_mapping = {
            'initial': 0, 'seedling': 0,
            'vegetative': 1, 'development': 1,
            'flowering': 2, 'heading': 2, 'mid': 2,
            'maturity': 3, 'late': 3, 'harvest': 3,
        }
        stage_idx = stage_mapping.get(stage_raw, 1)

        soil_mapping = {'sandy': 0, 'loamy': 1, 'clay': 2}
        soil_raw = str(features.get('soil_type', 'loamy')).lower()
        soil_idx = soil_mapping.get(soil_raw, 1)

        soil_moisture = float(features.get('soil_moisture', features.get('Soil_Moisture', 40.0)))
        ndvi = float(features.get('ndvi', features.get('NDVI', 0.55)))
        temperature = float(features.get('temperature', features.get('Temperature_C', 26.0)))
        humidity = float(features.get('humidity', features.get('Humidity', 60.0)))
        rainfall = float(features.get('rainfall_mm', features.get('Rainfall_mm', 0.0)))
        sunlight = float(features.get('sunlight_hours', features.get('Sunlight_Hours', 8.0)))
        prev_irr = float(features.get('prev_irrigation_mm', features.get('Previous_Irrigation_mm', 10.0)))
        field_area_ha = float(features.get('field_area_hectare', features.get('field_area', 1.0)))

        # Standard agronomic benchmarks by crop
        benchmarks = {
            0: {'name': 'Rice (Paddy)', 'base_t_ha': 4.8, 'days_total': 120, 'min_t': 3.5, 'max_t': 6.5},
            1: {'name': 'Wheat', 'base_t_ha': 4.2, 'days_total': 135, 'min_t': 3.0, 'max_t': 5.8},
            2: {'name': 'Maize (Corn)', 'base_t_ha': 6.2, 'days_total': 105, 'min_t': 4.5, 'max_t': 8.5},
            3: {'name': 'Cotton', 'base_t_ha': 2.4, 'days_total': 160, 'min_t': 1.6, 'max_t': 3.6},
            4: {'name': 'Pulses', 'base_t_ha': 1.6, 'days_total': 90, 'min_t': 1.0, 'max_t': 2.4},
            5: {'name': 'Sugarcane', 'base_t_ha': 75.0, 'days_total': 360, 'min_t': 55.0, 'max_t': 95.0},
        }
        b_info = benchmarks.get(crop_idx, benchmarks[1])

        # Prepare DataFrame for ML model
        X_df = pd.DataFrame([{
            'Crop_Type': crop_idx,
            'Crop_Growth_Stage': stage_idx,
            'Soil_Type': soil_idx,
            'Soil_Moisture': soil_moisture,
            'NDVI': ndvi,
            'Temperature_C': temperature,
            'Humidity': humidity,
            'Rainfall_mm': rainfall,
            'Sunlight_Hours': sunlight,
            'Previous_Irrigation_mm': prev_irr
        }])

        predicted_yield = None
        if self.yield_model:
            try:
                predicted_yield = float(self.yield_model.predict(X_df)[0])
            except Exception as e:
                logger.error(f"Error predicting yield with ML model: {e}")

        if predicted_yield is None:
            # Agronomic rule-based calculation
            ndvi_mult = 0.55 + (ndvi * 0.7)
            moist_mult = 1.10 if (40 <= soil_moisture <= 55) else (0.85 if soil_moisture < 30 else 0.95)
            predicted_yield = round(b_info['base_t_ha'] * ndvi_mult * moist_mult, 2)

        predicted_yield = round(max(0.5, predicted_yield), 2)
        base_benchmark = b_info['base_t_ha']
        pct_vs_benchmark = round(((predicted_yield - base_benchmark) / base_benchmark) * 100, 1)

        # 1 ton/ha = 4.04686 quintals per acre
        yield_quintals_per_acre = round(predicted_yield * 4.04686, 1)
        total_projected_tons = round(predicted_yield * field_area_ha, 2)
        total_projected_quintals = round(yield_quintals_per_acre * (field_area_ha * 2.47105), 1)

        # Stage progression & Days to harvest
        stage_progress = [15, 45, 75, 95][min(stage_idx, 3)]
        days_remaining = max(5, int(b_info['days_total'] * (1 - (stage_progress / 100.0))))

        # Key drivers
        drivers = []
        if ndvi >= 0.65:
            drivers.append({"factor": "High Canopy NDVI (Vigor)", "impact": "Positive (+12%)", "type": "boost"})
        elif ndvi < 0.40:
            drivers.append({"factor": "Low Vegetative Vigor (NDVI)", "impact": "Drag (-18%)", "type": "penalty"})

        if 38 <= soil_moisture <= 58:
            drivers.append({"factor": "Optimal Root Zone Moisture", "impact": "Positive (+8%)", "type": "boost"})
        elif soil_moisture < 30:
            drivers.append({"factor": "Soil Moisture Deficit", "impact": "Drag (-15%)", "type": "penalty"})
        elif soil_moisture > 65:
            drivers.append({"factor": "Root Zone Waterlogging", "impact": "Drag (-10%)", "type": "penalty"})

        if 21 <= temperature <= 31:
            drivers.append({"factor": "Favorable Thermal Range", "impact": "Optimal (+5%)", "type": "boost"})
        elif temperature > 36:
            drivers.append({"factor": "Heat Stress Exposure", "impact": "Drag (-12%)", "type": "penalty"})

        potential_pct = min(100, round((predicted_yield / b_info['max_t']) * 100, 1))
        quality_grade = "Grade A+ (Premium Export)" if potential_pct >= 85 else ("Grade A (Standard Market)" if potential_pct >= 70 else "Grade B (Fair)")

        return {
            "crop_name": b_info['name'],
            "growth_stage": stage_raw.capitalize(),
            "field_area_ha": field_area_ha,
            "field_area_acres": round(field_area_ha * 2.47105, 2),
            "estimated_yield_per_ha_tons": predicted_yield,
            "estimated_yield_per_acre_quintals": yield_quintals_per_acre,
            "total_estimated_yield_tons": total_projected_tons,
            "total_estimated_yield_quintals": total_projected_quintals,
            "regional_benchmark_tons_ha": base_benchmark,
            "benchmark_comparison_pct": pct_vs_benchmark,
            "yield_potential_pct": potential_pct,
            "growth_progress_pct": stage_progress,
            "days_to_harvest": days_remaining,
            "quality_grade": quality_grade,
            "confidence_score": 0.94 if self.yield_model else 0.82,
            "key_drivers": drivers,
            "harvest_window": f"{days_remaining} days (Estimated)"
        }

    # ─────────────────────────────────────────────────────────────────────────
    # DISEASE RISK PREDICTION
    # ─────────────────────────────────────────────────────────────────────────
    def predict_disease_risk(self, features: dict) -> dict:
        """
        Predict crop disease and pest outbreak risks based on microclimate
        infection indices (humidity, temperature, wetness, NDVI anomalies).
        """
        crop_raw = str(features.get('crop_type', features.get('crop', 'wheat'))).lower()
        crop_mapping = {
            'rice': 0, 'paddy': 0,
            'wheat': 1,
            'maize': 2, 'corn': 2,
            'cotton': 3,
            'pulse': 4, 'pulses': 4,
            'sugarcane': 5,
        }
        crop_idx = crop_mapping.get(crop_raw, 1)

        stage_raw = str(features.get('growth_stage', features.get('stage', 'development'))).lower()
        stage_mapping = {
            'initial': 0, 'vegetative': 1, 'development': 1,
            'flowering': 2, 'heading': 2, 'maturity': 3,
        }
        stage_idx = stage_mapping.get(stage_raw, 1)

        temp = float(features.get('temperature', features.get('Temperature_C', 26.0)))
        humidity = float(features.get('humidity', features.get('Humidity', 65.0)))
        rainfall = float(features.get('rainfall_mm', features.get('Rainfall_mm', 0.0)))
        soil_moisture = float(features.get('soil_moisture', features.get('Soil_Moisture', 40.0)))
        ndvi = float(features.get('ndvi', features.get('NDVI', 0.55)))
        wind_speed = float(features.get('wind_speed_kmh', features.get('Wind_Speed_kmh', 10.0)))

        # Prepare DataFrame for ML model
        X_df = pd.DataFrame([{
            'Crop_Type': crop_idx,
            'Crop_Growth_Stage': stage_idx,
            'Temperature_C': temp,
            'Humidity': humidity,
            'Rainfall_mm': rainfall,
            'Soil_Moisture': soil_moisture,
            'NDVI': ndvi,
            'Wind_Speed_kmh': wind_speed
        }])

        risk_score = None
        if self.disease_model:
            try:
                risk_score = float(self.disease_model.predict(X_df)[0])
            except Exception as e:
                logger.error(f"Error predicting disease with ML model: {e}")

        if risk_score is None:
            score = 15.0
            if humidity > 80: score += 35
            elif humidity > 70: score += 20
            if 20 <= temp <= 30: score += 25
            if rainfall > 2: score += 15
            if soil_moisture > 60: score += 10
            risk_score = min(98.0, max(5.0, score))

        risk_score = round(risk_score, 1)

        if risk_score < 30:
            level = "LOW"
            badge_color = "emerald"
            summary_action = "Routine monitoring. Field microclimate is safe from major fungal outbreaks."
        elif risk_score < 55:
            level = "MODERATE"
            badge_color = "yellow"
            summary_action = "Elevated humidity detected. Inspect field border rows and lower canopy for early spots."
        elif risk_score < 75:
            level = "HIGH"
            badge_color = "orange"
            summary_action = "High risk conditions for fungal spore germination. Preventative bio-control / fungicide recommended."
        else:
            level = "CRITICAL"
            badge_color = "rose"
            summary_action = "Critical epidemic alert: High humidity combined with optimal pathogen temperature. Immediate action required."

        # Environmental triggers
        triggers = []
        if humidity >= 75:
            triggers.append(f"Elevated air humidity ({humidity}%) exceeds 75% fungal germination threshold")
        if 20 <= temp <= 30:
            triggers.append(f"Ambient temperature ({temp}°C) in peak pathogen proliferation sweet spot (20-30°C)")
        if rainfall > 1.0:
            triggers.append(f"Canopy wetness from {rainfall}mm precipitation facilitates spore attachment")
        if soil_moisture > 65:
            triggers.append(f"Waterlogged soil ({soil_moisture}%) creates root rot & anoxic fungal conditions")
        if ndvi < 0.38:
            triggers.append("Satellite vegetation vigor drop detected — possible asymptomatic early stress")

        if not triggers:
            triggers.append("Current weather parameters are within safe dry thresholds")

        threats_data = DISEASE_KNOWLEDGE_BASE.get(
            'rice' if crop_idx == 0 else ('wheat' if crop_idx == 1 else ('maize' if crop_idx == 2 else ('cotton' if crop_idx == 3 else 'default'))),
            DISEASE_KNOWLEDGE_BASE['default']
        )

        evaluated_threats = []
        for t in threats_data:
            t_min, t_max = t.get('optimal_temp_range', [20, 30])
            h_min = t.get('optimal_humidity_min', 75)
            
            p_score = (risk_score * 0.6)
            if t_min <= temp <= t_max:
                p_score += 25
            if humidity >= h_min:
                p_score += 15
            p_score = min(99, max(5, round(p_score)))

            evaluated_threats.append({
                "name": t["name"],
                "pathogen": t["pathogen"],
                "probability_pct": p_score,
                "threat_level": "HIGH" if p_score >= 70 else ("MODERATE" if p_score >= 45 else "LOW"),
                "symptoms": t["symptoms"],
                "preventive_actions": t["preventive_actions"],
                "organic_options": t["organic_options"]
            })

        evaluated_threats = sorted(evaluated_threats, key=lambda x: x['probability_pct'], reverse=True)

        return {
            "overall_risk_score": risk_score,
            "overall_risk_level": level,
            "badge_color": badge_color,
            "summary_action": summary_action,
            "triggers_detected": triggers,
            "top_threats": evaluated_threats,
            "primary_threat": evaluated_threats[0]["name"] if evaluated_threats else "General Fungal Risk",
            "preventative_sprays": evaluated_threats[0]["preventive_actions"] if evaluated_threats else [],
            "organic_remedy": evaluated_threats[0]["organic_options"] if evaluated_threats else "Neem oil spray"
        }

    # ─────────────────────────────────────────────────────────────────────────
    # COMPREHENSIVE SMART IRRIGATION PREDICTION
    # ─────────────────────────────────────────────────────────────────────────
    def predict_irrigation_comprehensive(self, features: dict) -> dict:
        """
        Complete irrigation prediction combining ML classification, regression,
        and optimal scheduling window.
        """
        irr_result = self.predict_irrigation_need(features)
        water_mm = self.predict_water_requirement(features)
        needs_irrigation = irr_result.get("needs_irrigation", False)
        confidence = irr_result.get("confidence", 0.85)

        moisture = float(features.get('soil_moisture', features.get('Soil_Moisture', 40.0)))
        area_ha = float(features.get('field_area_hectare', features.get('field_area', 1.0)))
        rainfall = float(features.get('rainfall_mm', features.get('Rainfall_mm', 0.0)))

        # Rule overrides for safety
        if rainfall >= 8.0:
            needs_irrigation = False
            water_mm = 0.0
            urgency = "Delayed (Rain forecasted)"
        elif moisture >= 60.0:
            needs_irrigation = False
            water_mm = 0.0
            urgency = "Not Needed (Soil Saturated)"
        elif moisture < 28.0:
            needs_irrigation = True
            water_mm = max(water_mm, 25.0)
            urgency = "Immediate (< 2 Hours)"
        elif needs_irrigation:
            urgency = "Recommended Within 12-24 Hours"
        else:
            urgency = "Normal Maintenance"

        # Water volume calculations: 1 mm depth over 1 hectare = 10,000 Liters
        total_liters = round(water_mm * area_ha * 10000, 0)
        # Average pump flow rate ~60 L/min
        duration_minutes = round(total_liters / 60.0) if total_liters > 0 else 0

        # Optimal schedule window: avoid noon peak heat (11 AM - 3 PM)
        if needs_irrigation:
            next_window = "Early Morning 06:00 AM - 08:30 AM (Minimal Evaporation)"
            recommended_hours_away = 1 if urgency == "Immediate (< 2 Hours)" else 6
        else:
            next_window = "In 3 Days (Maintain current cycle)"
            recommended_hours_away = 72

        return {
            "needs_irrigation": needs_irrigation,
            "prediction_label": "Yes - Irrigate" if needs_irrigation else "No - Delay/Skip",
            "confidence_pct": round(confidence * 100, 1),
            "water_requirement_mm": round(water_mm, 1),
            "total_water_liters": int(total_liters),
            "recommended_duration_minutes": int(duration_minutes),
            "urgency": urgency,
            "optimal_window": next_window,
            "hours_until_next_cycle": recommended_hours_away,
            "current_soil_moisture": moisture,
            "target_soil_moisture": 55.0,
            "pump_action_suggested": "START" if (needs_irrigation and moisture < 35) else ("STANDBY" if needs_irrigation else "OFF")
        }

    # ─────────────────────────────────────────────────────────────────────────
    # UNIFIED MULTI-PREDICTION HUB
    # ─────────────────────────────────────────────────────────────────────────
    def get_full_predictions(self, features: dict, farm_state=None) -> dict:
        """
        Returns full multi-model prediction suite:
          1. Irrigation prediction & next optimal schedule
          2. Yield estimation & harvest outlook
          3. Crop disease & pest outbreak risk
        """
        irr = self.predict_irrigation_comprehensive(features)
        yd = self.predict_yield(features)
        dis = self.predict_disease_risk(features)

        return {
            "status": "success",
            "field_id": features.get("field_id", 1),
            "crop": yd.get("crop_name", "Crop"),
            "growth_stage": yd.get("growth_stage", "Development"),
            "irrigation": irr,
            "yield": yd,
            "disease": dis,
            "timestamp": pd.Timestamp.now().isoformat()
        }


# ─────────────────────────────────────────────────────────────────────────────
# DISEASE KNOWLEDGE BASE
# ─────────────────────────────────────────────────────────────────────────────
DISEASE_KNOWLEDGE_BASE = {
    "rice": [
        {
            "name": "Rice Blast (Leaf & Neck Blast)",
            "pathogen": "Magnaporthe oryzae",
            "optimal_temp_range": [22, 28],
            "optimal_humidity_min": 85,
            "symptoms": "Spindle-shaped elliptical lesions with grey/white center and brown-red margin.",
            "preventive_actions": [
                "Apply prophylactic spray of Tricyclazole 75% WP @ 0.6 g/L or Azoxystrobin @ 1 ml/L.",
                "Avoid excessive split application of nitrogenous fertilizers.",
                "Maintain thin water layer; avoid alternate severe drying and flooding."
            ],
            "organic_options": "Pseudomonas fluorescens (0.5%) seed & foliar treatment, Neem oil 3%."
        },
        {
            "name": "Sheath Blight",
            "pathogen": "Rhizoctonia solani",
            "optimal_temp_range": [28, 32],
            "optimal_humidity_min": 80,
            "symptoms": "Greenish-grey oval or irregular water-soaked spots on leaf sheaths near water line.",
            "preventive_actions": [
                "Apply Hexaconazole 5% EC @ 2 ml/L or Validamycin 3% L @ 2.5 ml/L.",
                "Ensure proper plant spacing for canopy aeration and sunlight penetration.",
                "Drain excess stagnant water if humidity stays elevated."
            ],
            "organic_options": "Trichoderma viride bio-fungicide @ 5 g/L."
        },
        {
            "name": "Bacterial Leaf Blight",
            "pathogen": "Xanthomonas oryzae pv. oryzae",
            "optimal_temp_range": [25, 34],
            "optimal_humidity_min": 75,
            "symptoms": "Water-soaked stripes turning yellow-to-white with wavy margins along leaf edges.",
            "preventive_actions": [
                "Spray Copper Oxychloride 50% WP @ 2.5 g/L mixed with Streptocycline @ 0.1 g/L.",
                "Avoid overhead irrigation or field-to-field flood transfer."
            ],
            "organic_options": "Bleaching powder @ 5 kg/ha applied with irrigation water."
        }
    ],
    "wheat": [
        {
            "name": "Yellow / Stripe Rust",
            "pathogen": "Puccinia striiformis",
            "optimal_temp_range": [12, 22],
            "optimal_humidity_min": 75,
            "symptoms": "Yellow-orange pustules arranged in linear stripes on upper leaf surface.",
            "preventive_actions": [
                "Spray Propiconazole 25% EC (Tilt) @ 1 ml/L at first appearance.",
                "Monitor cool, humid microclimates with frequent morning dew.",
                "Use certified rust-resistant wheat varieties (e.g. HD-2967, DBW-187)."
            ],
            "organic_options": "Foliar spray of cow urine formulation (1:10) + Neem cake extract."
        },
        {
            "name": "Powdery Mildew",
            "pathogen": "Blumeria graminis f. sp. tritici",
            "optimal_temp_range": [15, 23],
            "optimal_humidity_min": 80,
            "symptoms": "White cottony or powdery fungal patches on lower leaf blades and stems.",
            "preventive_actions": [
                "Apply Wettable Sulphur 80% WP @ 3 g/L or Tebuconazole @ 1 ml/L.",
                "Avoid excessive dense seeding to maintain canopy airflow."
            ],
            "organic_options": "Potassium bicarbonate spray or milk-water dilution (1:9)."
        },
        {
            "name": "Leaf Blight / Spot",
            "pathogen": "Bipolaris sorokiniana",
            "optimal_temp_range": [22, 30],
            "optimal_humidity_min": 70,
            "symptoms": "Small brownish-black lesions enlarging to oval spots with yellow halo.",
            "preventive_actions": [
                "Spray Mancozeb 75% WP @ 2.5 g/L.",
                "Ensure balanced potassium (potash) fertilization to strengthen leaf cuticle."
            ],
            "organic_options": "Trichoderma harzianum soil and seed treatment."
        }
    ],
    "maize": [
        {
            "name": "Turcicum Leaf Blight (NCLB)",
            "pathogen": "Exserohilum turcicum",
            "optimal_temp_range": [18, 27],
            "optimal_humidity_min": 80,
            "symptoms": "Long, elliptical grayish-green or tan lesions running parallel to leaf veins.",
            "preventive_actions": [
                "Spray Mancozeb 75% WP @ 2.5 g/L or Azoxystrobin @ 1 ml/L.",
                "Rotate with non-cereal crops and plow under old crop debris."
            ],
            "organic_options": "Pseudomonas fluorescens foliar spray."
        },
        {
            "name": "Common Rust",
            "pathogen": "Puccinia sorghi",
            "optimal_temp_range": [16, 25],
            "optimal_humidity_min": 75,
            "symptoms": "Golden brown to cinnamon brown powdery pustules scattered over leaves.",
            "preventive_actions": [
                "Spray Propiconazole @ 1 ml/L upon seeing initial pustules.",
                "Ensure proper field drainage."
            ],
            "organic_options": "Neem oil 3 ml/L preventative spray."
        }
    ],
    "cotton": [
        {
            "name": "Bacterial Blight / Angular Leaf Spot",
            "pathogen": "Xanthomonas citri pv. malvacearum",
            "optimal_temp_range": [28, 35],
            "optimal_humidity_min": 80,
            "symptoms": "Angular, water-soaked leaf spots bounded by veinlets, dark brown on drying.",
            "preventive_actions": [
                "Spray Copper Oxychloride 50 WP @ 2.5 g/L + Streptocycline @ 0.1 g/L.",
                "Avoid high nitrogen fertilizer dosage."
            ],
            "organic_options": "Seed treatment with Trichoderma viride."
        },
        {
            "name": "Fusarium Wilt / Root Rot",
            "pathogen": "Fusarium oxysporum / Rhizoctonia",
            "optimal_temp_range": [26, 33],
            "optimal_humidity_min": 70,
            "symptoms": "Yellowing of leaves starting from bottom, vascular browning of stem, drooping.",
            "preventive_actions": [
                "Drench root zone with Carbendazim 50 WP @ 2 g/L or Trichoderma @ 10 g/L.",
                "Prevent waterlogging in low-lying furrow spots."
            ],
            "organic_options": "Soil application of Trichoderma enriched in farmyard manure (FYM)."
        }
    ],
    "default": [
        {
            "name": "General Fungal Foliar Blight",
            "pathogen": "Alternaria / Phytophthora spp.",
            "optimal_temp_range": [20, 30],
            "optimal_humidity_min": 75,
            "symptoms": "Concentric dark brown rings or irregular water-soaked spots on foliage.",
            "preventive_actions": [
                "Apply broad-spectrum protective fungicide (Mancozeb 75 WP @ 2 g/L).",
                "Water early in the morning so sun can dry leaves quickly."
            ],
            "organic_options": "Neem oil 3 ml/L or bio-fungicide Trichoderma viride."
        },
        {
            "name": "Root Rot & Damping-off",
            "pathogen": "Pythium / Rhizoctonia spp.",
            "optimal_temp_range": [22, 32],
            "optimal_humidity_min": 70,
            "symptoms": "Soft, brown water-soaked roots, collar rot at soil level, stunting.",
            "preventive_actions": [
                "Improve field drainage and avoid over-irrigation.",
                "Drench root zone with bio-fungicide."
            ],
            "organic_options": "Apply Trichoderma viride mixed with decomposed compost."
        }
    ]
}

predictor = IrrigationPredictor()

