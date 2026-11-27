"""
Predictions API — Yield Estimation, Crop Disease Risk, and Smart Irrigation
===========================================================================
GET  /api/v1/predictions/{field_id}          → Full prediction suite (Irrigation + Yield + Disease)
POST /api/v1/predictions/predict             → Custom feature prediction for simulator
GET  /api/v1/predictions/yield/{field_id}    → Detailed Yield estimation & harvest outlook
GET  /api/v1/predictions/disease/{field_id}  → Crop Disease early warning & pathogen risk
GET  /api/v1/predictions/irrigation/{field_id} → ML Irrigation prediction & water volume
"""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db.session import get_db
from ..models.database import Field, SensorData, SatelliteData
from ..services.prediction import predictor
from ..services.satellite import satellite_service
from ..services.weather import get_weather

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predictions", tags=["Predictions"])


class CustomPredictionRequest(BaseModel):
    crop_type: Optional[str] = "Wheat"
    growth_stage: Optional[str] = "Development"
    soil_type: Optional[str] = "loamy"
    field_area_hectare: Optional[float] = 1.0
    soil_moisture: float = 40.0
    temperature: float = 25.0
    humidity: float = 60.0
    rainfall_mm: float = 0.0
    wind_speed_kmh: float = 10.0
    sunlight_hours: float = 8.0
    ndvi: float = 0.55
    prev_irrigation_mm: float = 10.0


async def _assemble_field_features(field_id: int, db: Session) -> dict:
    """Helper to collect latest sensor, satellite, and weather data for a field."""
    field = db.query(Field).filter(Field.id == field_id).first()
    if not field:
        # Fallback to farmer_id lookup
        field = db.query(Field).filter(Field.farmer_id == field_id).first()
    if not field:
        # Fallback to first field in database
        field = db.query(Field).first()
    if not field:
        raise HTTPException(status_code=404, detail=f"No field found for ID {field_id}")

    farmer = field.owner
    latest_sensor = (
        db.query(SensorData)
        .filter(SensorData.field_id == field_id)
        .order_by(SensorData.timestamp.desc())
        .first()
    )
    latest_satellite = (
        db.query(SatelliteData)
        .filter(SatelliteData.field_id == field_id)
        .order_by(SatelliteData.timestamp.desc())
        .first()
    )

    ndvi_val = 0.55
    if latest_satellite and latest_satellite.ndvi_value is not None:
        ndvi_val = latest_satellite.ndvi_value
    elif farmer and farmer.latitude and farmer.longitude:
        try:
            ndvi_info = await satellite_service.get_ndvi(farmer.latitude, farmer.longitude)
            ndvi_val = ndvi_info.get("ndvi_value", 0.55)
        except Exception:
            ndvi_val = 0.55

    weather_info = None
    if farmer and farmer.latitude and farmer.longitude:
        try:
            weather_info = await get_weather(farmer.latitude, farmer.longitude)
        except Exception:
            weather_info = None

    features = {
        "field_id": field.id,
        "crop_type": field.crop_type or "Wheat",
        "growth_stage": field.growth_stage or "Development",
        "soil_type": "loamy",
        "field_area_hectare": float(field.field_area or 1.0),
        "soil_moisture": float(latest_sensor.soil_moisture) if latest_sensor and latest_sensor.soil_moisture is not None else 40.0,
        "temperature": float(latest_sensor.temperature) if latest_sensor and latest_sensor.temperature is not None else (weather_info.get("temperature", 26.0) if weather_info else 26.0),
        "humidity": float(latest_sensor.humidity) if latest_sensor and latest_sensor.humidity is not None else (weather_info.get("humidity", 60.0) if weather_info else 60.0),
        "rainfall_mm": float(weather_info.get("precipitation", 0.0)) if weather_info else 0.0,
        "wind_speed_kmh": float(weather_info.get("windspeed", 10.0)) if weather_info else 10.0,
        "sunlight_hours": 8.5,
        "ndvi": float(ndvi_val),
        "prev_irrigation_mm": 10.0,
    }
    return features


@router.get("/{field_id}")
async def get_all_predictions(field_id: int, db: Session = Depends(get_db)):
    """
    Get full predictive suite for a given field:
      1. Yield estimation (tons/ha, quintals/acre, total yield, harvest countdown)
      2. Disease & pest outbreak risk (score, level, active pathogen threats, treatments)
      3. Smart Irrigation (irrigation need, volume in liters/mm, optimal schedule window)
    """
    features = await _assemble_field_features(field_id, db)
    predictions = predictor.get_full_predictions(features)
    return predictions


@router.post("/predict")
async def predict_custom_features(data: CustomPredictionRequest):
    """Run full prediction suite with custom user-supplied features."""
    features = data.dict()
    predictions = predictor.get_full_predictions(features)
    return predictions


@router.get("/yield/{field_id}")
async def get_yield_prediction(field_id: int, db: Session = Depends(get_db)):
    """Get standalone yield prediction for a field."""
    features = await _assemble_field_features(field_id, db)
    return {
        "field_id": field_id,
        "yield_prediction": predictor.predict_yield(features),
    }


@router.get("/disease/{field_id}")
async def get_disease_prediction(field_id: int, db: Session = Depends(get_db)):
    """Get standalone disease risk prediction for a field."""
    features = await _assemble_field_features(field_id, db)
    return {
        "field_id": field_id,
        "disease_prediction": predictor.predict_disease_risk(features),
    }


@router.get("/irrigation/{field_id}")
async def get_irrigation_prediction(field_id: int, db: Session = Depends(get_db)):
    """Get standalone comprehensive smart irrigation prediction for a field."""
    features = await _assemble_field_features(field_id, db)
    return {
        "field_id": field_id,
        "irrigation_prediction": predictor.predict_irrigation_comprehensive(features),
    }
