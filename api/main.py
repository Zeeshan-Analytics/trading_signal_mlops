"""
FastAPI application for trading signal predictions.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List
import numpy as np
import sys
import os
from pathlib import Path

# Setup paths - works from any directory
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Change to project root for relative paths to work
original_dir = os.getcwd()
os.chdir(project_root)

from src.models.predict import TradingSignalPredictor
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Load config
config = load_config()

# Initialize FastAPI app
app = FastAPI(
    title=config['api']['title'],
    description=config['api']['description'],
    version=config['api']['version']
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor (load model once at startup)
predictor = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global predictor
    try:
        logger.info("Loading model...")
        predictor = TradingSignalPredictor()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


# Pydantic models for request/response
class Features(BaseModel):
    """Features for prediction."""
    SMA_10: float = Field(..., description="10-period Simple Moving Average")
    SMA_20: float = Field(..., description="20-period Simple Moving Average")
    SMA_50: float = Field(..., description="50-period Simple Moving Average")
    EMA_12: float = Field(..., description="12-period Exponential Moving Average")
    EMA_26: float = Field(..., description="26-period Exponential Moving Average")
    RSI_14: float = Field(..., description="14-period Relative Strength Index")
    MACD: float = Field(..., description="MACD indicator")
    MACD_signal: float = Field(..., description="MACD Signal line")
    MACD_hist: float = Field(..., description="MACD Histogram")
    BB_upper: float = Field(..., description="Bollinger Band Upper")
    BB_middle: float = Field(..., description="Bollinger Band Middle")
    BB_lower: float = Field(..., description="Bollinger Band Lower")
    volume_sma_20: float = Field(..., description="20-period Volume SMA")
    price_change: float = Field(..., description="Price change percentage")
    
    class Config:
        json_schema_extra = {
            "example": {
                "SMA_10": 150.5,
                "SMA_20": 148.2,
                "SMA_50": 145.8,
                "EMA_12": 151.0,
                "EMA_26": 149.5,
                "RSI_14": 65.3,
                "MACD": 1.5,
                "MACD_signal": 1.2,
                "MACD_hist": 0.3,
                "BB_upper": 155.0,
                "BB_middle": 150.0,
                "BB_lower": 145.0,
                "volume_sma_20": 1000000.0,
                "price_change": 0.02
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response."""
    signal: str = Field(..., description="Predicted trading signal")
    confidence: float = Field(..., description="Confidence score (0-1)")
    probabilities: Dict[str, float] = Field(..., description="Probabilities for all classes")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    version: str


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_name: str
    features: List[str]
    num_features: int
    signal_classes: Dict[int, str]


# API Endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Trading Signal API",
        "version": config['api']['version'],
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if predictor is not None else "unhealthy",
        "model_loaded": predictor is not None,
        "version": config['api']['version']
    }


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """Get model information."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": "trading_signal_model",
        "features": predictor.feature_names,
        "num_features": len(predictor.feature_names),
        "signal_classes": predictor.signal_classes
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(features: Features):
    """
    Make a trading signal prediction.
    
    Returns the predicted signal (strong_buy, buy, hold, sell, strong_sell)
    along with confidence scores.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert features to dict
        feature_dict = features.dict()
        
        # Make prediction
        result = predictor.predict_from_dict(feature_dict)
        
        logger.info(f"Prediction made: {result['signal']} (confidence: {result['confidence']:.2%})")
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch-predict", response_model=List[PredictionResponse], tags=["Prediction"])
async def batch_predict(features_list: List[Features]):
    """
    Make predictions for multiple samples.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = []
        for features in features_list:
            feature_dict = features.dict()
            result = predictor.predict_from_dict(feature_dict)
            results.append(result)
        
        logger.info(f"Batch prediction completed: {len(results)} samples")
        
        return results
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    host = config['api']['host']
    port = config['api']['port']
    
    logger.info(f"Starting API server on {host}:{port}")
    logger.info(f"Working directory: {os.getcwd()}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False  # Set to False when running as script
    )