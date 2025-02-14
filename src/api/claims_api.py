from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List, Dict
import logging
from pathlib import Path
import aiofiles
import json
from datetime import datetime
import pandas as pd

from src.claims_processing.claims_processor import ClaimsProcessor
from src.services.parts_mapping_service import PartsMapper

app = FastAPI(title="Insurance Claims API")
logger = logging.getLogger(__name__)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize claims processor
claims_processor = ClaimsProcessor()

# Initialize parts mapper
parts_mapper = PartsMapper()

@app.post("/api/claims/process")
async def process_claim(claim_data: Dict, images: List[UploadFile] = None):
    """Process a new insurance claim"""
    try:
        # Save images if provided
        image_paths = []
        if images:
            for image in images:
                image_path = await save_image(image)
                image_paths.append(image_path)

        # Process claim
        result = await claims_processor.process_claim(
            claim_data=claim_data,
            images=image_paths
        )
        
        return result
    except Exception as e:
        logger.error(f"API Error - Process Claim: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def save_image(image: UploadFile) -> str:
    """Save uploaded image"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{image.filename}"
        filepath = Path("data/uploads") / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(filepath, 'wb') as f:
            content = await image.read()
            await f.write(content)
            
        return str(filepath)
    except Exception as e:
        logger.error(f"Failed to save image: {e}")
        raise

@app.get("/api/claims/{claim_id}")
async def get_claim(claim_id: str):
    """Get claim details by ID"""
    try:
        # Load historical claims
        claims_path = Path("data/historical_claims.csv")
        if not claims_path.exists():
            raise HTTPException(status_code=404, detail="No claims found")
            
        claims_df = pd.read_csv(claims_path)
        claim = claims_df[claims_df['claim_id'] == claim_id]
        
        if claim.empty:
            raise HTTPException(status_code=404, detail="Claim not found")
            
        return claim.to_dict('records')[0]
    except Exception as e:
        logger.error(f"API Error - Get Claim: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/claims/statistics")
async def get_statistics():
    """Get claims processing statistics"""
    try:
        claims_path = Path("data/historical_claims.csv")
        if not claims_path.exists():
            return {
                "total_claims": 0,
                "average_processing_time": 0,
                "fraud_detection_rate": 0
            }
            
        claims_df = pd.read_csv(claims_path)
        
        return {
            "total_claims": len(claims_df),
            "approved_claims": len(claims_df[claims_df['recommendation'] == 'approve']),
            "rejected_claims": len(claims_df[claims_df['recommendation'] == 'reject']),
            "under_review": len(claims_df[claims_df['recommendation'] == 'review']),
            "average_fraud_risk": claims_df['fraud_risk'].mean(),
            "total_amount": claims_df['total_amount'].sum()
        }
    except Exception as e:
        logger.error(f"API Error - Get Statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check if data files exist
        claims_path = Path("data/historical_claims.csv")
        parts_path = Path("data/Primary_Parts_Code.csv")
        
        if not claims_path.exists() or not parts_path.exists():
            raise HTTPException(
                status_code=503,
                detail="Required data files not found"
            )
            
        # Verify data file contents
        parts_df = pd.read_csv(parts_path)
        required_columns = ['Surveyor Part Code', 'Surveyor Part Name', 'Category', 'Average_Cost']
        missing_columns = [col for col in required_columns if col not in parts_df.columns]
        
        if missing_columns:
            raise HTTPException(
                status_code=503,
                detail=f"Missing columns in parts data: {missing_columns}"
            )
            
        return {
            "status": "healthy",
            "data_files": {
                "claims": str(claims_path),
                "parts": str(parts_path)
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/parts/mapping/stats")
async def get_mapping_stats():
    """Get parts mapping statistics"""
    try:
        stats = parts_mapper.get_mapping_statistics()
        return stats
    except Exception as e:
        logger.error(f"Failed to get mapping stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/parts/map")
async def map_part(part_description: str):
    """Map a part description to standard part"""
    try:
        result = await parts_mapper.map_part(part_description)
        return result
    except Exception as e:
        logger.error(f"Part mapping failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 