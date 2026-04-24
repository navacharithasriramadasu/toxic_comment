"""
Toxic Comment Detection System - FastAPI Backend
Summer Internship Project | Summer of AI (Swecha x IIIT Hyd)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
import logging
import os
import sys

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Toxic Comment Detection API",
    description="AI-powered toxic comment detection using BERT transformer model",
    version="1.0.0",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Models ---
class CommentRequest(BaseModel):
    text: str
    user_id: Optional[str] = "anonymous"

class ToxicityResult(BaseModel):
    text: str
    toxicity_score: float
    is_toxic: bool
    label: str
    confidence: float
    categories: dict
    action: str
    action_reason: str

# --- Load ML Model ---
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.model import ToxicityClassifier
from backend.preprocessor import TextPreprocessor

classifier = ToxicityClassifier()
preprocessor = TextPreprocessor()

# --- Moderation Rules ---
def apply_moderation_rules(score: float, categories: dict) -> tuple[str, str]:
    """Rule-based moderation logic based on toxicity score"""
    severe_toxic = categories.get("severe_toxic", 0)
    threat = categories.get("threat", 0)

    if score >= 0.85 or severe_toxic >= 0.7 or threat >= 0.6:
        return "BLOCK", "Content severely violates community guidelines"
    elif score >= 0.60:
        return "FLAG", "Content flagged for moderator review"
    elif score >= 0.35:
        return "WARN", "Please review and edit your comment before posting"
    else:
        return "ALLOW", "Comment meets community standards"


# --- Endpoints ---
@app.get("/")
def root():
    return FileResponse(os.path.join(os.path.dirname(__file__), "../frontend/index.html"))


@app.post("/api/analyze", response_model=ToxicityResult)
async def analyze_comment(request: CommentRequest):
    """Analyze a comment for toxicity using BERT model"""
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Comment text cannot be empty")

    if len(request.text) > 5000:
        raise HTTPException(status_code=400, detail="Comment exceeds maximum length of 5000 characters")

    try:
        # Preprocess text
        cleaned_text = preprocessor.clean(request.text)

        # Get model prediction
        result = classifier.predict(cleaned_text)

        # Apply moderation rules
        action, reason = apply_moderation_rules(result["toxicity_score"], result["categories"])

        logger.info(f"Analyzed comment | Score: {result['toxicity_score']:.3f} | Action: {action}")

        return ToxicityResult(
            text=request.text,
            toxicity_score=round(result["toxicity_score"], 4),
            is_toxic=result["is_toxic"],
            label=result["label"],
            confidence=round(result["confidence"], 4),
            categories=result["categories"],
            action=action,
            action_reason=reason,
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "model": classifier.model_name,
        "device": classifier.device,
    }


@app.get("/api/stats")
def get_stats():
    return classifier.get_stats()


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
