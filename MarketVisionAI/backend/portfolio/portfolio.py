from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from itertools import combinations
from typing import List, Dict, Union, Optional
from pydantic import BaseModel
import json
import os
import google.generativeai as genai

# Set Gemini API key
os.environ["GEMINI_API_KEY"] = "AIzaSyCD4D18O6PqfAkniAOB6d87ZY3evFX1MvI"
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Choose Gemini model
MODEL_NAME = "models/gemini-2.5-flash"

# Create Gemini model with JSON-only output
model = genai.GenerativeModel(
    MODEL_NAME,
    generation_config={"response_mime_type": "application/json"}
)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Fund(BaseModel):
    fundCode: str
    amount: float  # Fixed type error
    holdings: Dict[str, float]
    sectors: Dict[str, float]

class Portfolio(BaseModel):
    clientId: str
    currency: str
    funds: List[Fund]

def load_client_portfolios():
    try:
        with open("../ClientPortfolio.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading portfolio data: {e}")
        return []

def pairwise_overlap_matrix(funds: List[Fund]):
    overlaps = {}
    vals = []
    maxv = 0.0
    for (i, fa), (j, fb) in combinations(enumerate(funds), 2):
        ha = fa.holdings or {}
        hb = fb.holdings or {}
        ov = sum(min(float(ha.get(t,0.0)), float(hb.get(t,0.0))) for t in set(ha)|set(hb))
        overlaps[f"{fa.fundCode} vs {fb.fundCode}"] = round(ov*100, 2)
        vals.append(ov)
        maxv = max(maxv, ov)
    avg = (sum(vals)/len(vals)) if vals else 0.0
    return overlaps, avg, maxv

def weighted_sector_exposure(funds: List[Fund]):
    total = sum(f.amount for f in funds) or 1.0
    sector_w = {}
    for f in funds:
        share = f.amount / total
        for sec, w in f.sectors.items():
            try:
                w = float(w)
            except:
                continue
            sector_w[sec] = sector_w.get(sec, 0.0) + share * w
    return sector_w

def analyze_portfolio(portfolio: Portfolio):
    funds = portfolio.funds
    total_value = sum(f.amount for f in funds)

    overlaps, avg_ov, _ = pairwise_overlap_matrix(funds)
    overlap_score = (1 - avg_ov) * 100

    sector_w = weighted_sector_exposure(funds)
    sector_pct = {k: round(v*100, 2) for k,v in sorted(sector_w.items(), key=lambda kv:-kv[1])}
    hhi = sum(v*v for v in sector_w.values())
    sector_score = (1 - hhi) * 100

    final_score = 0.5*overlap_score + 0.5*sector_score

    return {
        "clientId": portfolio.clientId,
        "currency": portfolio.currency,
        "totalValue": total_value,
        "pairwiseOverlaps": overlaps,
        "averageOverlap": round(avg_ov*100, 2),
        "overlapScore": round(overlap_score, 2),
        "sectorDiversification": sector_pct,
        "HHI": round(hhi, 3),
        "sectorScore": round(sector_score, 2),
        "finalDiversificationScore": round(final_score, 2)
    }

@app.get("/clients")
async def get_clients():
    portfolios = load_client_portfolios()
    return [{"clientId": p["clientId"], "currency": p["currency"]} for p in portfolios]

@app.get("/analyze-portfolio/{client_id}")
async def analyze_portfolio_by_id(client_id: str):
    portfolios = load_client_portfolios()
    portfolio = next((p for p in portfolios if p["clientId"] == client_id), None)
    
    if not portfolio:
        raise HTTPException(status_code=404, detail=f"Client {client_id} not found")
    
    return analyze_portfolio(Portfolio(**portfolio))

def get_portfolio_analysis(client_data: dict) -> dict:
    """
    Sends client portfolio data to Gemini and gets structured JSON analysis.
    """
    prompt = f"""
    You are a financial analyst AI. Analyze the following client portfolio and return ONLY a valid JSON object.
Do NOT include markdown, code blocks, or text outside the JSON.

**Client Data:**
{json.dumps(client_data, indent=2)}

**Instructions:**
1. Compute total portfolio value.
2. Evaluate fund overlaps and diversification metrics.
3. Based on current diversification, suggest at least TWO additional sectors
   the client should consider for diversification.
4. For each suggested sector, include a clear recommendation.
5.give it in simple terms 

**Required JSON Format:**
{{
  "portfolioAnalysis": {{
    "clientId": "{client_data.get('clientId', 'Unknown')}",
    "currency": "{client_data.get('currency', '')}",
    "totalValue": 0.0
  }},
  "scores": {{
    "pairwiseOverlaps": {{"FUND_X vs FUND_Y": 0.0}},
    "averageOverlap": 0.0,
    "overlapScore": 0.0,
    "HHI": 0.000,
    "sectorScore": 0.0,
    "finalDiversificationScore": 0.0
  }},
  "possibleDiversification": [
    {{"sector": "Technology", "recommendation": "Increase allocation to technology for growth"}},
    {{"sector": "Healthcare", "recommendation": "Add healthcare exposure to reduce correlation"}}
  ],
  "traderType": "...",
  "summary": "..."
}}

Return ONLY this JSON. Nothing else.
    """

    try:
        response = model.generate_content(contents=prompt)
        return json.loads(response.text)  # already JSON, thanks to response_mime_type
    except Exception as e:
        print("❌ Gemini error:", e)
        if 'response' in locals():
            print("Raw response:", response.text)
        return None

@app.get("/ai-summary/{client_id}")
async def get_ai_summary(client_id: str):
    portfolios = load_client_portfolios()
    portfolio = next((p for p in portfolios if p["clientId"] == client_id), None)
    
    if not portfolio:
        raise HTTPException(status_code=404, detail=f"Client {client_id} not found")
    
    ai_analysis = get_portfolio_analysis(portfolio)
    if not ai_analysis:
        raise HTTPException(status_code=500, detail="Failed to generate AI analysis")
    
    return ai_analysis

@app.get("/")
async def root():
    return {"message": "Portfolio Analyzer API is running"}