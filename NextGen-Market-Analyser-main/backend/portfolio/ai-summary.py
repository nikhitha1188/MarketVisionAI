import os
import json
import google.generativeai as genai

# 🔑 Set your Gemini API key
os.environ["GEMINI_API_KEY"] = "AIzaSyCD4D18O6PqfAkniAOB6d87ZY3evFX1MvI"
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# ✅ Choose a stable Gemini model
MODEL_NAME = "models/gemini-2.5-lite"

# ✅ Create Gemini model with JSON-only output
model = genai.GenerativeModel(
    MODEL_NAME,
    generation_config={"response_mime_type": "application/json"}
)

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
    calculate based on the sectors they have invested and make the diversified and suggest them the sector with reason in simple english
  ],
  "traderType": "...",
  "summary": "..."
}}

Return ONLY this JSON. Nothing else. give only the 
    """

    try:
        response = model.generate_content(contents=prompt)
        return json.loads(response.text)  # already JSON, thanks to response_mime_type
    except Exception as e:
        print("❌ Gemini error:", e)
        if 'response' in locals():
            print("Raw response:", response.text)
        return None

# ✅ Example client portfolio
sample_data = {
    "clientId": "C103",
    "currency": "USD",
    "funds": [
      {
        "fundCode": "FUND_F",
        "amount": 2925000,
        "holdings": {
          "JPM": 0.23,
          "AMZN": 0.4,
          "NVDA": 0.37
        },
        "sectors": {
          "Financials": 0.23,
          "Consumer Discretionary": 0.4,
          "Technology": 0.37
        }
      },
      {
        "fundCode": "FUND_A",
        "amount": 7015000,
        "holdings": {
          "MSFT": 0.18,
          "AAPL": 0.34,
          "GOOGL": 0.33,
          "AMZN": 0.15
        },
        "sectors": {
          "Technology": 0.85,
          "Consumer Discretionary": 0.15
        }
      }
    ]
  }

# ✅ Run analysis
result = get_portfolio_analysis(sample_data)

# ✅ Print structured output
if result:
    print(json.dumps(result, indent=2))
