import asyncio
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from flask import json
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
import requests
from google import genai
from openai import OpenAI
from auth_utils import validate_jwt

app = FastAPI(title="Stock Evaluator API")

origins = [
    "http://localhost:5173",  # your React frontend
    "http://127.0.0.1:5173"
]

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can specify frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB Connection
MONGO_URI= "mongodb://localhost:27017/"

try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10000)
    client.admin.command("ping")
    print("✅ MongoDB connected successfully!")
except Exception as e:
    print("❌ MongoDB connection failed:", e)

db = client["stockdb"]
collection = db["stocks"]

# Stock Analysis Function
def analyze_stock_data(stock_data: dict):
    symbol = stock_data["stockSymbol"]
    pe_ratio = stock_data.get("priceEarningsRatio", 0)
    eps = stock_data.get("earningsPerShare", 0)
    dividend_yield = stock_data.get("dividendYield", 0)
    market_cap = stock_data.get("marketCap", 0)
    debt_to_equity = stock_data.get("debtToEquityRatio", 0)
    roe = stock_data.get("returnOnEquity", 0)
    roa = stock_data.get("returnOnAssets", 0)
    current_ratio = stock_data.get("currentRatio", 0)
    quick_ratio = stock_data.get("quickRatio", 0)
    book_value = stock_data.get("bookValuePerShare", 0)

    feedback = {}

    # P/E Ratio
    if pe_ratio < 15:
        feedback["priceEarningsRatio"] = f"The P/E ratio of {pe_ratio} suggests the stock is cheap relative to earnings."
    elif 15 <= pe_ratio <= 30:
        feedback["priceEarningsRatio"] = f"The P/E ratio of {pe_ratio} is fairly typical."
    else:
        feedback["priceEarningsRatio"] = f"The P/E ratio of {pe_ratio} indicates the stock is relatively expensive compared to its earnings."

    # EPS
    if eps < 1:
        feedback["earningsPerShare"] = f"The EPS of {eps} is low; profitability may be a concern."
    elif eps < 5:
        feedback["earningsPerShare"] = f"The EPS of {eps} shows modest profitability."
    else:
        feedback["earningsPerShare"] = f"The EPS of {eps} is a strong indicator of the company's profitability."

    # Dividend Yield
    if dividend_yield < 1:
        feedback["dividendYield"] = f"The dividend yield of {dividend_yield}% is lower than the market average."
    elif dividend_yield <= 3:
        feedback["dividendYield"] = f"The dividend yield of {dividend_yield}% is around the market norm."
    else:
        feedback["dividendYield"] = f"The dividend yield of {dividend_yield}% is attractive for income-focused investors."

    # Market Cap
    trillion = 1_000_000_000_000
    if market_cap >= 500 * trillion:
        feedback["marketCap"] = f"The market cap of ${market_cap/trillion:.2f} trillion makes it one of the world’s giants."
    elif market_cap >= 100 * trillion:
        feedback["marketCap"] = f"The market cap of ${market_cap/trillion:.2f} trillion indicates a very large, stable company."
    else:
        feedback["marketCap"] = f"The market capitalization of ${market_cap/1_000_000_000:.2f} billion indicates a sizable player."

    # Debt-to-Equity
    if debt_to_equity < 0.5:
        feedback["debtToEquityRatio"] = f"The debt-to-equity ratio of {debt_to_equity} suggests very little leverage."
    elif debt_to_equity <= 1.5:
        feedback["debtToEquityRatio"] = f"The debt-to-equity ratio of {debt_to_equity} suggests a moderate level of leverage."
    else:
        feedback["debtToEquityRatio"] = f"The debt-to-equity ratio of {debt_to_equity} indicates high leverage; watch for risk."

    # ROE
    pct_roe = roe * 100
    if pct_roe < 8:
        feedback["returnOnEquity"] = f"The ROE of {pct_roe:.2f}% is below average."
    elif pct_roe <= 15:
        feedback["returnOnEquity"] = f"The ROE of {pct_roe:.2f}% is healthy."
    else:
        feedback["returnOnEquity"] = f"The ROE of {pct_roe:.2f}% is very strong, showing efficient profit generation."

    # ROA
    pct_roa = roa * 100
    if pct_roa < 5:
        feedback["returnOnAssets"] = f"The ROA of {pct_roa:.2f}% is modest."
    elif pct_roa <= 10:
        feedback["returnOnAssets"] = f"The ROA of {pct_roa:.2f}% indicates efficient asset utilization."
    else:
        feedback["returnOnAssets"] = f"The ROA of {pct_roa:.2f}% is excellent, showing superb asset productivity."

    # Current Ratio
    if current_ratio < 1:
        feedback["currentRatio"] = f"The current ratio of {current_ratio} signals potential short-term liquidity issues."
    elif current_ratio <= 2:
        feedback["currentRatio"] = f"The current ratio of {current_ratio} suggests the company has a good short-term liquidity position."
    else:
        feedback["currentRatio"] = f"The current ratio of {current_ratio} indicates a very comfortable liquidity cushion."

    # Quick Ratio
    if quick_ratio < 1:
        feedback["quickRatio"] = f"The quick ratio of {quick_ratio} may be insufficient for immediate obligations."
    elif quick_ratio <= 2:
        feedback["quickRatio"] = f"The quick ratio of {quick_ratio} indicates a strong ability to meet short-term obligations."
    else:
        feedback["quickRatio"] = f"The quick ratio of {quick_ratio} shows an exceptionally strong liquidity position."

    # Book Value
    feedback["bookValuePerShare"] = f"The book value per share of {book_value:.2f} is a measure of the company's net asset value on a per-share basis."

    # Recommendation Logic
    buy_score = 0
    sell_score = 0
    if pe_ratio < 15: buy_score += 1
    if pe_ratio > 30: sell_score += 1
    if eps > 5: buy_score += 1
    if eps < 1: sell_score += 1
    if debt_to_equity < 0.5: buy_score += 1
    if debt_to_equity > 1.5: sell_score += 1
    if roe * 100 > 15: buy_score += 1
    if roe * 100 < 8: sell_score += 1

    if buy_score >= 3:
        recommendation = "Buy"
        summary = "This stock shows strong potential with multiple positive indicators pointing towards being undervalued and financially healthy."
    elif sell_score >= 2:
        recommendation = "Sell"
        summary = "This stock shows several red flags, such as high valuation or financial risks, that investors should be wary of."
    else:
        recommendation = "Hold"
        summary = "This stock has a mix of positive and negative indicators, suggesting a neutral stance. It is best to hold and monitor its performance."

    return {
        "stockSymbol": symbol,
        "values": stock_data,
        "feedback": feedback,
        "summary": summary,
        "recommendation": recommendation,
        "source": "MongoDB"
    }

# --- Updated /evaluate endpoint with fallback caching ---
@app.get("/evaluate/{symbol}")
def evaluate_stock(symbol: str):
    symbol = symbol.strip().upper()
    stock_doc = collection.find_one({"stockSymbol": symbol}, {"_id": 0})

    if stock_doc:
        # Stock found in DB
        stock_data = {"stockSymbol": symbol, **stock_doc.get("parameters", {})}
        return analyze_stock_data(stock_data)

    # Stock not found -> call fallback API
    try:
        fallback_url = f"http://127.0.0.1:8100/analyze?symbol={symbol}&days=10"
        res = requests.get(fallback_url)
        res.raise_for_status()
        data = res.json()

        # Normalize output
        normalized_values = data.get("fundamentals", {}).get("raw", {})
        feedback = data.get("fundamentals", {}).get("feedback", {})
        recommendation = data.get("recommendation", {}).get("recommendation", None)
        summary = f"Overall rating: {data.get('fundamentals', {}).get('overallRating', '-')}"

        # Save normalized result into MongoDB for future requests
        collection.insert_one({
            "stockSymbol": symbol,
            "parameters": normalized_values
        })

        # Return in same structure as analyze_stock_data output
        return {
            "stockSymbol": symbol,
            "values": normalized_values,
            "feedback": feedback,
            "summary": summary,
            "recommendation": recommendation,
            "source": "Fallback API (cached)"
        }

    except requests.RequestException as e:
        raise HTTPException(status_code=404,
            detail=f"Stock {symbol} not found and fallback API failed: {str(e)}")

# Get all stock symbols
# --- New endpoint: proxy feedback to external API ---
""" API_KEY = "AIzaSyAsxQ2l6tnGZr9LUq7PbZaFgxp4Wm6Js5c"
client = genai.Client(api_key=API_KEY) """


client1 = OpenAI(
    api_key="AIzaSyAsxQ2l6tnGZr9LUq7PbZaFgxp4Wm6Js5c",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

@app.post("/send-feedback-stream")
async def send_feedback_stream(stock_payload: dict):
    feedback = stock_payload.get("feedback", {})
    if not feedback:
        raise HTTPException(status_code=400, detail="No feedback to summarize.")

    feedback_text = "\n".join([f"{k}: {v}" for k, v in feedback.items()])
    prompt = f"""
You are a professional financial analyst. Based on the following 10 financial metrics of a company, synthesize them into **key, concise, and professional bullet points**. 

- Group related positive and negative points together.
- Highlight important metrics that indicate strengths or risks.
- Avoid writing a paragraph summary; output only **bullet points**.
- Keep the language formal, precise, and relevant for investors.
- Do not include irrelevant details.
-Not too long make it short and crisp and proper content.

{feedback_text}
"""

    async def event_stream():
        try:
            response = client1.chat.completions.create(
                model="gemini-2.5-flash",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                stream=True
            )

            # iterate through streaming chunks
            for chunk in response:
                for choice in chunk.choices:
                    if hasattr(choice.delta, "content") and choice.delta.content:
                        # SSE: send each chunk immediately
                        yield f"data: {json.dumps(choice.delta.content)}\n\n"
                        await asyncio.sleep(0.01)  # small delay to avoid blocking

        except Exception as e:
            yield f"data: {json.dumps('Error: ' + str(e))}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
    
    



    
@app.get("/stocks")
def get_stock_symbols(valid: bool = Depends(validate_jwt)):
    symbols = collection.distinct("stockSymbol")
    if not symbols:
        raise HTTPException(status_code=404, detail="No stocks found in database.")
    return {"stocks": symbols}
