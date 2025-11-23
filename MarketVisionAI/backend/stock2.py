import pandas as pd
import json
import os

# --- Data Loading ---
# Get the directory of this script, so we can reliably load the CSV
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "StocksTickerSymbols.csv")

try:
    stock_df = pd.read_csv(csv_path)
    stock_df.set_index('stockSymbol', inplace=True)
    print(f"Loaded CSV successfully from: {csv_path}")
except FileNotFoundError:
    print(f"FATAL ERROR: '{csv_path}' not found. The application cannot start without it.")
    stock_df = None

# --- Core Analysis Function ---
def analyze_stock_data(stock_data):
    symbol = stock_data.name
    pe_ratio = stock_data['priceEarningsRatio']
    eps = stock_data['earningsPerShare']
    dividend_yield = stock_data['dividendYield']
    market_cap = stock_data['marketCap']
    debt_to_equity = stock_data['debtToEquityRatio']
    roe = stock_data['returnOnEquity']
    roa = stock_data['returnOnAssets']
    current_ratio = stock_data['currentRatio']
    quick_ratio = stock_data['quickRatio']
    book_value = stock_data['bookValuePerShare']

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

    # --- Summary & Recommendation ---
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
        "values": stock_data.to_dict(),
        "feedback": feedback,
        "summary": summary,
        "recommendation": recommendation,
    }

# --- Backend API Function ---
def get_report_for_symbol(symbol: str):
    if stock_df is None:
        return json.dumps({"error": "Stock dataset not loaded. Please check server logs."})

    try:
        stock_data = stock_df.loc[symbol]
        report = analyze_stock_data(stock_data)
        return json.dumps(report, indent=4)
    except KeyError:
        return json.dumps({"error": f"Stock symbol '{symbol}' not found in the dataset."}, indent=4)

# --- Testing ---
if __name__ == "__main__":
    print(f"\n--- Generating Report for STK005 ---")
    print(get_report_for_symbol("STK005"))
