
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from openai import OpenAI
# 移除 Google AI SDK 的匯入，解決 ModuleNotFoundError
# import google.generativeai as genai

# ==================== 頁面配置 ====================
st.set_page_config(page_title="五線譜 + 樂活通道分析", layout="wide")
st.title("五線譜 + 樂活通道 分析系統")

# ==================== Sidebar 設定 ====================
st.sidebar.header("⚙️ 參數設定")

# 股票代號輸入欄位
stock_input = st.sidebar.text_input("股票代號", value="00675L", help="台股請輸入代號,系統會自動加上.TW或.TWO")

# 移除原始的簡單 if/else 判斷，由新的下載函數處理備援邏輯
# if stock_input and not ("." in stock_input):
#     stock_symbol = f"{stock_input}.TW"
# else:
#     stock_symbol = stock_input

# 移除 Gemini 選項，只保留 ChatGPT
ai_model = st.sidebar.selectbox("AI 模型選擇", ["ChatGPT (OpenAI)"])

# 💡 優化 2: API Key 安全處理 - 優先從 Streamlit Secrets 讀取
# 預期在 Streamlit Secrets 中配置為：
# [external_api]
# openai_api_key = "..."
# gemini_api_key = "..."

# 嘗試從 Secrets 讀取 Key
api_key = None
try:
    # 簡化 Key 讀取邏輯，只針對 OpenAI
    api_key = st.secrets["external_api"]["openai_api_key"]
except (KeyError, AttributeError):
    # 如果 Secrets 中沒有配置，則允許用戶通過側邊欄輸入 (主要用於本地測試或臨時輸入)
    pass

st.sidebar.markdown("### 🔑 API Key 配置")
if not api_key:
    st.sidebar.warning("⚠️ Secrets 未配置。請輸入 Key。")
    # 簡化 Key 輸入邏輯，只針對 OpenAI
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
else:
    st.sidebar.success("✅ API Key 已從 Secrets 安全載入。")
    
# --- 期間選擇部分保持不變 ---
period_options = {
    "短期 (0.5年)": 0.5,
    "中期 (1年)": 1.0,
    "長期 (3.5年)": 3.5,
    "超長期 (10年)": 10.0
}

period_type = st.sidebar.selectbox("五線譜分析期間", list(period_options.keys()) + ["自訂期間"], index=2)

if period_type == "自訂期間":
    st.sidebar.markdown("### 📅 自訂日期範圍")
    col_start, col_end = st.sidebar.columns(2)
    with col_start:
        start_date_custom = st.date_input("開始日期", value=datetime.now() - timedelta(days=365*3))
    with col_end:
        end_date_custom = st.date_input("結束日期", value=datetime.now())
    
    days = (end_date_custom - start_date_custom).days
    years = days / 365.0
else:
    years = period_options[period_type]
    days = int(years * 365)

analyze_button = st.sidebar.button("🚀 開始分析", type="primary")

# ==================== 技術指標計算函數 (保持不變) ====================
def calculate_rsi(data, period=14):
    """計算 RSI 指標"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """計算 MACD 指標"""
    ema_fast = data.ewm(span=fast, adjust=False).mean()
    ema_slow = data.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_kd(high, low, close, n=9, m1=3, m2=3):
    """計算 KD 指標"""
    llv = low.rolling(window=n).min()
    hhv = high.rolling(window=n).max()
    rsv = (close - llv) / (hhv - llv) * 100
    k = rsv.ewm(com=m1-1, adjust=False).mean()
    d = k.ewm(com=m2-1, adjust=False).mean()
    return k, d

def detect_rsi_divergence(price, rsi, window=20):
    """檢測 RSI 背離"""
    price_high = price.rolling(window=window).max()
    rsi_high = rsi.rolling(window=window).max()
    
    price_new_high = price == price_high
    rsi_new_high = rsi == rsi_high
    
    # 價格創新高但 RSI 未創新高 = 背離
    divergence = price_new_high & (~rsi_new_high)
    return divergence

# ==================== 🛠️ 數據下載與備援函數 (修正區) ====================

@st.cache_data(ttl=3600)
def get_stock_info(symbol):
    """安全地獲取股票名稱"""
    try:
        ticker = yf.Ticker(symbol)
        stock_info = ticker.info
        stock_name = stock_info.get('longName', symbol)
        return stock_name, symbol
    except:
        return symbol, symbol

# 替換舊的 load_stock_data 函數
@st.cache_data(ttl=3600)
def download_stock_data_with_fallback(stock_input, days):
    """
    下載股票資料並嘗試 .TW 和 .TWO 備援。
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days + 500)
    normalized_input = stock_input.strip().upper()
    
    # 潛在的代號列表：如果用戶沒有輸入後綴，則嘗試 .TW 和 .TWO
    if "." in normalized_input:
        symbol_attempts = [normalized_input]
    else:
        symbol_attempts = [f"{normalized_input}.TW", f"{normalized_input}.TWO"]

    final_symbol = None
    stock_data = None
    
    for symbol in symbol_attempts:
        if symbol == f"{normalized_input}.TWO" and symbol != normalized_input:
             st.warning(f"❌ {symbol_attempts[0]} 下載失敗，嘗試使用 {symbol}...")
        
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        if not data.empty:
            stock_data = data
            final_symbol = symbol
            break

    if stock_data is None:
        return pd.DataFrame(), None, normalized_input # 返回空數據和原始輸入
    
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.get_level_values(0)
    
    stock_name, _ = get_stock_info(final_symbol) # 獲取真實名稱
        
    return stock_data, stock_name, final_symbol

# ==================== 主要分析邏輯 (修正區) ====================
if analyze_button:
    if not stock_input: # 使用 stock_input 檢查是否為空
        st.error("❌ 請輸入股票代號")
    elif not api_key:
        st.error("❌ 請輸入或配置 API Key")
    else:
        try:
            with st.spinner("📥 下載股票資料中..."):
                
                # 呼叫新的健壯下載函數 (替換舊的 load_stock_data)
                stock_data, stock_name, stock_symbol_actual = download_stock_data_with_fallback(stock_input, days)
                
                if stock_data.empty or stock_symbol_actual is None:
                    st.error(f"❌ 無法取得 {stock_input.upper()} 的資料，請檢查代號是否正確。")
                    st.stop()
                
                # 只保留需要分析的區間數據（用於五線譜計算）
                regression_data = stock_data.tail(days).copy()
                regression_data = regression_data.dropna()
                
                st.success(f"✅ 成功載入 {stock_name} ({stock_symbol_actual}) 資料")
            
            # ==================== A. 五線譜計算 (保持不變) ====================
            with st.spinner("📈 計算五線譜..."):
                x_indices = np.arange(len(regression_data))
                y_values = regression_data['Close'].values
                
                slope, intercept = np.polyfit(x_indices, y_values, 1)
                trend_line = slope * x_indices + intercept
                
                residuals = y_values - trend_line
                sd = np.std(residuals)
                
                regression_data['TL'] = trend_line
                regression_data['TL+2SD'] = trend_line + 2 * sd
                regression_data['TL+1SD'] = trend_line + 1 * sd
