import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
# 移除 OpenAI 匯入， App 不再需要任何 API Key
# from openai import OpenAI

# ==================== 🛠️ 自訂 CSS 樣式 (終極日雜風格) ====================
custom_css = """
<style>
/* 隱藏 Streamlit 頁腳和菜單按鈕 */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* 全局背景色與字體：柔和的米白和深灰 */
body, .main, .st-emotion-cache-1dp6dkb {
    background-color: #fdfdfd; /* 極淺米白 */
    color: #5A5A5A; /* 柔和深灰 */
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans TC", sans-serif;
}

/* 標題調整：降低視覺重量，強調簡潔 */
.st-emotion-cache-10trblm {
    color: #4A4A4A; 
    font-weight: 400; /* 更纖細 */
    border-bottom: 1px solid #E5E5E5; /* 極細下劃線 */
    padding-bottom: 5px;
    margin-bottom: 15px;
}

/* 側邊欄調整 */
.st-emotion-cache-vk3ypz {
    background-color: #f7f7f7; /* 淺灰色側邊欄 */
    border-right: 1px solid #E0E0E0;
    padding-top: 1.5rem; /* 增加頂部留白 */
}

/* 輸入框/選擇框的樣式：圓潤且柔和的邊框 */
.st-emotion-cache-1cypcdb, .st-emotion-cache-1wmy99i { /* 涵蓋多種輸入元件 */
    border-radius: 8px; /* 柔和圓角 */
    border: 1px solid #D9D9D9; /* 淺色邊框 */
    box-shadow: none !important; /* 移除預設陰影 */
    background-color: white;
}

/* 調整主要的 Metric 區塊 (卡片風格) */
.st-emotion-cache-1cypcdb {
    border: 1px solid #EBEBEB; /* 更淺、更自然感的邊框 */
    border-radius: 12px;
    padding: 15px;
    background-color: #fffffe; 
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.02); /* 極輕微、分散的陰影 */
}

/* Metric 的指標文字顏色 (日雜強調色: 淺棕色/大地色) */
.css-1r6rthg {
    color: #9E8974 !important; /* 更深的柔和棕色 */
    font-weight: 600;
    font-size: 1.6rem !important;
}

/* 按鈕樣式 ( primary 按鈕使用強調色) */
.st-emotion-cache-hkqjaj button[data-testid="baseButton-primary"] {
    background-color: #B0A595; 
    color: white;
    border-radius: 8px;
    border: none;
    font-weight: 500;
    transition: background-color 0.2s;
}
.st-emotion-cache-hkqjaj button[data-testid="baseButton-primary"]:hover {
    background-color: #917C64; /* 懸停時略深 */
}

/* 資訊/警告框的樣式調整，使其更柔和 */
[data-testid="stAlert"] {
    border-left: 5px solid #EBD5D5; /* 警告色柔和化 */
    background-color: #FEFCFB;
    color: #5A5A5A;
    border-radius: 8px;
}
</style>
"""

# ==================== 頁面配置 ====================
st.set_page_config(page_title="五線譜 + 樂活通道分析")
st.title("五線譜 + 樂活通道 分析系統")

# 注入自訂 CSS
st.markdown(custom_css, unsafe_allow_html=True)


# ==================== Sidebar 設定 ====================
st.sidebar.header("⚙️ 參數設定")

stock_input = st.sidebar.text_input("股票代號", value="00675L", help="台股請輸入代號,系統會自動加上.TW或.TWO")

# 移除 AI 模型選擇
# ai_model = st.sidebar.selectbox("AI 模型選擇", ["ChatGPT (OpenAI)"])

# 💡 API Key 處理：全部移除， App 不再需要 API Key
# 相關程式碼已被移除

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

# ==================== 技術指標計算函數 (新增進階指標) ====================
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

# 🌟 新增指標 1: 計算 ADX (趨勢強度)
def calculate_adx(high, low, close, period=14):
    """計算 ADX, +DI, -DI 指標 (DMI)"""
    df = pd.DataFrame({'High': high, 'Low': low, 'Close': close})
    
    # 1. 計算 True Range (TR)
    df['TR'] = np.maximum.reduce([df['High'] - df['Low'], 
                                  abs(df['High'] - df['Close'].shift(1)), 
                                  abs(df['Low'] - df['Close'].shift(1))])
    
    # 2. 計算 Directional Movement (+DM, -DM)
    df['+DM'] = (df['High'] - df['High'].shift(1)).clip(lower=0)
    df['-DM'] = (df['Low'].shift(1) - df['Low']).clip(lower=0)
    
    idx = df['+DM'] > df['-DM']
    df.loc[idx, '-DM'] = 0
    df.loc[~idx, '+DM'] = 0
    
    # 3. 平滑處理 (Welles Wilder smoothing)
    alpha = 1/period
    df['ATR'] = df['TR'].ewm(alpha=alpha, adjust=False).mean()
    df['+DMI'] = df['+DM'].ewm(alpha=alpha, adjust=False).mean()
    df['-DMI'] = df['-DM'].ewm(alpha=alpha, adjust=False).mean()
    
    # 4. 計算 Directional Index (DI)
    df['+DI'] = (df['+DMI'] / df['ATR']) * 100
    df['-DI'] = (df['-DMI'] / df['ATR']) * 100
    
    # 5. 計算 Directional Movement Index (DX)
    sum_di = df['+DI'] + df['-DI']
    df['DX'] = (abs(df['+DI'] - df['-DI']) / sum_di.replace(0, np.nan)) * 100
    
    # 6. 計算 Average Directional Index (ADX)
    df['ADX'] = df['DX'].ewm(alpha=alpha, adjust=False).mean()
    
    return df['ADX'], df['+DI'], df['-DI']

# 🌟 新增指標 2: 計算 Bollinger Band Width (BBW)
def calculate_bbw(close, period=20, std_dev=2):
    """計算布林帶寬度 (BBW)"""
    ma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    # 避免除以零
    bbw = (2 * std_dev * std) / ma.replace(0, np.nan)
    return bbw


# 🌟 新增指標 3: 計算 Williams %R (威廉指標 - 情緒超買賣)
def calculate_williams_r(high, low, close, period=14):
    """計算 Williams %R"""
    hhv = high.rolling(window=period).max()
    llv = low.rolling(window=period).min()
    # 避免除以零
    range_hl = hhv - llv
    williams_r = -100 * (hhv - close) / range_hl.replace(0, np.nan)
    return williams_r


# ==================== 數據下載與備援函數 (保持不變) ====================

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

@st.cache_data(ttl=3600) 
def download_stock_data_with_fallback(stock_input, days):
    """
    下載股票資料並嘗試 .TW 和 .TWO 備援。
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days + 500)
    normalized_input = stock_input.strip().upper()
    
    if "." in normalized_input:
        symbol_attempts = [normalized_input]
    else:
        symbol_attempts = [f"{normalized_input}.TW", f"{normalized_input}.TWO"]

    final_symbol = None
    stock_data = None
    
    for symbol in symbol_attempts:
        if symbol.endswith(".TWO"):
             st.warning(f"❌ {normalized_input}.TW 下載失敗，嘗試使用 {symbol}...")
        
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        if not data.empty:
            stock_data = data
            final_symbol = symbol
            break

    if stock_data is None:
        return pd.DataFrame(), None, normalized_input
    
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.get_level_values(0)
    
    stock_name, _ = get_stock_info(final_symbol)
        
    return stock_data, stock_name, final_symbol

# ==================== 🛠️ 智能分析生成函數 (方案 B 核心 - 整合情緒) ====================

def generate_internal_analysis(stock_name, stock_symbol, slope_dir, sd_level, fiveline_zone, current, sell_signals, buy_signals, full_bbw_series):
    """
    根據多種技術指標的硬編碼規則，生成分析摘要。
    """
    analysis_text = []

    # 提取指標
    current_adx = current['ADX']
    current_plus_di = current['+DI']
    current_minus_di = current['-DI']
    current_bbw = current['BBW']
    current_williams_r = current['%R']
    current_v_ratio = current['Volume_Ratio']
    
    # 計算歷史 BBW 分位數 (修正點)
    bbw_quantile = full_bbw_series.quantile(0.1)
    
    # --- 1. 趨勢與動能判斷 (Trend & Momentum) ---
    analysis_text.append("### 1. 趨勢與動能判斷 (Trend & Momentum)")
    
    adx_strength = ""
    if current_adx > 30:
        adx_strength = f"ADX ({current_adx:.1f}) 顯示**趨勢強度非常高**，應順勢操作。"
    elif current_adx > 20:
        adx_strength = f"ADX ({current_adx:.1f}) 顯示趨勢強度中等，趨勢正在確立。"
    else:
        adx_strength = f"ADX ({current_adx:.1f}) 顯示**趨勢強度較弱**，可能處於盤整或反轉前夕。"
    
    if slope_dir == "上升":
        trend_summary = f"五線譜趨勢：明確為**上升**，股價位於 {fiveline_zone}。{adx_strength}"
    elif slope_dir == "下降":
        trend_summary = f"五線譜趨勢：明確為**下降**，股價位於 {fiveline_zone}。{adx_strength}"
    else:
        trend_summary = f"五線譜趨勢：**盤整或觀望**。{adx_strength}"
        
    analysis_text.append(trend_summary + "\n")

    # --- 2. 市場情緒與波動性分析 (Sentiment & Volatility) ---
    analysis_text.append("### 2. 市場情緒與波動性分析")
    
    sentiment_analysis = []
    
    # 2.1 威廉指標 (%R) 判斷極端情緒
    if current_williams_r > -20: 
        sentiment_analysis.append(f"🔴 **極度樂觀：** 威廉指標 (%R: {current_williams_r:.1f}%) 處於超買區，市場情緒過熱，存在回調壓力。")
    elif current_williams_r < -80:
        sentiment_analysis.append(f"🟢 **極度悲觀：** 威廉指標 (%R: {current_williams_r:.1f}%) 處於超賣區，市場情緒偏向恐慌，可能醞釀技術性反彈。")
    
    # 2.2 成交量比率判斷狂熱度
    if current_v_ratio > 1.8: # 更嚴格的熱度判斷
        sentiment_analysis.append(f"⚠️ **成交狂熱：** 成交量 ({current_v_ratio:.1f}倍均量) 異常放大，需警惕狂熱性追漲或恐慌性拋售。")
    
    # 2.3 BBW 判斷收縮
    if current_bbw < bbw_quantile: 
        sentiment_analysis.append(f"🔲 **波動性收縮：** 價格壓縮至極致，預期短期內將有**方向性大變動**。")
    
    if not sentiment_analysis:
        analysis_text.append("市場情緒和波動性指標處於正常範圍，無極端訊號。\n")
    else:
        analysis_text.append("\n".join(sentiment_analysis) + "\n")
    
    # --- 3. 綜合操作建議 (Trading Recommendation) ---
    analysis_text.append("### 3. 綜合操作建議")
    
    # 優先處理極端情緒下的操作
    if current_williams_r > -20 and sell_signals:
        rec = f"**極度危險**：情緒超買且有 {len(sell_signals)} 個賣出訊號。建議投資人**立即清倉或空手**，風險極高。"
    elif current_williams_r < -80 and buy_signals and current_adx < 25:
        rec = "**中線布局機會**：情緒極度悲觀。可考慮**極小額試單**，但需確認 ADX 是否開始上揚，設嚴格止損。"
    elif current_bbw < bbw_quantile and current_adx < 20:
        rec = "**靜待時機**：市場處於暴風雨前的寧靜。建議在價格突破盤整區間前，保持場外觀望。"
    elif sell_signals:
        rec = f"鑑於當前有 {len(sell_signals)} 個賣出訊號，建議投資人**減碼或空手觀望**，以順應趨勢。"
    elif buy_signals:
        rec = f"當前有 {len(buy_signals)} 個買入訊號，建議可考慮**分批進場**，並緊盯 ADX 確認趨勢強度。"
    else:
        rec = "多數指標訊號不明確。建議**保持觀望**，等待更明確的買賣轉折訊號出現。"
        
    analysis_text.append(rec + "\n")
    
    # 4. 風險提示
    analysis_text.append("### 4. 聲明與風險提示")
    analysis_text.append(f"本分析為基於多重技術指標 (KD/RSI/MACD/DMI/BBW/%R/V-Ratio) 的**程式碼硬編碼判斷**，**不依賴外部 AI**，且不構成任何投資建議。所有交易決策請自行承擔風險。")
    
    return "\n".join(analysis_text)


# ==================== 主要分析邏輯 (修正點) ====================
if analyze_button:
    if not stock_input:
        st.error("❌ 請輸入股票代號")
    else:
        try:
            with st.spinner("📥 下載股票資料中..."):
                
                stock_data, stock_name, stock_symbol_actual = download_stock_data_with_fallback(stock_input, days)
                
                if stock_data.empty or stock_symbol_actual is None:
                    st.error(f"❌ 嚴重錯誤：無法取得 {stock_input.upper()} 的資料，請檢查代號是否正確。")
                    st.stop()
                
                regression_data = stock_data.tail(days).copy()
                regression_data = regression_data.dropna()
                
                st.success(f"✅ 成功載入 {stock_name} ({stock_symbol_actual}) 資料")
            
            # (中略: 五線譜、樂活通道計算保持不變)
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
                regression_data['TL-1SD'] = trend_line - 1 * sd
                regression_data['TL-2SD'] = trend_line - 2 * sd
                
            with st.spinner("📊 計算樂活通道..."):
                window = 100
                regression_data['MA20W'] = regression_data['Close'].rolling(window=window, min_periods=window).mean()
                rolling_std = regression_data['Close'].rolling(window=window, min_periods=window).std()
                regression_data['UB'] = regression_data['MA20W'] + 2 * rolling_std
                regression_data['LB'] = regression_data['MA20W'] - 2 * rolling_std
                regression_data['Zone'] = np.where(regression_data['Close'] > regression_data['MA20W'], '樂活區(多頭)', '毅力區(空頭)')

            # 🌟 新增指標計算區
            with st.spinner("🔧 計算所有技術指標..."):
                # 舊指標
                regression_data['RSI'] = calculate_rsi(regression_data['Close'], 14)
                macd, signal, hist = calculate_macd(regression_data['Close'])
                regression_data['MACD'] = macd
                regression_data['MACD_Signal'] = signal
                regression_data['MACD_Hist'] = hist
                k, d = calculate_kd(regression_data['High'], regression_data['Low'], regression_data['Close'])
                regression_data['K'] = k
                regression_data['D'] = d
                
                # 🛠️ 修正 4: 在此處計算所有移動平均線，包括 MA60
                regression_data['MA5'] = regression_data['Close'].rolling(5).mean()
                regression_data['MA10'] = regression_data['Close'].rolling(10).mean()
                regression_data['MA20'] = regression_data['Close'].rolling(20).mean()
                regression_data['MA60'] = regression_data['Close'].rolling(60).mean() # 新增 MA60
                
                regression_data['Volume_MA5'] = regression_data['Volume'].rolling(5).mean()
                regression_data['Volume_Ratio'] = regression_data['Volume'] / regression_data['Volume_MA5']
                
                regression_data['RSI_Divergence'] = detect_rsi_divergence(regression_data['Close'], regression_data['RSI'])
                
                # 新增指標 (ADX, BBW, %R)
                adx, plus_di, minus_di = calculate_adx(regression_data['High'], regression_data['Low'], regression_data['Close'])
                regression_data['ADX'] = adx
                regression_data['+DI'] = plus_di
                regression_data['-DI'] = minus_di
                
                bbw = calculate_bbw(regression_data['Close'])
                regression_data['BBW'] = bbw
                
                williams_r = calculate_williams_r(regression_data['High'], regression_data['Low'], regression_data['Close'])
                regression_data['%R'] = williams_r
            
            # ==================== D. 買賣訊號判斷 (保持不變) ====================
            with st.spinner("🎯 生成買賣訊號..."):
                # 確保 valid_data 在計算完所有指標後再進行 dropna
                valid_data = regression_data.dropna(subset=['MA20W', 'UB', 'LB', 'RSI', 'K', 'D', 'ADX', 'BBW', '%R', 'MA60']) 
                
                if valid_data.empty:
                    st.error("❌ 資料不足")
                    st.stop()
                
                current = valid_data.iloc[-1]
                previous = valid_data.iloc[-2] if len(valid_data) > 1 else current
                current_price = float(current['Close'])
                current_tl = float(current['TL'])
                current_ma20w = float(current['MA20W'])
                
                slope_dir = "上升" if slope > 0 else "下降"
                
                deviation = current_price - current_tl
                sd_level = deviation / sd
                
                if sd_level >= 2:
                    fiveline_zone = "極度及樂觀 (+2SD以上)"
                elif sd_level >= 1:
                    fiveline_zone = "樂觀 (+1SD~+2SD)"
                elif sd_level >= 0:
                    fiveline_zone = "合理區 (TL~+1SD)"
                elif sd_level >= -1:
                    fiveline_zone = "悲觀 (-1SD~TL)"
                else:
                    fiveline_zone = "極度悲觀 (-2SD以下)"
                
                # ===== 賣出訊號判斷 (整合新指標) =====
                sell_signals = []
                # 1. 高檔訊號
                if sd_level >= 2:
                    if current['RSI_Divergence']:
                        sell_signals.append("⚠️ RSI 背離 (高檔)")
                    if current['RSI'] > 70 and current['RSI'] < previous['RSI']:
                        sell_signals.append("⚠️ RSI 從高檔回落 (超買區)")
                    if current['K'] < current['D'] and current['K'] > 80:
                        sell_signals.append("⚠️ KD 高檔死叉")
                # 2. DMI 轉空訊號
                if current['+DI'] < current['-DI'] and current['ADX'] > 25:
                    sell_signals.append("🚨 DMI 趨勢轉空 (+DI < -DI 且 ADX 強)")
                # 3. 爆量滯漲
                if current['Volume_Ratio'] > 2.0 and (current['Close'] - current['Open']) / current['Open'] < 0.005:
                    sell_signals.append("⚠️ 爆量滯漲 (V-Ratio > 2.0)")
                # 4. 威廉指標極度超買
                if current['%R'] > -20: 
                    sell_signals.append("🚨 威廉指標 (%R) 顯示極度樂觀情緒，潛在反轉")
                # 5. 跌破均線
                if current['Close'] < current['MA10']:
                    sell_signals.append("🚨 跌破 MA10")

                
                # ===== 買入訊號判斷 (整合新指標) =====
                buy_signals = []
                # 1. 低檔訊號
                if sd_level <= -1.0:
                    if current['RSI'] < 30 and current['RSI'] > previous['RSI']:
                        buy_signals.append("✅ RSI 從超賣區反彈")
                    if current['K'] > current['D'] and current['K'] < 20:
                        buy_signals.append("✅ KD 低檔金叉")
                # 2. DMI 轉多訊號
                if current['+DI'] > current['-DI'] and current['ADX'] > 25:
                    buy_signals.append("✅ DMI 趨勢轉多 (+DI > -DI 且 ADX 強)")
                # 3. 波動性收縮
                if current['BBW'] < valid_data['BBW'].quantile(0.1): # 修正：從 valid_data 獲取 quantile
                    buy_signals.append("⚠️ BBW 波動性極端收縮 (潛在爆發點)")
                # 4. 威廉指標極度超賣
                if current['%R'] < -80:
                    buy_signals.append("✅ 威廉指標 (%R) 顯示極度悲觀情緒，潛在反彈")
                # 5. 趨勢確認
                if 0.5 <= sd_level <= 1.5:
                    if slope > 0:
                        buy_signals.append("✅ 趨勢向上 (Slope > 0) 且股價合理")
                    if current['Close'] > current['MA20W']:
                        buy_signals.append("✅ 站上生命線")
                    if current['K'] > current['D'] and 40 <= current['K'] <= 60:
                        buy_signals.append("💚 KD 中段黃金交叉")
            
            # ===== 綜合建議 (保持不變) =====
            if sell_signals:
                action = "🔴 **賣出訊號**"
                action_detail = "建議減碼或觀望"
            elif buy_signals:
                action = "🟢 **買入訊號**"
                action_detail = "可考慮進場或加碼"
            else:
                action = "⚪ **觀望**"
                action_detail = "暫無明確訊號"
            
            # ==================== 介面顯示 (行動版優化) ====================
            st.subheader(f"📈 {stock_name} ({stock_symbol_actual})")
            
            # 顯示關鍵指標
            col1, col2, col3 = st.columns(3)
            col1.metric("股價", f"{current_price:.2f}")
            col2.metric("五線譜", fiveline_zone)
            col3.metric("RSI(14)", f"{current['RSI']:.1f}")

            col4, col5, col6 = st.columns(3) # 新增一個欄位
            col4.metric("KD", f"K:{current['K']:.1f} D:{current['D']:.1f}")
            col5.metric("ADX (強度)", f"{current['ADX']:.1f}")
            col6.metric("%R (情緒)", f"{current['%R']:.1f}") # 顯示 %R 指標
            
            st.divider()
            st.markdown(f"### {action}")
            st.info(action_detail)
            
            if sell_signals:
                st.warning("**賣出理由：**\n" + "\n".join([f"- {s}" for s in sell_signals]))
            
            if buy_signals:
                st.success("**買入理由：**\n" + "\n".join([f"- {s}" for s in buy_signals]))
            
            # ==================== 圖表分頁 (保持不變) ====================
            tab1, tab2, tab3, tab4 = st.tabs(["🎼 五線譜", "🌈 樂活通道", "📊 震盪指標", "🚀 波動與情緒"]) # Tab 標題修改

            with tab1:
                st.markdown(f"趨勢斜率: **{slope:.4f} ({slope_dir})**")
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=valid_data.index, y=valid_data['Close'], mode='lines', name='股價', line=dict(color='#4A4A4A', width=2)))
                fig1.add_trace(go.Scatter(x=valid_data.index, y=valid_data['TL+2SD'], mode='lines', name='TL+2SD', line=dict(color='#C8A2C8', width=2))) 
                fig1.add_trace(go.Scatter(x=valid_data.index, y=valid_data['TL+1SD'], mode='lines', name='TL+1SD', line=dict(color='#DDA0DD', width=2)))
                fig1.add_trace(go.Scatter(x=valid_data.index, y=valid_data['TL'], mode='lines', name='TL', line=dict(color='#B0A595', width=2))) 
                fig1.add_trace(go.Scatter(x=valid_data.index, y=valid_data['TL-1SD'], mode='lines', name='TL-1SD', line=dict(color='#A3C1AD', width=2))) 
                fig1.add_trace(go.Scatter(x=valid_data.index, y=valid_data['TL-2SD'], mode='lines', name='TL-2SD', line=dict(color='#8FBC8F', width=2))) 
                fig1.update_layout(title="五線譜走勢圖", height=500, hovermode='x unified', template='plotly_white')
                st.plotly_chart(fig1, use_container_width=True)
            
            with tab2:
                plot_data = valid_data.copy()
                
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=plot_data.index,
                    y=plot_data['Close'],
                    mode='lines',
                    name='股價',
                    line=dict(color='#4A4A4A', width=2),
                    hovertemplate='股價: %{y:.2f}<extra></extra>'
                ))
                fig2.add_trace(go.Scatter(
                    x=plot_data.index,
                    y=plot_data['UB'],
                    mode='lines',
                    name='上通道',
                    line=dict(color='#DDA0DD', width=2),
                    hovertemplate='上通道: %{y:.2f}<extra></extra>'
                ))
                fig2.add_trace(go.Scatter(
                    x=plot_data.index,
                    y=plot_data['MA20W'],
                    mode='lines',
                    name='20週均線',
                    line=dict(color='#B0A595', width=2),
                    hovertemplate='20週MA: %{y:.2f}<extra></extra>'
                ))
                fig2.add_trace(go.Scatter(
                    x=plot_data.index,
                    y=plot_data['LB'],
                    mode='lines',
                    name='下通道',
                    line=dict(color='#A3C1AD', width=2),
                    hovertemplate='下通道: %{y:.2f}<extra></extra>'
                ))
                
                if current_price > current_ma20w:
                    zone_text = "目前處於：樂活區 (多頭) 🚀"
                else:
                    zone_text = "目前處於：毅力區 (空頭) 🐻"
                    
                fig2.update_layout(
                    title=f"樂活通道走勢圖 - {zone_text}",
                    height=500,
                    hovermode='x unified',
                    template='plotly_white',
                    showlegend=True,
                    legend=dict(x=0, y=1, orientation='h')
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            with tab3:
                st.markdown("### 📊 震盪指標 (RSI, KD, MACD)")
                
                fig_ma = go.Figure()
                fig_ma.add_trace(go.Scatter(x=valid_data.index, y=valid_data['Close'], mode='lines', name='股價', line=dict(color='#4A4A4A', width=2)))
                # 🛠️ 修正 5: 確保 MA60 在 valid_data 中存在
                fig_ma.add_trace(go.Scatter(x=valid_data.index, y=valid_data['MA5'], mode='lines', name='MA5', line=dict(color='#FF8C66', width=1.5))) 
                fig_ma.add_trace(go.Scatter(x=valid_data.index, y=valid_data['MA10'], mode='lines', name='MA10', line=dict(color='#C8A2C8', width=1.5)))
                fig_ma.add_trace(go.Scatter(x=valid_data.index, y=valid_data['MA20'], mode='lines', name='MA20', line=dict(color='#B0A595', width=1.5)))
                fig_ma.add_trace(go.Scatter(x=valid_data.index, y=valid_data['MA60'], mode='lines', name='MA60', line=dict(color='#A3C1AD', width=1.5)))
                fig_ma.update_layout(title="移動平均線 (MA5/MA10/MA20/MA60)", height=350, hovermode='x unified', template='plotly_white')
                st.plotly_chart(fig_ma, use_container_width=True)
                
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=valid_data.index, y=valid_data['RSI'], mode='lines', name='RSI(14)', line=dict(color='#DDA0DD', width=2)))
                fig3.add_hline(y=70, line_dash="dash", line_color="#FF8C66", annotation_text="超買")
                fig3.add_hline(y=50, line_dash="dot", line_color="#B0A595", annotation_text="中線")
                fig3.add_hline(y=30, line_dash="dash", line_color="#A3C1AD", annotation_text="超賣")
                fig3.update_layout(title="RSI 相對強弱指標 (週期: 14天)", height=300, hovermode='x unified', template='plotly_white')
                st.plotly_chart(fig3, use_container_width=True)
                
                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(x=valid_data.index, y=valid_data['K'], mode='lines', name='K', line=dict(color='#FF8C66', width=2)))
                fig4.add_trace(go.Scatter(x=valid_data.index, y=valid_data['D'], mode='lines', name='D', line=dict(color='#DDA0DD', width=2)))
                fig4.add_hline(y=80, line_dash="dash", line_color="#FF8C66", annotation_text="超買")
                fig4.add_hline(y=20, line_dash="dash", line_color="#A3C1AD", annotation_text="超賣")
                fig4.update_layout(title="KD 隨機指標", height=300, hovermode='x unified', template='plotly_white')
                st.plotly_chart(fig4, use_container_width=True)

            with tab4:
                st.markdown("### 🚀 波動與趨勢動能 (ADX, BBW, %R)")
                
                col_williams, col_bbw_ratio = st.columns(2)
                col_williams.metric("當前威廉 %R", f"{current['%R']:.2f}%")
                col_bbw_ratio.metric("當前成交量比", f"{current['Volume_Ratio']:.2f}倍均量")
                
                st.markdown("---")
                
                # 繪製 ADX
                fig_adx = go.Figure()
                fig_adx.add_trace(go.Scatter(x=valid_data.index, y=valid_data['ADX'], mode='lines', name='ADX (趨勢強度)', line=dict(color='#B0A595', width=2)))
                fig_adx.add_trace(go.Scatter(x=valid_data.index, y=valid_data['+DI'], mode='lines', name='+DI (多頭)', line=dict(color='#A3C1AD', width=1.5)))
                fig_adx.add_trace(go.Scatter(x=valid_data.index, y=valid_data['-DI'], mode='lines', name='-DI (空頭)', line=dict(color='#DDA0DD', width=1.5)))
                fig_adx.add_hline(y=25, line_dash="dash", line_color="#4A4A4A", annotation_text="趨勢強弱分界線 (25)")
                fig_adx.update_layout(title="趨向指標 ADX, +DI, -DI", height=300, hovermode='x unified', template='plotly_white')
                st.plotly_chart(fig_adx, use_container_width=True)
                
                # 繪製 BBW
                fig_bbw = go.Figure()
                fig_bbw.add_trace(go.Scatter(x=valid_data.index, y=valid_data['BBW'] * 100, mode='lines', name='BBW %', line=dict(color='#FF8C66', width=2)))
                bbw_low_quantile = valid_data['BBW'].quantile(0.1) * 100
                fig_bbw.add_hline(y=bbw_low_quantile, line_dash="dash", line_color="#4A4A4A", annotation_text=f"歷史低點 ({bbw_low_quantile:.2f}%)")
                fig_bbw.update_layout(title="布林帶寬度 (BBW)", height=300, hovermode='x unified', template='plotly_white', yaxis_title="BBW (%)")
                st.plotly_chart(fig_bbw, use_container_width=True)

                # 繪製 Williams %R
                fig_williams = go.Figure()
                fig_williams.add_trace(go.Scatter(x=valid_data.index, y=valid_data['%R'], mode='lines', name='Williams %R', line=dict(color='#C8A2C8', width=2)))
                fig_williams.add_hline(y=-20, line_dash="dash", line_color="#FF8C66", annotation_text="超買線 (-20)")
                fig_williams.add_hline(y=-80, line_dash="dash", line_color="#A3C1AD", annotation_text="超賣線 (-80)")
                fig_williams.update_layout(title="威廉指標 (Williams %R)", height=300, hovermode='x unified', template='plotly_white')
                st.plotly_chart(fig_williams, use_container_width=True)

            
            # ==================== 智能分析摘要 (方案 B - 零 Key) ====================
            st.divider()
            st.subheader("🧠 智能深度分析 (無需 Key)")
            
            with st.spinner("🧠 智能分析生成中..."):
                analysis_result = generate_internal_analysis(
                    stock_name, 
                    stock_symbol_actual, 
                    slope_dir, 
                    sd_level, 
                    fiveline_zone, 
                    current, 
                    sell_signals, 
                    buy_signals,
                    valid_data['BBW'] # 傳入完整的 BBW 序列
                )
                st.markdown(analysis_result)
        
        except Exception as e:
            st.error(f"❌ 錯誤：{str(e)}")
            import traceback
            st.code(traceback.format_exc())

else:
    st.info("👈 請設定參數後點擊「開始分析」")
    st.markdown("""
    ### 🎯 智能交易系統特色
    
    **五線譜分析**
    - 價值位階判斷（昂貴/合理/便宜）
    - 趨勢線斜率分析
    
    **樂活通道**
    - 布林通道上下軌
    - 20週移動平均生命線
    
    **智能訊號 (新增)**
    - ✅ 趨向指標 (ADX, DMI) 判斷趨勢強度和多空轉換
    - ✅ 布林帶寬度 (BBW) 偵測波動性收縮（爆發點）
    - ✅ 威廉指標 (%R) 捕捉極端市場情緒
    - ✅ RSI, MACD, KD, 量價關係
    
    **Python 內部分析 (零 Key)**
    - 整合所有指標給出操作建議，不依賴外部 API。
    """)
