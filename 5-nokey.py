import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import shioaji as sj
import time

# =========================================================
# 🌸 B — Sakura Latte Theme（櫻花霧面奶茶主題）- 最終版
# =========================================================
custom_css = """
<style>
/* 隱藏 Streamlit 頁腳與右上角選單 */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* 全局背景：霧面奶茶米白 */
body, .main, .st-emotion-cache-1dp6dkb {
    background-color: #FAF7F4 !important;
    color: #5F5A58;
    font-family: "Noto Sans TC", "Noto Sans JP", "Hiragino Sans", sans-serif;
}

/* 隱藏側邊欄，實現單欄模式 */
section[data-testid="stSidebar"] {
    display: none;
}
/* 確保主內容區佔滿整個寬度 */
.block-container {
    padding-top: 3.5rem !important;
    padding-bottom: 0rem;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}
.main {
    max-width: 1180px;
}

/* 標題樣式（放在參數設定上方的小字）*/
.app-title {
    color: #A07C8C !important;
    font-weight: 500 !important;
    font-size: 2rem !important; /* 比 st.title 小一些 */
    line-height: 1.1;
    margin-bottom: 0.6rem !important;
    word-break: break-word !important;
    white-space: normal !important;
    font-family: "Noto Sans TC", "PingFang TC", "Hiragino Sans", "Noto Sans JP", "Helvetica Neue", sans-serif !important;
    text-align: center !important; /* 置中 */
    display: block !important;
    width: 100% !important;
    }

/* 🎯 最終修正 1: 大標題 CSS 調整，確保完整顯示中文標題（移除 brittle hashed-class-only 規則） */
h1, .stAppHeader h1, .st-emotion-cache-10trblm, .css-10trblm {
    color: #A07C8C !important;
    font-weight: 500 !important;
    font-size: 2.0rem !important; /* 保留原來的大標題規則（不再用於主標題呈現） */
    line-height: 1.2;
    border-bottom: 1px solid #E7D8D8;
    padding-bottom: 6px;
    margin-bottom: 15px;
    white-space: normal !important;
    overflow: visible !important;
    max-width: 100% !important;
    word-break: break-word !important;
    hyphens: auto;
}

/* 卡片統一風格：柔白 + 淡粉邊框 + 櫻花陰影 */
[data-testid="stContainer"], [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] { 
    border-radius: 12px;
    background-color: #FFFFFFF2;
    border: 1px solid #F1E6E6;
    box-shadow: 0 3px 10px rgba(210,170,160,0.10);
    padding: 0.8rem;
}
/* Metric card specific style */
.st-emotion-cache-1cypcdb {
    border: 1px solid #F1E6E6;
    background-color: #FFFFFFF2;
    border-radius: 12px;
    box-shadow: 0 3px 10px rgba(210,170,160,0.10);
    padding: 15px;
}

/* H3/H4 小標題風格：粉棕灰 */
h3, h4 {
    font-size: 1.2rem !important;
    color: #8B6F77 !important;
    font-weight: 500 !important;
    margin-top: 0.8rem !important;
}

/* 確保分析報告中的標題層次拉平，移除粗體 */
[data-testid="stMarkdownContainer"] h3, [data-testid="stMarkdownContainer"] h4 {
    font-weight: 500 !important; 
    color: #8B6F77 !important;
    font-size: 1.1rem !important; 
    margin-bottom: 0.2rem !important;
}

/* 文字 */
p, label {
    font-size: 1rem;
    color: #5F5A58 !important;
}

/* Tabs：選取底線為粉紫 */
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #A07C8C !important;
    border-bottom: 3px solid #C7A5B5 !important;
}

/* 按鈕樣式改為淡紫色 (primary button) */
button[kind="primary"], .st-emotion-cache-hkqjaj button[data-testid="baseButton-primary"] {
    background-color: #C8A2C8 !important; /* 柔和淡紫色 */
    color: white !important;
    border-radius: 10px !important;
    border: none !important;
    padding: 0.45rem 1rem !important;
    font-weight: 500 !important;
}

button[kind="primary"]:hover,
.st-emotion-cache-hkqjaj button[data-testid="baseButton-primary"]:hover {
    background-color: #B28FB2 !important;
}

/* 綠色框框修改：Metric 主字變小，防止截斷，並維持深色 */
.css-1r6rthg, [data-testid="stMetricValue"] div {
    color: #5F5A58 !important;       /* 改為深灰/黑色 */
    font-weight: 600 !important;
    font-size: 1.2rem !important;    /* 縮小字體 */
    white-space: normal !important;  /* 允許換行 */
    overflow: visible !important;    /* 顯示完整文字 */
    line-height: 1.3 !important;
    word-wrap: break-word !important;
}
</style>
"""

# ==================== 頁面配置與 CSS 注入 ====================
st.set_page_config(page_title="樂活五線譜", layout="wide")
st.markdown(custom_css, unsafe_allow_html=True)

# ==================== 🚀 永豐 API 初始化 (讀取雲端 st.secrets) ====================
@st.cache_resource
def init_api():
    api = sj.Shioaji()
    try:
        api.login(
            api_key=st.secrets["SHIOAJI_API_KEY"],
            secret_key=st.secrets["SHIOAJI_SECRET_KEY"]
        )
        # 等待合約下載防呆
        timeout = 10
        start = time.time()
        while not api.Contracts.Stocks:
            time.sleep(0.5)
            if time.time() - start > timeout: break
        return api
    except Exception as e:
        return None

api = init_api()


# ==================== 🌟 核心計算函數 ====================

def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    ema_fast = data.ewm(span=fast, adjust=False).mean()
    ema_slow = data.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_kd(high, low, close, n=9, m1=3, m2=3):
    llv = low.rolling(window=n).min()
    hhv = high.rolling(window=n).max()
    rsv = (close - llv) / (hhv - llv) * 100
    k = rsv.ewm(com=m1-1, adjust=False).mean()
    d = k.ewm(com=m2-1, adjust=False).mean()
    return k, d

def detect_rsi_divergence(price, rsi, window=20):
    price_high = price.rolling(window=window).max()
    rsi_high = rsi.rolling(window=window).max()
    price_new_high = price == price_high
    rsi_new_high = rsi == rsi_high
    divergence = price_new_high & (~rsi_new_high)
    return divergence

def calculate_adx(high, low, close, period=14):
    df = pd.DataFrame({'High': high, 'Low': low, 'Close': close})
    df['TR'] = np.maximum.reduce([df['High'] - df['Low'], abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1))])
    df['+DM'] = (df['High'] - df['High'].shift(1)).clip(lower=0)
    df['-DM'] = (df['Low'].shift(1) - df['Low']).clip(lower=0)
    idx = df['+DM'] > df['-DM']
    df.loc[idx, '-DM'] = 0
    df.loc[~idx, '+DM'] = 0
    alpha = 1/period
    df['ATR'] = df['TR'].ewm(alpha=alpha, adjust=False).mean()
    df['+DMI'] = df['+DM'].ewm(alpha=alpha, adjust=False).mean()
    df['-DMI'] = df['-DM'].ewm(alpha=alpha, adjust=False).mean()
    df['+DI'] = (df['+DMI'] / df['ATR']) * 100
    df['-DI'] = (df['-DMI'] / df['ATR']) * 100
    sum_di = df['+DI'] + df['-DI']
    df['DX'] = (abs(df['+DI'] - df['-DI']) / sum_di.replace(0, np.nan)) * 100
    df['ADX'] = df['DX'].ewm(alpha=alpha, adjust=False).mean()
    return df['ADX'], df['+DI'], df['-DI']

def calculate_bbw(close, period=20, std_dev=2):
    ma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    bbw = (2 * std_dev * std) / ma.replace(0, np.nan)
    return bbw

def calculate_williams_r(high, low, close, period=14):
    hhv = high.rolling(window=period).max()
    llv = low.rolling(window=period).min()
    range_hl = hhv - llv
    williams_r = -100 * (hhv - close) / range_hl.replace(0, np.nan)
    return williams_r

def detect_market(symbol):
    if not symbol or not isinstance(symbol, str): return 'US'
    s = symbol.upper().strip()
    if '.TW' in s or '.TWO' in s: return 'TW'
    clean = s.replace('.TW', '').replace('.TWO', '').replace('.US','').strip()
    if len(clean) >= 4 and clean[:4].isdigit(): return 'TW'
    if clean.isdigit() and len(clean) in (3,4,5): return 'TW'
    return 'US'

def get_stock_data_yfinance(symbol, start_date, end_date, market='US'):
    try:
        sym = symbol.strip().upper()
        if market == 'TW' and '.TW' not in sym and '.TWO' not in sym:
            sym = f"{sym}.TW"
        df = yf.download(sym, start=start_date, end=end_date, progress=False)
        if df is None or df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        df.rename(columns=lambda x: x.capitalize(), inplace=True)
        keep = [c for c in ['Open','High','Low','Close','Volume'] if c in df.columns]
        return df[keep]
    except Exception as e: return None

# 🚀 這裡已經幫你把「永豐即時補丁」寫進去了！
def get_stock_data_auto(stock_input, days):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days + 500)
    normalized_input = stock_input.strip()
    market = detect_market(normalized_input)
    
    df = None
    actual_symbol = normalized_input
    
    # --- A. 永豐 API (台股優先) ---
    if market == 'TW' and api:
        try:
            clean_sym = normalized_input.replace('.TW', '').replace('.TWO', '').strip()
            contract = api.Contracts.Stocks[clean_sym]
            if contract:
                # 1. 抓取歷史 K 線
                kbars = api.kbars(contract, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
                df_sj = pd.DataFrame({**kbars})
                
                if not df_sj.empty:
                    df_sj.index = pd.to_datetime(df_sj.ts)
                    df_sj = df_sj[['Open', 'High', 'Low', 'Close', 'Volume']]
                    for col in df_sj.columns:
                        df_sj[col] = pd.to_numeric(df_sj[col], errors='coerce')
                        
                    # 2. 永豐 Snapshot 即時報價補丁
                    try:
                        snap = api.snapshots([contract])[0]
                        if snap.total_volume > 0:
                            today_date = pd.Timestamp(datetime.now().date())
                            if df_sj.index[-1].date() != today_date.date():
                                new_row = pd.DataFrame({
                                    'Open': [snap.open],
                                    'High': [snap.high],
                                    'Low': [snap.low],
                                    'Close': [snap.close],
                                    'Volume': [snap.total_volume]
                                }, index=[today_date])
                                df_sj = pd.concat([df_sj, new_row])
                            else:
                                df_sj.iloc[-1, df_sj.columns.get_loc('High')] = max(df_sj['High'].iloc[-1], snap.high)
                                df_sj.iloc[-1, df_sj.columns.get_loc('Low')] = min(df_sj['Low'].iloc[-1], snap.low)
                                df_sj.iloc[-1, df_sj.columns.get_loc('Close')] = snap.close
                                df_sj.iloc[-1, df_sj.columns.get_loc('Volume')] = snap.total_volume
                    except: pass
                        
                    df = df_sj
                    actual_symbol = clean_sym
        except: df = None
    
    # --- B. Yahoo Finance (若永豐過短無法算均線，或為美股時備援) ---
    if df is None or df.empty or len(df) < 150:
        sym = normalized_input
        if market == 'TW' and '.TW' not in sym and '.TWO' not in sym: sym = f"{sym}.TW"
        df = get_stock_data_yfinance(sym, start_date, end_date, market=market)
        actual_symbol = sym
        
        # Yahoo 的即時報價補丁
        try:
            info_sym = actual_symbol
            if market == 'TW' and not actual_symbol.endswith('.TW') and not actual_symbol.endswith('.TWO'):
                info_sym = f"{actual_symbol}.TW"
            
            ticker_info = yf.Ticker(info_sym).info
            rt_price = ticker_info.get('regularMarketPrice')
            if rt_price is not None and not df.empty:
                today_date = pd.Timestamp(datetime.now().date())
                if df.index[-1].date() != today_date.date():
                    new_row = pd.DataFrame({
                        'Open':   [float(ticker_info.get('regularMarketOpen', rt_price))],
                        'High':   [float(ticker_info.get('dayHigh', rt_price))],
                        'Low':    [float(ticker_info.get('dayLow', rt_price))],
                        'Close':  [float(rt_price)],
                        'Volume': [int(ticker_info.get('regularMarketVolume', 0))]
                    }, index=[today_date])
                    df = pd.concat([df, new_row])
        except: pass
    
    if df is None or df.empty: return pd.DataFrame(), None, normalized_input
    
    stock_name = None
    try:
        info = yf.Ticker(actual_symbol).info
        stock_name = info.get('longName') or info.get('shortName') or normalized_input
    except: stock_name = normalized_input
    
    return df, stock_name, actual_symbol

# ==================== 訊號與圖表渲染 ====================
def generate_signals(current, valid_data, sd_level, slope):
    previous = valid_data.iloc[-2] if len(valid_data) > 1 else current
    sell_signals = []
    buy_signals = []
    
    if sd_level >= 2:
        if current['RSI_Divergence']: sell_signals.append("RSI 背離 (高檔)")
        if current['RSI'] > 70 and current['RSI'] < previous['RSI']: sell_signals.append("RSI 從高檔回落 (超買區)")
        if current['K'] < current['D'] and current['K'] > 80: sell_signals.append("KD 高檔死叉")
    if current['+DI'] < current['-DI'] and current['ADX'] > 25: sell_signals.append("DMI 趨勢轉空 (+DI < -DI 且 ADX 強)")
    if current['Volume_Ratio'] > 2.0 and (current['Close'] - current['Open']) / current['Open'] < 0.005: sell_signals.append("爆量滯漲 (V-Ratio > 2.0)")
    if current['%R'] > -20: sell_signals.append("威廉指標 (%R) 顯示極度樂觀情緒，潛在反轉")
    if current['Close'] < current['MA10']: sell_signals.append("跌破 MA10")

    if sd_level <= -1.0:
        if current['RSI'] < 30 and current['RSI'] > previous['RSI']: buy_signals.append("RSI 從超賣區反彈")
        if current['K'] > current['D'] and current['K'] < 20: buy_signals.append("KD 低檔金叉")
    if current['+DI'] > current['-DI'] and current['ADX'] > 25: buy_signals.append("DMI 趨勢轉多 (+DI > -DI 且 ADX 強)")
    if current['BBW'] < valid_data['BBW'].quantile(0.1): buy_signals.append("BBW 波動性極端收縮 (潛在爆發點)")
    if current['%R'] < -80: buy_signals.append("威廉指標 (%R) 顯示極度悲觀情緒，潛在反彈")
    if 0.5 <= sd_level <= 1.5:
        if slope > 0: buy_signals.append("趨勢向上 (Slope > 0) 且股價合理")
        if current['Close'] > current['MA20W']: buy_signals.append("站上生命線")
        if current['K'] > current['D'] and 40 <= current['K'] <= 60: buy_signals.append("KD 中段黃金交叉")
        
    return sell_signals, buy_signals

def render_metric_cards(current, fiveline_zone, action_detail):
    current_price = current['Close']
    with st.container(border=True):
        st.markdown("#### 關鍵數據摘要")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("股價 (收盤)", f"{current_price:.2f}") 
        fiveline_zone_clean = fiveline_zone.replace("及", "")
        col2.metric("五線譜位階", fiveline_zone_clean)
        sentiment_val = current['%R']
        if sentiment_val > -20: sentiment_text = "極度樂觀 "
        elif sentiment_val < -80: sentiment_text = "極度悲觀 "
        else: sentiment_text = "均衡 "
        col3.metric("市場情緒", sentiment_text)
        col4.metric("綜合建議", action_detail)

def generate_internal_analysis(stock_name, stock_symbol, slope_dir, sd_level, fiveline_zone, current, sell_signals, buy_signals, full_bbw_series):
    analysis_text = []
    current_adx = current['ADX']
    current_williams_r = current['%R']
    current_bbw = current['BBW']
    current_v_ratio = current['Volume_Ratio']
    bbw_quantile = full_bbw_series.quantile(0.1)
    
    analysis_text.append("#### 1. 趨勢與動能判斷 (Trend & Momentum)")
    if current_adx > 30: adx_strength = f"ADX ({current_adx:.1f}) 顯示趨勢強度非常高。"
    elif current_adx > 20: adx_strength = f"ADX ({current_adx:.1f}) 顯示趨勢強度中等。"
    else: adx_strength = f"ADX ({current_adx:.1f}) 顯示趨勢強度較弱，可能處於盤整或反轉前夕。"
    fiveline_zone_clean = fiveline_zone.replace("及", "")
    if slope_dir == "上升": trend_summary = f"五線譜趨勢：明確為上升，股價位於 {fiveline_zone_clean}。"
    elif slope_dir == "下降": trend_summary = f"五線譜趨勢：明確為下降，股價位於 {fiveline_zone_clean}。"
    else: trend_summary = f"五線譜趨勢：盤整或觀望。"
    analysis_text.append(trend_summary + " " + adx_strength + "\n")

    analysis_text.append("#### 2. 市場情緒與波動性分析")
    sentiment_analysis = []
    if current_williams_r > -20: sentiment_analysis.append(f"極度樂觀：威廉指標 (%R: {current_williams_r:.1f}%) 處於超買區。")
    elif current_williams_r < -80: sentiment_analysis.append(f"極度悲觀：威廉指標 (%R: {current_williams_r:.1f}%) 處於超賣區。")
    if current_v_ratio > 1.8: sentiment_analysis.append(f"成交狂熱：成交量 ({current_v_ratio:.1f}倍均量) 異常放大。")
    if current_bbw < bbw_quantile: sentiment_analysis.append(f"波動性收縮：價格壓縮至極致，預期短期內將有方向性大變動。")
    if not sentiment_analysis: analysis_text.append("市場情緒和波動性指標處於正常範圍，無極端訊號。\n")
    else: analysis_text.append("\n".join(sentiment_analysis) + "\n")
    
    analysis_text.append("#### 3. 綜合操作建議")
    if current_williams_r > -20 and sell_signals: rec = f"極度危險：情緒超買且有 {len(sell_signals)} 個賣出訊號。建議投資人立即清倉或空手，風險極高。"
    elif current_williams_r < -80 and buy_signals and current_adx < 25: rec = "中線布局機會：情緒極度悲觀。可考慮極小額試單，但需確認 ADX 是否開始上揚。"
    elif current_bbw < bbw_quantile and current_adx < 20: rec = "靜待時機：市場處於暴風雨前的寧靜。建議保持場外觀望。"
    elif sell_signals: rec = f"鑑於當前有 {len(sell_signals)} 個賣出訊號，建議投資人減碼或空手觀望。"
    elif buy_signals: rec = f"當前有 {len(buy_signals)} 個買入訊號，建議可考慮分批進場，並緊盯 ADX 確認趨勢強度。"
    else: rec = "多數指標訊號不明確。建議保持觀望，等待更明確的買賣轉折訊號出現。"
    analysis_text.append(rec + "\n")
    analysis_text.append("#### 4. 聲明與風險提示\n本分析為基於多重技術指標的程式碼硬編碼判斷，不構成任何投資建議。所有交易決策請自行承擔風險。")
    return "\n".join(analysis_text)

def render_fiveline_plot(valid_data, slope_dir, slope):
    st.markdown(f"趨勢斜率: **{slope:.4f} ({slope_dir})**")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=valid_data.index, y=valid_data['Close'], mode='lines', name='股價', line=dict(color='#8A6F68', width=2.0)))
    fig1.add_trace(go.Scatter(x=valid_data.index, y=valid_data['TL+2SD'], mode='lines', name='TL+2SD', line=dict(color='#C7A5B5', width=1.8)))
    fig1.add_trace(go.Scatter(x=valid_data.index, y=valid_data['TL+1SD'], mode='lines', name='TL+1SD', line=dict(color='#DCC7D6', width=1.8)))
    fig1.add_trace(go.Scatter(x=valid_data.index, y=valid_data['TL'], mode='lines', name='TL', line=dict(color='#BBA6A0', width=2)))
    fig1.add_trace(go.Scatter(x=valid_data.index, y=valid_data['TL-1SD'], mode='lines', name='TL-1SD', line=dict(color='#D7CFCB', width=1.8)))
    fig1.add_trace(go.Scatter(x=valid_data.index, y=valid_data['TL-2SD'], mode='lines', name='TL-2SD', line=dict(color='#E5DDDA', width=1.8)))
    fig1.update_layout(title="五線譜走勢圖", height=500, hovermode='x unified', template='plotly_white', showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

def render_lohas_plot(valid_data, current_price, current_ma20w):
    plot_data = valid_data.copy()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=plot_data.index, y=plot_data['Close'], mode='lines', name='股價', line=dict(color='#8A6F68', width=2), hovertemplate='股價: %{y:.2f}<extra></extra>'))
    fig2.add_trace(go.Scatter(x=plot_data.index, y=plot_data['UB'], mode='lines', name='上通道', line=dict(color='#DDA0DD', width=2), hovertemplate='上通道: %{y:.2f}<extra></extra>'))
    fig2.add_trace(go.Scatter(x=plot_data.index, y=plot_data['MA20W'], mode='lines', name='20週均線', line=dict(color='#B0A595', width=2), hovertemplate='20週MA: %{y:.2f}<extra></extra>'))
    fig2.add_trace(go.Scatter(x=plot_data.index, y=plot_data['LB'], mode='lines', name='下通道', line=dict(color='#A3C1AD', width=2), hovertemplate='下通道: %{y:.2f}<extra></extra>'))
    zone_text = "目前處於：樂活區 (多頭)" if current_price > current_ma20w else "目前處於：毅力區 (空頭)"
    fig2.update_layout(title=f"樂活通道走勢圖 - {zone_text}", height=500, hovermode='x unified', template='plotly_white', showlegend=True, legend=dict(x=0, y=1, orientation='h'))
    st.plotly_chart(fig2, use_container_width=True)

def render_oscillator_plots(valid_data):
    st.markdown("### 📊 震盪指標 (RSI, KD, MACD)")
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=valid_data.index, y=valid_data['Close'], mode='lines', name='股價', line=dict(color='#8A6F68', width=2)))
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
    fig3.update_layout(title="RSI 相對強弱指標", height=300, hovermode='x unified', template='plotly_white')
    st.plotly_chart(fig3, use_container_width=True)
    
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=valid_data.index, y=valid_data['K'], mode='lines', name='K', line=dict(color='#FF8C66', width=2)))
    fig4.add_trace(go.Scatter(x=valid_data.index, y=valid_data['D'], mode='lines', name='D', line=dict(color='#DDA0DD', width=2)))
    fig4.add_hline(y=80, line_dash="dash", line_color="#FF8C66", annotation_text="超買")
    fig4.add_hline(y=20, line_dash="dash", line_color="#A3C1AD", annotation_text="超賣")
    fig4.update_layout(title="KD 隨機指標", height=300, hovermode='x unified', template='plotly_white')
    st.plotly_chart(fig4, use_container_width=True)

def render_volatility_plots(valid_data, current):
    st.markdown("### 波動與趨勢動能 (ADX, BBW, %R)")
    col_williams, col_bbw_ratio = st.columns(2)
    col_williams.metric("當前威廉 %R", f"{current['%R']:.2f}%")
    col_bbw_ratio.metric("當前成交量比", f"{current['Volume_Ratio']:.2f}倍均量")
    st.markdown("---")
    
    fig_adx = go.Figure()
    fig_adx.add_trace(go.Scatter(x=valid_data.index, y=valid_data['ADX'], mode='lines', name='ADX (趨勢強度)', line=dict(color='#B0A595', width=2)))
    fig_adx.add_trace(go.Scatter(x=valid_data.index, y=valid_data['+DI'], mode='lines', name='+DI (多頭)', line=dict(color='#A3C1AD', width=1.5)))
    fig_adx.add_trace(go.Scatter(x=valid_data.index, y=valid_data['-DI'], mode='lines', name='-DI (空頭)', line=dict(color='#DDA0DD', width=1.5)))
    fig_adx.add_hline(y=25, line_dash="dash", line_color="#4A4A4A", annotation_text="趨勢強弱分界線 (25)")
    fig_adx.update_layout(title="趨向指標 ADX, +DI, -DI", height=300, hovermode='x unified', template='plotly_white')
    st.plotly_chart(fig_adx, use_container_width=True)
    
    fig_bbw = go.Figure()
    fig_bbw.add_trace(go.Scatter(x=valid_data.index, y=valid_data['BBW'] * 100, mode='lines', name='BBW %', line=dict(color='#FF8C66', width=2)))
    bbw_low_quantile = valid_data['BBW'].quantile(0.1) * 100
    fig_bbw.add_hline(y=bbw_low_quantile, line_dash="dash", line_color="#4A4A4A", annotation_text=f"歷史低點 ({bbw_low_quantile:.2f}%)")
    fig_bbw.update_layout(title="布林帶寬度 (BBW)", height=300, hovermode='x unified', template='plotly_white', yaxis_title="BBW (%)")
    st.plotly_chart(fig_bbw, use_container_width=True)

    fig_williams = go.Figure()
    fig_williams.add_trace(go.Scatter(x=valid_data.index, y=valid_data['%R'], mode='lines', name='Williams %R', line=dict(color='#C8A2C8', width=2)))
    fig_williams.add_hline(y=-20, line_dash="dash", line_color="#FF8C66", annotation_text="超買線 (-20)")
    fig_williams.add_hline(y=-80, line_dash="dash", line_color="#A3C1AD", annotation_text="超賣線 (-80)")
    fig_williams.update_layout(title="威廉指標 (Williams %R)", height=300, hovermode='x unified', template='plotly_white')
    st.plotly_chart(fig_williams, use_container_width=True)


# ----------------------------------------------------
# 🌟 參數輸入區 (左欄內容)
# ----------------------------------------------------
def render_input_sidebar(initial_stock_input, initial_period_type):
    with st.container():
        stock_input = st.text_input("輸入股票代碼", value=initial_stock_input, key="stock_input_key")
        period_options = {"短期 (0.5年)": 0.5, "中期 (1年)": 1.0, "長期 (3.5年)": 3.5, "超長期 (10年)": 10.0}
        period_type = st.selectbox("選擇分析期間", list(period_options.keys()) + ["自訂期間"], index=list(period_options.keys()).index(initial_period_type), key="period_type_key")

        if period_type == "自訂期間":
            col_start, col_end = st.columns(2)
            with col_start: start_date_custom = st.date_input("開始日", value=datetime.now().date() - timedelta(days=365*3), key="start_date_custom_key") 
            with col_end: end_date_custom = st.date_input("結束日", value=datetime.now().date(), key="end_date_custom_key")
            days = (end_date_custom - start_date_custom).days
        else:
            days = int(period_options[period_type] * 365)
            current_end_date = datetime.now().date()
            current_start_date = current_end_date - timedelta(days=days)
            col_start, col_end = st.columns(2)
            with col_start: st.markdown(f"開始日：{current_start_date}")
            with col_end: st.markdown(f"結束日：{current_end_date}")
        
        st.markdown("---")
        analyze_button = st.button("開始分析", type="primary", use_container_width=True, key="analyze_button_key") 
    return stock_input, days, analyze_button

# ----------------------------------------------------
# 🌟 主要內容分析區 (右欄內容)
# ----------------------------------------------------
def render_analysis_main(stock_input, days, analyze_button):
    if analyze_button or st.session_state.get('app_initialized', False):
        st.session_state.app_initialized = True
        
        if not stock_input:
            st.error("❌ 請輸入股票代號後點擊「開始分析」")
            return
        
        try:
            with st.spinner("📥 正在下載與計算資料..."):
                stock_data, stock_name, stock_symbol_actual = get_stock_data_auto(stock_input, days)
                
                if stock_data.empty or stock_symbol_actual is None:
                    st.error("❌ 無法取得股票資料，請檢查股票代碼是否正確。")
                    return
                
                regression_data = stock_data.tail(days).copy().dropna()
                
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
                window = 100
                regression_data['MA20W'] = regression_data['Close'].rolling(window=window, min_periods=window).mean()
                rolling_std = regression_data['Close'].rolling(window=window, min_periods=window).std()
                regression_data['UB'] = regression_data['MA20W'] + 2 * rolling_std
                regression_data['LB'] = regression_data['MA20W'] - 2 * rolling_std
                regression_data['Zone'] = np.where(regression_data['Close'] > regression_data['MA20W'], '樂活區(多頭)', '毅力區(空頭)')

                regression_data['RSI'] = calculate_rsi(regression_data['Close'], 14)
                macd, signal, hist = calculate_macd(regression_data['Close'])
                regression_data['MACD'] = macd
                regression_data['MACD_Signal'] = signal
                regression_data['MACD_Hist'] = hist
                k, d = calculate_kd(regression_data['High'], regression_data['Low'], regression_data['Close'])
                regression_data['K'] = k
                regression_data['D'] = d
                regression_data['MA5'] = regression_data['Close'].rolling(5).mean()
                regression_data['MA10'] = regression_data['Close'].rolling(10).mean()
                regression_data['MA20'] = regression_data['Close'].rolling(20).mean()
                regression_data['MA60'] = regression_data['Close'].rolling(60).mean()
                regression_data['Volume_MA5'] = regression_data['Volume'].rolling(5).mean()
                regression_data['Volume_Ratio'] = regression_data['Volume'] / regression_data['Volume_MA5']
                regression_data['RSI_Divergence'] = detect_rsi_divergence(regression_data['Close'], regression_data['RSI'])
                adx, plus_di, minus_di = calculate_adx(regression_data['High'], regression_data['Low'], regression_data['Close'])
                regression_data['ADX'] = adx
                regression_data['+DI'] = plus_di
                regression_data['-DI'] = minus_di
                bbw = calculate_bbw(regression_data['Close'])
                regression_data['BBW'] = bbw
                williams_r = calculate_williams_r(regression_data['High'], regression_data['Low'], regression_data['Close'])
                regression_data['%R'] = williams_r
                
                valid_data = regression_data.dropna(subset=['MA20W', 'UB', 'LB', 'RSI', 'K', 'D', 'ADX', 'BBW', '%R', 'MA60']) 
                if valid_data.empty: st.error("❌ 資料不足"); return
                
                current = valid_data.iloc[-1]
                slope_dir = "上升" if slope > 0 else "下降"
                deviation = current['Close'] - current['TL']
                sd_level = deviation / sd
                
                if sd_level >= 2: fiveline_zone = "極度樂觀"
                elif sd_level >= 1: fiveline_zone = "樂觀"
                elif sd_level >= 0: fiveline_zone = "合理區"
                elif sd_level >= -1: fiveline_zone = "悲觀"
                else: fiveline_zone = "極度悲觀"
                
                sell_signals, buy_signals = generate_signals(current, valid_data, sd_level, slope)
                
                if sell_signals:
                    action = "**賣出訊號**"
                    action_detail = "建議減碼或觀望"
                elif buy_signals:
                    action = "**買入訊號**"
                    action_detail = "可考慮進場或加碼"
                else:
                    action = "**觀望**"
                    action_detail = "暫無明確訊號"
                
                st.subheader(f"📈 {stock_name} ({stock_symbol_actual})")
                render_metric_cards(current, fiveline_zone, action_detail)
                
                st.divider()
                st.markdown(f"### {action}")
                st.info(action_detail)

                if sell_signals: st.warning("**賣出理由：**\n" + "\n".join([f"- {s}" for s in sell_signals]))
                if buy_signals: st.success("**買入理由：**\n" + "\n".join([f"- {s}" for s in buy_signals]))
                
                tab1, tab2, tab3, tab4 = st.tabs(["🎼 五線譜", "🌈 樂活通道", "📊 震盪指標", "波動與情緒"]) 

                with tab1: render_fiveline_plot(valid_data, slope_dir, slope);
                with tab2: render_lohas_plot(valid_data, current['Close'], current['MA20W']);
                with tab3: render_oscillator_plots(valid_data);
                with tab4: render_volatility_plots(valid_data, current);

                st.divider()
                st.markdown("### 深度分析：")
                analysis_result = generate_internal_analysis(stock_name, stock_symbol_actual, slope_dir, sd_level, fiveline_zone, current, sell_signals, buy_signals, valid_data['BBW'])
                st.markdown(analysis_result)

        except Exception as e:
            st.error(f"❌ 錯誤：{str(e)}")

# ----------------------------------------------------
# 🌟 主執行區塊
# ----------------------------------------------------
if 'stock_input_value' not in st.session_state:
    st.session_state.stock_input_value = "00675L"
if 'period_type_value' not in st.session_state:
    st.session_state.period_type_value = "長期 (3.5年)"

col_left, col_right = st.columns([1, 2.5]) 

with col_left:
    st.markdown('<div class="app-title">樂活五線譜</div>', unsafe_allow_html=True)
    stock_input, days, analyze_button = render_input_sidebar(st.session_state.stock_input_value, st.session_state.period_type_value)

with col_right:
    stock_input = st.session_state.stock_input_key if 'stock_input_key' in st.session_state else st.session_state.stock_input_value
    analyze_button = st.session_state.analyze_button_key if 'analyze_button_key' in st.session_state else False
    
    period_type = st.session_state.period_type_key if 'period_type_key' in st.session_state else st.session_state.period_type_value
    period_options = {"短期 (0.5年)": 0.5,"中期 (1年)": 1.0,"長期 (3.5年)": 3.5,"超長期 (10年)": 10.0}

    if period_type == "自訂期間" and 'start_date_custom_key' in st.session_state:
        start_date = st.session_state.start_date_custom_key
        end_date = st.session_state.end_date_custom_key
        days = (end_date - start_date).days
    else:
        years = period_options.get(period_type, 3.5)
        days = int(years * 365)
    
    render_analysis_main(stock_input, days, analyze_button)
