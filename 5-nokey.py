# Complete integrated version (V10 -> full ready-to-paste)
# - 支援台股與美股 (auto detect)
# - FinMind token 從 st.secrets["FINMIND_TOKEN"] 或環境變數回退到內嵌（如沒有）
# - 取得台股中文名稱的備援（FinMind）
# - 修正 yfinance wrapper 參數
# - generate_internal_analysis 放置於呼叫前
# - 圖表：隱藏五線譜圖例、移除 autoscale/resetScale 按鈕
# - 標題會顯示「中文名稱 (代碼)」若中文名稱可用
# - 依你之前 UI 要求微調 CSS（小標題、metric 字級等）
import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests

# ==================== CONFIG / CSS ====================
custom_css = """
<style>
/* 隱藏 Streamlit 頁腳與右上角選單 */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* 全局背景與字體 */
body, .main {
    background-color: #FAF7F4 !important;
    color: #5F5A58;
    font-family: "Noto Sans TC", "Noto Sans JP", "Hiragino Sans", sans-serif;
}

/* 隱藏側邊欄，單欄呈現 */
section[data-testid="stSidebar"] { display: none; }
.block-container { padding: 1rem 2rem !important; max-width: 1180px; }

/* 小標題（放在參數設定上方） */
.app-title {
    color: #A07C8C !important;
    font-weight: 600 !important;
    font-size: 1.3rem !important;
    margin-bottom: 0.6rem !important;
}

/* 收緊 h3/h4 與內文間距（符合你的第二張圖要求）*/
h3, h4 {
    font-size: 1.2rem !important;
    color: #8B6F77 !important;
    font-weight: 500 !important;
    margin-top: 0.2rem !important;
    margin-bottom: 0.10rem !important;
}
[data-testid="stMarkdownContainer"] h3, [data-testid="stMarkdownContainer"] h4 {
    font-weight: 500 !important;
    color: #8B6F77 !important;
    font-size: 1.1rem !important;
    margin-top: 0.2rem !important;
    margin-bottom: 0.10rem !important;
}

/* Metric 主字：縮小避免被截斷（符合第三張圖）*/
.css-1r6rthg {
    color: #A07C8C !important;
    font-weight: 600 !important;
    font-size: 1.4rem !important;
}

/* 卡片風格 */
[data-testid="stContainer"] { border-radius: 12px; background-color: #FFFFFFF2; border:1px solid #F1E6E6; box-shadow:0 3px 10px rgba(210,170,160,0.08); padding:0.8rem; }

</style>
"""

st.set_page_config(page_title="樂活五線譜", layout="wide")
st.markdown(custom_css, unsafe_allow_html=True)

# Plotly config (remove autoscale/resetScale)
PLOTLY_CONFIG = {"modeBarButtonsToRemove": ["autoScale2d", "resetScale2d"], "displaylogo": False}

# ==================== UTIL / CALC FUNCTIONS ====================
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
    hist = macd - signal_line
    return macd, signal_line, hist

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
    df['TR'] = np.maximum.reduce([df['High'] - df['Low'], (df['High'] - df['Close'].shift(1)).abs(), (df['Low'] - df['Close'].shift(1)).abs()])
    df['+DM'] = (df['High'] - df['High'].shift(1)).clip(lower=0)
    df['-DM'] = (df['Low'].shift(1) - df['Low']).clip(lower=0)
    mask = df['+DM'] > df['-DM']
    df.loc[mask, '-DM'] = 0
    df.loc[~mask, '+DM'] = 0
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

# ==================== DATA / INFO FUNCTIONS ====================
# FinMind token: prefer st.secrets, then environment, then default empty
FINMIND_TOKEN = ""
if hasattr(st, "secrets") and st.secrets.get("FINMIND_TOKEN"):
    FINMIND_TOKEN = st.secrets["FINMIND_TOKEN"]
elif os.environ.get("FINMIND_TOKEN"):
    FINMIND_TOKEN = os.environ.get("FINMIND_TOKEN")
else:
    # optional default (if you intentionally want it here, keep; else leave as "")
    FINMIND_TOKEN = ""

@st.cache_data(ttl=3600)
def get_stock_info_yf(symbol):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        name = info.get('longName') or info.get('shortName') or symbol
        return name
    except Exception:
        return symbol

def detect_market(symbol):
    if not symbol or not isinstance(symbol, str):
        return 'US'
    s = symbol.upper().strip()
    if '.TW' in s or '.TWO' in s:
        return 'TW'
    clean = s.replace('.TW', '').replace('.TWO', '').replace('.US','').strip()
    # If first 4 chars are digits -> TW (handles 00675L, 2330, etc.)
    if len(clean) >= 4 and clean[:4].isdigit():
        return 'TW'
    # If fully digits with typical lengths
    if clean.isdigit() and len(clean) in (3,4,5):
        return 'TW'
    return 'US'

@st.cache_data(ttl=3600)
def get_tw_stock_data_finmind(symbol, start_date, end_date, api_token=None):
    try:
        clean_symbol = symbol.replace('.TW','').replace('.TWO','').strip()
        url = "https://api.finmindtrade.com/api/v4/data"
        params = {
            "dataset": "TaiwanStockPrice",
            "data_id": clean_symbol,
            "start_date": start_date.strftime('%Y-%m-%d'),
            "end_date": end_date.strftime('%Y-%m-%d')
        }
        if api_token and api_token.strip():
            params["token"] = api_token
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        j = resp.json()
        if j.get('status') != 200 or not j.get('data'):
            return None
        df = pd.DataFrame(j['data'])
        # Rename and choose columns if present
        rename_map = {'date':'date','open':'open','max':'high','min':'low','close':'close','Trading_Volume':'volume'}
        df = df.rename(columns=rename_map)
        cols_needed = ['date','open','high','low','close','volume']
        # Keep existing ones
        existing = [c for c in cols_needed if c in df.columns]
        if 'date' in df.columns:
            df = df[['date'] + [c for c in existing if c!='date']]
        df['date'] = pd.to_datetime(df['date'])
        for col in ['open','high','low','close','volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.sort_values('date').set_index('date')
        df.rename(columns=lambda x: x.capitalize(), inplace=True)
        return df
    except Exception:
        return None

def get_stock_data_yfinance(symbol, start_date, end_date, market='US'):
    try:
        sym = symbol.strip().upper()
        # add .TW if TW market and no suffix
        if market == 'TW' and '.TW' not in sym and '.TWO' not in sym:
            sym = f"{sym}.TW"
        df = yf.download(sym, start=start_date, end=end_date, progress=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        df.rename(columns=lambda x: x.capitalize(), inplace=True)
        keep = [c for c in ['Open','High','Low','Close','Volume'] if c in df.columns]
        df = df[keep]
        return df
    except Exception:
        return None

@st.cache_data(ttl=3600)
def get_tw_stock_name_finmind(symbol, api_token=None):
    try:
        clean_symbol = symbol.replace('.TW','').replace('.TWO','').strip()
        url = "https://api.finmindtrade.com/api/v4/data"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        params = {"dataset":"TaiwanStockPrice","data_id":clean_symbol,"start_date":start_date.strftime('%Y-%m-%d'),"end_date":end_date.strftime('%Y-%m-%d'),"limit":1}
        if api_token and api_token.strip():
            params["token"] = api_token
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        j = resp.json()
        if j.get('status') != 200 or not j.get('data'):
            return None
        row = j['data'][0]
        for key in ['stock_name','Stock_Name','name','Name','stockName','公司名稱']:
            if key in row and row[key]:
                return str(row[key]).strip()
        return None
    except Exception:
        return None

@st.cache_data(ttl=3600)
def get_stock_data_auto(stock_input, days, data_source='auto', finmind_token=None):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days + 500)
    normalized_input = stock_input.strip()
    market = detect_market(normalized_input)
    if data_source == 'auto':
        if market == 'TW' and finmind_token and finmind_token.strip():
            actual_source = 'finmind'
        else:
            actual_source = 'yfinance'
    else:
        actual_source = data_source
    df = None
    actual_symbol = normalized_input
    if actual_source == 'finmind' and market == 'TW':
        df = get_tw_stock_data_finmind(normalized_input, start_date, end_date, api_token=finmind_token)
        if df is not None:
            actual_symbol = normalized_input.replace('.TW','').replace('.TWO','').strip()
    if df is None:
        sym = normalized_input
        if market == 'TW' and '.TW' not in sym and '.TWO' not in sym:
            sym = f"{sym}.TW"
        df = get_stock_data_yfinance(sym, start_date, end_date, market=market)
        actual_symbol = sym
        if df is None and market == 'TW' and not sym.endswith('.TWO'):
            sym2 = sym.replace('.TW','') + '.TWO'
            df = get_stock_data_yfinance(sym2, start_date, end_date, market=market)
            if df is not None:
                actual_symbol = sym2
    if df is None or df.empty:
        return pd.DataFrame(), None, normalized_input
    # Try get stock name from yfinance first
    stock_name = None
    try:
        stock_name = get_stock_info_yf(actual_symbol)
    except Exception:
        stock_name = normalized_input
    # If TW and name seems to be just code or empty, try FinMind name
    if detect_market(actual_symbol) == 'TW':
        name_upper = (stock_name or "").strip().upper()
        sym_upper = actual_symbol.strip().upper()
        if not stock_name or name_upper == sym_upper or name_upper.isdigit():
            fm_name = None
            try:
                fm_name = get_tw_stock_name_finmind(actual_symbol, finmind_token)
            except Exception:
                fm_name = None
            if fm_name:
                stock_name = fm_name
    return df, stock_name, actual_symbol

# ==================== SIGNALS / RENDER HELPERS ====================
def generate_signals(current, valid_data, sd_level, slope):
    previous = valid_data.iloc[-2] if len(valid_data) > 1 else current
    sell_signals = []
    buy_signals = []
    if sd_level >= 2:
        if current.get('RSI_Divergence', False): sell_signals.append("⚠️ RSI 背離 (高檔)")
        if current.get('RSI', 0) > 70 and current.get('RSI', 0) < previous.get('RSI', 0): sell_signals.append("⚠️ RSI 從高檔回落 (超買區)")
        if current.get('K', 0) < current.get('D', 0) and current.get('K', 0) > 80: sell_signals.append("⚠️ KD 高檔死叉")
    if current.get('+DI', 0) < current.get('-DI', 0) and current.get('ADX', 0) > 25:
        sell_signals.append("🚨 DMI 趨勢轉空 (+DI < -DI 且 ADX 強)")
    if current.get('Volume_Ratio', 0) > 2.0 and ((current['Close'] - current['Open']) / current['Open'] if current['Open'] else 0) < 0.005:
        sell_signals.append("⚠️ 爆量滯漲 (V-Ratio > 2.0)")
    if current.get('%R', -999) > -20: sell_signals.append("🚨 威廉指標 (%R) 顯示極度樂觀情緒，潛在反轉")
    if current['Close'] < current.get('MA10', current['Close']): sell_signals.append("🚨 跌破 MA10")
    if sd_level <= -1.0:
        if current.get('RSI', 0) < 30 and current.get('RSI', 0) > previous.get('RSI', 0): buy_signals.append("✅ RSI 從超賣區反彈")
        if current.get('K', 0) > current.get('D', 0) and current.get('K', 0) < 20: buy_signals.append("✅ KD 低檔金叉")
    if current.get('+DI', 0) > current.get('-DI', 0) and current.get('ADX', 0) > 25:
        buy_signals.append("✅ DMI 趨勢轉多 (+DI > -DI 且 ADX 強)")
    if current.get('BBW', 999) < valid_data['BBW'].quantile(0.1): buy_signals.append("⚠️ BBW 波動性極端收縮 (潛在爆發點)")
    if current.get('%R', -999) < -80: buy_signals.append("✅ 威廉指標 (%R) 顯示極度悲觀情緒，潛在反彈")
    if 0.5 <= sd_level <= 1.5:
        if slope > 0: buy_signals.append("✅ 趨勢向上 (Slope > 0) 且股價合理")
        if current['Close'] > current.get('MA20W', current['Close']): buy_signals.append("✅ 站上生命線")
        if current.get('K',0) > current.get('D',0) and 40 <= current.get('K',0) <= 60: buy_signals.append("💚 KD 中段黃金交叉")
    return sell_signals, buy_signals

def render_metric_cards(current, fiveline_zone, action_detail):
    current_price = current['Close']
    with st.container():
        st.markdown("#### 關鍵數據摘要")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("股價 (收盤)", f"{current_price:.2f}")
        col2.metric("五線譜位階", fiveline_zone.replace("及",""))
        sentiment_val = current.get('%R', 0)
        if sentiment_val > -20:
            sentiment_text = "極度樂觀 🔴"
        elif sentiment_val < -80:
            sentiment_text = "極度悲觀 🟢"
        else:
            sentiment_text = "均衡 ⚪"
        col3.metric("市場情緒", sentiment_text)
        col4.metric("綜合建議", action_detail)

# ==================== PLOT RENDER FUNCTIONS ====================
def render_fiveline_plot(valid_data, slope_dir, slope):
    st.markdown(f"趨勢斜率: **{slope:.4f} ({slope_dir})**")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=valid_data.index, y=valid_data['Close'], mode='lines', name='股價', line=dict(color='#8A6F68', width=2)))
    for col, color, width in [('TL+2SD','#C7A5B5',1.8),('TL+1SD','#DCC7D6',1.8),('TL','#BBA6A0',2),('TL-1SD','#D7CFCB',1.8),('TL-2SD','#E5DDDA',1.8)]:
        if col in valid_data.columns:
            fig.add_trace(go.Scatter(x=valid_data.index, y=valid_data[col], mode='lines', name=col, line=dict(color=color, width=width)))
    # hide legend for five-line plot as requested, keep tidy
    fig.update_layout(title="五線譜走勢圖", height=520, hovermode='x unified', template='plotly_white', showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    st.caption("圖表工具說明：可放大/縮小、平移、截圖等（autoscale 已移除以避免誤觸）。")

def render_lohas_plot(valid_data, current_price, current_ma20w):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=valid_data.index, y=valid_data['Close'], mode='lines', name='股價', line=dict(color='#8A6F68', width=2)))
    for col,color in [('UB','#DDA0DD'),('MA20W','#B0A595'),('LB','#A3C1AD')]:
        if col in valid_data.columns:
            fig.add_trace(go.Scatter(x=valid_data.index, y=valid_data[col], mode='lines', name=col, line=dict(color=color, width=2)))
    zone_text = "目前處於：樂活區 (多頭)" if current_price > current_ma20w else "目前處於：毅力區 (空頭)"
    fig.update_layout(title=f"樂活通道走勢圖 - {zone_text}", height=500, hovermode='x unified', template='plotly_white', showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

def render_oscillator_plots(valid_data):
    st.markdown("### 📊 震盪指標 (RSI, KD, MACD)")
    fig_ma = go.Figure()
    for col,color in [('Close','#8A6F68'),('MA5','#FF8C66'),('MA10','#C8A2C8'),('MA20','#B0A595'),('MA60','#A3C1AD')]:
        if col in valid_data.columns:
            fig_ma.add_trace(go.Scatter(x=valid_data.index, y=valid_data[col], mode='lines', name=col, line=dict(color=color, width=1.5)))
    fig_ma.update_layout(title="移動平均線 (MA5/MA10/MA20/MA60)", height=350, hovermode='x unified', template='plotly_white', showlegend=False)
    st.plotly_chart(fig_ma, use_container_width=True, config=PLOTLY_CONFIG)
    # RSI
    fig_rsi = go.Figure()
    if 'RSI' in valid_data.columns:
        fig_rsi.add_trace(go.Scatter(x=valid_data.index, y=valid_data['RSI'], mode='lines', name='RSI', line=dict(color='#DDA0DD', width=2)))
        fig_rsi.add_hline(y=70, line_dash='dash', line_color='#FF8C66')
        fig_rsi.add_hline(y=50, line_dash='dot', line_color='#B0A595')
        fig_rsi.add_hline(y=30, line_dash='dash', line_color='#A3C1AD')
        fig_rsi.update_layout(title="RSI(14)", height=300, template='plotly_white', showlegend=False)
        st.plotly_chart(fig_rsi, use_container_width=True, config=PLOTLY_CONFIG)
    # KD
    fig_kd = go.Figure()
    if 'K' in valid_data.columns and 'D' in valid_data.columns:
        fig_kd.add_trace(go.Scatter(x=valid_data.index, y=valid_data['K'], mode='lines', name='K', line=dict(color='#FF8C66', width=2)))
        fig_kd.add_trace(go.Scatter(x=valid_data.index, y=valid_data['D'], mode='lines', name='D', line=dict(color='#DDA0DD', width=2)))
        fig_kd.add_hline(y=80, line_dash='dash', line_color='#FF8C66')
        fig_kd.add_hline(y=20, line_dash='dash', line_color='#A3C1AD')
        fig_kd.update_layout(title="KD 隨機指標", height=300, template='plotly_white', showlegend=False)
        st.plotly_chart(fig_kd, use_container_width=True, config=PLOTLY_CONFIG)

def render_volatility_plots(valid_data, current):
    st.markdown("### 波動與趨勢動能 (ADX, BBW, %R)")
    col1, col2 = st.columns(2)
    col1.metric("當前威廉 %R", f"{current.get('%R',np.nan):.2f}%")
    col2.metric("當前成交量比", f"{current.get('Volume_Ratio',np.nan):.2f}倍均量")
    fig_adx = go.Figure()
    for col,color in [('ADX','#B0A595'),('+DI','#A3C1AD'),('-DI','#DDA0DD')]:
        if col in valid_data.columns:
            fig_adx.add_trace(go.Scatter(x=valid_data.index, y=valid_data[col], mode='lines', name=col, line=dict(color=color, width=1.5)))
    fig_adx.add_hline(y=25, line_dash='dash', line_color='#4A4A4A')
    fig_adx.update_layout(title="趨向指標 ADX, +DI, -DI", height=300, template='plotly_white', showlegend=False)
    st.plotly_chart(fig_adx, use_container_width=True, config=PLOTLY_CONFIG)
    # BBW
    if 'BBW' in valid_data.columns:
        fig_bbw = go.Figure()
        fig_bbw.add_trace(go.Scatter(x=valid_data.index, y=valid_data['BBW']*100, mode='lines', name='BBW %', line=dict(color='#FF8C66', width=2)))
        bbw_low = valid_data['BBW'].quantile(0.1)*100
        fig_bbw.add_hline(y=bbw_low, line_dash='dash', line_color='#4A4A4A')
        fig_bbw.update_layout(title="布林帶寬度 (BBW)", height=300, template='plotly_white', showlegend=False, yaxis_title="BBW (%)")
        st.plotly_chart(fig_bbw, use_container_width=True, config=PLOTLY_CONFIG)
    # Williams %R
    if '%R' in valid_data.columns:
        fig_w = go.Figure()
        fig_w.add_trace(go.Scatter(x=valid_data.index, y=valid_data['%R'], mode='lines', name='%R', line=dict(color='#C8A2C8', width=2)))
        fig_w.add_hline(y=-20, line_dash='dash', line_color='#FF8C66')
        fig_w.add_hline(y=-80, line_dash='dash', line_color='#A3C1AD')
        fig_w.update_layout(title="威廉指標 (Williams %R)", height=300, template='plotly_white', showlegend=False)
        st.plotly_chart(fig_w, use_container_width=True, config=PLOTLY_CONFIG)

# ==================== ANALYSIS GENERATOR (defined before usage) ====================
def generate_internal_analysis(stock_name, stock_symbol, slope_dir, sd_level, fiveline_zone, current, sell_signals, buy_signals, full_bbw_series):
    analysis_text = []
    current_adx = current.get('ADX', np.nan)
    current_williams_r = current.get('%R', np.nan)
    current_bbw = current.get('BBW', np.nan)
    current_v_ratio = current.get('Volume_Ratio', np.nan)
    bbw_quantile = full_bbw_series.quantile(0.1) if hasattr(full_bbw_series, 'quantile') else np.nan
    analysis_text.append("#### 1. 趨勢與動能判斷 (Trend & Momentum)")
    if current_adx > 30:
        adx_strength = f"ADX ({current_adx:.1f}) 顯示趨勢強度非常高。"
    elif current_adx > 20:
        adx_strength = f"ADX ({current_adx:.1f}) 顯示趨勢強度中等。"
    else:
        adx_strength = f"ADX ({current_adx:.1f}) 顯示趨勢強度較弱，可能處於盤整或反轉前夕。"
    fiveline_zone_clean = fiveline_zone.replace("及","")
    if slope_dir == "上升":
        trend_summary = f"五線譜趨勢：明確為上升，股價位於 {fiveline_zone_clean}。"
    elif slope_dir == "下降":
        trend_summary = f"五線譜趨勢：明確為下降，股價位於 {fiveline_zone_clean}。"
    else:
        trend_summary = "五線譜趨勢：盤整或觀望。"
    analysis_text.append(trend_summary + " " + adx_strength + "\n")
    analysis_text.append("#### 2. 市場情緒與波動性分析")
    sentiment_analysis = []
    if current_williams_r > -20:
        sentiment_analysis.append(f"🔴 極度樂觀：威廉指標 (%R: {current_williams_r:.1f}%) 處於超買區。")
    elif current_williams_r < -80:
        sentiment_analysis.append(f"🟢 極度悲觀：威廉指標 (%R: {current_williams_r:.1f}%) 處於超賣區。")
    if current_v_ratio > 1.8:
        sentiment_analysis.append(f"⚠️ 成交狂熱：成交量 ({current_v_ratio:.1f}倍均量) 異常放大。")
    if current_bbw < bbw_quantile:
        sentiment_analysis.append("🔲 波動性收縮：價格壓縮至極致，預期短期內將有方向性大變動。")
    if not sentiment_analysis:
        analysis_text.append("市場情緒和波動性指標處於正常範圍，無極端訊號。\n")
    else:
        analysis_text.append("\n".join(sentiment_analysis) + "\n")
    analysis_text.append("#### 3. 綜合操作建議")
    if current_williams_r > -20 and sell_signals:
        rec = f"極度危險：情緒超買且有 {len(sell_signals)} 個賣出訊號。建議投資人立即清倉或空手，風險極高。"
    elif current_williams_r < -80 and buy_signals and current_adx < 25:
        rec = "中線布局機會：情緒極度悲觀。可考慮極小額試單，但需確認 ADX 是否開始上揚。"
    elif current_bbw < bbw_quantile and current_adx < 20:
        rec = "靜待時機：市場處於暴風雨前的寧靜。建議保持場外觀望。"
    elif sell_signals:
        rec = f"鑑於當前有 {len(sell_signals)} 個賣出訊號，建議投資人減碼或空手觀望。"
    elif buy_signals:
        rec = f"當前有 {len(buy_signals)} 個買入訊號，建議可考慮分批進場，並緊盯 ADX 確認趨勢強度。"
    else:
        rec = "多數指標訊號不明確。建議保持觀望，等待更明確的買賣轉折訊號出現。"
    analysis_text.append(rec + "\n")
    analysis_text.append("#### 4. 聲明與風險提示")
    analysis_text.append("本分析為基於多重技術指標的程式碼硬編碼判斷，不構成任何投資建議。所有交易決策請自行承擔風險。")
    return "\n".join(analysis_text)

# ==================== INPUT SIDEBAR ====================
def render_input_sidebar(initial_stock_input, initial_period_type):
    with st.container():
        st.markdown("### 🔍 參數設定")
        stock_input = st.text_input("輸入股票代碼", value=initial_stock_input, key="stock_input_key")
        period_options = {"短期 (0.5年)":0.5,"中期 (1年)":1.0,"長期 (3.5年)":3.5,"超長期 (10年)":10.0}
        period_type = st.selectbox("選擇分析期間", list(period_options.keys()) + ["自訂期間"], index=list(period_options.keys()).index(initial_period_type), key="period_type_key")
        if period_type == "自訂期間":
            col_start, col_end = st.columns(2)
            with col_start:
                start_date_custom = st.date_input("開始日", value=datetime.now().date() - timedelta(days=365*3), key="start_date_custom_key")
            with col_end:
                end_date_custom = st.date_input("結束日", value=datetime.now().date(), key="end_date_custom_key")
            days = (end_date_custom - start_date_custom).days
        else:
            days = int(period_options[period_type] * 365)
            current_end_date = datetime.now().date()
            current_start_date = current_end_date - timedelta(days=days)
            col_start, col_end = st.columns(2)
            with col_start:
                st.markdown(f"開始日：{current_start_date}")
            with col_end:
                st.markdown(f"結束日：{current_end_date}")
        st.markdown("---")
        analyze_button = st.button("開始分析", type="primary", use_container_width=True, key="analyze_button_key")
    return stock_input, days, analyze_button

# ==================== MAIN ANALYSIS & RENDER ====================
def render_analysis_main(stock_input, days, analyze_button):
    if analyze_button or st.session_state.get('app_initialized', False):
        st.session_state.app_initialized = True
        if not stock_input:
            st.error("❌ 請輸入股票代號後點擊「開始分析」")
            return
        try:
            with st.spinner("📥 正在下載與計算資料..."):
                stock_data, stock_name, stock_symbol_actual = get_stock_data_auto(stock_input, days, data_source='auto', finmind_token=FINMIND_TOKEN)
                if stock_data.empty or stock_symbol_actual is None:
                    st.error("❌ 無法取得股票資料，請檢查股票代號是否正確。")
                    return
                # Display title: show "name (symbol)" if name is meaningful
                try:
                    if stock_name and stock_name.strip() and stock_name.strip().upper() != stock_symbol_actual.strip().upper():
                        st.subheader(f"📈 {stock_name} ({stock_symbol_actual})")
                    else:
                        st.subheader(f"📈 {stock_symbol_actual}")
                except:
                    st.subheader(f"📈 {stock_symbol_actual}")
                # --- calculations (same as original flow) ---
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
                valid_data = regression_data.dropna(subset=['MA20W','UB','LB','RSI','K','D','ADX','BBW','%R','MA60'])
                if valid_data.empty:
                    st.error("❌ 資料不足")
                    return
                current = valid_data.iloc[-1]
                slope_dir = "上升" if slope > 0 else "下降"
                deviation = current['Close'] - current['TL']
                sd_level = deviation / sd if sd != 0 else 0
                if sd_level >= 2:
                    fiveline_zone = "極度樂觀 (+2SD以上)"
                elif sd_level >= 1:
                    fiveline_zone = "樂觀 (+1SD~+2SD)"
                elif sd_level >= 0:
                    fiveline_zone = "合理區 (TL~+1SD)"
                elif sd_level >= -1:
                    fiveline_zone = "悲觀 (-1SD~TL)"
                else:
                    fiveline_zone = "極度悲觀 (-2SD以下)"
                sell_signals, buy_signals = generate_signals(current, valid_data, sd_level, slope)
                if sell_signals:
                    action = "🔴 **賣出訊號**"
                    action_detail = "建議減碼或觀望"
                elif buy_signals:
                    action = "🟢 **買入訊號**"
                    action_detail = "可考慮進場或加碼"
                else:
                    action = "⚪ **觀望**"
                    action_detail = "暫無明確訊號"
                # render cards & charts
                render_metric_cards(current, fiveline_zone, action_detail)
                st.divider()
                st.markdown(f"### {action}")
                st.info(action_detail)
                if sell_signals: st.warning("**賣出理由：**\n" + "\n".join([f"- {s}" for s in sell_signals]))
                if buy_signals: st.success("**買入理由：**\n" + "\n".join([f"- {s}" for s in buy_signals]))
                tab1, tab2, tab3, tab4 = st.tabs(["🎼 五線譜", "🌈 樂活通道", "📊 震盪指標", "波動與情緒"])
                with tab1:
                    render_fiveline_plot(valid_data, slope_dir, slope)
                with tab2:
                    render_lohas_plot(valid_data, current['Close'], current['MA20W'])
                with tab3:
                    render_oscillator_plots(valid_data)
                with tab4:
                    render_volatility_plots(valid_data, current)
                st.divider()
                # Use requested wording "深度分析：" and show generated analysis
                st.markdown("#### 深度分析：")
                analysis_result = generate_internal_analysis(stock_name, stock_symbol_actual, slope_dir, sd_level, fiveline_zone, current, sell_signals, buy_signals, valid_data['BBW'])
                st.markdown(analysis_result)
        except Exception as e:
            st.error(f"❌ 錯誤：{str(e)}")
            import traceback
            st.code(traceback.format_exc())

# ==================== APP ENTRY ====================
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
    period_options = {"短期 (0.5年)":0.5,"中期 (1年)":1.0,"長期 (3.5年)":3.5,"超長期 (10年)":10.0}
    if period_type == "自訂期間" and 'start_date_custom_key' in st.session_state:
        start_date = st.session_state.start_date_custom_key
        end_date = st.session_state.end_date_custom_key
        days = (end_date - start_date).days
    else:
        years = period_options.get(period_type, 3.5)
        days = int(years * 365)
    render_analysis_main(stock_input, days, analyze_button)
