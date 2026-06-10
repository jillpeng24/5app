import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import shioaji as sj  
import time            
import os              
from dotenv import load_dotenv 

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
"""

# ==================== 頁面配置與 CSS 注入 ====================
st.set_page_config(page_title="樂活五線譜", layout="wide")

# 注入自訂 CSS
st.markdown(custom_css, unsafe_allow_html=True)

# ==================== 🚀 永豐 API 初始化 (支援雲端與本機) ====================
@st.cache_resource(ttl=43200) # ⏳ 設定 12 小時 (43200秒) 自動過期重連
def init_api():
    import os
    from dotenv import load_dotenv
    import shioaji as sj
    import time
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(current_dir, "config", ".env")
    load_dotenv(dotenv_path=env_path)

    api_key = os.getenv("SHIOAJI_API_KEY")
    secret_key = os.getenv("SHIOAJI_SECRET_KEY")
    
    if not api_key:
        try:
            api_key = st.secrets["SHIOAJI_API_KEY"]
            secret_key = st.secrets["SHIOAJI_SECRET_KEY"]
        except Exception:
            pass

    if not api_key or not secret_key:
        print("❌ [後台偵錯] 致命錯誤：完全抓不到任何金鑰字串！")
        return None

    try:
        print("⏳ [後台偵錯] 找到金鑰了！正在嘗試登入永豐 API...")
        api = sj.Shioaji()
        api.login(api_key, secret_key)
        
        timeout = 10
        start = time.time()
        while not api.Contracts.Stocks:
            time.sleep(0.5)
            if time.time() - start > timeout: 
                print("❌ [後台偵錯] 登入成功，但下載股票合約超時 (Timeout)！")
                break
                
        print("✅ [後台偵錯] 永豐 API 登入與合約下載成功！")
        return api
        
    except Exception as e:
        print(f"❌ [後台偵錯] 永豐登入失敗，真正死因：{str(e)}")
        return None

api = init_api()

if api is None and "api_warned" not in st.session_state:
    st.toast("⚠️ 找不到永豐金鑰或連線失敗，目前使用 Yahoo 歷史報價", icon="⚠️")
    st.session_state.api_warned = True


# ==================== 🌟 核心計算函數 🌟 ====================

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

@st.cache_data(ttl=3600)
def get_stock_info(symbol):
    try:
        ticker = yf.Ticker(symbol)
        stock_info = ticker.info
        stock_name = stock_info.get('longName', symbol)
        return stock_name, symbol
    except:
        return symbol, symbol


def detect_market(symbol):
    if not symbol or not isinstance(symbol, str):
        return 'US'
    s = symbol.upper().strip()
    if '.TW' in s or '.TWO' in s:
        return 'TW'
    clean = s.replace('.TW', '').replace('.TWO', '').replace('.US','').strip()
    if len(clean) >= 4 and clean[:4].isdigit():
        return 'TW'
    if clean.isdigit() and len(clean) in (3,4,5):
        return 'TW'
    return 'US'

def get_stock_data_auto(stock_input, days):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days + 500)
    normalized_input = stock_input.strip()
    market = detect_market(normalized_input)
    
    sym = normalized_input
    if market == 'TW' and '.TW' not in sym and '.TWO' not in sym:
        sym = f"{sym}.TW"
    
    with st.status(f"🚀 正在處理 {sym}...", expanded=False) as status:
        status.write("📡 正在從 Yahoo Finance 抓取歷史 K 線...")
        fetch_end_date = end_date + timedelta(days=1)
        df = yf.download(sym, start=start_date, end=fetch_end_date, progress=False, auto_adjust=True)
        
        if (df is None or df.empty) and market == 'TW' and sym.endswith('.TW'):
            status.write("🔄 嘗試切換至 .TWO 下載...")
            sym2 = sym.replace('.TW', '.TWO')
            df = yf.download(sym2, start=start_date, end=fetch_end_date, progress=False, auto_adjust=True)
            if df is not None and not df.empty: sym = sym2

        if df is None or df.empty:
            status.update(label="❌ Yahoo 資料下載失敗", state="error")
            return pd.DataFrame(), None, normalized_input

        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        df.rename(columns=lambda x: x.capitalize(), inplace=True)

        df = df[['Open','High','Low','Close','Volume']].copy()
        df = df.sort_index()

        pct_change = df['Close'].pct_change()
        split_dates = pct_change[pct_change < -0.3].index.tolist()

        if split_dates:
            for split_date in split_dates:
                idx = df.index.get_loc(split_date)
                if isinstance(idx, slice) or isinstance(idx, type(np.array([]))):
                    idx = np.where(idx)[0][0]
                if idx > 0:
                    price_before = df['Close'].iloc[idx - 1]
                    price_after = df['Close'].iloc[idx]
                    ratio = round(price_before / price_after)
                    if ratio >= 2:
                        status.write(f"🔀 偵測到分割斷層：{split_date.date()} 約 {ratio}:1，正在還原歷史資料...")
                        df.loc[df.index < split_date, ['Open','High','Low','Close']] /= ratio
                        df.loc[df.index < split_date, 'Volume'] *= ratio

        status.write(f"✅ 已抓到 {len(df)} 筆歷史數據")

        if market == 'TW':
            if not api:
                status.write("ℹ️ 永豐 API 未連線，將顯示 Yahoo 盤後/延遲資料")
            else:
                status.write("💉 正在啟動永豐 API 盤中即時補丁...")
                try:
                    clean_sym = normalized_input.replace('.TW', '').replace('.TWO', '').strip()
                    contract = api.Contracts.Stocks[clean_sym]
                    if contract:
                        snap = api.snapshots([contract])[0]
                        if snap.close > 0:
                            today_date = pd.Timestamp(datetime.now().date())
                            if df.index.tz is not None: today_date = today_date.tz_localize(df.index.tz)
                            
                            sj_vol = snap.total_volume * 1000 
                            
                            if df.index[-1].date() != today_date.date():
                                status.write("🆕 發現今日新 K 棒，正在插入...")
                                new_row = pd.DataFrame({
                                    'Open': [snap.open], 'High': [snap.high], 'Low': [snap.low],
                                    'Close': [snap.close], 'Volume': [sj_vol]
                                }, index=[today_date])
                                df = pd.concat([df, new_row])
                            else:
                                status.write("🆙 正在更新今日最新價格...")
                                df.iloc[-1, df.columns.get_loc('High')] = max(df['High'].iloc[-1], snap.high)
                                df.iloc[-1, df.columns.get_loc('Low')] = min(df['Low'].iloc[-1], snap.low)
                                df.iloc[-1, df.columns.get_loc('Close')] = snap.close
                                df.iloc[-1, df.columns.get_loc('Volume')] = sj_vol
                            status.write(f"💎 補丁成功！目前現價：{snap.close}")
                        else:
                            status.write("⚠️ 永豐 API 回傳今日成交量為 0，跳過補丁")
                except Exception as e:
                    status.write(f"⚠️ 補丁失敗：{e}")
        
        if market == 'US':
            status.write("⚡ 正在抓取美股即時價格...")
            try:
                ticker = yf.Ticker(sym)
                rt_price = ticker.fast_info.last_price
                if rt_price:
                    df.iloc[-1, df.columns.get_loc('Close')] = rt_price
                    status.write(f"💎 美股現價已更新：{rt_price:.2f}")
            except: pass

        status.update(label=f"✅ {sym} 資料準備就緒", state="complete", expanded=False)

    try: name = yf.Ticker(sym).info.get('longName', normalized_input)
    except: name = normalized_input
        
    return df, name, sym

# 輔助：隱藏原始雜亂的買賣訊號判斷，僅產出簡潔摘要狀態供 metric 卡片使用
def generate_summary_action(current, valid_data):
    previous = valid_data.iloc[-2] if len(valid_data) > 1 else current
    
    macd_hist = current['MACD_Hist']
    prev_macd_hist = previous['MACD_Hist']
    w_r = current['%R']
    
    # 加入天氣符號 🌧️ 讓卡片燈號一致
    if w_r > -20 or (macd_hist < 0 and prev_macd_hist >= 0):
        return "🌧️ 警戒防守"
    elif w_r < -80 or (macd_hist > 0 and prev_macd_hist <= 0):
        return "🌸 佈局訊號"
    else:
        return "🌾 區間觀望"

# 輔助：呈現數據卡片
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


# 🎯 輔助：【全新改版】三段式智能分析生成函數（加入 RSI 與天氣符號）
def generate_internal_analysis(stock_name, stock_symbol, slope_dir, sd_level, fiveline_zone, current, valid_data):
    previous = valid_data.iloc[-2] if len(valid_data) > 1 else current
    
    adx = current['ADX']
    p_di = current['+DI']
    m_di = current['-DI']
    macd = current['MACD']
    macd_sig = current['MACD_Signal']
    macd_hist = current['MACD_Hist']
    w_r = current['%R']
    rsi = current['RSI']  # 🎯 抓取 RSI 數值
    bbw = current['BBW'] * 100
    close = current['Close']
    ma10 = current['MA10']
    
    analysis_text = []

    is_overheated = w_r > -20 or sd_level >= 1.5
    is_oversold = w_r < -80 or sd_level <= -1.5
    is_trend_up = adx > 25 and p_di > m_di
    is_trend_down = adx > 25 and m_di > p_di
    is_macd_weak = macd_hist < 0 or (macd_hist < previous['MACD_Hist'])
    is_macd_strong = macd_hist > 0 or (macd_hist > previous['MACD_Hist'])
    
    # 1. 🎯 戰情總結
    if is_overheated and is_macd_weak:
        summary = "### 🎯 戰情總結：高檔動能衰退，建議【減碼防守】"
        script_hold = "長線雖有保護，但短線情緒已達沸點且推升動能出現背離轉弱。建議可適度將 1/3 或 1/2 部位獲利入袋，剩餘部位嚴守 MA10 作為最後防線，破線請無條件出場。"
        script_empty = "目前位階追高風險極大，盈虧比不佳。強烈建議空手觀望，耐心等待拉回量縮或指標降溫後的佈局機會。"
    elif is_oversold and is_macd_strong:
        summary = "### 🎯 戰情總結：低檔恐慌蔓延但動能回溫，建議【留意築底 / 逢低試單】"
        script_hold = "短線已達超賣區間，且 MACD 動能有止跌回穩跡象。若未跌破前低，不建議在此恐慌殺低，可觀察後續反彈力道。"
        script_empty = "市場情緒極度悲觀，但指標顯示下檔空間可能有限。可考慮以極小資金分批試單，並嚴設跌破前低停損。"
    elif is_trend_up and is_macd_strong:
        summary = "### 🎯 戰情總結：多方趨勢明確且動能強勁，建議【偏多操作 / 順勢抱牢】"
        script_hold = "長短線指標皆屬多方掌控，大趨勢極強。請安心順勢抱牢，未跌破短天期均線 (MA5/MA10) 前，不需過早預設高點。"
        script_empty = "趨勢強勁，拉回均線有撐皆是進場機會。惟須注意資金控管，避免於乖離過大時單筆重壓。"
    elif is_trend_down and is_macd_weak:
        summary = "### 🎯 戰情總結：空方趨勢確立且跌勢延續，建議【空手迴避 / 嚴格停損】"
        script_hold = "趨勢與動能皆處於弱勢，若已觸及停損點位請果斷執行，切勿凹單攤平。"
        script_empty = "正處於「接飛刀」的高風險期，請保持空手，等待 MACD 於低檔明顯金叉再做打算。"
    else:
        summary = "### 🎯 戰情總結：多空陷入膠著或震盪，建議【區間操作 / 觀望為主】"
        script_hold = "目前市場無明顯單向大趨勢，建議依賴均線進行紀律防守，或於五線譜箱型上下緣來回操作。"
        script_empty = "方向尚未明朗，此時進場容易遭受雙巴。建議保留現金，待帶量突破或指標表態後再行進場。"

    analysis_text.append(summary)
    analysis_text.append("\n### ⚖️ 多空雷達掃描：")
    
    # 2. 多空雷達掃描 (改用 🌧️ 與 ❄️)
    fiveline_clean = fiveline_zone.replace('及', '')
    if is_trend_up:
        analysis_text.append(f"* **【長線趨勢】🌸 多方掌控：** 五線譜位於「{fiveline_clean}」，且 ADX 高達 **{adx:.1f}** (+DI: {p_di:.1f}, -DI: {m_di:.1f})，顯示多方大趨勢極強。")
    elif is_trend_down:
        analysis_text.append(f"* **【長線趨勢】🌧️ 空方壓制：** 五線譜位於「{fiveline_clean}」，且 ADX 高達 **{adx:.1f}** (+DI: {p_di:.1f}, -DI: {m_di:.1f})，顯示空方大趨勢極強。")
    else:
        analysis_text.append(f"* **【長線趨勢】🌾 區間震盪：** 五線譜位於「{fiveline_clean}」，ADX 為 **{adx:.1f}** (趨勢偏弱)，長線無明顯單向動能。")

    if macd > 0 and macd_hist < 0 and previous['MACD_Hist'] >= 0:
        analysis_text.append(f"* **【短線動能】❄️ 高檔死叉 (引擎冷卻)：** MACD 於零軸上死叉 (DIF: **{macd:.2f}**, DEA: **{macd_sig:.2f}**)，紅柱轉綠 (**{macd_hist:.2f}**)，短線推升引擎失去動能。")
    elif macd < 0 and macd_hist > 0 and previous['MACD_Hist'] <= 0:
        analysis_text.append(f"* **【短線動能】🌸 低檔金叉 (引擎發動)：** MACD 於零軸下金叉 (DIF: **{macd:.2f}**, DEA: **{macd_sig:.2f}**)，綠柱轉紅 (**{macd_hist:.2f}**)，短線有築底反彈跡象。")
    elif macd_hist > previous['MACD_Hist']:
        analysis_text.append(f"* **【短線動能】🌸 動能延續：** MACD 柱狀體維持擴張態勢 (DIF: **{macd:.2f}**, 柱狀體: **{macd_hist:.2f}**)。")
    else:
        analysis_text.append(f"* **【短線動能】🌧️ 動能衰退：** MACD 柱狀體出現縮減 (DIF: **{macd:.2f}**, 柱狀體: **{macd_hist:.2f}**)，短線上攻力道減弱。")
        
    if w_r > -20:
        analysis_text.append(f"* **【情緒與波動】🌧️ 極度沸騰 (超買)：** 威廉指標 W%R 來到 **{w_r:.2f}**，**短天期 RSI 高達 {rsi:.1f}**，且布林帶寬 BBW 為 **{bbw:.2f}%**，隨時可能面臨漲多回檔。")
    elif w_r < -80:
        analysis_text.append(f"* **【情緒與波動】🌸 極度恐慌 (超賣)：** 威廉指標 W%R 來到 **{w_r:.2f}**，**目前 RSI 降至 {rsi:.1f}**，且布林帶寬 BBW 為 **{bbw:.2f}%**，市場情緒過度悲觀。")
    else:
        analysis_text.append(f"* **【情緒與波動】🌾 情緒穩定：** 威廉指標 W%R 為 **{w_r:.2f}**，**目前 RSI 為 {rsi:.1f}**，布林帶寬 BBW 為 **{bbw:.2f}%**，未見極端情緒。")

    # 3. 行動劇本
    analysis_text.append("\n### 🛡️ 行動劇本與防守點位：")
    analysis_text.append(f"* **若持有部位：** {script_hold}")
    analysis_text.append(f"* **若空手觀望：** {script_empty}")
    
    return "\n".join(analysis_text)


# 輔助：圖表函數 (保持新顏色)

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
    
    dt_all = pd.date_range(start=valid_data.index[0], end=valid_data.index[-1])
    dt_obs = [d.strftime("%Y-%m-%d") for d in valid_data.index]
    dt_breaks = [d for d in dt_all.strftime("%Y-%m-%d").tolist() if d not in dt_obs]
    
    fig1.update_xaxes(rangebreaks=[dict(values=dt_breaks)]) 
    
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
    st.markdown("### 📊 震盪指標 (MA, RSI, MACD)")
    
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
    
    # 🎯 把原本打架的 KD 換成了極具參考價值的 MACD 圖表
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=valid_data.index, y=valid_data['MACD'], mode='lines', name='DIF (快線)', line=dict(color='#FF8C66', width=2)))
    fig4.add_trace(go.Scatter(x=valid_data.index, y=valid_data['MACD_Signal'], mode='lines', name='DEA (慢線)', line=dict(color='#DDA0DD', width=2)))
    fig4.add_trace(go.Bar(x=valid_data.index, y=valid_data['MACD_Hist'], name='MACD 柱狀體', marker_color=np.where(valid_data['MACD_Hist'] > 0, '#C8A2C8', '#B0A595')))
    fig4.update_layout(title="MACD 指數平滑異同移動平均線", height=300, hovermode='x unified', template='plotly_white')
    st.plotly_chart(fig4, use_container_width=True)

def render_volatility_plots(valid_data, current):
    st.markdown("### 波動與趨勢動能 (ADX, BBW, %R)")
    
    col_adx, col_bbw, col_williams, col_v_ratio = st.columns(4)
    
    col_adx.metric("當前 ADX (強度)", f"{current['ADX']:.1f}")
    col_bbw.metric("當前 BBW (帶寬)", f"{current['BBW'] * 100:.2f}%")
    col_williams.metric("當前威廉 %R", f"{current['%R']:.2f}%")
    col_v_ratio.metric("成交量比", f"{current['Volume_Ratio']:.2f}倍")
    
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
        
        stock_list = ["00631L", "00675L", "QQQ", "QLD", "TQQQ", "➕ 自填其他代碼"]
        
        if initial_stock_input in stock_list:
            default_idx = stock_list.index(initial_stock_input)
        else:
            default_idx = stock_list.index("➕ 自填其他代碼")
            
        selected_option = st.selectbox("選擇或輸入股票代碼：", stock_list, index=default_idx, key="stock_select")

        if selected_option == "➕ 自填其他代碼":
            display_val = initial_stock_input if initial_stock_input not in stock_list else ""
            stock_input = st.text_input("⌨️ 請輸入代碼 (例如 AAPL 或 2330)：", value=display_val, key="stock_manual").strip().upper()
            if not stock_input:
                stock_input = "00675L"
        else:
            stock_input = selected_option

        period_options = {
            "短期 (0.5年)": 0.5, "中期 (1年)": 1.0, "長期 (3.5年)": 3.5, "超長期 (10年)": 10.0
        }
        period_type = st.selectbox("選擇分析期間", list(period_options.keys()) + ["自訂期間"], 
                                   index=list(period_options.keys()).index(initial_period_type), 
                                   key="period_type_key")

        if period_type == "自訂期間":
            col_start, col_end = st.columns(2)
            with col_start:
                start_date_custom = st.date_input("開始日", value=datetime.now().date() - timedelta(days=365*3), key="start_date_custom_key") 
            with col_end:
                end_date_custom = st.date_input("結束日", value=datetime.now().date(), key="end_date_custom_key")
            days = (end_date_custom - start_date_custom).days
        else:
            days = int(period_options[period_type] * 365)
            st.markdown(f"<div style='color:#A07C8C; font-size:0.9rem; margin-top:5px; margin-bottom:10px;'>期間：{datetime.now().date() - timedelta(days=days)} ~ {datetime.now().date()}</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        analyze_button = st.button("開始分析", type="primary", use_container_width=True, key="analyze_button_key") 
    
    return stock_input, days, analyze_button

# ----------------------------------------------------
# 🌟 主要內容分析區 (右欄內容)
# ----------------------------------------------------
def render_analysis_main(stock_input, start_date, end_date, analyze_button):
    if analyze_button or st.session_state.get('app_initialized', False):
        st.session_state.app_initialized = True
        
        if not stock_input:
            st.error("❌ 請輸入股票代號後點擊「開始分析」")
            return
        
        try:
            if api is None and "api_warned" not in st.session_state:
                st.toast("⚠️ 找不到永豐金鑰或連線失敗，目前使用 Yahoo 歷史報價", icon="⚠️")
                st.session_state.api_warned = True

            with st.spinner("📥 正在下載與計算資料..."):
                fetch_days = (datetime.now().date() - start_date).days
                stock_data, stock_name, stock_symbol_actual = get_stock_data_auto(stock_input, fetch_days)
                
                if stock_data.empty or stock_symbol_actual is None:
                    st.error("❌ 無法取得股票資料，請檢查股票代碼是否正確。")
                    return
                
                df_calc = stock_data.copy()
                
                window = 100
                df_calc['MA20W'] = df_calc['Close'].rolling(window=window, min_periods=window).mean()
                rolling_std = df_calc['Close'].rolling(window=window, min_periods=window).std()
                df_calc['UB'] = df_calc['MA20W'] + 2 * rolling_std
                df_calc['LB'] = df_calc['MA20W'] - 2 * rolling_std
                df_calc['Zone'] = np.where(df_calc['Close'] > df_calc['MA20W'], '樂活區(多頭)', '毅力區(空頭)')

                df_calc['RSI'] = calculate_rsi(df_calc['Close'], 14)
                macd, signal, hist = calculate_macd(df_calc['Close'])
                df_calc['MACD'] = macd
                df_calc['MACD_Signal'] = signal
                df_calc['MACD_Hist'] = hist
                
                df_calc['MA5'] = df_calc['Close'].rolling(5).mean()
                df_calc['MA10'] = df_calc['Close'].rolling(10).mean()
                df_calc['MA20'] = df_calc['Close'].rolling(20).mean()
                df_calc['MA60'] = df_calc['Close'].rolling(60).mean()
                df_calc['Volume_MA5'] = df_calc['Volume'].rolling(5).mean()
                df_calc['Volume_Ratio'] = df_calc['Volume'] / df_calc['Volume_MA5'].replace(0, np.nan)
                df_calc['RSI_Divergence'] = detect_rsi_divergence(df_calc['Close'], df_calc['RSI'])
                adx, plus_di, minus_di = calculate_adx(df_calc['High'], df_calc['Low'], df_calc['Close'])
                df_calc['ADX'] = adx
                df_calc['+DI'] = plus_di
                df_calc['-DI'] = minus_di
                df_calc['BBW'] = calculate_bbw(df_calc['Close'])
                df_calc['%R'] = calculate_williams_r(df_calc['High'], df_calc['Low'], df_calc['Close'])

                # 移除 KD，改為確保 MACD 無空值
                df_calc = df_calc.dropna(subset=['MA20W', 'UB', 'LB', 'RSI', 'MACD', 'ADX', 'BBW', '%R', 'MA60'])

                target_start = pd.Timestamp(start_date)
                target_end = pd.Timestamp(end_date)
                if df_calc.index.tz is not None:
                    target_start = target_start.tz_localize(df_calc.index.tz)
                    target_end = target_end.tz_localize(df_calc.index.tz)

                valid_data = df_calc[(df_calc.index >= target_start) & (df_calc.index <= target_end)].copy()

                if valid_data.empty: 
                    st.error("❌ 該區間內資料不足以繪製五線譜 (可能因假日或無交易紀錄)")
                    return

                x_indices = np.arange(len(valid_data))
                y_values = valid_data['Close'].values
                slope, intercept = np.polyfit(x_indices, y_values, 1)
                trend_line = slope * x_indices + intercept
                sd = np.std(y_values - trend_line)

                valid_data['TL'] = trend_line
                valid_data['TL+2SD'] = trend_line + 2 * sd
                valid_data['TL+1SD'] = trend_line + 1 * sd
                valid_data['TL-1SD'] = trend_line - 1 * sd
                valid_data['TL-2SD'] = trend_line - 2 * sd
                
                current = valid_data.iloc[-1]
                slope_dir = "上升" if slope > 0 else "下降"
                deviation = current['Close'] - current['TL']
                sd_level = deviation / sd
                
                if sd_level >= 2: fiveline_zone = "極度樂觀"
                elif sd_level >= 1: fiveline_zone = "樂觀"
                elif sd_level >= 0: fiveline_zone = "合理區"
                elif sd_level >= -1: fiveline_zone = "悲觀"
                else: fiveline_zone = "極度悲觀"
                
                # 取得上方卡片要顯示的精簡行動建議
                action_detail = generate_summary_action(current, valid_data)
                
                # --- 結果呈現 ---
                st.subheader(f"📈 {stock_name} ({stock_symbol_actual})")
                
                render_metric_cards(current, fiveline_zone, action_detail)
                
                tab1, tab2, tab3, tab4 = st.tabs(["🎼 五線譜", "🌈 樂活通道", "📊 震盪指標", "波動與情緒"]) 

                with tab1: render_fiveline_plot(valid_data, slope_dir, slope);
                with tab2: render_lohas_plot(valid_data, current['Close'], current['MA20W']);
                with tab3: render_oscillator_plots(valid_data);
                with tab4: render_volatility_plots(valid_data, current);

                st.divider()
                
                # 🎯 全新的分析報告模組（戰情總結 -> 數據雷達 -> 劇本）
                analysis_result = generate_internal_analysis(stock_name, stock_symbol_actual, slope_dir, sd_level, fiveline_zone, current, valid_data)
                st.markdown(analysis_result)

        except Exception as e:
            st.error(f"❌ 錯誤：{str(e)}")
            import traceback
            st.code(traceback.format_exc())


# ----------------------------------------------------
# 🌟 主執行區塊
# ----------------------------------------------------

if 'stock_input_value' not in st.session_state:
    st.session_state.stock_input_value = "00631L"
if 'period_type_value' not in st.session_state:
    st.session_state.period_type_value = "短期 (0.5年)"

col_left, col_right = st.columns([1, 2.5]) 

with col_left:
    st.markdown('<div class="app-title">樂活五線譜</div>', unsafe_allow_html=True)
    stock_input, days, analyze_button = render_input_sidebar(st.session_state.stock_input_value, st.session_state.period_type_value)

with col_right:
    st.session_state.stock_input_value = stock_input

    period_type = st.session_state.period_type_key if 'period_type_key' in st.session_state else st.session_state.period_type_value
    period_options = {"短期 (0.5年)": 0.5,"中期 (1年)": 1.0,"長期 (3.5年)": 3.5,"超長期 (10年)": 10.0}

    if period_type == "自訂期間" and 'start_date_custom_key' in st.session_state:
        start_date = st.session_state.start_date_custom_key
        end_date = st.session_state.end_date_custom_key
    else:
        years = period_options.get(period_type, 3.5)
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=int(years * 365))
    
    render_analysis_main(stock_input, start_date, end_date, analyze_button)
