import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import shioaji as sj
import time
import os
import traceback
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
h1, .stAppHeader h1 {
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
button[kind="primary"] {
    background-color: #C8A2C8 !important; /* 柔和淡紫色 */
    color: white !important;
    border-radius: 10px !important;
    border: none !important;
    padding: 0.45rem 1rem !important;
    font-weight: 500 !important;
}

button[kind="primary"]:hover {
    background-color: #B28FB2 !important;
}

/* 綠色框框修改：Metric 主字變小，防止截斷，並維持深色 */
[data-testid="stMetricValue"] div {
    color: #5F5A58 !important;       /* 改為深灰/黑色 */
    font-weight: 600 !important;
    font-size: 1.2rem !important;    /* 縮小字體 */
    white-space: normal !important;  /* 允許換行 */
    overflow: visible !important;    /* 顯示完整文字 */
    line-height: 1.3 !important;
    word-wrap: break-word !important;
}
[data-testid="stMarkdownContainer"] ul {
    list-style: none !important;
    padding-left: 0 !important;
    margin: 0.4rem 0 !important;
}

[data-testid="stMarkdownContainer"] li {
    position: relative !important;
    padding-left: 1.5em !important;
    margin-bottom: 0.8rem !important;
    line-height: 1.7 !important;
    word-break: break-word !important;
    overflow-wrap: break-word !important;
}

[data-testid="stMarkdownContainer"] li::before {
    content: "•" !important;
    position: absolute !important;
    left: 0 !important;
    top: 0 !important;
    color: #8B6F77 !important;
}

[data-testid="stMarkdownContainer"] li > p,
[data-testid="stMarkdownContainer"] li > div {
    margin: 0 !important;
    padding: 0 !important;
}

@media (max-width: 768px) {
    [data-testid="stMarkdownContainer"] li {
        padding-left: 1.3em !important;
    }
}
</style>
"""

# ==================== 頁面配置與 CSS 注入 ====================
st.set_page_config(page_title="樂活五線譜", layout="wide")

# 注入自訂 CSS
st.markdown(custom_css, unsafe_allow_html=True)

# ==================== 🚀 永豐 API 初始化 (支援雲端與本機) ====================
@st.cache_resource(ttl=43200) # ⏳ 設定 12 小時 (43200秒) 自動過期重連
def init_api():
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
        # 沒有金鑰：結果不會因重試而改變，可以安心快取
        print("❌ [後台偵錯] 致命錯誤：完全抓不到任何金鑰字串！")
        return None

    # 🔧 登入失敗時直接拋出例外（不 return None），
    #    Streamlit 不會快取例外，下次重新整理會自動再試一次登入
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

try:
    api = init_api()
except Exception as e:
    print(f"❌ [後台偵錯] 永豐登入失敗，真正死因：{str(e)}")
    api = None

if api is None and "api_warned" not in st.session_state:
    st.toast("⚠️ 找不到永豐金鑰或連線失敗，目前使用 Yahoo 歷史報價", icon="⚠️")
    st.session_state.api_warned = True


# ==================== 🌟 核心計算函數 🌟 ====================

def calculate_rsi(data, period=14):
    # 🔧 改用 Wilder 平滑法（與一般看盤軟體 / 券商一致）
    delta = data.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    ema_fast = data.ewm(span=fast, adjust=False).mean()
    ema_slow = data.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

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
        stock_name = stock_info.get('longName') or symbol
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


# ==================== 🔀 股票分割還原（台美股共用） ====================
PRICE_COLS = ['Open', 'High', 'Low', 'Close']

def _to_naive(ts):
    """把有時區的時間轉成無時區（保留當地日期），方便與 K 線日期比較"""
    ts = pd.Timestamp(ts)
    return ts.tz_localize(None) if ts.tzinfo is not None else ts

def _ratio_text(ratio):
    return f"1 拆 {ratio:.4g}" if ratio >= 1 else f"{1/ratio:.4g} 合 1"

def _snap_ratio(x):
    """比例很接近整數（誤差 5% 內）就取整數，例如 3.97 → 4；否則保留原值"""
    n = round(x)
    return float(n) if n >= 1 and abs(x / n - 1) < 0.05 else float(x)

def _apply_split(df, pos, ratio):
    """把第 pos 根 K 棒「之前」的價格除以 ratio、成交量乘以 ratio"""
    p = [df.columns.get_loc(c) for c in PRICE_COLS]
    v = df.columns.get_loc('Volume')
    df.iloc[:pos, p] = df.iloc[:pos, p] / ratio
    df.iloc[:pos, v] = df.iloc[:pos, v] * ratio

def fix_unadjusted_splits(df, sym, market):
    """
    兩道關卡，台美股都適用：
    1) Yahoo 官方分割紀錄：逐筆檢查「價格是否真的還沒還原」，沒還原才處理（避免重複還原）
    2) 台股備援：台股有漲跌幅限制，一天內價格變成原本的 1.8 倍以上或 55% 以下
       在正常交易中不可能發生，只會是分割 / 合併（減資）→ 用實際價差還原
       美股沒有漲跌幅限制（例如 TQQQ 2020/3/16 單日跌約 34%），所以只信任官方紀錄
    """
    logs = []
    handled = set()

    # --- 第 1 道：官方分割紀錄 ---
    try:
        splits = yf.Ticker(sym).splits
    except Exception:
        splits = None
    if splits is not None and len(splits) > 0:
        for dt, ratio in splits.items():
            ratio = float(ratio)
            # 只處理明顯的分割 / 合併；小比例（多為配股）Yahoo 會自行還原
            if not (ratio >= 1.5 or 0 < ratio <= 1 / 1.5):
                continue
            pos = df.index.searchsorted(_to_naive(dt).normalize())
            if pos <= 0 or pos >= len(df):
                continue
            raw = df['Close'].iloc[pos - 1] / df['Close'].iloc[pos]
            if abs(raw / ratio - 1) < 0.15:   # 價差與分割比例吻合 → 確實還沒還原
                _apply_split(df, pos, ratio)
                logs.append(f"🔀 依官方分割紀錄還原：{df.index[pos].date()} {_ratio_text(ratio)}")
            handled.add(pos)

    # --- 第 2 道：台股備援偵測（Yahoo 漏記分割時） ---
    if market == 'TW':
        r = (df['Close'].shift(1) / df['Close']).to_numpy()
        for pos in range(1, len(df)):
            if pos in handled or not np.isfinite(r[pos]):
                continue
            x = r[pos]
            if x >= 1.8:
                ratio = _snap_ratio(x)
            elif x <= 1 / 1.8:
                ratio = 1 / _snap_ratio(1 / x)
            else:
                continue
            _apply_split(df, pos, ratio)
            logs.append(f"🔀 偵測到分割斷層：{df.index[pos].date()} 約 {_ratio_text(ratio)}，正在還原歷史資料...")

    return logs


# ==================== 📡 資料抓取（含快取） ====================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_history(sym, market, start_date):
    """Yahoo 歷史 K 線，快取 1 小時（盤中最新價由下方即時補丁負責）"""
    logs = []
    fetch_end_date = datetime.now().date() + timedelta(days=1)
    df = yf.download(sym, start=start_date, end=fetch_end_date, progress=False, auto_adjust=True)

    if (df is None or df.empty) and market == 'TW' and sym.endswith('.TW'):
        logs.append("🔄 嘗試切換至 .TWO 下載...")
        sym2 = sym.replace('.TW', '.TWO')
        df2 = yf.download(sym2, start=start_date, end=fetch_end_date, progress=False, auto_adjust=True)
        if df2 is not None and not df2.empty:
            df, sym = df2, sym2

    if df is None or df.empty:
        return pd.DataFrame(), sym, logs

    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.rename(columns=lambda x: x.capitalize(), inplace=True)

    df = df[['Open','High','Low','Close','Volume']].copy()
    df = df.sort_index()
    df['Volume'] = df['Volume'].astype(float)

    logs += fix_unadjusted_splits(df, sym, market)
    return df, sym, logs

@st.cache_data(ttl=30, show_spinner=False)
def fetch_tw_snapshot(code):
    """永豐即時快照，快取 30 秒。日期取自快照本身的時間戳（台灣時間）"""
    contract = api.Contracts.Stocks[code]
    if not contract:
        return None
    snap = api.snapshots([contract])[0]
    return {
        'date': pd.to_datetime(snap.ts).normalize(),
        'open': snap.open, 'high': snap.high, 'low': snap.low, 'close': snap.close,
        'volume': snap.total_volume * 1000,   # 張 → 股，與 Yahoo 一致
    }

@st.cache_data(ttl=30, show_spinner=False)
def fetch_us_latest_bar(sym):
    """美股最新一根日 K（含日期與高低點），快取 30 秒"""
    h = yf.Ticker(sym).history(period='1d', auto_adjust=True)
    if h is None or h.empty:
        return None
    row = h.iloc[-1]
    return {
        'date': _to_naive(h.index[-1]).normalize(),
        'open': float(row['Open']), 'high': float(row['High']), 'low': float(row['Low']),
        'close': float(row['Close']), 'volume': float(row['Volume']),
    }

def apply_latest_bar(df, bar):
    """依即時資料「本身的日期」決定新增或更新 K 棒，避免週末 / 開盤前插入假 K 棒"""
    last_date = df.index[-1].normalize()
    if bar['date'] > last_date:
        new_row = pd.DataFrame({
            'Open': [bar['open']], 'High': [bar['high']], 'Low': [bar['low']],
            'Close': [bar['close']], 'Volume': [float(bar['volume'])]
        }, index=[bar['date']])
        return pd.concat([df, new_row]), f"🆕 發現今日新 K 棒（{bar['date'].date()}），正在插入...", True
    if bar['date'] == last_date:
        df.iloc[-1, df.columns.get_loc('High')] = max(df['High'].iloc[-1], bar['high'])
        df.iloc[-1, df.columns.get_loc('Low')] = min(df['Low'].iloc[-1], bar['low'])
        df.iloc[-1, df.columns.get_loc('Close')] = bar['close']
        df.iloc[-1, df.columns.get_loc('Volume')] = float(bar['volume'])
        return df, f"🆙 正在更新最新一根 K 棒（{bar['date'].date()}）...", True
    return df, "ℹ️ 即時資料日期比歷史資料舊，略過補丁", False

def get_stock_data_auto(stock_input, days):
    start_date = (datetime.now() - timedelta(days=days + 500)).date()
    normalized_input = stock_input.strip()
    market = detect_market(normalized_input)
    
    sym = normalized_input
    if market == 'TW' and '.TW' not in sym and '.TWO' not in sym:
        sym = f"{sym}.TW"
    
    with st.status(f"🚀 正在處理 {sym}...", expanded=False) as status:
        status.write("📡 正在從 Yahoo Finance 抓取歷史 K 線...")
        df, sym, logs = fetch_history(sym, market, start_date)
        for msg in logs:
            status.write(msg)

        if df.empty:
            status.update(label="❌ Yahoo 資料下載失敗", state="error")
            return pd.DataFrame(), None, normalized_input

        status.write(f"✅ 已抓到 {len(df)} 筆歷史數據")

        if market == 'TW':
            if not api:
                status.write("ℹ️ 永豐 API 未連線，將顯示 Yahoo 盤後/延遲資料")
            else:
                status.write("💉 正在啟動永豐 API 盤中即時補丁...")
                try:
                    clean_sym = normalized_input.replace('.TW', '').replace('.TWO', '').strip()
                    bar = fetch_tw_snapshot(clean_sym)
                    if bar is None:
                        status.write("⚠️ 永豐 API 找不到此代碼，跳過補丁")
                    elif bar['close'] <= 0 or bar['volume'] <= 0:
                        status.write("⚠️ 永豐 API 回傳今日成交量為 0，跳過補丁")
                    else:
                        df, msg, applied = apply_latest_bar(df, bar)
                        status.write(msg)
                        if applied:
                            status.write(f"💎 補丁成功！目前現價：{bar['close']}")
                except Exception as e:
                    status.write(f"⚠️ 補丁失敗：{e}")
        
        if market == 'US':
            status.write("⚡ 正在抓取美股即時價格...")
            try:
                bar = fetch_us_latest_bar(sym)
                if bar and bar['close'] > 0:
                    df, msg, applied = apply_latest_bar(df, bar)
                    status.write(msg)
                    if applied:
                        status.write(f"💎 美股現價已更新：{bar['close']:.2f}")
            except Exception:
                pass

        status.update(label=f"✅ {sym} 資料準備就緒", state="complete", expanded=False)

    name, _ = get_stock_info(sym)
    if name == sym:
        name = normalized_input
        
    return df, name, sym


# ==================== 🧭 情境判斷（卡片與報告共用，確保結論一致） ====================
#    🌱 播種 → 🌾 成長 → 💰 收成：依「五線譜位階」分五段，「樂活通道」出界時直接拉到最外側
SCENARIO_LABEL = {
    'buy_more':  "🌱 加碼布局",
    'buy':       "🌿 分批布局",
    'hold':      "🌾 按兵不動",
    'trim':      "🧺 分批減碼",
    'sell_more': "💰 大幅減碼",
}

ZONE_TO_SCENARIO = {
    "極度悲觀": 'buy_more',
    "悲觀":     'buy',
    "合理區":   'hold',
    "樂觀":     'trim',
    "極度樂觀": 'sell_more',
}

def channel_position(current):
    """股價相對樂活通道的位置：below（跌破下緣）/ above（突破上緣）/ inside"""
    if current['Close'] < current['LB']:
        return 'below'
    if current['Close'] > current['UB']:
        return 'above'
    return 'inside'

def decide_scenario(fiveline_zone, current):
    scenario = ZONE_TO_SCENARIO.get(fiveline_zone.replace('及', ''), 'hold')
    channel = channel_position(current)
    buy_side, sell_side = ('buy_more', 'buy'), ('trim', 'sell_more')
    if channel == 'below':
        # 跌破下緣 → 加碼布局；但五線譜位階偏高時兩者矛盾，保守處理為按兵不動
        scenario = 'hold' if scenario in sell_side else 'buy_more'
    elif channel == 'above':
        scenario = 'hold' if scenario in buy_side else 'sell_more'
    return scenario, channel

# 輔助：產出卡片用的精簡建議（與下方戰情總結使用同一套判斷）
def generate_summary_action(current, fiveline_zone):
    scenario, _ = decide_scenario(fiveline_zone, current)
    return SCENARIO_LABEL[scenario]

# 輔助：呈現數據卡片
def render_metric_cards(current, fiveline_zone, action_detail):
    current_price = current['Close']
    
    with st.container(border=True):
        st.markdown("#### 關鍵數據摘要")
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("股價 (收盤)", f"{current_price:.2f}") 

        fiveline_zone_clean = fiveline_zone.replace("及", "")
        col2.metric("五線譜位階", fiveline_zone_clean)
        
        # 🔧 改名「短線情緒」：威廉指標只看最近 14 天，避免跟五線譜位階的用詞混淆
        sentiment_val = current['%R']
        if sentiment_val > -20: sentiment_text = "超買"
        elif sentiment_val < -80: sentiment_text = "超賣"
        else: sentiment_text = "中性"
        col3.metric("短線情緒", sentiment_text)
        
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
    
    analysis_text = []

    is_trend_up = adx > 25 and p_di > m_di
    is_trend_down = adx > 25 and m_di > p_di
    
    # 1. 🎯 戰情總結（與上方卡片共用 decide_scenario：五線譜位階 + 樂活通道）
    scenario, channel = decide_scenario(fiveline_zone, current)
    ub, ma20w, lb = f"{current['UB']:.2f}", f"{current['MA20W']:.2f}", f"{current['LB']:.2f}"
    prev_wr = previous['%R']
    if scenario == 'buy_more':
        summary = "### 🎯 戰情總結：市場極度恐慌，建議【🌱 加碼布局】"
        script_hold = f"已有部位者可在此加碼，拉低平均成本。請分多批進場、預留資金，不要一次投入；價格在通道下緣 {lb} 附近或以下，都是可以分批承接的區域。"
        script_empty = f"市場恐慌時往往是長線較好的買點，可開始分批建立部位，並預留資金因應續跌；回到 20 週均線 {ma20w} 之前都還算相對低檔。"
    elif scenario == 'buy':
        summary = "### 🎯 戰情總結：位階偏低，建議【🌿 分批布局】"
        script_hold = f"可小量加碼，保留大部分資金；若再跌到極度悲觀區或跌破通道下緣 {lb}，再加大力道。"
        script_empty = f"可先小量買進建立部位，不必等最低點；之後若跌近通道下緣 {lb}，再分批加碼。"
    elif scenario == 'hold':
        summary = "### 🎯 戰情總結：沒有明顯的買賣訊號，建議【🌾 按兵不動】"
        script_hold = f"有部位就續抱，以 20 週均線 {ma20w} 作為防守參考，不必頻繁進出。"
        script_empty = f"價格不便宜也不算貴，不追高也不急著買；等回到悲觀區，或接近通道下緣 {lb} 再分批布局。"
    elif scenario == 'trim':
        summary = "### 🎯 戰情總結：位階偏高，建議【🧺 分批減碼】"
        script_hold = f"停止加碼，可先賣出一小部分獲利入袋；剩餘部位續抱，若漲到極度樂觀區或突破通道上緣 {ub}，再多賣一些。"
        script_empty = f"價格已偏貴，不建議此時追高，耐心等待回檔到 20 週均線 {ma20w} 附近。"
    else:
        summary = "### 🎯 戰情總結：市場情緒過熱，建議【💰 大幅減碼】"
        script_hold = f"情緒過熱，可分批賣出較多部位、落袋為安，保留少部分參與後續行情；若跌回 20 週均線 {ma20w} 以下，再重新檢視剩餘部位。"
        script_empty = f"追高風險極大，請保持耐心，等待價格回到合理區或悲觀區，至少回到 20 週均線 {ma20w} 附近再考慮。"

    analysis_text.append(summary)

    # 判斷依據
    channel_text = {'below': "已跌破樂活通道下緣", 'above': "已突破樂活通道上緣", 'inside': "位於樂活通道內"}[channel]
    analysis_text.append(f"* 判斷依據： 五線譜位於「{fiveline_zone.replace('及', '')}」，股價{channel_text}"
                         f"（上緣 {ub}／20週均線 {ma20w}／下緣 {lb}）。")
    if channel != 'inside':
        analysis_text.append("* 提醒： 樂活通道訊號以週收盤為準，本週收盤前仍可能變動。")

    # 時機提醒：威廉指標與 MACD 只用來調整節奏，不改變建議方向；兩者同時出現時合併成一句
    if scenario in ('buy_more', 'buy'):
        wr_ok = prev_wr < -80 <= w_r                       # 剛離開超賣區
        macd_weak = macd_hist < previous['MACD_Hist']       # 下跌動能未止
        if wr_ok and macd_weak:
            analysis_text.append("* ⏱️ 時機提醒： 威廉指標剛離開超賣區，短線出現止跌訊號；但 MACD 下跌動能還沒停，建議先小量開始，並拉長分批的間隔。")
        elif wr_ok:
            analysis_text.append("* ⏱️ 時機提醒： 威廉指標剛離開超賣區，短線出現止跌訊號，可以開始執行分批布局。")
        elif macd_weak:
            analysis_text.append("* ⏱️ 時機提醒： MACD 顯示下跌動能還沒停止，建議拉長分批的間隔，不必急著一次買足。")
    elif scenario in ('trim', 'sell_more'):
        wr_ok = prev_wr > -20 >= w_r                       # 剛離開超買區
        macd_strong = macd_hist > previous['MACD_Hist']     # 上漲動能仍強
        if wr_ok and macd_strong:
            analysis_text.append("* ⏱️ 時機提醒： 威廉指標剛離開超買區，短線漲勢轉弱；但 MACD 上漲動能仍強，建議先賣一小部分，再慢慢分批。")
        elif wr_ok:
            analysis_text.append("* ⏱️ 時機提醒： 威廉指標剛離開超買區，短線漲勢轉弱，可以開始執行分批減碼。")
        elif macd_strong:
            analysis_text.append("* ⏱️ 時機提醒： MACD 顯示上漲動能仍強，可以慢慢分批賣，不必一次賣太多。")
    analysis_text.append("\n### ⚖️ 多空雷達掃描：")
    
    # 2. 多空雷達掃描（輔助參考）
    fiveline_clean = fiveline_zone.replace('及', '')
    if is_trend_up:
        analysis_text.append(f"* 【趨勢強度】🌸 多方掌控： 五線譜位於「{fiveline_clean}」，且 ADX 高達 {adx:.1f} (+DI: {p_di:.1f}, -DI: {m_di:.1f})，顯示多方大趨勢極強。")
    elif is_trend_down:
        analysis_text.append(f"* 【趨勢強度】🌧️ 空方壓制： 五線譜位於「{fiveline_clean}」，且 ADX 高達 {adx:.1f} (+DI: {p_di:.1f}, -DI: {m_di:.1f})，顯示空方大趨勢極強。")
    else:
        analysis_text.append(f"* 【趨勢強度】🌾 區間震盪： 五線譜位於「{fiveline_clean}」，ADX 為 {adx:.1f} (趨勢偏弱)，目前無明顯單向趨勢。")

    if macd > 0 and macd_hist < 0 and previous['MACD_Hist'] >= 0:
        analysis_text.append(f"* 【短線動能】❄️ 高檔死叉 (引擎冷卻)： MACD 於零軸上死叉 (DIF: {macd:.2f}, DEA: {macd_sig:.2f})，紅柱轉綠 ({macd_hist:.2f})，短線推升引擎失去動能。")
    elif macd < 0 and macd_hist > 0 and previous['MACD_Hist'] <= 0:
        analysis_text.append(f"* 【短線動能】🌸 低檔金叉 (引擎發動)： MACD 於零軸下金叉 (DIF: {macd:.2f}, DEA: {macd_sig:.2f})，綠柱轉紅 ({macd_hist:.2f})，短線有築底反彈跡象。")
    # 🔧 柱狀體正負分開描述：紅柱（>0）代表多方、綠柱（<0）代表空方
    elif macd_hist >= 0 and macd_hist > previous['MACD_Hist']:
        analysis_text.append(f"* 【短線動能】🌸 紅柱擴張： MACD 柱狀體為正且變長 (DIF: {macd:.2f}, 柱狀體: {macd_hist:.2f})，上漲動能增強。")
    elif macd_hist >= 0:
        analysis_text.append(f"* 【短線動能】🌧️ 紅柱縮短： MACD 柱狀體為正但變短 (DIF: {macd:.2f}, 柱狀體: {macd_hist:.2f})，上漲動能減弱。")
    elif macd_hist < previous['MACD_Hist']:
        analysis_text.append(f"* 【短線動能】🌧️ 綠柱擴張： MACD 柱狀體為負且變長 (DIF: {macd:.2f}, 柱狀體: {macd_hist:.2f})，下跌動能增強。")
    else:
        analysis_text.append(f"* 【短線動能】🌾 綠柱縮短： MACD 柱狀體為負但變短 (DIF: {macd:.2f}, 柱狀體: {macd_hist:.2f})，下跌動能減弱，尚未轉為多方。")

    # 🔧 威廉指標改看「進入／離開」超買超賣區：進入只代表正在漲／跌，離開才是轉折訊號
    vol_text = f"RSI(14) 為 {rsi:.1f}，布林帶寬 BBW 為 {bbw:.2f}%"
    if w_r > -20 and prev_wr <= -20:
        analysis_text.append(f"* 【情緒與波動】🌾 進入超買區： 威廉指標 W%R 升至 {w_r:.2f}，{vol_text}。代表正在強勢上漲，先觀察，強勢時可能持續超買一段時間。")
    elif w_r > -20:
        analysis_text.append(f"* 【情緒與波動】🌾 持續超買： 威廉指標 W%R 為 {w_r:.2f}，{vol_text}。漲勢仍在延續，等跌回 -20 以下才代表轉弱。")
    elif prev_wr > -20:
        analysis_text.append(f"* 【情緒與波動】🌧️ 離開超買區： 威廉指標 W%R 從超買區跌回 {w_r:.2f}，{vol_text}。漲勢出現轉弱訊號。")
    elif w_r < -80 and prev_wr >= -80:
        analysis_text.append(f"* 【情緒與波動】🌾 進入超賣區： 威廉指標 W%R 降至 {w_r:.2f}，{vol_text}。代表正在下跌，先觀察，跌勢中可能持續超賣一段時間。")
    elif w_r < -80:
        analysis_text.append(f"* 【情緒與波動】🌾 持續超賣： 威廉指標 W%R 為 {w_r:.2f}，{vol_text}。跌勢仍在延續，等回升到 -80 以上才代表止跌。")
    elif prev_wr < -80:
        analysis_text.append(f"* 【情緒與波動】🌸 離開超賣區： 威廉指標 W%R 從超賣區回升到 {w_r:.2f}，{vol_text}。出現止跌反彈訊號。")
    else:
        analysis_text.append(f"* 【情緒與波動】🌾 情緒穩定： 威廉指標 W%R 為 {w_r:.2f}，{vol_text}，未見極端情緒。")
    # 3. 行動劇本
    analysis_text.append("\n### 🛡️ 行動劇本與防守點位：")
    analysis_text.append(f"* 若持有部位： {script_hold}")
    analysis_text.append(f"* 若空手觀望： {script_empty}")
    
    return "\n".join(analysis_text)


# 輔助：圖表函數 (保持新顏色)

def render_fiveline_plot(valid_data, slope_dir, slope):
    if st.session_state.get('fiveline_mode', '一般') == '對數':
        annual = (np.exp(slope * 252) - 1) * 100   # 每日對數斜率 → 年化成長率（一年約 252 個交易日）
        st.markdown(f"趨勢年化成長率: **{annual:+.1f}% ({slope_dir})**")
    else:
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
    dt_obs = {d.strftime("%Y-%m-%d") for d in valid_data.index}   # 🔧 改用 set，查找速度快上百倍
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
                st.date_input("開始日", value=datetime.now().date() - timedelta(days=365*3), key="start_date_custom_key") 
            with col_end:
                st.date_input("結束日", value=datetime.now().date(), key="end_date_custom_key")
        else:
            days = int(period_options[period_type] * 365)
            st.markdown(f"<div style='color:#A07C8C; font-size:0.9rem; margin-top:5px; margin-bottom:10px;'>期間：{datetime.now().date() - timedelta(days=days)} ~ {datetime.now().date()}</div>", unsafe_allow_html=True)

        # 🔧 五線譜算法切換：一般（直線，與參考網站一致）／對數（適合長期成長、槓桿型標的）
        st.radio("五線譜算法", ["一般", "對數"], horizontal=True, key="fiveline_mode")

        st.markdown("---")
        analyze_button = st.button("開始分析", type="primary", use_container_width=True, key="analyze_button_key") 
    
    return stock_input, analyze_button

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

            fetch_days = (datetime.now().date() - start_date).days
            stock_data, stock_name, stock_symbol_actual = get_stock_data_auto(stock_input, fetch_days)
            
            if stock_data.empty or stock_symbol_actual is None:
                st.error("❌ 無法取得股票資料，請檢查股票代碼是否正確。")
                return
            
            df_calc = stock_data.copy()
            
            # 🔧 樂活通道：週 K + 20 週 + Price Headley 加速通道（倍數 4）
            #    已用 sentimentinsideout.com 的 0050 六個日期數值驗證，誤差約 0.1~0.3
            #    先把日 K 合併成週 K（休市的週自然不會出現），算完再對回每一天
            week = df_calc.index.to_period('W-SUN')
            wk = df_calc.groupby(week).agg(High=('High', 'max'), Low=('Low', 'min'), Close=('Close', 'last'))
            hl_ratio = (wk['High'] - wk['Low']) / (wk['High'] + wk['Low'])
            wk['MA20W'] = wk['Close'].rolling(20).mean()
            wk['UB'] = (wk['High'] * (1 + 4 * hl_ratio)).rolling(20).mean()
            wk['LB'] = (wk['Low'] * (1 - 4 * hl_ratio)).rolling(20).mean()
            df_calc[['MA20W', 'UB', 'LB']] = wk.loc[week, ['MA20W', 'UB', 'LB']].to_numpy()
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

            # 🔧 五線譜：「一般」對股價做直線回歸；「對數」對 log(股價) 回歸後再換算回股價（線條微彎、不會變負數）
            is_log = st.session_state.get('fiveline_mode', '一般') == '對數'
            to_scale = np.log if is_log else (lambda v: v)
            from_scale = np.exp if is_log else (lambda v: v)

            x_indices = np.arange(len(valid_data))
            y_values = to_scale(valid_data['Close'].values)
            slope, intercept = np.polyfit(x_indices, y_values, 1)
            trend_line = slope * x_indices + intercept
            sd = np.std(y_values - trend_line)

            valid_data['TL'] = from_scale(trend_line)
            valid_data['TL+2SD'] = from_scale(trend_line + 2 * sd)
            valid_data['TL+1SD'] = from_scale(trend_line + 1 * sd)
            valid_data['TL-1SD'] = from_scale(trend_line - 1 * sd)
            valid_data['TL-2SD'] = from_scale(trend_line - 2 * sd)

            current = valid_data.iloc[-1]
            slope_dir = "上升" if slope > 0 else "下降"
            sd_level = (y_values[-1] - trend_line[-1]) / sd
            
            # 🔧 改為上下對稱：±1 標準差內為合理區，±1~±2 為樂觀／悲觀，超過 ±2 為極度
            if sd_level >= 2: fiveline_zone = "極度樂觀"
            elif sd_level >= 1: fiveline_zone = "樂觀"
            elif sd_level > -1: fiveline_zone = "合理區"
            elif sd_level > -2: fiveline_zone = "悲觀"
            else: fiveline_zone = "極度悲觀"
            
            # 取得上方卡片要顯示的精簡行動建議（與下方報告同一套判斷）
            action_detail = generate_summary_action(current, fiveline_zone)
            
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
            # 🔧 完整錯誤細節只印在後台 log，畫面只顯示簡短訊息
            print(traceback.format_exc())
            st.error(f"❌ 發生錯誤，請稍後再試（{str(e)}）")


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
    stock_input, analyze_button = render_input_sidebar(st.session_state.stock_input_value, st.session_state.period_type_value)

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
