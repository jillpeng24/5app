"""
樂活五線譜 v2
- 常用清單 6 檔＋搜尋其他代號
- 依研究結果給建議：0.5 年五線譜循環、威廉、樂活通道、防大跌（200 日線）、VIX
- 有永豐金鑰（Secrets 或 config/.env）→ 台股用永豐即時報價；沒有 → 用 Yahoo（延遲報價）
  同一份程式部署兩次：自己的網址放金鑰，朋友的網址不放
"""
import json
import os
import time
from datetime import datetime, timedelta, timezone
from html import escape

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import requests
import yfinance as yf

try:
    import shioaji as sj
except Exception:          # 沒裝 shioaji 也能跑（朋友版用不到）
    sj = None
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

st.set_page_config(page_title="樂活五線譜", page_icon="🎼", layout="centered")
TW_TZ = timezone(timedelta(hours=8))
HERE = os.path.dirname(os.path.abspath(__file__))

# ==================== 常用清單與規則 ====================
WATCH = {
    "00631L": "元大台灣50正2",
    "00675L": "富邦臺灣加權正2",
    "00685L": "群益臺灣加權正2",
    "QQQ": "Invesco QQQ",
    "QLD": "ProShares 那斯達克 2 倍",
    "TQQQ": "ProShares 那斯達克 3 倍",
}
TW_LEV = dict(sell_z=2.5, sell="賣 1/2", buy_z=-1.0, wr=-80,
              idx="^TWII", idx_name="加權", dd=0.20, defend="賣一半")
RULES = {
    "00631L": TW_LEV, "00675L": TW_LEV, "00685L": TW_LEV,
    "QLD": dict(sell_z=2.5, sell="賣 1/3", buy_z=-2.0, wr=-80,
                idx="^NDX", idx_name="那斯達克", dd=0.20, defend="全部賣出"),
    "TQQQ": dict(sell_z=2.5, sell="賣 1/3", buy_z=None, wr=-80,
                 idx="^NDX", idx_name="那斯達克", dd=0.30, defend="全部賣出"),
    "QQQ": dict(sell_z=None, idx="^NDX", idx_name="那斯達克", dd=None),
}

# ==================== 外觀 ====================
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Huninn&display=swap" rel="stylesheet">
<style>
:root{--paper:#F6F1E8;--sheet:#FFFDF8;--ink:#4A4038;--muted:#9A8F82;--rule:#EAE1D3;
  --sell:#B8407F;--buy:#3F63C8;--up:#D0705E;--down:#6E9E80}
#MainMenu,footer,header[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"],
.stAppDeployButton,[data-testid="manage-app-button"]{display:none!important;visibility:hidden}
html,body,.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"]{background:var(--paper)!important}
.stApp, .stApp p, .stApp span, .stApp div, .stApp button, .stApp input{font-family:"Huninn","PingFang TC","Noto Sans TC",system-ui,sans-serif!important}
.stApp [data-testid="stIconMaterial"],.stApp [class*="material-symbols"],.stApp [class*="material-icons"]{font-family:"Material Symbols Rounded"!important}  /* 圖示要用圖示字型，不然會變成英文字 */
.stApp{color:var(--ink)}
.block-container{max-width:780px;padding-top:1.2rem!important;padding-bottom:2rem!important}
@media (max-width:640px){.block-container{padding-left:12px!important;padding-right:12px!important}}
.num{font-variant-numeric:tabular-nums}
.masthead{display:flex;align-items:baseline;justify-content:space-between;gap:12px}
.masthead .title{margin:0;padding:0;font-size:26px!important;line-height:1.3;font-weight:400;letter-spacing:.08em;color:var(--ink)}
.masthead span{color:var(--muted);font-size:13px}
/* 標的切換（st.pills） */
[data-testid="stPills"] button,[data-testid="stButtonGroup"] button{border-radius:999px!important;border:1px solid var(--rule)!important;background:var(--sheet)!important;color:var(--muted)!important}
[data-testid="stPills"] button[kind$="Active"],[data-testid="stButtonGroup"] button[kind$="Active"],
[data-testid="stPills"] button[aria-checked="true"],[data-testid="stButtonGroup"] button[aria-checked="true"]{background:var(--ink)!important;color:var(--sheet)!important;border-color:var(--ink)!important}
/* 期間、算法：小一點的分段按鈕 */
.st-key-per button,.st-key-alg button{font-size:14px!important;min-height:0!important;padding:2px 12px!important;border-color:var(--rule)!important;background:var(--sheet)!important;color:var(--muted)!important}
.st-key-per button p,.st-key-alg button p{font-size:14px!important}
.st-key-per button[kind$="Active"],.st-key-alg button[kind$="Active"],
.st-key-per button[aria-checked="true"],.st-key-alg button[aria-checked="true"]{background:var(--paper)!important;color:var(--ink)!important;border-color:var(--staff,#D6CAB6)!important}
.cus-lab{color:var(--muted);font-size:14px;white-space:nowrap}
/* 月曆上方的月份、年份選單：加寬，避免「7 月」被擠成兩行 */
[data-testid="stDateInputHeaderPickerPopover"],[data-testid="stDateInputHeaderPickerPopover"] [role="listbox"]{min-width:0!important;width:max-content!important}
[data-testid="stDateInputHeaderPickerPopover"] [role="option"]{white-space:nowrap!important;width:auto!important;padding-left:10px!important;padding-right:10px!important}
.st-key-d0 input,.st-key-d1 input{font-size:15px!important;padding-left:8px!important;padding-right:4px!important}
[data-testid="stSelectbox"] div[data-baseweb="select"]>div{border-radius:999px!important;background:var(--sheet)!important}
[data-testid="stDateInput"] input{border-radius:999px!important;background:var(--sheet)!important}
[data-testid="stTextInput"] input{border-radius:999px!important;background:var(--sheet)!important}
.head{display:grid;grid-template-columns:1fr auto;align-items:end;gap:2px 16px;margin:6px 0 0}
.head .name{font-size:15px;color:var(--muted)}
.head .price{font-size:46px;line-height:1.05;color:var(--ink)}
.head .chg{font-size:17px;margin-left:8px}
.head .date{grid-column:2;grid-row:1/3;text-align:right;color:var(--muted);font-size:14px;align-self:start;line-height:1.5;white-space:nowrap}
.head .delay{display:inline-block;margin-top:2px;font-size:12px;border:1px solid var(--rule);border-radius:999px;padding:0 8px}
.up{color:var(--up)} .down{color:var(--down)}
.verdict{margin:14px 0 16px;padding:14px 18px;background:var(--sheet);border-radius:14px}
.verdict .lab{font-size:13.5px;color:var(--muted);margin:0}
.verdict .t{display:flex;align-items:baseline;gap:10px;font-size:26px;margin:0;color:var(--ink)}
.verdict .t .z{font-size:18px}
.verdict p{margin:2px 0 0;color:var(--muted);font-size:16px}
.verdict p b{font-weight:400}
.cycles{display:flex;gap:6px;margin-top:12px;padding-top:10px;border-top:1px dashed var(--rule);align-items:stretch;flex-wrap:wrap}
.cycles .h{font-size:13px;color:var(--muted);align-self:center;margin-right:2px;line-height:1.4}
.cycles a{flex:1;min-width:0;background:var(--paper);border-radius:10px;padding:5px 8px;text-align:center;line-height:1.35;text-decoration:none!important}
.cycles a small{display:block;font-size:12.5px;color:var(--muted)}
.cycles a span{font-size:15px;white-space:nowrap}
@media (max-width:640px){.cycles .h{flex-basis:100%;margin:0 0 2px}.cycles .h br{display:none}}  /* 手機：標題一行、三格同一排 */
.hi{color:var(--sell)!important} .lo{color:var(--buy)!important} .mid{color:var(--muted)!important}
ul.rules{list-style:none;margin:-8px 0 0;padding:0}
ul.rules li{display:grid;grid-template-columns:5em 1fr auto;gap:0 12px;padding:12px 4px;border-bottom:1px solid var(--rule);margin:0}
ul.rules .k{font-size:17px} ul.rules .v{font-size:16px}
ul.rules .s{grid-column:2/4;color:var(--muted);font-size:14.5px}
ul.rules .st{align-self:start;white-space:nowrap}
.pill{display:inline-block;border-radius:999px;padding:0 10px;font-size:13.5px}
.pill.buy{color:var(--buy);background:#E6ECFA} .pill.idle{color:var(--muted);background:#F0EAE0} .pill.sell{color:var(--sell);background:#F8E4EE}
.lv.sell{color:var(--sell)} .lv.buy{color:var(--buy)}
.foot{margin-top:22px;color:var(--muted);font-size:13px;text-align:center}
</style>
""", unsafe_allow_html=True)


# ==================== 永豐 API（有金鑰才連） ====================
@st.cache_resource(ttl=43200)
def init_api():
    if sj is None:
        return None
    if load_dotenv:
        load_dotenv(dotenv_path=os.path.join(HERE, "config", ".env"))
    key, secret = os.getenv("SHIOAJI_API_KEY"), os.getenv("SHIOAJI_SECRET_KEY")
    if not key:
        try:
            key, secret = st.secrets["SHIOAJI_API_KEY"], st.secrets["SHIOAJI_SECRET_KEY"]
        except Exception:
            pass
    if not key or not secret:
        return None                      # 朋友版：沒有金鑰，安靜改用 Yahoo
    api = sj.Shioaji()
    api.login(key, secret)               # 失敗會拋例外，不會被快取，下次重新整理再試
    t0 = time.time()
    while not api.Contracts.Stocks and time.time() - t0 < 10:
        time.sleep(0.5)
    return api


try:
    api = init_api()
except Exception as e:
    print(f"永豐登入失敗：{e}")
    api = None


# ==================== 資料 ====================
def is_tw(code):
    c = code.upper().replace(".TWO", "").replace(".TW", "")
    return code.upper().endswith((".TW", ".TWO")) or (len(c) >= 4 and c[:4].isdigit())


def _clean(df):
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.rename(columns=lambda x: x.capitalize())
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"]).sort_index()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_history(sym, years=11):
    """Yahoo 歷史日 K（還原除權息）。台股 .TW 抓不到再試 .TWO"""
    start = (datetime.now() - timedelta(days=int(years * 365) + 200)).date()
    end = datetime.now().date() + timedelta(days=1)
    tries = [sym] + ([sym.replace(".TW", ".TWO")] if sym.endswith(".TW") else [])
    for s in tries:
        try:
            df = _clean(yf.download(s, start=start, end=end, progress=False, auto_adjust=True))
        except Exception:
            df = pd.DataFrame()
        if not df.empty:
            return df, s
    return pd.DataFrame(), sym


@st.cache_data(ttl=30, show_spinner=False)
def fetch_tw_snapshot(code):
    """永豐即時快照，30 秒更新一次"""
    contract = api.Contracts.Stocks[code]
    if not contract:
        return None
    snap = api.snapshots([contract])[0]
    ts = pd.to_datetime(snap.ts)
    return {"ts": ts, "date": ts.normalize(), "open": snap.open, "high": snap.high,
            "low": snap.low, "close": snap.close, "name": getattr(contract, "name", ""),
            "chg": getattr(snap, "change_rate", None)}   # 永豐自己算的漲跌幅（%），比用 Yahoo 前一天收盤算準


@st.cache_data(ttl=60, show_spinner=False)
def fetch_us_latest(sym):
    h = _clean(yf.Ticker(sym).history(period="1d", auto_adjust=True))
    if h.empty:
        return None
    r = h.iloc[-1]
    return {"ts": None, "date": h.index[-1].normalize(), "open": r.Open, "high": r.High, "low": r.Low, "close": r.Close}


@st.cache_data(ttl=86400, show_spinner=False)
def yahoo_name(sym):
    try:
        info = yf.Ticker(sym).info
        return info.get("shortName") or info.get("longName") or sym
    except Exception:
        return sym


def apply_bar(df, bar):
    """把即時價補到最後一根（同一天就更新，新的一天就新增）"""
    last = df.index[-1].normalize()
    if bar["date"] > last:
        row = pd.DataFrame({"Open": [bar["open"]], "High": [bar["high"]], "Low": [bar["low"]],
                            "Close": [bar["close"]], "Volume": [0.0]}, index=[bar["date"]])
        return pd.concat([df, row])
    if bar["date"] == last:
        df = df.copy()
        df.iloc[-1, df.columns.get_loc("High")] = max(df["High"].iloc[-1], bar["high"])
        df.iloc[-1, df.columns.get_loc("Low")] = min(df["Low"].iloc[-1], bar["low"])
        df.iloc[-1, df.columns.get_loc("Close")] = bar["close"]
    return df


@st.cache_data(ttl=20, show_spinner=False)
def fetch_mis_quote(code):
    """證交所即時行情（沒有永豐金鑰時用；延遲約幾秒到一分鐘）"""
    js = requests.get(f"https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch=tse_{code}.tw|otc_{code}.tw&json=1&delay=0",
                      timeout=6, headers={"User-Agent": "Mozilla/5.0"}).json()
    for r in js.get("msgArray") or []:
        num = lambda k: float(r[k]) if r.get(k) not in (None, "", "-") else None
        px = num("z")
        if px is None:                                   # 這幾秒沒成交：用最佳買價
            b = (r.get("b") or "").split("_")[0]
            px = float(b) if b not in ("", "-") else None
        if not px:
            continue
        ts = pd.Timestamp(datetime.strptime(f'{r["d"]} {r.get("t", "13:30:00")}', "%Y%m%d %H:%M:%S"))
        y = num("y")
        return {"ts": ts, "date": ts.normalize(), "close": px, "open": num("o") or px,
                "high": num("h") or px, "low": num("l") or px, "name": r.get("n", ""),
                "chg": (px / y - 1) * 100 if y else None}
    return None


def load(code):
    """回傳 (日K, 名稱, 是否即時, 報價時間文字)"""
    tw = is_tw(code)
    sym = code.upper()
    if tw and not sym.endswith((".TW", ".TWO")):
        sym += ".TW"
    df, sym = fetch_history(sym)
    if df.empty:
        return None
    live, when, live_chg, name = False, None, None, WATCH.get(code.upper())
    clean = sym.replace(".TWO", "").replace(".TW", "")
    if tw and api is not None:
        try:
            bar = fetch_tw_snapshot(clean)
            if bar and bar["close"] > 0:
                df = apply_bar(df, bar)
                live, when, live_chg = True, bar["ts"], bar.get("chg")
                name = name or bar["name"]
        except Exception as e:
            print(f"永豐快照失敗：{e}")
    elif tw:                                             # 朋友版：證交所即時行情
        try:
            bar = fetch_mis_quote(clean)
            if bar and bar["close"] > 0:
                df = apply_bar(df, bar)
                when, live_chg = bar["ts"], bar.get("chg")
                name = name or bar["name"]
        except Exception as e:
            print(f"證交所即時行情失敗：{e}")
    else:
        try:
            bar = fetch_us_latest(sym)
            if bar and bar["close"] > 0:
                df = apply_bar(df, bar)
        except Exception:
            pass
    return df, name or yahoo_name(sym), live, when, sym, live_chg


# ==================== 指標 ====================
def williams_r(df, n=14):
    hh, ll = df["High"].rolling(n).max(), df["Low"].rolling(n).min()
    return -100 * (hh - df["Close"]) / (hh - ll).replace(0, np.nan)


def lohas(df):
    """樂活通道：週 K、20 週、倍數 4（跟原本 App 同一套公式），再對回每一天"""
    week = df.index.to_period("W-SUN")
    wk = df.groupby(week).agg(High=("High", "max"), Low=("Low", "min"), Close=("Close", "last"))
    r = (wk["High"] - wk["Low"]) / (wk["High"] + wk["Low"])
    wk["MA"] = wk["Close"].rolling(20).mean()
    wk["UB"] = (wk["High"] * (1 + 4 * r)).rolling(20).mean()
    wk["LB"] = (wk["Low"] * (1 - 4 * r)).rolling(20).mean()
    return pd.DataFrame(wk.loc[week, ["MA", "UB", "LB"]].to_numpy(), index=df.index, columns=["MA", "UB", "LB"])


def zscore(close, years=None, today=None, start=None, end=None, log=False):
    """五線譜位階：從今天往回 N×365 天（或自訂起訖日），以交易日序號回歸；對數模式先取 log"""
    if years is not None:
        start, end = pd.Timestamp(today) - pd.Timedelta(days=int(years * 365)), None
    w = close[close.index >= pd.Timestamp(start)]
    if end is not None:
        w = w[w.index <= pd.Timestamp(end)]
    if len(w) < 20:
        return None
    y = np.log(w.values) if log else w.values
    x = np.arange(len(w))
    b, a = np.polyfit(x, y, 1)
    sd = np.std(y - (b * x + a))
    return float((y[-1] - (b * x[-1] + a)) / sd)


@st.cache_data(ttl=86400, show_spinner=False)
def _twse_month(yyyymm, day_key):
    """證交所：某個月的加權指數每日收盤（day_key 讓「這個月」每天重抓，過去的月份只抓一次）"""
    js = requests.get(f"https://www.twse.com.tw/rwd/zh/TAIEX/MI_5MINS_HIST?date={yyyymm}01&response=json",
                      timeout=6, headers={"User-Agent": "Mozilla/5.0"}).json()
    out = {}
    for r in js.get("data") or []:
        y, m, dd = r[0].split("/")
        out[pd.Timestamp(int(y) + 1911, int(m), int(dd))] = float(r[4].replace(",", ""))
    return out


@st.cache_data(ttl=600, show_spinner=False)
def twse_taiex(today):
    """加權指數最近約 11 個月的收盤（證交所官方資料，算 200 日線用）"""
    out = {}
    for k in range(12):                       # 從這個月往回抓；抓不到就停，改用 Yahoo
        mth = (pd.Timestamp(today) - pd.DateOffset(months=k)).strftime("%Y%m")
        try:
            out.update(_twse_month(mth, f"{today}-{datetime.now(TW_TZ):%H}" if k == 0 else ""))
        except Exception:
            break
    return pd.Series(out, dtype=float).sort_index()


@st.cache_data(ttl=60, show_spinner=False)
def taiex_live():
    """盤中加權指數：有永豐用永豐，沒有用證交所即時行情（約 5 秒更新）"""
    try:
        if api is not None:
            snap = api.snapshots([api.Contracts.Indexs.TSE.TSE001])[0]
            if snap.close > 0:
                return pd.to_datetime(snap.ts).normalize(), float(snap.close)
    except Exception:
        pass
    try:
        js = requests.get("https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch=tse_t00.tw&json=1&delay=0",
                          timeout=8, headers={"User-Agent": "Mozilla/5.0"}).json()
        r = js["msgArray"][0]
        z = r.get("z")
        if z and z != "-":
            return pd.Timestamp(datetime.strptime(r["d"], "%Y%m%d")), float(z)
    except Exception:
        pass
    return None


def index_closes(idx, today):
    """大盤每日收盤：長歷史用 Yahoo；加權最近一年用證交所官方資料蓋過去（Yahoo 常晚一天），再補盤中即時"""
    df, _ = fetch_history(idx, years=3)
    s = df["Close"].copy() if not df.empty else pd.Series(dtype=float)
    if idx == "^TWII":
        try:
            tw = twse_taiex(today)
            if len(tw):
                s = pd.concat([s[s.index < tw.index[0]], tw])
        except Exception:
            pass
        live = taiex_live()
        if live and (s.empty or live[0] >= s.index[-1]):
            s = s.copy()
            s[live[0]] = live[1]
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s if len(s) else None


def index_status(idx, today):
    """大盤指數：收盤、200 日線、最近 5 天內有沒有剛站回、資料日期"""
    s = index_closes(idx, today)
    if s is None or len(s) < 210:
        return None
    ma = s.rolling(200).mean()
    above = s > ma
    return {"close": float(s.iloc[-1]), "ma": float(ma.iloc[-1]), "date": s.index[-1].normalize(),
            "above": bool(above.iloc[-1]), "reclaimed": bool(above.iloc[-1] and not above.iloc[-6:-1].all())}


@st.cache_data(ttl=1800, show_spinner=False)
def vix_now():
    df, _ = fetch_history("^VIX", years=3)
    return None if df.empty else float(df["Close"].iloc[-1])


@st.cache_data(ttl=1800, show_spinner=False)
def vix_series():
    df, _ = fetch_history("^VIX", years=3)
    return None if df.empty else df["Close"]


@st.cache_data(ttl=600, show_spinner=False)
def replay(close, wr, idx_s, vix_s, rule, yrs, tw):
    """
    把規則從大約兩年前開始，一天一天重新跑一遍，記住之前建議過什麼：
    - 防大跌：哪天開始防守（建議賣出）、哪天解除（站回 200 日線或 VIX ≥ 40）
    - 循環：哪天在 +2.5 建議賣出、哪天建議買回；防守中先不買回
    """
    dates, n = close.index, len(close)
    out = {"defend": False, "d_start": None, "d_end": None, "d_reason": None,
           "sold": False, "s_date": None, "b_date": None, "blocked": False}
    if n < 60:
        return out
    c, w = close.to_numpy(), wr.to_numpy()
    hi = close.rolling(252, min_periods=1).max().to_numpy()
    # 大盤與 200 日線、VIX 對齊到每個交易日（台股看前一個美股交易日的 VIX）
    if idx_s is not None and rule.get("dd"):
        ma = idx_s.rolling(200).mean()
        al = lambda x: x.reindex(x.index.union(dates)).ffill().reindex(dates).to_numpy()
        ix, ixma = al(idx_s), al(ma)
    else:
        ix = ixma = None
    if vix_s is not None:
        v = vix_s.copy()
        if tw:
            v.index = v.index + pd.Timedelta(days=1)
        vx = v.reindex(v.index.union(dates)).ffill().reindex(dates).to_numpy()
    else:
        vx = np.full(n, np.nan)
    ns = dates.asi8
    span = int(yrs * 365 * 86400e9)
    for t in range(max(1, n - 520), n):
        # 防大跌
        if ix is not None and np.isfinite(ixma[t]):
            below = ix[t] < ixma[t]
            if not out["defend"] and c[t] / hi[t] - 1 <= -rule["dd"] and below:
                out.update(defend=True, d_start=dates[t])
            elif out["defend"] and (not below or vx[t] >= 40):
                out.update(defend=False, d_end=dates[t], d_reason="vix" if below else "ma")
        # 循環
        if rule.get("sell_z") is not None:
            i0 = int(np.searchsorted(ns, ns[t] - span))
            if t - i0 < 20:
                continue
            y = c[i0:t + 1]
            x = np.arange(len(y))
            b, a = np.polyfit(x, y, 1)
            sd = np.std(y - (b * x + a))
            z = (y[-1] - (b * x[-1] + a)) / sd
            buy_ok = (rule["buy_z"] is not None and z <= rule["buy_z"]) or w[t] <= rule["wr"]
            if not out["sold"] and z >= rule["sell_z"]:
                out.update(sold=True, s_date=dates[t], blocked=False)
            elif out["sold"] and buy_ok:
                if out["defend"]:
                    out["blocked"] = True          # 防守中：先不買回
                else:
                    out.update(sold=False, b_date=dates[t], blocked=False)
            elif out["sold"]:
                out["blocked"] = False
    return out


# ==================== 文字 ====================
def sgn(z):
    return ("−" if z < 0 else "+") + f"{abs(z):.2f}"


def p2(v):
    return f"{v:,.0f}" if v >= 1000 else f"{v:.2f}"


def zcls(z):
    return "hi" if z >= 1 else "lo" if z <= -1 else "mid"


def rule_row(k, v, pill, pill_cls, sub=""):
    s = f'<span class="s">{sub}</span>' if sub else ""
    return (f'<li><span class="k">{k}</span><span class="v">{v}</span>'
            f'<span class="st"><span class="pill {pill_cls}">{pill}</span></span>{s}</li>')


def weekday_tw(d):
    return "一二三四五六日"[d.weekday()]


# ==================== 畫面 ====================
qp = st.query_params
default = qp.get("t", "00631L").upper()
st.markdown('<div class="masthead"><div class="title">樂活五線譜</div></div>', unsafe_allow_html=True)

def _on_pick():
    st.session_state.typed = ""


def _on_type():
    if st.session_state.typed.strip():
        st.session_state.pick = None


if "pick" not in st.session_state:
    st.session_state.pick = default if default in WATCH else None
    st.session_state.typed = "" if default in WATCH else default
c1, c2 = st.columns([4, 1.3], vertical_alignment="center")
with c1:
    st.pills("常用標的", list(WATCH), label_visibility="collapsed", key="pick", on_change=_on_pick)
with c2:
    st.text_input("其他代號", placeholder="🔍 其他代號", label_visibility="collapsed", key="typed", on_change=_on_type)
code = st.session_state.typed.strip().upper() or st.session_state.pick or "00631L"
if qp.get("t", "").upper() != code:
    st.query_params["t"] = code

with st.spinner("讀取資料中…"):
    got = load(code)
if got is None:
    st.error(f"找不到「{escape(code)}」的資料，請確認代號。")
    st.stop()
df, name, live, when, sym, live_chg = got
today = datetime.now(TW_TZ).date()

close = df["Close"]
wr = williams_r(df)
lh = lohas(df)
last, prev = float(close.iloc[-1]), float(close.iloc[-2])
chg = live_chg / 100 if live_chg is not None else last / prev - 1
d = df.index[-1]

# --- 標頭 ---
if when is not None and when.date() == today and when.hour * 60 + when.minute < 13 * 60 + 30:
    when_txt = f"盤中 {when:%H:%M}"
else:
    when_txt = "收盤"
delay = "" if live or not is_tw(code) else '<br><span class="delay">延遲報價</span>'
st.markdown(f"""
<section class="head">
  <div class="name">{escape(name)}　{escape(code)}</div>
  <div class="date num">{d.month}/{d.day}（{weekday_tw(d)}）{when_txt}{delay}</div>
  <div class="num"><span class="price">{p2(last)}</span><span class="chg {'up' if chg >= 0 else 'down'}">{'+' if chg >= 0 else '−'}{abs(chg):.2%}</span></div>
</section>""", unsafe_allow_html=True)

# --- 期間與算法（最上面的建議、圖、規則清單都跟著它） ---
PERIODS = {"0.5 年": 0.5, "1 年": 1, "3.5 年": 3.5, "10 年": 10, "自訂": None}
if "per" not in st.session_state:
    st.session_state.per = next((k for k, v in PERIODS.items() if v is not None and str(v) == qp.get("p")), "0.5 年")
    st.session_state.alg = "一般"
if not st.session_state.get("per"):
    st.session_state.per = "0.5 年"
if not st.session_state.get("alg"):
    st.session_state.alg = "一般"
yrs = PERIODS[st.session_state.per]
is_log = st.session_state.alg == "對數"
# 自訂期間：起日、迄日分成兩個日期框，各點一次就好（不用在同一個月曆點兩次）
D_MIN, D_MAX = df.index[0].date(), d.date()
if "d0" not in st.session_state:
    st.session_state.d0 = max((d - pd.DateOffset(years=1)).date(), D_MIN)
    st.session_state.d1 = D_MAX
_s, _e = st.session_state.d0, st.session_state.d1
if not _s or not _e or _e <= _s:
    _s, _e = max((d - pd.DateOffset(years=1)).date(), D_MIN), D_MAX
rng = (_s, _e)
z_now = zscore(close, yrs, today, log=is_log) if yrs else zscore(close, start=rng[0], end=rng[1], log=is_log)
per_label = f"{yrs:g} 年" if yrs else f"{rng[0]:%Y/%m/%d}～{rng[1]:%Y/%m/%d} "

rule = RULES.get(code)
w_now = float(wr.iloc[-1])
md = lambda t: f"{t.month}/{t.day}"
last_day = d.normalize()

# 大盤（防大跌用）與歷史重跑
ix_s = index_closes(rule["idx"], today) if rule and rule.get("idx") else None
ix = None
if ix_s is not None and len(ix_s) >= 210:
    _ma = ix_s.rolling(200).mean()
    ix = {"close": float(ix_s.iloc[-1]), "ma": float(_ma.iloc[-1]), "date": ix_s.index[-1].normalize(),
          "above": bool(ix_s.iloc[-1] > _ma.iloc[-1])}
stale = ix is not None and is_tw(code) and ix["date"] < last_day      # 大盤資料比股價舊 → 先不判斷
H = replay(close, wr, ix_s if not stale else None, vix_series(), rule, yrs or 0.5, is_tw(code)) if rule else None
Hc = H if yrs else None          # 自訂期間：循環不推歷史（防大跌照樣推）

cyc_pill, cyc_sub = None, ""
if z_now is None:
    v_title, v_z, v_sub = "資料不足", "", ""
else:
    v_z = f'<span class="z num {zcls(z_now)}">{sgn(z_now)}</span>'
    if rule and rule.get("sell_z") is not None:
        in_sell = z_now >= rule["sell_z"]
        in_buy = (rule["buy_z"] is not None and z_now <= rule["buy_z"]) or w_now <= rule["wr"]
        sold = Hc and Hc["sold"]
        if in_sell:
            v_title, cyc_pill = "賣點", ("賣點", "sell")
            if sold and Hc["s_date"] is not None and Hc["s_date"] < last_day:
                v_sub = f'{md(Hc["s_date"])} 已建議{rule["sell"]}，剩下的繼續抱著。'
            else:
                v_sub = f'<b class="hi">可以{rule["sell"]}</b>，剩下的繼續抱著。'
            cyc_sub = v_sub
        elif in_buy:
            v_title, cyc_pill = "買回區", ("買回區", "buy")
            if Hc is None:
                v_sub = '先前賣出的部分，<b class="lo">可以買回</b>；沒賣的，繼續抱著。'
            elif sold and Hc["defend"]:
                v_sub = '先前賣出的部分，大盤防守中，<b class="lo">建議先不買回</b>，請自行斟酌。'
            elif Hc["b_date"] is not None and Hc["b_date"] == last_day:
                v_sub = '先前賣出的部分，<b class="lo">可以買回</b>；沒賣的，繼續抱著。'
            elif Hc["b_date"] is not None and (last_day - Hc["b_date"]).days <= 31:
                v_sub = f'{md(Hc["b_date"])} 已建議買回，繼續抱著。'
            else:
                v_sub = "最近沒有賣出，不用買回，繼續抱著。"
            cyc_sub = v_sub
        else:
            v_title, cyc_pill = "持有", ("持有", "idle")
            if sold:
                v_sub = f'{md(Hc["s_date"])} 已{rule["sell"]}，還沒到買回點，剩下的繼續抱著。'
            else:
                v_sub = "繼續抱著。"
            cyc_sub = v_sub if sold else ""
    else:
        v_title, v_sub = "位階", ""

# 最上面的建議要把防大跌一起考慮：防守剛解除（5 個交易日內）就提醒買回防守賣出的部分
recent_end = bool(H and not H["defend"] and H["d_end"] is not None and len(close.loc[H["d_end"]:]) <= 5 and not stale)
# 防守中或剛解除時，多一行防大跌的建議；循環那行前面標「循環：」，兩個分開講
def_line = ""
if H and H["defend"] and not stale and rule.get("dd"):
    def_line = (f'防大跌：{md(H["d_start"])} 起防守中（已建議{rule["defend"]}），'
                f'等{rule["idx_name"]}站回 200 日線再買回。')
elif recent_end:
    why = f'{rule["idx_name"]}站回 200 日線' if H["d_reason"] == "ma" else "VIX 破 40"
    def_line = f'防大跌：{md(H["d_end"])} {why}，先前防守賣出的部分，<b class="lo">可以買回</b>。'
if def_line and v_sub:
    v_sub = f"循環：{v_sub}</p><p>{def_line}"
elif def_line:
    v_sub = def_line

others = {y: zscore(close, y, today, log=is_log) for y in (0.5, 1, 3.5, 10) if y != yrs}
cyc_html = "".join(
    f'<a href="?t={code}&p={y:g}" target="_self"><small>{y:g} 年</small><span class="{zcls(o)}">{sgn(o)} SD</span></a>'
    for y, o in others.items() if o is not None)
st.markdown(f"""
<section class="verdict">
  <p class="lab">{per_label}{"循環" if rule and rule.get("sell_z") else "五線譜"}</p>
  <p class="t">{v_title}{v_z}</p>
  {f'<p>{v_sub}</p>' if v_sub else ''}
  {f'<div class="cycles num"><span class="h">其他週期<br>（參考）</span>{cyc_html}</div>' if cyc_html else ''}
</section>""", unsafe_allow_html=True)

try:
    opts = st.container(horizontal=True, gap="small")
except TypeError:                    # 舊版 Streamlit 沒有 horizontal
    opts = st.container()
with opts:
    st.segmented_control("期間", list(PERIODS), key="per", label_visibility="collapsed")
    st.segmented_control("算法", ["一般", "對數"], key="alg", label_visibility="collapsed")
if yrs is None:
    try:                                              # 起日、迄日放同一排
        row = st.container(horizontal=True, gap="xsmall", vertical_alignment="center")
    except Exception:                                 # 舊版 Streamlit 不支援時，照一般排法
        row = st.container()
    with row:
        for lab, ky in (("起日", "d0"), ("迄日", "d1")):
            st.markdown(f'<span class="cus-lab">{lab}</span>', unsafe_allow_html=True)
            st.date_input(lab, key=ky, min_value=D_MIN, max_value=D_MAX, format="YYYY/MM/DD",
                          label_visibility="collapsed", width=136)

# --- 圖 ---
keep = df.index >= df.index[-1] - pd.Timedelta(days=3653)
rnd = lambda s, n=4: [None if not np.isfinite(v) else round(float(v), n) for v in s]
data = {
    "today": today.isoformat(), "period": f"{yrs:g}" if yrs else "c", "log": is_log,
    "d0": rng[0].isoformat() if yrs is None else None, "d1": rng[1].isoformat() if yrs is None else None,
    "dates": [x.strftime("%Y-%m-%d") for x in df.index[keep]],
    "close": rnd(close[keep]), "ma": rnd(lh["MA"][keep]), "ub": rnd(lh["UB"][keep]),
    "lb": rnd(lh["LB"][keep]), "wr": rnd(wr[keep], 1),
}
with open(os.path.join(HERE, "chart_template.html"), encoding="utf-8") as fh:
    tpl = fh.read()
chart_html = tpl.replace("__DATA__", json.dumps(data, separators=(",", ":")))
if hasattr(st, "iframe"):          # 新版 Streamlit：高度自動配合內容
    st.iframe(chart_html, height="content")
else:
    components.html(chart_html, height=430)   # 舊版只能固定高度

# --- 規則清單 ---
rows = []
if z_now is not None and rule and cyc_pill:
    sub = cyc_sub.replace("<b", "<span").replace("</b>", "</span>").rstrip("。")
    rows.append(rule_row("🔁 循環", f'目前 <span class="{zcls(z_now)}">{sgn(z_now)} SD</span>', cyc_pill[0], cyc_pill[1], sub))
if np.isfinite(w_now):
    wp = ("超賣", "buy") if w_now <= -80 else ("超買", "sell") if w_now >= -20 else ("中間", "idle")
    rows.append(rule_row("📉 威廉", f"{w_now:.0f}", *wp))
ma, ub, lb = (float(lh[c].iloc[-1]) for c in ("MA", "UB", "LB"))
if np.isfinite(ub):
    edges = f'上緣 <span class="lv sell">{p2(ub)}</span>｜下緣 <span class="lv buy">{p2(lb)}</span>'
    if last > ub:
        rows.append(rule_row("🌈 樂活", "突破上緣", "突破上緣", "sell",
                             f"股價飆漲中，若要避免錯失漲幅，跌回上緣 {p2(ub)} 以下再賣出<br>{edges}"))
    elif last < lb:
        rows.append(rule_row("🌈 樂活", "跌破下緣", "跌破下緣", "buy",
                             f"股價狂跌中，若要避免接刀，漲回下緣 {p2(lb)} 以上再買進<br>{edges}"))
    elif last < ma:
        rows.append(rule_row("🌈 樂活", "在通道內", "通道內", "idle",
                             f"回到中線 {p2(ma)} 之前，都還算相對低檔<br>{edges}"))
    else:
        rows.append(rule_row("🌈 樂活", "在通道內", "通道內", "idle", f"已回到中線 {p2(ma)} 之上<br>{edges}"))
if rule and rule.get("idx") and ix:
    nm = rule["idx_name"]
    if stale:
        rows.append(rule_row("🛡 防大跌", f"{nm}資料到 {md(ix['date'])}", "資料更新中", "idle",
                             f"{nm}還沒更新到 {md(last_day)}，先不判斷"))
    else:
        hi = float(close.iloc[-252:].max())
        dd = last / hi - 1
        gap = ix["close"] / ix["ma"] - 1
        v = f"離高點 {'−' if dd < 0 else '+'}{abs(dd):.1%}，{nm}在 200 日線{'上' if gap >= 0 else '下'} {'+' if gap >= 0 else '−'}{abs(gap):.1%}"
        if H and H["defend"] and H["d_start"] == last_day:
            rows.append(rule_row("🛡 防大跌", v, rule["defend"], "sell",
                                 f"{nm}跌破 200 日線，而且離高點跌超過 {rule['dd']:.0%}：可以{rule['defend']}"))
        elif H and H["defend"]:
            rows.append(rule_row("🛡 防大跌", v, "防守中", "sell",
                                 f"{md(H['d_start'])} 起防守中（已建議{rule['defend']}），等{nm}站回 200 日線 {p2(ix['ma'])} 再買回"))
        elif recent_end:
            why = f"{nm}站回 200 日線" if H["d_reason"] == "ma" else "VIX 破 40"
            rows.append(rule_row("🛡 防大跌", v, "可買回", "buy",
                                 f"{md(H['d_end'])} {why}，之前防守賣出的，可以買回"))
        elif not ix["above"]:
            rows.append(rule_row("🛡 防大跌", v, "不動作", "idle",
                                 f"{nm}已跌破 200 日線，但離高點還沒跌 {rule['dd']:.0%}，繼續抱著" if rule.get("dd")
                                 else f"{nm}在 200 日線之下"))
        else:
            rows.append(rule_row("🛡 防大跌", v, "不動作", "idle", f"{nm}還在 200 日線之上，繼續抱著"))
vx = vix_now()
if vx is not None:
    vp = ("極度恐慌", "buy") if vx >= 40 else ("恐慌", "idle") if vx >= 30 else ("偏緊張", "idle") if vx >= 20 else ("平靜", "idle")
    rows.append(rule_row("😨 VIX", f"{vx:.1f}", *vp, "之前防守賣出的部分，可以買回" if vx >= 40 else ""))
st.markdown(f'<ul class="rules num">{"".join(rows)}</ul><div class="foot">僅供參考，請自行判斷。</div>',
            unsafe_allow_html=True)
