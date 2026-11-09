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
    elif not tw:
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


@st.cache_data(ttl=1800, show_spinner=False)
def index_status(idx):
    """大盤指數：收盤、200 日線、最近 5 天內有沒有剛站回"""
    df, _ = fetch_history(idx, years=2)
    if df.empty or len(df) < 210:
        return None
    ma = df["Close"].rolling(200).mean()
    above = df["Close"] > ma
    return {"close": float(df["Close"].iloc[-1]), "ma": float(ma.iloc[-1]),
            "above": bool(above.iloc[-1]), "reclaimed": bool(above.iloc[-1] and not above.iloc[-6:-1].all())}


@st.cache_data(ttl=1800, show_spinner=False)
def vix_now():
    df, _ = fetch_history("^VIX", years=1)
    return None if df.empty else float(df["Close"].iloc[-1])


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
chg = live_chg / 100 if live and live_chg is not None else last / prev - 1
d = df.index[-1]

# --- 標頭 ---
if live and when is not None and when.date() == today and when.hour * 60 + when.minute < 13 * 60 + 30:
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
rng = st.session_state.get("rng")
if yrs is None and (not rng or len(rng) != 2):
    rng = (d.date() - timedelta(days=365), d.date())
z_now = zscore(close, yrs, today, log=is_log) if yrs else zscore(close, start=rng[0], end=rng[1], log=is_log)
per_label = f"{yrs:g} 年" if yrs else f"{rng[0]:%Y/%m/%d}～{rng[1]:%Y/%m/%d}"

rule = RULES.get(code)
w_now = float(wr.iloc[-1])
cyc_pill = None
if z_now is None:
    v_title, v_z, v_sub = "資料不足", "", ""
else:
    v_z = f'<span class="z num {zcls(z_now)}">{sgn(z_now)}</span>'
    if rule and rule.get("sell_z") is not None:
        if z_now >= rule["sell_z"]:
            v_title, v_sub, cyc_pill = "賣點", f'<b class="hi">可以{rule["sell"]}</b>，剩下的繼續抱著。', ("賣點", "sell")
        elif (rule["buy_z"] is not None and z_now <= rule["buy_z"]) or w_now <= rule["wr"]:
            v_title, v_sub, cyc_pill = "買回區", '之前賣出的部分，<b class="lo">可以買回</b>；沒賣的，繼續抱著。', ("買回區", "buy")
        else:
            v_title, v_sub, cyc_pill = "持有", "繼續抱著。", ("持有", "idle")
    else:
        v_title, v_sub = "位階", ""

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
    st.date_input("自訂期間", value=rng, key="rng", min_value=df.index[0].date(), max_value=d.date(),
                  format="YYYY/MM/DD", label_visibility="collapsed")

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
    sub = {"賣點": f"可以{rule['sell']}", "買回區": "之前賣出的部分，可以買回", "持有": ""}[cyc_pill[0]]
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
if rule and rule.get("idx"):
    ix = index_status(rule["idx"])
    if ix:
        hi = float(close.iloc[-252:].max())
        dd = last / hi - 1
        gap = ix["close"] / ix["ma"] - 1
        nm = rule["idx_name"]
        v = f"離高點 {'−' if dd < 0 else '+'}{abs(dd):.1%}，{nm}在 200 日線{'上' if gap >= 0 else '下'} {'+' if gap >= 0 else '−'}{abs(gap):.1%}"
        if rule.get("dd") and dd <= -rule["dd"] and not ix["above"]:
            rows.append(rule_row("🛡 防大跌", v, rule["defend"], "sell",
                                 f"{nm}跌破 200 日線，而且離高點跌超過 {rule['dd']:.0%}：可以{rule['defend']}"))
        elif ix["reclaimed"]:
            rows.append(rule_row("🛡 防大跌", v, "站回", "buy", f"{nm}剛站回 200 日線，之前防守賣出的，可以買回"))
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
