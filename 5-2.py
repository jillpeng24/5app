import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from openai import OpenAI
# import google.generativeai as genai # 保持註釋，解決部署問題

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

# 🛠️ 修正 1: 移除殘留的股票代號處理邏輯（確保不會與新函數衝突）

ai_model = st.sidebar.selectbox("AI 模型選擇", ["ChatGPT (OpenAI)"])

# 💡 API Key 安全處理 (維持不變)
api_key = None
try:
    api_key = st.secrets["external_api"]["openai_api_key"]
except (KeyError, AttributeError):
    pass

st.sidebar.markdown("### 🔑 API Key 配置")
if not api_key:
    st.sidebar.warning("⚠️ Secrets 未配置。請輸入 Key。")
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

# ==================== 🛠️ 數據下載與備援函數 (核心邏輯) ====================

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
        # 🛠️ 修正 2: 僅在嘗試 .TWO 時顯示警告
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

# ==================== 主要分析邏輯 (修正區) ====================
if analyze_button:
    if not stock_input:
        st.error("❌ 請輸入股票代號")
    elif not api_key:
        st.error("❌ 請輸入或配置 API Key")
    else:
        try:
            with st.spinner("📥 下載股票資料中..."):
                
                stock_data, stock_name, stock_symbol_actual = download_stock_data_with_fallback(stock_input, days)
                
                if stock_data.empty or stock_symbol_actual is None:
                    # 🛠️ 修正 3: 只在最終失敗時顯示一個錯誤訊息
                    st.error(f"❌ 嚴重錯誤：無法取得 {stock_input.upper()} 的資料，請檢查代號是否正確。")
                    st.stop()
                
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
                regression_data['TL-1SD'] = trend_line - 1 * sd
                regression_data['TL-2SD'] = trend_line - 2 * sd
                
            # ==================== B. 樂活通道計算 (保持不變) ====================
            with st.spinner("📊 計算樂活通道..."):
                window = 100
                regression_data['MA20W'] = regression_data['Close'].rolling(window=window, min_periods=window).mean()
                rolling_std = regression_data['Close'].rolling(window=window, min_periods=window).std()
                regression_data['UB'] = regression_data['MA20W'] + 2 * rolling_std
                regression_data['LB'] = regression_data['MA20W'] - 2 * rolling_std
                regression_data['Zone'] = np.where(regression_data['Close'] > regression_data['MA20W'], '樂活區(多頭)', '毅力區(空頭)')
            
            # ==================== C. 技術指標計算 (保持不變) ====================
            with st.spinner("🔧 計算技術指標 (RSI, MACD, KD)..."):
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
                
                regression_data['Volume_MA5'] = regression_data['Volume'].rolling(5).mean()
                regression_data['Volume_Ratio'] = regression_data['Volume'] / regression_data['Volume_MA5']
                
                regression_data['RSI_Divergence'] = detect_rsi_divergence(regression_data['Close'], regression_data['RSI'])
            
            # ==================== D. 買賣訊號判斷 (保持不變) ====================
            with st.spinner("🎯 生成買賣訊號..."):
                valid_data = regression_data.dropna(subset=['MA20W', 'UB', 'LB', 'RSI', 'K', 'D'])
                
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
                
                # ===== 賣出訊號判斷 (在 +2SD 高檔) =====
                sell_signals = []
                if sd_level >= 2:
                    if current['RSI_Divergence']:
                        sell_signals.append("⚠️ RSI 背離")
                    if current['RSI'] > 70 and current['RSI'] < previous['RSI']:
                        sell_signals.append("⚠️ RSI 從高檔回落")
                    if current['MACD_Hist'] < 0 and previous['MACD_Hist'] > 0:
                        sell_signals.append("⚠️ MACD 死亡交叉")
                    if current['Close'] < current['MA10']:
                        sell_signals.append("🚨 跌破 MA10")
                    if current['Volume_Ratio'] > 2.0 and (current['Close'] - current['Open']) / current['Open'] < 0.005:
                        sell_signals.append("⚠️ 爆量滯漲")
                    if current['K'] < current['D'] and current['K'] > 80:
                        sell_signals.append("⚠️ KD 高檔死叉")
                
                # ===== 買入訊號判斷 (回到 +1SD) =====
                buy_signals = []
                if 0.5 <= sd_level <= 1.5:
                    if slope > 0:
                        buy_signals.append("✅ 趨勢向上 (Slope > 0)")
                    if current['Close'] > current['MA20W']:
                        buy_signals.append("✅ 站上生命線")
                    if 45 <= current['RSI'] <= 55:
                        buy_signals.append("✅ RSI 中段整理")
                    if current['RSI'] > 50 and previous['RSI'] <= 50:
                        buy_signals.append("💚 RSI 突破 50")
                    if current['K'] > current['D'] and 40 <= current['K'] <= 60:
                        buy_signals.append("💚 KD 中段黃金交叉")
                    if (current['Low'] - current['Open']) / current['Open'] < -0.02 and current['Close'] > current['Open']:
                        buy_signals.append("✅ 長下影線反轉")
                
                # ===== 綜合建議 =====
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
            
            # 🚀 優化: 使用 st.columns(3) 替代 st.columns(5)，讓 metrics 在手機上排版更優雅
            col1, col2, col3 = st.columns(3)
            col1.metric("股價", f"{current_price:.2f}")
            col2.metric("五線譜", fiveline_zone)
            col3.metric("RSI(14)", f"{current['RSI']:.1f}")

            # 剩下的 metrics 放在第二排，確保手機上的空間足夠
            col4, col5 = st.columns(2)
            col4.metric("KD", f"K:{current['K']:.1f} D:{current['D']:.1f}")
            col5.metric("Slope", f"{slope:.4f}", delta="上升" if slope > 0 else "下降")
            
            st.divider()
            st.markdown(f"### {action}")
            st.info(action_detail)
            
            if sell_signals:
                st.warning("**賣出理由：**\n" + "\n".join([f"- {s}" for s in sell_signals]))
            
            if buy_signals:
                st.success("**買入理由：**\n" + "\n".join([f"- {s}" for s in buy_signals]))
            
            # ==================== 圖表分頁 (保持不變) ====================
            tab1, tab2, tab3 = st.tabs(["🎼 五線譜", "🌈 樂活通道", "📊 技術指標"])
            
            with tab1:
                st.markdown(f"趨勢斜率: **{slope:.4f} ({slope_dir})**")
                # 圖表配色採用柔和大地色系
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
                st.markdown("### 📊 技術指標分析")
                
                valid_data['MA5'] = valid_data['Close'].rolling(5).mean()
                valid_data['MA10'] = valid_data['Close'].rolling(10).mean()
                valid_data['MA20'] = valid_data['Close'].rolling(20).mean()
                valid_data['MA60'] = valid_data['Close'].rolling(60).mean()
                
                fig_ma = go.Figure()
                fig_ma.add_trace(go.Scatter(x=valid_data.index, y=valid_data['Close'], mode='lines', name='股價', line=dict(color='#4A4A4A', width=2)))
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
            
            # ==================== AI 分析 (保持不變) ====================
            st.subheader("🤖 AI 深度分析")
            
            prompt = f"""
你是專業股票分析師。請分析 {stock_name} ({stock_symbol_actual})：

【技術狀態】
- 股價：{current_price:.2f}
- 五線譜位置：{sd_level:.2f}SD ({fiveline_zone})
- Slope：{slope:.4f}
- RSI：{current['RSI']:.1f}
- KD：K={current['K']:.1f}, D={current['D']:.1f}
- 樂活通道：{"站上生命線" if current_price > current_ma20w else "跌破生命線"}

【訊號】
賣出訊號：{', '.join(sell_signals) if sell_signals else '無'}
買入訊號：{', '.join(buy_signals) if buy_signals else '無'}

請提供：
1. 趨勢判斷
2. 操作建議
3. 風險提示
"""
            
            with st.spinner("🧠 AI 分析中..."):
                try:
                    client = OpenAI(api_key=api_key)
                    response = client.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "system", "content": "你是專業股市分析師。"}, {"role": "user", "content": prompt}],
                        temperature=0.7
                    )
                    ai_response = response.choices[0].message.content
                    st.markdown(ai_response)
                except Exception as e:
                    st.error(f"❌ AI 分析失敗：{str(e)}。請檢查 API Key 是否正確。")

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
    
    **智能訊號**
    - ✅ RSI 背離偵測
    - ✅ MACD 動能判斷
    - ✅ KD 黃金/死亡交叉
    - ✅ 量價背離分析
    - ✅ 趨勢反轉訊號
    
    **AI 分析**
    - 整合所有指標給出操作建議
    """)
