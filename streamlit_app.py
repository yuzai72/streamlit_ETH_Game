- 加入自動播放（每 5 秒前進一根 K 線）的功能（checkbox 控制）
"""
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
import random

st.set_page_config(page_title="AI vs Human 交易遊戲", layout="wide")

# ---------- 技術指標 ----------
def add_kdj(df, n=9, k_period=3, d_period=3):
    low_min = df['Low'].rolling(n, min_periods=1).min()
    high_max = df['High'].rolling(n, min_periods=1).max()
    rsv = (df['Close'] - low_min) / (high_max - low_min + 1e-9) * 100
    df['K'] = rsv.ewm(span=k_period, adjust=False).mean()
    df['D'] = df['K'].ewm(span=d_period, adjust=False).mean()
    df['J'] = 3*df['K'] - 2*df['D']
    return df

def add_rsi(df, period=6):
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=1).mean()
    avg_loss = loss.rolling(period, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['RSI'] = 100 - 100/(1+rs)
    return df

def add_macd(df):
    df['EMA_short'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_long'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_short'] - df['EMA_long']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    return df

# ---------- 抓資料 ----------
@st.cache_data
def fetch_data(ticker="ETH-USD", period="60d", interval="5m"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = add_kdj(df)
    df = add_rsi(df)
    df = add_macd(df)
    df.dropna(inplace=True)
    return df

# ---------- 訓練 AI ----------
def train_ai(train_df):
    df = train_df.copy()
    if df.empty:
        return None, []
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(subset=['target'], inplace=True)
    feats = ["K","D","J","RSI","MACD","MACD_signal","MACD_hist"]
    for f in feats:
        if f not in df.columns:
            return None, []
    X, y = df[feats], df['target']
    model = RandomForestClassifier(n_estimators=80, random_state=42)
    model.fit(X, y)
    return model, feats

# ---------- 初始化 / 重置 ----------
def init_game():
    full_df = fetch_data()
    if full_df.empty or len(full_df) < 30:
        st.error("抓取到的資料不足（至少需要 30 筆完整資料）。請檢查網路或 ticker 設定。")
        st.stop()
    max_start = len(full_df) - 30
    if max_start < 0:
        st.error("資料不足以切分為遊戲區段。")
        st.stop()
    game_start_index = random.randint(0, max_start)
    st.session_state.game_df = full_df.iloc[game_start_index:game_start_index+30].copy()
    train_df = pd.concat([full_df.iloc[:game_start_index], full_df.iloc[game_start_index+30:]])
    model, feats = train_ai(train_df)
    if model is None or not feats:
        st.error("訓練 AI 失敗（特徵不足或訓練資料不完整）。")
        st.stop()
    st.session_state.model = model
    st.session_state.feats = feats
    st.session_state.human_cash = 10000.0
    st.session_state.human_asset = 0.0
    st.session_state.ai_cash = 10000.0
    st.session_state.ai_asset = 0.0
    st.session_state.step = 20
    st.session_state.game_started = False
    st.session_state.autoplay = False

# 重置按鈕
if st.button("重置遊戲 (重新抓資料並重新訓練)"):
    keys = list(st.session_state.keys())
    for k in keys:
        del st.session_state[k]
    init_game()

if "game_df" not in st.session_state:
    init_game()

st.title("💹 AI vs Human 交易遊戲 (ETH-USD)")

# 啟動遊戲（手動）
if not st.session_state.get("game_started", False):
    st.write("按下「立即開始遊戲 ⏩」以開始")
    if st.button("立即開始遊戲 ⏩"):
        st.session_state.game_started = True
    else:
        st.stop()
else:
    st.write("遊戲開始！")

# 顯示 K 線
step = int(st.session_state.get("step", 20))
game_df = st.session_state.get("game_df")
if game_df is None or game_df.empty:
    st.error("遊戲資料不存在或為空。請重置遊戲。")
    st.stop()
step = max(0, min(step, len(game_df)-1))
st.session_state.step = step
sub_df = game_df.iloc[:step+1]
fig = go.Figure(data=[go.Candlestick(
    x=sub_df.index,
    open=sub_df['Open'], high=sub_df['High'],
    low=sub_df['Low'], close=sub_df['Close']
)])
st.plotly_chart(fig, use_container_width=True)

# 玩家操作（買/賣/下一步/上一步）與自動播放開關
col1, col2, col3 = st.columns([1,1,1])
with col1:
    if st.button("買入 🟢"):
        current_price = game_df.iloc[step]['Close']
        if st.session_state.human_cash > 0 and current_price > 0:
            st.session_state.human_asset += st.session_state.human_cash / current_price
            st.session_state.human_cash = 0.0
with col2:
    if st.button("賣出 🔴"):
        current_price = game_df.iloc[step]['Close']
        if st.session_state.human_asset > 0 and current_price > 0:
            st.session_state.human_cash += st.session_state.human_asset * current_price
            st.session_state.human_asset = 0.0
with col3:
    next_step = st.button("下一步 ▶️")
    prev_step = st.button("上一步 ◀️")
    autoplay_checked = st.checkbox("自動播放（每 5 秒）", value=st.session_state.get("autoplay", False))
    st.session_state.autoplay = bool(autoplay_checked)

# AI 操作
current_price = game_df.iloc[step]['Close']
feats = st.session_state.get("feats", [])
model = st.session_state.get("model", None)
if model is not None and feats:
    try:
        ai_feats = game_df[feats].iloc[step].to_frame().T
    except Exception:
        ai_feats = pd.DataFrame()
    if ai_feats.empty or ai_feats.isnull().any(axis=None):
        ai_pred = 0
    else:
        try:
            ai_pred = int(model.predict(ai_feats)[0])
        except Exception:
            ai_pred = 0
    if ai_pred == 1 and st.session_state.ai_cash > 0 and current_price > 0:
        st.session_state.ai_asset += st.session_state.ai_cash / current_price
        st.session_state.ai_cash = 0.0
    elif ai_pred == 0 and st.session_state.ai_asset > 0 and current_price > 0:
        st.session_state.ai_cash += st.session_state.ai_asset * current_price
        st.session_state.ai_asset = 0.0

# 步數更新邏輯（手動按鈕）
if 'next_step' in locals() and next_step:
    if st.session_state.step < len(game_df)-1:
        st.session_state.step += 1
        st.experimental_rerun()
elif 'prev_step' in locals() and prev_step:
    if st.session_state.step > 0:
        st.session_state.step -= 1
        st.experimental_rerun()

# 自動播放：每 5 秒前進一格（會阻塞該 worker）
if st.session_state.get("autoplay", False):
    if st.session_state.step < len(game_df) - 1:
        import time
        time.sleep(5)
        st.session_state.step += 1
        st.experimental_rerun()

# 如果已到最後一步，顯示結果
if st.session_state.get("step", 0) >= len(game_df)-1:
    human_value = st.session_state.human_cash + st.session_state.human_asset * current_price
    ai_value = st.session_state.ai_cash + st.session_state.ai_asset * current_price
    winner = "人類" if human_value > ai_value else ("AI" if ai_value > human_value else "平手")
    st.success(f"遊戲已到最後一步。人類資產: {human_value:.2f}, AI資產: {ai_value:.2f}, 贏家: {winner}")

# 顯示資產
st.write(f"人類現金: {st.session_state.human_cash:.2f} | 人類持有數量: {st.session_state.human_asset:.6f}")
st.write(f"AI現金: {st.session_state.ai_cash:.2f} | AI持有數量: {st.session_state.ai_asset:.6f}")
