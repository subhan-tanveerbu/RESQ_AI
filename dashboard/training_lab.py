import streamlit as st
import pandas as pd
import plotly.express as px
import subprocess, sys, os
from pathlib import Path
from dashboard.components import page_header

ROOT = Path(__file__).resolve().parents[1]
page_header("RESQ AI / INTELLIGENCE", "RL Training Lab", "Train Double DQN in a separate process and inspect saved learning history.")
episodes = st.number_input("Training episodes", min_value=10, max_value=20000, value=500, step=100)
seed = st.number_input("Training seed", min_value=0, value=7)
if st.button("Start training", type="primary"):
    log = ROOT/"data/training_logs/training.csv"
    log.parent.mkdir(parents=True, exist_ok=True)
    out = ROOT/"data/training_logs/process.log"
    with out.open("a", encoding="utf-8") as f:
        proc = subprocess.Popen([sys.executable, str(ROOT/"training/train.py"), "--episodes", str(int(episodes)), "--seed", str(int(seed))],
                                cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    st.session_state.training_pid = proc.pid
    st.success(f"Training process started (PID {proc.pid}). It runs independently of the dashboard.")
log_path = ROOT/"data/training_logs/training.csv"
if log_path.exists():
    df = pd.read_csv(log_path)
    if not df.empty:
        st.subheader("Training history")
        df["moving_average"] = df["reward"].rolling(25, min_periods=1).mean()
        st.plotly_chart(px.line(df, x="episode", y=["reward","moving_average"], title="Episode reward and 25-episode moving average"), use_container_width=True)
        c1,c2,c3 = st.columns(3)
        c1.metric("Episodes logged",len(df))
        c2.metric("Best reward",round(df.reward.max(),2))
        c3.metric("Latest epsilon",round(df.epsilon.iloc[-1],3))
        st.plotly_chart(px.line(df,x="episode",y="rescued",title="Victims rescued per episode"),use_container_width=True)
        st.dataframe(df.tail(15),use_container_width=True)
else:
    st.info("No training history yet. Start a training run to generate real learning data.")
logfile = ROOT/"data/training_logs/process.log"
if logfile.exists():
    with st.expander("Training process output"):
        st.text(logfile.read_text(encoding="utf-8",errors="ignore")[-6000:])
st.caption("Training runs in a separate Python process. Refresh this page to update the charts. Closing the browser does not necessarily stop the worker.")
