import streamlit as st
import numpy as np
from config import RescueConfig
from environment.rescue_env import RescueEnv, ACTION_NAMES
from agents.random_agent import RandomAgent
from agents.rule_based_agent import RuleBasedAgent
from agents.dqn_agent import DoubleDQNAgent
from simulation.pygame_renderer import render_rgb
from dashboard.components import page_header, metric_strip

page_header("RESQ AI / OPERATIONS", "Live Rescue Simulation", "Pygame-powered tactical environment embedded in Streamlit.")
if "env" not in st.session_state:
    st.session_state.env = RescueEnv(RescueConfig())
if "mode" not in st.session_state:
    st.session_state.mode = "Manual"
env = st.session_state.env
mode = st.selectbox("Agent mode",["Manual","Random","Rule-Based","Double DQN"], key="mode")
left,right = st.columns([1.8,1])
with left:
    st.image(render_rgb(env), caption="Sector 07 · Pygame renderer", use_container_width=True)
    metric_strip(env.info())
with right:
    st.subheader("Mission controls")
    if mode == "Manual":
        action = st.selectbox("Action", list(enumerate(ACTION_NAMES)), format_func=lambda x:x[1])[0]
        if st.button("Execute action", type="primary", use_container_width=True):
            _,_,_,_ = env.step(action); st.rerun()
    else:
        if st.button("Execute one agent step", type="primary", use_container_width=True):
            if mode == "Random":
                agent = RandomAgent()
                action = agent.act(env.observe(),env)
            elif mode == "Rule-Based":
                agent = RuleBasedAgent()
                action = agent.act(env.observe(),env)
            else:
                agent = DoubleDQNAgent(env.observation_size)
                path = "models/best_model/resq_double_dqn.pt"
                try:
                    agent.load(path); agent.epsilon = 0
                    action = agent.act(env.observe(),env,explore=False)
                except FileNotFoundError:
                    st.error("No trained model found. Train or load a model first.")
                    action = None
            if action is not None:
                env.step(action); st.rerun()
    if st.button("Reset mission", use_container_width=True):
        st.session_state.env = RescueEnv(RescueConfig(seed=env.config.seed+1)); st.rerun()
    st.divider()
    st.markdown("**Map legend**")
    st.write("🟩 Evacuation / safe · 🟠 Fire · 🟡 Gas · ⚫ Unknown / blocked · 🔴 Critical victim · 🟢 Robot")
    st.caption("Manual mode is for interactive testing. Agent modes execute one action per click.")
if env.done:
    st.warning(f"Mission ended: {env.reason}")
