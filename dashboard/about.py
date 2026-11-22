import streamlit as st
from dashboard.components import page_header

page_header("RESQ AI / SYSTEM", "Project & Research", "Intelligent Autonomous Rescue Agent Using Reinforcement Learning.")
st.markdown("## Project overview")
st.write("RESQ AI is a simulated earthquake-response environment in which an autonomous agent learns to locate, prioritize, collect, and evacuate victims under resource constraints and changing hazards.")
st.markdown("## Research formulation")
st.latex(r"MDP = (S, A, P, R, \\gamma)")
st.write("The agent receives a compact partially observable numerical observation and chooses among seven discrete actions. Double DQN estimates action values and improves through replay-based temporal-difference learning.")
st.markdown("## Agent comparison")
st.write("- Random Agent: stochastic baseline.")
st.write("- Rule-Based Agent: priority and pathfinding baseline.")
st.write("- Double DQN: experience-driven learning agent.")
st.markdown("## Technology")
st.write("Python · Pygame · Streamlit · PyTorch · NumPy · Pandas · Plotly")
st.warning("This is a simulation for academic research and demonstration, not a real-world emergency response system.")
