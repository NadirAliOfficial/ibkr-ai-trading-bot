import streamlit as st
import json

# Load or create config
CONFIG_FILE = 'config.json'
with open(CONFIG_FILE) as f:
    cfg = json.load(f)

st.title("0DTE Strategy Config")

# Allocation slider
cfg['allocation_per_trade'] = st.slider("Allocation per Trade", 0.01, 0.2, cfg['allocation_per_trade'], 0.01)
# VWAP window
cfg['vwap_window'] = st.number_input("VWAP Window", 5, 60, cfg['vwap_window'])
# RSI window
cfg['rsi_window'] = st.number_input("RSI Window", 5, 30, cfg['rsi_window'])
# Profit tiers
options = [0.25,0.5,0.75,1.0,1.5,2.0]
cfg['tp_tiers'] = st.multiselect("Profit Tiers", options, cfg['tp_tiers'])
# Stop-loss
cfg['sl_pct'] = st.slider("Stop-Loss %", 0.05, 0.5, cfg['sl_pct'], 0.05)
# Black-Scholes toggle
cfg['use_black_scholes'] = st.checkbox("Use Black-Scholes Pricing", cfg['use_black_scholes'])

if st.button("Save & Apply"):
    with open(CONFIG_FILE,'w') as f:
        json.dump(cfg, f, indent=2)
    st.success("Configuration saved! Restart bot to apply.")

st.markdown("---")
st.subheader("Current Settings")
st.json(cfg)
