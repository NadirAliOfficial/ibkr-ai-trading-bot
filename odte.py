import json, tkinter as tk
from tkinter import ttk, messagebox

CONFIG_FILE = 'strategy_config.json'
# load or default
try:
    with open(CONFIG_FILE) as f: cfg = json.load(f)
except:
    cfg = {}

root = tk.Tk()
root.title("0DTE Strategy Config")
frm = ttk.Frame(root, padding=20); frm.pack()

# helper to save
def save():
    with open(CONFIG_FILE,'w') as f: json.dump(cfg,f,indent=2)
    messagebox.showinfo("Saved","Config saved."); root.destroy()

# Allocation
ttk.Label(frm,text="Allocation %").grid(row=0,column=0,sticky="w")
alloc = tk.DoubleVar(value=cfg.get("allocation_per_trade",0.05)*100)
ttk.Scale(frm,from_=1,to=20,variable=alloc,orient="horizontal").grid(row=0,column=1)

# VWAP window
ttk.Label(frm,text="VWAP Window").grid(row=1,column=0,sticky="w")
vwap = tk.IntVar(value=cfg.get("vwap_window",20))
ttk.Spinbox(frm,from_=5,to=60,textvariable=vwap,width=5).grid(row=1,column=1)

# RSI window
ttk.Label(frm,text="RSI Window").grid(row=2,column=0,sticky="w")
rsi = tk.IntVar(value=cfg.get("rsi_window",14))
ttk.Spinbox(frm,from_=5,to=30,textvariable=rsi,width=5).grid(row=2,column=1)

# Profit tiers
ttk.Label(frm,text="TP Tiers (%)").grid(row=3,column=0,sticky="w")
tiers = tk.StringVar(value=",".join(str(int(t*100)) for t in cfg.get("tp_tiers",[25,75,150])))
ttk.Entry(frm,textvariable=tiers).grid(row=3,column=1)

# Stop-loss %
ttk.Label(frm,text="Stop-Loss %").grid(row=4,column=0,sticky="w")
sl = tk.DoubleVar(value=cfg.get("sl_pct",0.20)*100)
ttk.Scale(frm,from_=5,to=50,variable=sl,orient="horizontal").grid(row=4,column=1)

# Black-Scholes toggle
bs = tk.BooleanVar(value=cfg.get("use_black_scholes",True))
ttk.Checkbutton(frm,text="Use Black-Scholes",variable=bs).grid(row=5,column=0,columnspan=2)

# Save button
ttk.Button(frm,text="Save & Close",command=lambda: [
    cfg.update({
      "allocation_per_trade": alloc.get()/100,
      "vwap_window":          vwap.get(),
      "rsi_window":           rsi.get(),
      "tp_tiers":             [int(x)/100 for x in tiers.get().split(",")],
      "sl_pct":               sl.get()/100,
      "use_black_scholes":    bs.get()
    }),
    save()
]).grid(row=6,column=0,columnspan=2,pady=10)

root.mainloop()
