import io
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

import engine as E

try:
    import yfinance as yf
except ImportError:
    yf = None

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Swing PRO V7 - Top-Down", page_icon="🎯", layout="wide")
st.title("🎯 AI Accumulation & Breakout Scanner — V8")
st.caption("World → India → Sectors → Market-cap & quality filters → Top 5 → Reward:Risk plan → Backtest → Journal")
st.caption("Data: free, end-of-day only (Yahoo Finance via yfinance). There is no live/intraday feed here - "
          "every price you see is from the last completed daily candle, timestamped below after each scan.")
st.warning("Decision-support tool, not a profit guarantee or investment advice. Verify live prices, results dates and "
           "corporate actions with NSE / your broker before trading.")


def show_df(df, **kw):          # works on old and new Streamlit versions
    try:
        st.dataframe(df, width="stretch", **kw)
    except Exception:
        st.dataframe(df, use_container_width=True, **kw)


def show_plot(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except Exception:
        st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------------ data loaders
UA = {"User-Agent": "Mozilla/5.0"}
NSE_FILES = {"Nifty 500": "ind_nifty500list.csv", "Nifty 200": "ind_nifty200list.csv"}
ALL_EQ_URLS = ["https://nsearchives.nseindia.com/content/equity/EQUITY_L.csv",
               "https://archives.nseindia.com/content/equity/EQUITY_L.csv"]
ALL_EQ_LABEL = "All NSE equities (EQ series, wider than Nifty 500)"


@st.cache_data(ttl=86400, show_spinner=False)
def index_list(name):
    for u in (f"https://archives.nseindia.com/content/indices/{NSE_FILES[name]}",
              f"https://www.niftyindices.com/IndexConstituent/{NSE_FILES[name]}"):
        try:
            r = requests.get(u, headers=UA, timeout=15)
            r.raise_for_status()
            d = pd.read_csv(io.StringIO(r.text))
            d.columns = [c.strip() for c in d.columns]
            return d[["Symbol", "Industry"]]
        except Exception:
            continue
    try:                                                     # fallback: Wikipedia table (no sector info)
        html = requests.get("https://en.wikipedia.org/wiki/" + name.upper().replace(" ", "_"), headers=UA, timeout=20).text
        for t in pd.read_html(io.StringIO(html)):
            t.columns = [str(c).strip() for c in t.columns]
            sc = next((c for c in t.columns if c.lower() == "symbol"), None)
            if sc:
                out = pd.DataFrame({"Symbol": t[sc].astype(str).str.replace(".", "-", regex=False).str.strip()})
                out["Industry"] = "Unknown"
                return out.drop_duplicates("Symbol")
    except Exception:
        pass
    return None


@st.cache_data(ttl=86400, show_spinner=False)
def all_nse_equities():
    """Full EQ-series list (no sector data) - used with the market-cap filter to go wider than Nifty 500."""
    for u in ALL_EQ_URLS:
        try:
            r = requests.get(u, headers=UA, timeout=20)
            r.raise_for_status()
            d = pd.read_csv(io.StringIO(r.text))
            d.columns = [c.strip().upper() for c in d.columns]
            if "SERIES" in d.columns:
                d = d[d["SERIES"].astype(str).str.strip() == "EQ"]
            out = pd.DataFrame({"Symbol": d["SYMBOL"].astype(str).str.strip()}).drop_duplicates("Symbol")
            out["Industry"] = "Unknown"
            return out.reset_index(drop=True)
        except Exception:
            continue
    return None


def build_universe(mode):
    if mode == ALL_EQ_LABEL:
        return all_nse_equities()
    n500 = index_list("Nifty 500")
    if n500 is None:
        return None
    if mode == "Nifty 500":
        return n500
    n200 = index_list("Nifty 200")
    if n200 is None:
        return None
    return n200 if mode == "Nifty 200" else n500[~n500["Symbol"].isin(n200["Symbol"])]


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def market_caps(symbols):
    """Current market cap in ₹ Cr per NSE symbol, via yfinance fast_info. Best-effort: symbols Yahoo has no
    data for are simply absent from the result (treated as 'unknown', not zero)."""
    out = {}
    for s in symbols:
        try:
            fi = yf.Ticker(s + ".NS").fast_info
            mc = None
            for k in ("market_cap", "marketCap"):
                try:
                    mc = fi[k]
                except Exception:
                    mc = getattr(fi, k, None)
                if mc:
                    break
            if mc:
                out[s] = float(mc) / 1e7                     # rupees -> crore
        except Exception:
            pass
    return out


def _download(tickers, period):
    out, tickers = {}, list(tickers)
    for i in range(0, len(tickers), 50):                     # batches of 50
        chunk = tickers[i:i + 50]
        try:
            data = yf.download(chunk, period=period, interval="1d", auto_adjust=True,
                               group_by="ticker", threads=True, progress=False)
        except Exception:
            continue
        multi = isinstance(data.columns, pd.MultiIndex)
        for t in chunk:
            try:
                d = (data[t] if multi else data).dropna(how="all").reset_index()
                if len(d):
                    out[t] = d
            except KeyError:
                pass
    return out


yf_batch = st.cache_data(ttl=6 * 3600, show_spinner=False)(_download)


@st.cache_data(ttl=86400, show_spinner=False)
def fundamentals(sym):
    try:
        i = yf.Ticker(sym + ".NS").get_info()
    except Exception:
        return {}
    g = lambda k, m: i[k] * m if isinstance(i.get(k), (int, float)) else None
    return {"ROE%": g("returnOnEquity", 100), "DebtEq": g("debtToEquity", 0.01), "EPS_g%": g("earningsGrowth", 100),
            "Rev_g%": g("revenueGrowth", 100), "PE": g("trailingPE", 1)}


def fund_status(f, sector):
    ck = []
    if f.get("ROE%") is not None:
        ck.append(f["ROE%"] >= 12)
    if f.get("DebtEq") is not None and "financ" not in str(sector).lower():   # D/E meaningless for banks/NBFC
        ck.append(f["DebtEq"] <= 1.0)
    if f.get("EPS_g%") is not None:
        ck.append(f["EPS_g%"] >= 0)
    if f.get("Rev_g%") is not None:
        ck.append(f["Rev_g%"] >= 0)
    return "NA" if not ck else ("PASS" if all(ck) else "FAIL")


def fund_line(f):
    parts = [(k, f.get(k), fmt) for k, fmt in (("ROE%", "ROE {:.0f}%"), ("DebtEq", "D/E {:.2f}"),
                                                ("EPS_g%", "EPS g {:.0f}%"), ("Rev_g%", "Sales g {:.0f}%"), ("PE", "PE {:.0f}"))]
    return " · ".join(fmt.format(v) for _, v, fmt in parts if v is not None) or "no data"


# ------------------------------------------------------------------ sidebar
sb = st.sidebar
simple_mode = sb.toggle("🟢 Simple mode (sensible defaults, fewer knobs)", True,
                        help="Turn off to see and tune every filter. Either way the same engine and the same "
                             "safety gates run - this only hides sliders, it never skips a check.")
sb.header("🌐 Data & universe")
src = sb.radio("Data source", ["Auto (yfinance)", "Upload CSV"])
period = sb.selectbox("History", ["2y", "3y", "5y", "10y", "max"], index=3,
                      help="The Cup Base Breakout setup needs years of history to find a genuine multi-year base. "
                           "Use 10y or max for it; 2y/3y are fine (and faster) if you only care about the "
                           "shorter setups (Trendline, 63D Breakout, Sweep, Pullback, Weekly Accumulation).")
drop_live = sb.checkbox("Ignore today's incomplete candle", True)
uni_mode, my_syms, limit, files, idx_file = "", "", 0, None, None
if src == "Auto (yfinance)":
    uni_mode = sb.selectbox("Universe", ["Nifty 500", "Nifty 500 minus Nifty 200 (mid/small)", "Nifty 200",
                                         ALL_EQ_LABEL, "My symbols"])
    if uni_mode == "My symbols":
        my_syms = sb.text_area("NSE symbols (comma separated)", "RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, PFC, COFORGE")
    limit = sb.number_input("Scan first N stocks (0 = all)", 0, 2000, 0,
                            help="Bounds BOTH the market-cap check and the price download. Use this for the wider "
                                 "'All NSE equities' universe or the free scan will be slow.")
else:
    files = sb.file_uploader("Daily OHLCV CSVs (one per stock)", type="csv", accept_multiple_files=True)
    idx_file = sb.file_uploader("Index CSV (Date, Close) - optional", type="csv")

adv = sb.expander("⚙️ Advanced settings (optional)", expanded=not simple_mode)
with adv:
    st.markdown("**🏷️ Quality filters**")
    enforce_mcap = st.checkbox("Enforce minimum market cap (extra fetch time)", True, disabled=(src != "Auto (yfinance)"),
                               help="Replaces 'only Nifty 500' with 'any NSE stock above this size' - set the "
                                    "universe above to the wider list to actually get MORE candidates.")
    min_mcap = st.number_input("Minimum market cap ₹ Cr", 0.0, value=1000.0, step=100.0, disabled=not enforce_mcap)
    min_price = st.number_input("Minimum share price ₹ (avoids penny / frequent-circuit stocks)", 0.0, value=20.0, step=5.0)

    st.markdown("**🎛️ Trade filters**")
    entry_label = st.selectbox("Entry style", list(E.ENTRY_MODES), index=1,
                               help="Pullback entry gives the best reward:risk but may not fill. Compare in Backtest.")
    min_score = st.slider("Minimum score", 50, 95, 70)
    min_vol = st.slider("Breakout volume ×", 1.5, 5.0, 2.0, 0.5)
    max_risk = st.slider("Max stop distance %", 3.0, 15.0, 8.0, 0.5)
    min_rr = st.slider("Min reward:risk room to resistance (R)", 1.5, 5.0, 2.5, 0.5)
    min_turnover = st.slider("Min avg turnover ₹ Cr/day", 1, 100, 10)
    use_mkt = st.checkbox("Stock-level market filter (index > EMA50)", True)

    st.markdown("**🧭 Top-down rules**")
    use_regime = st.checkbox("Obey world/India regime (RISK-OFF = no trade, NEUTRAL = half risk)", True)
    only_top = st.checkbox("Trade only the strongest sectors", True)
    top_n = st.slider("Number of top sectors", 1, 6, 3)
p = dict(min_score=min_score, min_vol=min_vol, max_risk=max_risk, min_rr=min_rr, min_turnover=min_turnover,
         use_mkt=use_mkt, entry_mode=E.ENTRY_MODES[entry_label],
         min_mcap_cr=min_mcap if enforce_mcap else 0.0, min_price=min_price)

sb.header("💰 Risk")
capital = sb.number_input("Capital ₹", 10000.0, value=100000.0, step=10000.0)
risk_pct = sb.number_input("Risk per trade %", 0.25, 5.0, 1.0, 0.25)
max_pos = sb.slider("Max % of capital in one stock", 5, 50, 25)
max_trades = sb.slider("Trades to show (Top N)", 1, 10, 5)
max_sector = sb.slider("Max per sector", 1, 5, 2)
use_fund = sb.checkbox("Fundamental check on the shortlisted stocks", True, disabled=yf is None)
need_fund = sb.checkbox("Drop stocks that FAIL fundamentals", False)


# ------------------------------------------------------------------ load + scan
def load_inputs():
    sectors, raws, idx_raw = {}, {}, None
    if src == "Auto (yfinance)":
        if yf is None:
            st.error("yfinance is not installed. Run: pip install yfinance")
            return None
        if uni_mode == "My symbols":
            syms = [s.strip().upper().replace(".NS", "") for s in my_syms.replace("\n", ",").split(",") if s.strip()]
        else:
            u = build_universe(uni_mode)
            if u is None:
                st.error("Could not load the index list (NSE / Wikipedia). Choose 'My symbols' or upload CSVs.")
                return None
            syms, sectors = u["Symbol"].tolist(), dict(zip(u["Symbol"], u["Industry"]))
        if limit:
            syms = syms[:int(limit)]
        mcaps = {}
        if enforce_mcap:
            with st.spinner(f"Checking market cap for {len(syms)} stocks (cached 24h)..."):
                mcaps = market_caps(tuple(syms))
            before = len(syms)
            syms = [s for s in syms if mcaps.get(s, 0) >= min_mcap]
            st.caption(f"Market cap filter: {before} → {len(syms)} stocks at or above ₹{min_mcap:,.0f} Cr "
                       f"({before - len(mcaps)} had no market-cap data and were dropped to be safe).")
            if not syms:
                st.error("No stocks passed the market cap filter. Lower the threshold or turn it off.")
                return None
        with st.spinner(f"Downloading {len(syms)} stocks..."):
            got = yf_batch(tuple(s + ".NS" for s in syms), period)
            raws = {t[:-3]: d for t, d in got.items()}
            for t in ("^CRSLDX", "^NSEI"):                    # NIFTY 500, fallback NIFTY 50
                idx_raw = yf_batch((t,), period).get(t)
                if idx_raw is not None and len(idx_raw) > 100:
                    break
    else:
        if not files:
            st.error("Upload at least one CSV.")
            return None
        for f in files:
            raw = pd.read_csv(f)
            name = f.name.rsplit(".", 1)[0]
            raws[name] = raw
            if "Sector" in raw.columns and raw["Sector"].notna().any():
                sectors[name] = str(raw["Sector"].dropna().iloc[-1])
        idx_raw = pd.read_csv(idx_file) if idx_file else None
        mcaps = {}
    return raws, sectors, idx_raw, mcaps


def load_world():
    if src != "Auto (yfinance)" or yf is None:
        return None, {}
    names = {**E.GLOBAL, **E.INDIA}
    with st.spinner("Reading world & India markets..."):
        got = yf_batch(tuple(names.values()), "1y")
    stats = {n: E.index_stats(got[t]) for n, t in names.items() if t in got}
    stats = {n: s for n, s in stats.items() if s}
    return (E.regime(stats) if stats else None), stats


if st.button("🚀 RUN FULL SCAN", type="primary"):
    loaded = load_inputs()
    if loaded:
        raws, sectors, idx_raw, mcaps = loaded
        idx = E.prep_index(idx_raw) if idx_raw is not None else None
        feats, norm, skipped = {}, {}, []
        bar = st.progress(0.0, text="Computing indicators...")
        for i, (sym, raw) in enumerate(raws.items()):
            try:
                r = E.normalize(raw)
                r = E.drop_live_candle(r) if drop_live else r
                if len(r) < 220:
                    skipped.append(f"{sym}: only {len(r)} rows (need 220+)")
                else:
                    norm[sym] = r
                    f = E.features(r, idx).tail(260).reset_index(drop=True)
                    feats[sym] = E.with_mcap(f, mcaps[sym]) if sym in mcaps else f
            except Exception as e:
                skipped.append(f"{sym}: {e}")
            bar.progress((i + 1) / max(1, len(raws)))
        bar.empty()
        reg, wstats = load_world()
        last_dates = [f["Date"].iloc[-1] for f in feats.values() if "Date" in f.columns and len(f)]
        data_asof = max(last_dates).strftime("%d-%b-%Y") if last_dates else "unknown (no Date column in data)"
        st.session_state["S"] = dict(feats=feats, norm=norm, idx=idx, sectors=sectors, skipped=skipped,
                                     regime=reg, wstats=wstats, sec=E.sector_strength(feats, sectors), mcaps=mcaps,
                                     data_asof=data_asof, scanned_at=datetime.now().strftime("%d-%b-%Y %H:%M"))
        st.session_state.pop("BT", None)

S = st.session_state.get("S")
if not S or not S["feats"]:
    st.info("Pick the data source in the sidebar, then press **RUN FULL SCAN**. After the scan, all sliders update "
            "results instantly (no re-download).")
    st.stop()

st.caption(f"📅 Price data as of **{S.get('data_asof', 'unknown')}** (last completed daily candle) · "
          f"scan run at {S.get('scanned_at', 'unknown')} local time. This is end-of-day data, not live.")

if S["skipped"]:
    with st.expander(f"⚠️ Skipped ({len(S['skipped'])})"):
        st.write("\n".join(S["skipped"]))

# ------------------------------------------------------------------ top-down decisions
R, sec = S["regime"], S["sec"]
mult = R["mult"] if (R and use_regime) else 1.0
if R:
    banner = f"World/India regime: **{R['label']}** ({R['score']}/100)"
    (st.success if R["label"] == "RISK-ON" else st.warning if R["label"] == "NEUTRAL" else st.error)(
        banner + (" — no new trades today; protecting capital is also a trade." if (use_regime and mult == 0) else
                  " — half risk, max 3 trades." if (use_regime and mult == 0.5) else ""))
else:
    st.info("World/India regime not available (CSV mode or download failed): regime rules are OFF.")
top_secs = sec["Sector"].head(top_n).tolist() if len(sec) else []

sig = {s: E.signals(d, p) for s, d in S["feats"].items()}
res = pd.DataFrame([E.summarize(s, S["sectors"].get(s, "Unknown"), d, p) for s, d in sig.items()])

cand = res[res["BUY"]].copy()
cand["Rank"] = cand["Score"] + 3 * cand["RR_room"].clip(upper=5).fillna(0) + np.where(cand["Sector"].isin(top_secs), 5, 0)
if only_top and top_secs:
    cand = cand[cand["Sector"].isin(top_secs)]
risk_eff = risk_pct * mult
trades_eff = max_trades if mult >= 1 else min(max_trades, 3)
picks, cnt = [], {}
if mult > 0:
    for _, r in cand.sort_values(["Rank", "Vol_x"], ascending=False).iterrows():
        if len(picks) >= trades_eff:
            break
        if r["Sector"] != "Unknown" and cnt.get(r["Sector"], 0) >= max_sector:
            continue
        f, fs = {}, "-"
        if use_fund and yf is not None:
            f = fundamentals(r["Symbol"])
            fs = fund_status(f, r["Sector"])
            if need_fund and fs == "FAIL":
                continue
        cnt[r["Sector"]] = cnt.get(r["Sector"], 0) + 1
        q = E.size(r["Entry"], r["Stop"], capital, risk_eff, max_pos)
        picks.append({**r.to_dict(), "Fund": fs, "FundLine": fund_line(f), "Qty": q,
                      "Invest": round(q * r["Entry"]), "RiskRs": round(q * (r["Entry"] - r["Stop"]))})

m = st.columns(4)
m[0].metric("Stocks scanned", len(res))
m[1].metric("Pass all gates", int(res["BUY"].sum()))
m[2].metric("Top sectors", ", ".join(top_secs[:3]) if top_secs else "n/a")
m[3].metric("Final picks", len(picks))

tabs = st.tabs(["🎯 Top trades", "🌍 Market & sectors", "👀 Watchlist", "📈 Chart", "🧪 Backtest", "📒 Journal"])

# ------------------------------------------------------------------ Top trades
with tabs[0]:
    if not picks:
        st.info("NO TRADE today. That is a valid decision. See the Watchlist tab for why stocks were rejected.")
    for k, pk in enumerate(picks, 1):
        with st.container(border=True):
            c = st.columns([2.4, 1, 1, 1, 1])
            c[0].markdown(f"### {k}. {pk['Symbol']} · {pk['Score']}/100")
            mc = f" · Mcap ₹{pk['McapCr']:,.0f} Cr" if pd.notna(pk.get("McapCr")) else ""
            c[0].caption(f"{pk['Sector']} · {pk['Setup']} · {pk['Candle']} · Vol {pk['Vol_x']}× · RSI {pk['RSI']} · Base {pk['Base']}{mc}")
            if pd.notna(pk.get("CupAgeYears")):
                st.caption(f"📈 Cup base: {pk['CupAgeYears']}y old neckline, {pk['CupDepthPct']:.0f}% deep, "
                          f"{pk['CupVolX']}× breakout-month volume. Classic video-method targets (not the "
                          f"official T1/T2 above): +10% ₹{pk['Classic_T1_10pct']:,.0f} · +20% ₹{pk['Classic_T2_20pct']:,.0f} "
                          f"· +30% ₹{pk['Classic_T3_30pct']:,.0f} - unverified rule, treat as a reference only.")
            c[1].metric("Entry", f"₹{pk['Entry']:,.2f}")
            c[2].metric("Stop", f"₹{pk['Stop']:,.2f}", f"-{pk['Risk%']}%")
            c[3].metric("Target 1 (2R)", f"₹{pk['T1_2R']:,.2f}")
            c[4].metric("Room to resistance", f"{pk['RR_room']}R",
                        f"₹{pk['Resistance']:,.2f}" if pd.notna(pk["Resistance"]) else "blue sky", delta_color="off")
            st.markdown(f"**Trigger:** {pk['Trigger']}")
            st.caption(f"Target 2 (3R) ₹{pk['T2_3R']:,.2f} · Qty **{pk['Qty']}** (₹{pk['Invest']:,}, risk ₹{pk['RiskRs']:,}) · "
                       f"Fundamentals {pk['Fund']}: {pk['FundLine']}")
    if picks:
        st.info("Exit plan: at Target 1 sell 50% and move the stop to your entry. Trail the rest below EMA20 (exit on a daily "
                "close under it). Never move a stop down. No follow-through in ~15 sessions → exit.")
    with st.expander("📖 वापरण्याची पद्धत (मराठी)"):
        st.markdown("""
1. संध्याकाळी 4 नंतर **RUN FULL SCAN** दाबा. वरचा Regime **RISK-OFF** असेल तर नवीन खरेदी नाही.
2. फक्त सर्वात मजबूत sectors मधील स्टॉक निवडले जातात. यादीतील शेवटचे 5 स्टॉक तुमचे उमेदवार.
3. प्रत्येक स्टॉकसाठी **Trigger** वाचा. रात्री ऑर्डर तयार ठेवा, सकाळी 9:30 नंतर किंमत trigger च्या वर गेली तरच खरेदी करा.
4. खरेदी होताच stop-loss ऑर्डर लावा. Qty कार्डवर दिली आहे (जोखीम फक्त 1% भांडवल).
5. Target 1 ला अर्धी विक्री, stop entry वर आणा, उरलेले EMA20 खाली बंद झाल्यावर विका.
6. दर आठवड्याला **Journal** भरा आणि **Backtest** ने खात्री करा.""")

# ------------------------------------------------------------------ Market & sectors
with tabs[1]:
    if S["wstats"]:
        st.markdown(f"**Regime score {R['score']}/100** (world {R['parts'][0]}/40 · India trend {R['parts'][1]}/45 · VIX {R['parts'][2]}/15)")
        w = pd.DataFrame(S["wstats"]).T.reset_index().rename(columns={"index": "Market"})
        show_df(w, hide_index=True)
    else:
        st.caption("World/India data needs the Auto (yfinance) source.")
    st.markdown("**Sector strength** (from the stocks in your universe: 60D/20D median return + % above EMA50)")
    if len(sec):
        s2 = sec.copy()
        s2.insert(0, "Top", np.where(s2["Sector"].isin(top_secs), "✅", ""))
        show_df(s2, hide_index=True)
    else:
        st.caption("Sector data needs the NSE list universe (not 'My symbols' / CSV without a Sector column).")

# ------------------------------------------------------------------ Watchlist
with tabs[2]:
    w = res[~res["BUY"] & (res["Score"] >= p["min_score"] - 15)].sort_values("Score", ascending=False)
    show_df(w[["Symbol", "Sector", "McapCr", "Score", "Setup", "Vol_x", "Risk%", "RR_room", "Why_not_BUY"]].head(50), hide_index=True)
    allr = res.sort_values(["BUY", "Score"], ascending=False)
    st.download_button("⬇️ Download full scan (CSV)", allr.to_csv(index=False).encode(), "swing_v6_scan.csv", "text/csv")

# ------------------------------------------------------------------ Chart
with tabs[3]:
    sym = st.selectbox("Stock", res.sort_values("Score", ascending=False)["Symbol"].tolist())
    row, t = res[res["Symbol"] == sym].iloc[0], sig[sym].tail(150)
    x = t["Date"] if "Date" in t else t.index
    fig = go.Figure(go.Candlestick(x=x, open=t["Open"], high=t["High"], low=t["Low"], close=t["Close"], name="Price"))
    for col, name in (("EMA20", "EMA20"), ("EMA50", "EMA50"), ("EMA200", "EMA200")):
        fig.add_trace(go.Scatter(x=x, y=t[col], name=name, line=dict(width=1)))
    levels = [("Entry", row["Entry"], "#3b82f6"), ("Stop", row["Stop"], "#dc2626"),
              ("T1 2R", row["T1_2R"], "#16a34a"), ("T2 3R", row["T2_3R"], "#16a34a")]
    if pd.notna(row["Resistance"]):
        levels.append(("Resistance", row["Resistance"], "#f59e0b"))
    for name, y, colr in levels:
        fig.add_hline(y=y, line_dash="dot", line_color=colr, annotation_text=name)
    fig.update_layout(height=520, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
    show_plot(fig)
    st.caption(f"{sym}: {row['Setup']} · Score {row['Score']} · " + (row["Why_not_BUY"] or "all gates passed") +
               f" · levels use the '{entry_label}' plan")

# ------------------------------------------------------------------ Backtest
with tabs[4]:
    st.caption("Same rules as the sidebar. Compares all 3 entry styles × 2 exit styles so you can pick the best "
               "reward:risk on real history. Stop is checked first when stop and target hit on one bar.")
    b = st.columns(4)
    nbt = b[0].number_input("Stocks to test", 1, 500, max(1, min(100, len(S["norm"]))))
    rr = b[1].selectbox("Target", [2.0, 3.0], format_func=lambda v: f"{v:.0f}R")
    hold = b[2].slider("Max holding (days)", 5, 40, 20)
    cost = b[3].number_input("Round-trip cost %", 0.0, 2.0, 0.2, 0.05) / 100
    if st.button("▶ Run backtest (6 combinations)"):
        syms, F = list(S["norm"])[:int(nbt)], {}
        bar = st.progress(0.0, text="Preparing history...")
        for i, s in enumerate(syms):
            F[s] = E.features(S["norm"][s], S["idx"], full_tl=True)
            bar.progress((i + 1) / len(syms))
        bar.empty()
        rows, trades = [], {}
        with st.spinner("Testing entry × exit combinations..."):
            for lab, mode in E.ENTRY_MODES.items():
                fr = {s: E.signals(f, {**p, "entry_mode": mode}) for s, f in F.items()}
                for style, sname in (("fixed", "Fixed target"), ("scale", "50% at target + trail EMA20")):
                    tr = E.backtest(fr, mode, style, rr, hold, cost)
                    rows.append({"Entry": lab, "Exit": sname, **E.backtest_stats(tr)})
                    trades[(lab, sname)] = tr
        tab = pd.DataFrame(rows)
        pool = tab[tab["Trades"].fillna(0) >= 20]
        pool = pool if len(pool) else tab.dropna(subset=["Avg R"])
        best = (tab.loc[pool["Avg R"].idxmax(), "Entry"], tab.loc[pool["Avg R"].idxmax(), "Exit"]) if len(pool) else None
        st.session_state["BT"] = dict(tab=tab, trades=trades, best=best)
    BT = st.session_state.get("BT")
    if BT:
        show_df(BT["tab"], hide_index=True)
        if BT["best"]:
            bt = BT["trades"][BT["best"]]
            st.success(f"Best average R on this data (min 20 trades): **{BT['best'][0]}** + **{BT['best'][1]}**. "
                       "Set the sidebar Entry style to match, and trust it only if it is positive across 3–5 years.")
            st.line_chart(bt.sort_values("Date")["R"].cumsum().reset_index(drop=True))
            show_df(bt.groupby("Setup")["R"].agg(Trades="count", Win_pct=lambda s: round((s > 0).mean() * 100, 1),
                                                 Avg_R="mean").round(2))
        else:
            st.info("No trades with these filters. Loosen the filters or test more stocks.")
        st.warning("Limits: today's index members only (survivorship bias), free data, one trade per stock at a time, no "
                   "portfolio limits. The world/India regime and sector rules are NOT part of the backtest (no history). "
                   "Paper-trade before real money.")

# ------------------------------------------------------------------ Journal
JC = ["Date", "Symbol", "Setup", "Entry", "Stop", "T1", "T2", "Qty", "Price", "R", "Status"]
with tabs[5]:
    J = st.session_state.get("J")
    if J is None:
        J = pd.DataFrame(columns=JC)
    c1, c2, c3 = st.columns(3)
    if c1.button("➕ Save today's Top picks") and picks:
        today = datetime.now().strftime("%Y-%m-%d")
        new = pd.DataFrame([{"Date": today, "Symbol": k["Symbol"], "Setup": k["Setup"], "Entry": k["Entry"], "Stop": k["Stop"],
                             "T1": k["T1_2R"], "T2": k["T2_3R"], "Qty": k["Qty"], "Price": np.nan, "R": np.nan,
                             "Status": "PLANNED"} for k in picks])
        J = pd.concat([J, new]).drop_duplicates(["Date", "Symbol"], keep="last").reset_index(drop=True)
    if c2.button("🔄 Update prices") and len(J) and yf is not None:
        got = _download(tuple(s + ".NS" for s in J["Symbol"].unique()), "5d")
        for i, r in J.iterrows():
            d = got.get(r["Symbol"] + ".NS")
            if d is not None and len(d):
                px = float(d["Close"].iloc[-1])
                J.loc[i, "Price"] = round(px, 2)
                J.loc[i, "R"] = round((px - r["Entry"]) / (r["Entry"] - r["Stop"]), 2)
                J.loc[i, "Status"] = ("STOP HIT" if px <= r["Stop"] else "T2 HIT" if px >= r["T2"] else
                                      "T1 HIT" if px >= r["T1"] else "ABOVE ENTRY" if px >= r["Entry"] else "BELOW ENTRY")
    up = c3.file_uploader("Restore journal CSV", type="csv")
    if up is not None and st.session_state.get("J_src") != up.name:
        J = pd.read_csv(up)
        st.session_state["J_src"] = up.name
    st.session_state["J"] = J
    show_df(J, hide_index=True)
    if len(J):
        st.download_button("⬇️ Download journal (CSV)", J.to_csv(index=False).encode(), "swing_journal.csv", "text/csv")
    st.caption("Reference P/L uses the last close vs your planned levels; it does not know if your order actually filled. "
               "Streamlit Cloud storage is temporary: download the CSV every week and re-upload it next time.")

st.divider()
st.caption("Educational screening tool. Score = setup quality, NOT a win probability.")
