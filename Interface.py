import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import datetime

st.set_page_config(page_title="Oil-INR Influence Explorer", layout="wide")

st.title("Does Oil Influence the Rupee?")
st.write(
    "An analysis of crude oil price movements vs. the USD/INR exchange rate, "
    "2020–2026. The relationship is **not constant** — it depends heavily on "
    "how turbulent markets are."
)

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def load_data():
    full = pd.read_csv("clean_with_returns.csv", index_col=0, parse_dates=True)
    full["Oil_volatility"] = full["Oil_return"].rolling(20).std()
    threshold = full["Oil_volatility"].quantile(0.75)
    full["Regime"] = full["Oil_volatility"].apply(
        lambda x: "High Volatility" if x > threshold else "Normal"
    )
    full["Rolling_corr"] = full["Oil_return"].rolling(180).corr(full["INR_return"])
    full = full.dropna(subset=["Oil_volatility"])
    return full, threshold

full, threshold = load_data()

@st.cache_data(ttl=1800)
def get_live_prices():
    oil = yf.download("BZ=F", period="5d")
    oil.columns = oil.columns.get_level_values(0)
    inr = yf.download("USDINR=X", period="5d")
    inr.columns = inr.columns.get_level_values(0)
    oil_now, oil_prev = oil["Close"].iloc[-1], oil["Close"].iloc[-2]
    inr_now, inr_prev = inr["Close"].iloc[-1], inr["Close"].iloc[-2]
    return oil_now, (oil_now-oil_prev)/oil_prev, inr_now, (inr_now-inr_prev)/inr_prev

# ---------------------------------------------------------
# Live ticker strip
# ---------------------------------------------------------
try:
    oil_now, oil_chg, inr_now, inr_chg = get_live_prices()
    c1, c2 = st.columns(2)
    c1.metric("Brent Crude Oil", f"${oil_now:.2f}", f"{oil_chg*100:+.2f}%")
    c2.metric("USD/INR", f"₹{inr_now:.2f}", f"{inr_chg*100:+.2f}%")
except Exception:
    st.caption("Live prices temporarily unavailable.")

st.divider()

EVENTS = {
    "2020-03-01": "COVID Crash",
    "2022-02-24": "Russia-Ukraine War Begins",
    "2023-10-07": "Israel-Gaza Conflict",
}

# ---------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------
st.sidebar.header("Controls")
freq = st.sidebar.radio("Data frequency", ["Daily", "Weekly", "Monthly", "Yearly"])
year_range = st.sidebar.slider("Year range", 2020, 2026, (2020, 2026))

freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "ME", "Yearly": "YE"}
if freq != "Daily":
    display_df = full[["Oil", "INR", "Oil_return", "INR_return"]].resample(freq_map[freq]).last()
    display_df["Oil_return"] = display_df["Oil"].pct_change()
    display_df["INR_return"] = display_df["INR"].pct_change()
    display_df = display_df.dropna()
else:
    display_df = full[["Oil", "INR", "Oil_return", "INR_return"]]

display_df = display_df[
    (display_df.index.year >= year_range[0]) & (display_df.index.year <= year_range[1])
]

normal = full[full["Regime"] == "Normal"]
volatile = full[full["Regime"] == "High Volatility"]

# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📈 Prices", "🔗 Correlation", "🌪️ Calm vs Turbulent",
    "🔍 Look Up a Date", "🎛️ Simulator", "🤖 Models",
    "🌍 Energy Transition", "📋 Conclusion"
])

# ---------------------------------------------------------
# TAB 1: Raw prices
# ---------------------------------------------------------
with tab1:
    st.header(f"Raw Prices ({freq})")
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax2 = ax1.twinx()
    ax1.plot(display_df.index, display_df["Oil"], color="#d97706", label="Oil (USD/barrel)")
    ax2.plot(display_df.index, display_df["INR"], color="#2563eb", label="USD/INR")
    ax1.set_ylabel("Oil price (USD/barrel)", color="#d97706")
    ax2.set_ylabel("USD/INR", color="#2563eb")
    fig1.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    st.pyplot(fig1)

    with st.expander("ℹ️ What does this graph mean?"):
        st.write(
            "The orange line shows Brent crude oil's price in US dollars per barrel. "
            "The blue line shows how many rupees one US dollar buys — when it rises, "
            "the rupee is weakening. Similar-looking trends don't necessarily mean "
            "one causes the other."
        )

    st.subheader(f"Current {freq} Snapshot")
    latest_row = display_df.iloc[-1]
    s1, s2, s3 = st.columns(3)
    s1.metric(f"Latest {freq.lower()} period", str(display_df.index[-1].date()))
    s2.metric("Oil return", f"{latest_row['Oil_return']*100:+.2f}%")
    s3.metric("INR return", f"{latest_row['INR_return']*100:+.4f}%")

    st.write(f"Most recent {freq.lower()} entries:")
    st.dataframe(
        display_df.tail(10).style.format({"Oil": "{:.2f}", "INR": "{:.2f}",
                                           "Oil_return": "{:.4%}", "INR_return": "{:.4%}"}),
        height=280
    )
    csv = display_df.to_csv().encode("utf-8")
    st.download_button(f"Download full {freq.lower()} history as CSV", csv, "oil_inr_data.csv", "text/csv")
    st.caption("Only the 10 most recent entries are shown here. Use the download button for the complete dataset.")

# ---------------------------------------------------------
# TAB 2: Rolling correlation with events
# ---------------------------------------------------------
with tab2:
    st.header("How Strongly Do They Move Together?")
    fig2, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(full.index, full["Rolling_corr"], color="#1e40af")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.fill_between(full.index, full["Rolling_corr"], 0,
                     where=(full["Rolling_corr"] > 0), color="#86efac", alpha=0.4)
    ax.fill_between(full.index, full["Rolling_corr"], 0,
                     where=(full["Rolling_corr"] < 0), color="#fca5a5", alpha=0.4)
    for date_str, label in EVENTS.items():
        event_date = pd.Timestamp(date_str)
        if full.index.min() <= event_date <= full.index.max():
            ax.axvline(event_date, color="black", linestyle=":", linewidth=1)
            ax.text(event_date, ax.get_ylim()[1]*0.9, label, rotation=90,
                    fontsize=8, va="top", ha="right")
    ax.set_ylabel("Correlation (-1 to +1)")
    ax.set_xlabel("Year")
    st.pyplot(fig2)
    st.caption("Green = moving together. Red = moving opposite. Dotted lines mark major global events.")

    with st.expander("ℹ️ What does this graph mean?"):
        st.write(
            "This measures how strongly oil and INR moved together over the "
            "previous 180 trading days, recalculated daily. The line swinging "
            "between positive and negative is the core finding of this project: "
            "**oil's relationship with the rupee is not fixed** — it changes "
            "depending on the market environment, often spiking near major global shocks."
        )

# ---------------------------------------------------------
# TAB 3: Regime comparison + gauge
# ---------------------------------------------------------
with tab3:
    st.header("Calm Markets vs. Turbulent Markets")
    corr_normal, p_normal = pearsonr(normal["Oil_return"], normal["INR_return"])
    corr_volatile, p_volatile = pearsonr(volatile["Oil_return"], volatile["INR_return"])

    col1, col2 = st.columns(2)
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    bars = ax3.bar(["Calm Markets", "Turbulent Markets"], [corr_normal, corr_volatile],
                   color=["#93c5fd", "#f87171"])
    ax3.axhline(0, color="gray", linewidth=0.8)
    ax3.set_ylabel("Correlation")
    for bar, val in zip(bars, [corr_normal, corr_volatile]):
        ax3.text(bar.get_x() + bar.get_width()/2, val + 0.005 if val > 0 else val - 0.015,
                 f"{val:+.3f}", ha="center", fontweight="bold")
    col1.pyplot(fig3)

    with col2:
        st.write(f"**Calm Markets** — correlation {corr_normal:+.4f}, "
                 f"{'significant' if p_normal < 0.05 else 'not significant'} (p={p_normal:.4f}), n={len(normal)}")
        st.write(f"**Turbulent Markets** — correlation {corr_volatile:+.4f}, "
                 f"{'significant' if p_volatile < 0.05 else 'not significant'} (p={p_volatile:.4f}), n={len(volatile)}")

    st.subheader("Today's Oil Influence Meter")
    current_vol = full["Oil_volatility"].iloc[-1]
    pct_position = (full["Oil_volatility"] < current_vol).mean() * 100
    st.progress(min(int(pct_position), 100), text=f"Current oil volatility is higher than {pct_position:.0f}% of historical days")
    regime_now = "High Volatility" if current_vol > threshold else "Normal"
    st.write("🌪️ **Turbulent — oil's influence is likely elevated right now**" if regime_now == "High Volatility"
             else "☀️ **Calm — oil's influence is likely minimal right now**")

    with st.expander("ℹ️ What does this comparison mean?"):
        st.write(
            "This is the core finding of this project. During calm markets, oil "
            "price moves and rupee moves show no statistically reliable relationship. "
            "During turbulent markets, a real, statistically significant connection appears."
        )

    st.subheader("Future Tendency — Calm vs. Turbulent Trend")
    st.write("How oil's correlation with INR has trended in recent years, separately for calm and turbulent markets.")

    recent_years = full[full.index.year >= 2021]
    trend_data = {"Calm": [], "Turbulent": []}
    for year in sorted(recent_years.index.year.unique()):
        year_data = recent_years[recent_years.index.year == year]
        calm_yr = year_data[year_data["Regime"] == "Normal"]
        turb_yr = year_data[year_data["Regime"] == "High Volatility"]
        if len(calm_yr) > 15:
            c, _ = pearsonr(calm_yr["Oil_return"], calm_yr["INR_return"])
            trend_data["Calm"].append((year, c))
        if len(turb_yr) > 10:
            c, _ = pearsonr(turb_yr["Oil_return"], turb_yr["INR_return"])
            trend_data["Turbulent"].append((year, c))

    fig_trend, ax_t = plt.subplots(figsize=(9, 4))
    if trend_data["Calm"]:
        yrs, vals = zip(*trend_data["Calm"])
        ax_t.plot(yrs, vals, marker="o", color="#2563eb", label="Calm Markets")
    if trend_data["Turbulent"]:
        yrs, vals = zip(*trend_data["Turbulent"])
        ax_t.plot(yrs, vals, marker="o", color="#f87171", label="Turbulent Markets")
    ax_t.axhline(0, color="gray", linestyle="--")
    ax_t.set_ylabel("Correlation")
    ax_t.legend()
    st.pyplot(fig_trend)

    st.warning(
        "⚠️ **This is not a forecast.** Sample sizes per year are small, so year-to-year "
        "swings shouldn't be over-interpreted. Oil's influence — in both calm and turbulent "
        "markets — may weaken further as India reduces oil dependence (see Energy Transition tab), "
        "but current data cannot confirm this trend statistically."
    )

    st.subheader("Projected Future Influence (Illustrative Only)")
    st.write(
        "Extending the recent trend lines above forward using simple linear extrapolation — "
        "this is a mathematical projection of the pattern already in the data, **not a real forecast.**"
    )

    from numpy.polynomial import polynomial as P
    future_years = np.arange(2027, 2031)
    fig_proj, ax_p = plt.subplots(figsize=(9, 4))

    for regime_name, color in [("Calm", "#2563eb"), ("Turbulent", "#f87171")]:
        if len(trend_data[regime_name]) >= 2:
            yrs, vals = zip(*trend_data[regime_name])
            coeffs = np.polyfit(yrs, vals, 1)  # simple straight-line fit
            projected = np.polyval(coeffs, future_years)
            ax_p.plot(yrs, vals, marker="o", color=color, label=f"{regime_name} (observed)")
            ax_p.plot(future_years, projected, marker="o", linestyle="--", color=color, alpha=0.5,
                      label=f"{regime_name} (projected)")

    ax_p.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax_p.set_ylabel("Correlation")
    ax_p.legend()
    st.pyplot(fig_proj)

    st.error(
        "🚫 **Not a real prediction.** This dotted line is a straight-line extrapolation of a "
        "noisy, small-sample trend — it assumes the recent pattern continues unchanged, which "
        "financial relationships rarely do. Treat this only as 'if the current trend held steady, "
        "here's roughly where it would point' — not as a genuine forecast of 2027-2030 behavior."
    )

# ---------------------------------------------------------
# TAB 4: Look up a date
# ---------------------------------------------------------
with tab4:
    st.header("What Happened on a Specific Day?")
    min_date, max_date = full.index.min().date(), full.index.max().date()
    picked_date = st.date_input("Choose a date", value=datetime.date(2022, 3, 1),
                                 min_value=min_date, max_value=max_date)

    closest_idx = full.index.get_indexer([pd.Timestamp(picked_date)], method="nearest")[0]
    closest_date = full.index[closest_idx]
    row = full.loc[closest_date]

    if closest_date.date() != picked_date:
        st.caption(f"No trading data for {picked_date} — showing nearest trading day: {closest_date.date()}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Oil return", f"{row['Oil_return']*100:+.2f}%")
    c2.metric("INR return", f"{row['INR_return']*100:+.4f}%")
    c3.metric("Regime", "🌪️ Turbulent" if row["Regime"] == "High Volatility" else "☀️ Calm")
    c4.metric("Rolling correlation", f"{row['Rolling_corr']:+.3f}" if not pd.isna(row['Rolling_corr']) else "N/A")

    nearby_events = {d: l for d, l in EVENTS.items() if abs((pd.Timestamp(d) - closest_date).days) < 30}
    for d, l in nearby_events.items():
        st.info(f"📌 Nearby event: **{l}** ({d})")

    context = full.loc[closest_date - pd.Timedelta(days=15):closest_date + pd.Timedelta(days=15), ["Oil", "INR"]]
    fig4, ax4 = plt.subplots(figsize=(10, 3))
    ax4b = ax4.twinx()
    ax4.plot(context.index, context["Oil"], color="#d97706")
    ax4b.plot(context.index, context["INR"], color="#2563eb")
    ax4.axvline(closest_date, color="gray", linestyle="--")
    st.pyplot(fig4)

# ---------------------------------------------------------
# TAB 5: Simulator
# ---------------------------------------------------------
with tab5:
    st.header("Historical Simulator")
    st.write("Enter a hypothetical oil price change to see what historically tended to happen — this is **not a forecast.**")

    hypothetical_move = st.slider("Oil price change (%)", -15, 15, 5)

    lr_normal = LinearRegression().fit(normal[["Oil_return"]], normal["INR_return"])
    lr_volatile = LinearRegression().fit(volatile[["Oil_return"]], volatile["INR_return"])

    pred_normal = lr_normal.predict([[hypothetical_move/100]])[0]
    pred_volatile = lr_volatile.predict([[hypothetical_move/100]])[0]

    c1, c2 = st.columns(2)
    c1.metric("Historical INR reaction — Calm markets", f"{pred_normal*100:+.4f}%")
    c2.metric("Historical INR reaction — Turbulent markets", f"{pred_volatile*100:+.4f}%")

    st.caption(
        "These numbers reflect historical average tendencies only (based on weak, "
        "regime-specific correlations), not a prediction of what will actually happen "
        "if oil moves this much tomorrow."
    )

# ---------------------------------------------------------
# TAB 6: Model comparison
# ---------------------------------------------------------
with tab6:
    st.header("Model Comparison")
    st.write("Two modeling approaches were tested on daily oil→INR prediction, using a proper time-based train/test split.")
    st.table(pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest"],
        "R² Score": [0.0109, 0.0045],
        "MAE": [0.002993, 0.002943]
    }))
    st.success("✅ **Best model: Linear Regression** — Random Forest's added complexity overfit the weak, noisy signal rather than improving it, suggesting the true relationship is closer to linear than non-linear.")

# ---------------------------------------------------------
# TAB 7: Energy Transition & Future Outlook (illustrative, not data-derived)
# ---------------------------------------------------------
with tab7:
    st.header("🌍 India's Energy Transition & Future Outlook")
    st.caption(
        "⚠️ **This section is illustrative context, not computed from this project's dataset.** "
        "It's meant to explain *why* oil's influence on the rupee may change over time — "
        "not to calculate or predict a specific number."
    )

    st.subheader("The Basic Mechanism")
    st.markdown(""" Higher Oil Prices → India spends more USD → USD demand rises → Rupee weakens
                 This is the economic theory behind this entire project. **How strongly this chain
    operates depends on how much oil India actually needs to import.**
    """)

    st.subheader("Explore: What If India's Oil Dependency Changes?")
    dependency = st.slider(
        "Hypothetical oil import dependency (%)",
        min_value=30, max_value=90, value=80, step=5,
        help="This is an illustrative slider, not a real measured figure for any specific year."
    )

    if dependency >= 70:
        level, msg = "🔴 High dependency", (
            "At this level, oil price shocks are likely to have a **stronger** economic "
            "impact on USD demand, and by extension, the rupee — consistent with the "
            "turbulent-market findings from this analysis."
        )
    elif dependency >= 50:
        level, msg = "🟡 Moderate dependency", (
            "At this level, oil still matters, but other factors — interest rates, "
            "foreign investment flows, global risk sentiment — likely play a comparably "
            "large or larger role."
        )
    else:
        level, msg = "🟢 Low dependency", (
            "At this level, oil is unlikely to be a dominant driver of INR movements — "
            "other economic factors would likely dominate rupee behavior."
        )

    st.metric("Dependency Level", level)
    st.write(msg)

    st.subheader("Forces Currently Reducing Oil Dependency")
    st.markdown("""
    | Factor | Direction |
    |---|---|
    | 🚗 EV Adoption | Increasing |
    | 🌾 Ethanol Blending (E20 / E100) | Increasing |
    | ☀️ Solar Capacity | Increasing |
    | 🌬️ Wind Capacity | Increasing |
    | 🚆 Railway Electrification | Increasing |
    | 🔋 Battery Storage | Improving |
    """)
    st.caption("Directional trends based on publicly known Indian energy policy goals — not quantified in this project's dataset.")

    st.info(
        "**Honest takeaway:** if India's dependence on imported crude oil continues to "
        "decline, economic theory suggests the rupee's sensitivity to oil price shocks "
        "may also decrease over time. This dashboard does not calculate or forecast the "
        "size or timing of that change — it only explains the underlying mechanism, using "
        "this project's own findings (stronger oil-INR correlation during turbulent "
        "periods) as the current-day baseline."
    )

# ---------------------------------------------------------
# TAB 8: Research Conclusion
# ---------------------------------------------------------
with tab8:
    st.header("Final Findings")
    st.markdown("""
    **✔ Daily frequency:** Weak relationship (correlation ≈ 0.10), barely statistically significant.

    **✔ Weekly/Monthly/Yearly:** No improvement — weekly showed a slightly higher raw correlation
    but failed to generalize on unseen data; monthly and yearly samples were too small to test reliably.

    **✔ Model comparison:** Simple Linear Regression outperformed Random Forest, indicating the
    (weak) relationship is closer to linear than complex, and added model complexity overfits
    rather than helps with this amount of data.

    **✔ Geopolitical dependence (key finding):** Oil shows **no reliable relationship** with INR during
    calm markets, but a **real, statistically significant relationship** during high-volatility periods —
    consistent with oil and INR both reacting to shared geopolitical/risk shocks rather than oil
    directly driving the rupee.

    **✔ Future outlook:** This relationship may weaken further as India reduces oil import dependence
    through ethanol blending and EV adoption — though current data is insufficient to confirm
    this trend statistically; it remains a plausible, testable hypothesis for future analysis.
    """)

st.divider()
st.caption("Data: Yahoo Finance (USD/INR, Brent Crude), 2020–2026. This tool describes a historical pattern; it does not predict future rupee movements.")