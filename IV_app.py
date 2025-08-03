import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from datetime import timedelta
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import griddata
import plotly.graph_objects as go
import streamlit as st

### ------- Get the Call and Put Option from yfinance --------

@st.cache_data(ttl=300)   # cache for 5 minutes
def get_option_data(ticker_symbol, option_type='call'):
    """
    Returns call or put options data for the ticker.
    option_type: 'call' (default) or 'put'
    """
    ticker = yf.Ticker(ticker_symbol)
    today = pd.Timestamp.today().normalize()
    option_data = []

    for exp in ticker.options:
        exp_date = pd.Timestamp(exp)
        if exp_date > today + timedelta(days=10):

            chain = ticker.option_chain(exp)
            
            if option_type == 'call':
                options = chain.calls
            else:
                options = chain.puts

            options = options[(options['bid'] > 0) & (options['ask'] > 0)].copy()
            
            options = options[options['openInterest'] > 200]
            
            if not options.empty:
                options['mid'] = (options['bid'] + options['ask']) / 2
                options['yf_Iv'] = options.get('impliedVolatility')
                options['oi'] = options.get('openInterest')
                options['optionType'] = option_type
                options['expiryDate'] = exp_date
                option_data.append(options)
    
    if option_data:
        df = pd.concat(option_data, ignore_index=True)

        df['daysToExpiry'] = (df['expiryDate'] - today).dt.days
        df['timeToExpiry'] = df['daysToExpiry'] / 365.0

        return df[['optionType', 'strike', 'bid', 'ask', 'mid', 'yf_Iv', 
                   'oi', 'expiryDate', 'daysToExpiry', 'timeToExpiry']]
    else:
        print(f"No options found for '{ticker_symbol}' matching the criteria.")
        return pd.DataFrame(columns=['optionType', 'strike', 'bid', 'ask', 'mid', 'yf_Iv', 
                   'oi', 'expiryDate', 'daysToExpiry', 'timeToExpiry'])
    
### ---- Get the spot price ----

@st.cache_data(ttl=300)   # cache for 5 minutes
def get_spot_price(symbol):
    try:
        hist = yf.Ticker(symbol).history(period='1d')
        if hist.empty:
            print(f"Error: No historical data found for ticker '{symbol}'.")
            return None
        else:
            return hist['Close'].iloc[-1]
    except Exception as e:
        print(f"An error occurred while fetching data for ticker '{symbol}': {e}")
        return None
    
### ---- Black-Scholes price for a European option ----

def bs_option_price(S, K, T, r, sigma, q=0, option_type="call"):
    """
    Computes the Black-Scholes price for a European call or put option.

    Parameters:
    S           : float : Spot price of the underlying asset
    K           : float : Strike price
    T           : float : Time to expiration (years)
    r           : float : Risk-free interest rate (as decimal, e.g., 0.05)
    sigma       : float : Volatility (annualized standard deviation)
    q           : float : Continuous dividend yield (as decimal, optional, default 0)
    option_type : str   : 'call' (default) or 'put'

    Returns:
    price : float : Theoretical fair value of the specified option type
    """
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return price

### --- Implied volatility Function ---
def implied_volatility(market_price, S, K, T, r, q=0, option_type='call'):
    """
    Implied volatility using Black-Scholes, for either a call or put.
    """
    if T <= 0 or market_price <= 0 or S <= 0 or K <= 0:
        return np.nan
    def objective_function(sigma):
        return bs_option_price(S, K, T, r, sigma, q, option_type) - market_price
    try:
        implied_vol = brentq(objective_function, 1e-6, 5)
    except (ValueError, RuntimeError):
        implied_vol = np.nan
    return implied_vol

### --- Calculated Implied volatility ---

def calculate_implied_vols(df, spot, r=0.04, q=0.0):
    """
    Adds 'bs_Iv' column with Black-Scholes implied volatility.
    Expects columns: 'mid', 'strike', 'timeToExpiry', 'optionType'.
    """
    df = df.copy()

    def get_iv(row):
        return implied_volatility( market_price=row['mid'],
            S=spot,K=row['strike'],
            T=row['timeToExpiry'],
            r=r,q=q,
            option_type=row.get('optionType', 'call'))

    df['bs_Iv'] = df.apply(get_iv, axis=1)
    return df.dropna(subset=['bs_Iv'])

### --- 3d plot for ImpliedVolatility for Call or Put option ---

def plot_iv_surface(df, ticker_symbol, spot_price, spot_line=False, strike_limits=None):
    """
    Plots a 3D implied volatility surface from a DataFrame of options.

    Parameters:
        df            : DataFrame with columns 'timeToExpiration', 'strike', 'bs_Iv', 'optionType'
        ticker_symbol : e.g. 'AAPL'
        spot_price    : float, the current spot price of the underlying
        spot_line     : bool, default False. Draw a vertical line at spot price if True.
        strike_limits : tuple or None, (min_strike, max_strike) to restrict plotted strikes

    All IVs are decimals (e.g., 0.25 means 25%, not percent).
    """
    # LIMIT STRIKES IF REQUESTED
    if strike_limits is not None:
        strike_min, strike_max = strike_limits
        df = df[(df['strike'] >= strike_min) & (df['strike'] <= strike_max)]

    # Extract axes
    X = df['timeToExpiry'].values
    Y = df['strike'].values
    Z = df['bs_Iv'].values

    opt_type = df['optionType'].iloc[0].capitalize() if 'optionType' in df.columns else 'Option'

    # Surface grid
    ti = np.linspace(X.min(), X.max(), 50)
    ki = np.linspace(Y.min(), Y.max(), 50)
    T, K = np.meshgrid(ti, ki)
    Zi = griddata((X, Y), Z, (T, K), method='linear')

    fig = go.Figure(data=[go.Surface(
        x=T, y=K, z=Zi,
        colorscale='Viridis',
        colorbar_title='Implied Volatility'
    )])

    # Spot price vertical line at T=0 and strike=spot_price
    if spot_line:
        z_range = [np.nanmin(Z), np.nanmax(Z)]
        fig.add_trace(go.Scatter3d(
            x=[0, 0],
            y=[spot_price, spot_price],
            z=z_range,
            mode='lines',
            line=dict(color='red', width=2),
            name=f'Spot Price at T=0 ({spot_price:.2f})'
        ))

    # Axis range fix on y ("strike") if you filtered
    scene_dict = dict(
        xaxis_title='Time to Expiration (years)',
        yaxis_title='Strike',
        zaxis_title='Implied Volatility'
    )
    if strike_limits is not None:
        scene_dict['yaxis'] = dict(range=[strike_min, strike_max])

    fig.update_layout(
        title=f'Implied Volatility Surface for {ticker_symbol} {opt_type}s Option',
        scene=scene_dict,
        width=660, height=660,
        margin=dict(l=65, r=65, b=65, t=90)
    )
    return fig

### --- 3d plot for ImpliedVolatility for Call and Put options both ---

def plot_iv_surface_both(calls_df, puts_df, ticker_symbol, spot_price, spot_line=False, strike_limits=None):
    """
    Plots 3D implied volatility surfaces for both calls and puts on the same figure,
    with the Call colorbar on the right and Put colorbar on the left.
    """
    if strike_limits is not None:
        strike_min, strike_max = strike_limits
        calls_df = calls_df[(calls_df['strike'] >= strike_min) & (calls_df['strike'] <= strike_max)]
        puts_df  = puts_df[(puts_df['strike'] >= strike_min) & (puts_df['strike'] <= strike_max)]

    Xc, Yc, Zc = calls_df['timeToExpiry'].values, calls_df['strike'].values, calls_df['bs_Iv'].values
    if len(Xc) > 0 and len(Yc) > 0:
        ti = np.linspace(Xc.min(), Xc.max(), 50)
        ki = np.linspace(Yc.min(), Yc.max(), 50)
        Tc, Kc = np.meshgrid(ti, ki)
        Zci = griddata((Xc, Yc), Zc, (Tc, Kc), method='linear')
    else:
        Tc = Kc = Zci = None

    Xp, Yp, Zp = puts_df['timeToExpiry'].values, puts_df['strike'].values, puts_df['bs_Iv'].values
    if len(Xp) > 0 and len(Yp) > 0:
        ti = np.linspace(Xp.min(), Xp.max(), 50)
        ki = np.linspace(Yp.min(), Yp.max(), 50)
        Tp, Kp = np.meshgrid(ti, ki)
        Zpi = griddata((Xp, Yp), Zp, (Tp, Kp), method='linear')
    else:
        Tp = Kp = Zpi = None

    fig = go.Figure()

    # Call surface: colorbar on the RIGHT
    if Tc is not None and Kc is not None:
        fig.add_trace(go.Surface(
            x=Tc, y=Kc, z=Zci,
            # colorscale='Blues', opacity=0.8,
            colorscale='Viridis', opacity=0.8,
            colorbar=dict(
                title='Call IV',
                x=1.0,                 # all the way right
                xanchor='left'
            ),
            showscale=True,
            name='Call IV'
        ))

    # Put surface: colorbar on the LEFT
    if Tp is not None and Kp is not None:
        fig.add_trace(go.Surface(
            x=Tp, y=Kp, z=Zpi,
            # colorscale='Greens', opacity=0.8,
            colorscale='Plasma', opacity=0.8,
            colorbar=dict(
                title='Put IV',
                x=0.0,                 # all the way left
                xanchor='right'
            ),
            showscale=True,
            name='Put IV'
        ))

    if spot_line:
        z_all = np.concatenate([Zc, Zp])
        z_range = [np.nanmin(z_all), np.nanmax(z_all)]
        fig.add_trace(go.Scatter3d(
            x=[0, 0],
            y=[spot_price, spot_price],
            z=z_range,
            mode='lines',
            line=dict(color='red', width=2),
            name=f'Spot Price ({spot_price:.2f})'
        ))

    scene_dict = {
        "xaxis_title": 'Time to Expiration (years)',
        "yaxis_title": 'Strike',
        "zaxis_title": 'Implied Volatility'
    }
    if strike_limits is not None:
        scene_dict['yaxis'] = dict(range=[strike_min, strike_max])

    fig.update_layout(
        title=f'Implied Volatility Surface for {ticker_symbol} Calls & Puts',
        scene=scene_dict,
        width=800, height=600,
        margin=dict(l=65, r=65, b=65, t=90)
    )
    return fig

# ------------------- STREAMLIT APP -----------------------

st.title("Implied Volatility Surface Explorer")

# On main panel
with st.container():
    col1, col2, col3, col4, col5 = st.columns([.6, .5, .5, 1.2, 1.])
    with col1:
        ticker_input = st.text_input("Ticker", value='AAPL')
    with col2:
        r_val = st.number_input("r (rate)", value=0.04, format="%.4f")
    with col3:
        q_val = st.number_input("q (div yield)", value=0.01, format="%.4f")
    with col4:
        range_pct = st.slider("Strike Range (% of Spot)", 50, 150, (80, 120), step=5)
    with col5:
        show_spot_line = st.checkbox("Spot Price Line", value=True)

# Center the button beneath the control bar, NOT inside any column
colA, colB, colC = st.columns([2, 1, 2])
with colB:
    plot_btn = st.button("Plot IV Surfaces")

if plot_btn:
    spot_price = get_spot_price(ticker_input)
    if spot_price is None:
        st.error("Couldn't fetch spot price for that ticker. Try another.")
    else:
        calls_df = get_option_data(ticker_input, 'call')
        puts_df = get_option_data(ticker_input, 'put')
        if calls_df.empty or puts_df.empty:
            st.warning("Insufficient data! Try a lower min open interest or a different symbol.")
        else:
            calls_with_iv = calculate_implied_vols(calls_df, spot_price, r=r_val, q=q_val)
            puts_with_iv = calculate_implied_vols(puts_df, spot_price, r=r_val, q=q_val)
            strike_window = (spot_price * range_pct[0] / 100, spot_price * range_pct[1] / 100)
            tabs = st.tabs(["Calls Only", "Puts Only", "Calls & Puts"])
            with tabs[0]:
                fig_calls = plot_iv_surface(calls_with_iv, ticker_input, spot_price, 
                                            spot_line=show_spot_line, strike_limits=strike_window)
                st.plotly_chart(fig_calls, use_container_width=True)
            with tabs[1]:
                fig_puts = plot_iv_surface(puts_with_iv, ticker_input, spot_price, 
                                           spot_line=show_spot_line, strike_limits=strike_window)
                st.plotly_chart(fig_puts, use_container_width=True)
            with tabs[2]:
                fig_both = plot_iv_surface_both(calls_with_iv, puts_with_iv, ticker_input, spot_price, 
                                                spot_line=show_spot_line, strike_limits=strike_window)
                st.plotly_chart(fig_both, use_container_width=True)

#st.write("---")
#st.markdown("Created by Tanmoy Ghosh ")
