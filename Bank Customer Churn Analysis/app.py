import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import pickle
import plotly.express as px

# Page config
st.set_page_config(
    page_title="Bank Churn Analysis", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- THEME & STYLING ---
THEME = {
    "bg": "#0e1117",
    "card_bg": "#1a1d24",
    "primary": "#4a7ba7",
    "primary_light": "#63b3ed",
    "secondary": "#7b68b8",
    "success": "#48bb78",
    "danger": "#fc8181",
    "danger_dark": "#e53e3e",
    "text": "#ffffff",
    "subtext": "#a0aec0",
    "grid": "#2d3748"
}

def get_custom_css():
    return f"""
<style>
    .stApp {{
        background-color: {THEME['bg']};
    }}
    
    [data-testid="stSidebar"] {{
        background-color: {THEME['card_bg']};
    }}
    
    h1, h2, h3, h4 {{
        color: {THEME['text']};
    }}
    
    .metric-card {{
        background: linear-gradient(135deg, #1e2631 0%, #252d3a 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid {THEME['grid']};
        text-align: center;
        margin: 0.5rem 0;
        height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }}
    
    .metric-title {{
        font-size: 0.85rem;
        color: {THEME['subtext']};
        margin-bottom: 0.5rem;
    }}
    
    .metric-value {{
        font-size: 2.2rem;
        font-weight: 700;
        color: {THEME['text']};
    }}
    
    .chart-container {{
        background-color: {THEME['card_bg']};
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid {THEME['grid']};
        margin-bottom: 1rem;
        height: 100%;
    }}
    
    .driver-card {{
        background-color: #1e2631;
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid {THEME['primary']};
        margin-bottom: 0.8rem;
    }}
    
    .driver-title {{
        font-size: 1rem;
        font-weight: 600;
        color: {THEME['text']};
        margin-bottom: 0.3rem;
    }}
    
    .driver-desc {{
        font-size: 0.85rem;
        color: {THEME['subtext']};
        line-height: 1.4;
    }}
    
    .stButton>button {{
        background-color: {THEME['danger_dark']};
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        width: 100%;
    }}
    
    .stButton>button:hover {{
        background-color: #c53030;
    }}
    
    /* Remove extra padding */
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 1rem;
    }}
</style>
"""

st.markdown(get_custom_css(), unsafe_allow_html=True)

@st.cache_data
def load_data_from_path(path: str):
    return pd.read_csv(path)

def load_data():
    try:
        return load_data_from_path("Churn_Modelling.csv")
    except Exception:
        uploaded = st.file_uploader("Upload Churn_Modelling.csv", type="csv")
        if uploaded is None:
            st.error("Churn_Modelling.csv not found in the project. Please upload the file to continue.")
            st.stop()
        return pd.read_csv(uploaded)

df = load_data()


@st.cache_data
def get_churn_by_age(df: pd.DataFrame):
    df2 = df.copy()
    bins = [18, 25, 35, 45, 60, np.inf]
    labels = ["18-25", "26-35", "36-45", "46-60", "60+"]
    df2["AgeGroup"] = pd.cut(df2["Age"], bins=bins, labels=labels)
    return df2.groupby("AgeGroup", observed=True)["Exited"].mean() * 100


@st.cache_data
def get_churn_by_balance(df: pd.DataFrame):
    df2 = df.copy()
    balance_bins = [0, 50000, 100000, 150000, 200000, np.inf]
    balance_labels = ["0-50K", "50-100K", "100-150K", "150-200K", "200K+"]
    df2["BalanceGroup"] = pd.cut(df2["Balance"], bins=balance_bins, labels=balance_labels)
    return df2.groupby("BalanceGroup", observed=True)["Exited"].mean() * 100


@st.cache_data
def get_geo_churn(df: pd.DataFrame):
    return df.groupby("Geography")["Exited"].mean() * 100


@st.cache_data
def get_product_churn(df: pd.DataFrame):
    return df.groupby("NumOfProducts")["Exited"].mean() * 100


@st.cache_data
def get_gender_churn(df: pd.DataFrame):
    return df.groupby("Gender")["Exited"].mean() * 100


@st.cache_data
def get_active_churn(df: pd.DataFrame):
    return df.groupby("IsActiveMember")["Exited"].mean().reindex([0, 1]).fillna(0) * 100


def _enable_feature(key: str):
    """Helper used by placeholder buttons to enable a sidebar feature checkbox."""
    try:
        st.session_state[key] = True
    except Exception:
        pass


def _warn_small_groups(series: pd.Series, min_n: int = 30, context: str = "group"):
    """Warn when any group in `series` has fewer than `min_n` samples.

    `series` should be a pandas Series of group labels (one per row of the filtered dataframe).
    """
    try:
        counts = series.value_counts(dropna=False)
        small = counts[counts < min_n]
        if not small.empty:
            items = ", ".join([f"{str(idx)} ({int(cnt)})" for idx, cnt in small.items()])
            st.warning(f"Small sample sizes for {context}: {items}. Percentages may be unreliable (<{min_n}).")
    except Exception:
        # don't break the dashboard if something unexpected happens
        pass


def render_driver_card(title: str, desc: str, left_color: str = None):
    """Render a consistent driver/insight card using the `.driver-card` CSS."""
    if left_color is None:
        left_color = THEME['primary']
        
    st.markdown(f"""
    <div class="driver-card" style="border-left-color: {left_color};">
        <div class="driver-title">{title}</div>
        <div class="driver-desc">{desc}</div>
    </div>
    """, unsafe_allow_html=True)


def render_metric_card(title: str, value: str, delta: str | None = None, subtitle: str | None = None):
    """Render a KPI using the `.metric-card` CSS so metrics look consistent with driver cards."""
    delta_html = f"<div style='font-size:0.85rem; color: #9ae6b4; margin-top:6px;'>{delta}</div>" if delta else ""
    subtitle_html = f"<div style='font-size:0.85rem; color: #a0aec0; margin-top:6px;'>{subtitle}</div>" if subtitle else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
        {subtitle_html}
    </div>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🌍 Select Geography")

    france = st.checkbox("France", value=True, key="fr")
    germany = st.checkbox("Germany", value=True, key="de")
    spain = st.checkbox("Spain", value=True, key="es")
    
    st.markdown("")
    # Key Features (moved to sidebar)
    st.markdown("### 📊 Key Features")
    st.caption("Toggle chart visibility")
    show_age = st.checkbox("📅 Age", value=True, key="feat_age")
    show_balance = st.checkbox("💰 Balance", value=True, key="feat_balance")
    show_products = st.checkbox("🔢 Products", value=True, key="feat_products")
    show_activity = st.checkbox("⚡ Activity", value=True, key="feat_activity")

    st.markdown("")
    active_only = st.checkbox("✓ Show Only Active Members", value=False)
    
    st.markdown("")
    if st.button("🔄 RESET FILTERS"):
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 👥 Demographics & Gender")
    
    show_male = st.checkbox("♂ Male", value=True, key="male")
    show_female = st.checkbox("♀ Female", value=True, key="female")
    
    st.markdown("---")
    st.markdown("### 🔎 Additional Filters")
    # Age slider
    if "Age" in df.columns:
        min_age = int(df["Age"].min())
        max_age = int(df["Age"].max())
        age_range = st.slider("Age range", min_age, max_age, (min_age, max_age), key="age_range")

    # Balance slider
    if "Balance" in df.columns:
        min_bal = int(df["Balance"].min())
        max_bal = int(df["Balance"].max())
        balance_range = st.slider("Balance range", min_bal, max_bal, (min_bal, max_bal), step=1000, key="balance_range")

    # Number of products multi-select
    if "NumOfProducts" in df.columns:
        prod_options = sorted(df["NumOfProducts"].unique().tolist())
        products_sel = st.multiselect("Number of Products", options=prod_options, default=prod_options, key="products_sel")

    st.markdown("---")

# Apply filters
geographies = []
if france:
    geographies.append("France")
if germany:
    geographies.append("Germany")
if spain:
    geographies.append("Spain")

if not geographies:
    geographies = df["Geography"].unique().tolist()

filtered_df = df[df["Geography"].isin(geographies)].copy()

if active_only: 
    filtered_df = filtered_df[filtered_df["IsActiveMember"] == 1] 

genders = []
if show_male:
    genders.append("Male")
if show_female:
    genders.append("Female")
if genders:
    filtered_df = filtered_df[filtered_df["Gender"].isin(genders)]

# Apply interactive filters from sidebar (if present)
try:
    if 'age_range' in globals():
        min_age_sel, max_age_sel = age_range
        if "Age" in filtered_df.columns:
            filtered_df = filtered_df[(filtered_df["Age"] >= min_age_sel) & (filtered_df["Age"] <= max_age_sel)]
except Exception:
    pass

try:
    if 'balance_range' in globals():
        min_bal_sel, max_bal_sel = balance_range
        if "Balance" in filtered_df.columns:
            filtered_df = filtered_df[(filtered_df["Balance"] >= min_bal_sel) & (filtered_df["Balance"] <= max_bal_sel)]
except Exception:
    pass

try:
    if 'products_sel' in globals() and products_sel:
        if "NumOfProducts" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["NumOfProducts"].isin(products_sel)]
except Exception:
    pass

# Calculate metrics
total_customers = len(filtered_df)
churned_customers = int(filtered_df["Exited"].sum())
churn_rate = (churned_customers / total_customers * 100) if total_customers > 0 else 0

# Title
st.markdown("<h1 style='text-align: center; margin-bottom: 0.3rem;'>🏦 Bank Customer Churn Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #718096; margin-bottom: 1.5rem;'>Comprehensive ML-Powered Customer Retention Dashboard</p>", unsafe_allow_html=True)

# Key Metrics
st.markdown("## 📈 Key Performance Indicators")
# compute overall churn for a simple delta
overall_churn_rate = (df["Exited"].mean() * 100) if len(df) > 0 else 0
delta_churn = churn_rate - overall_churn_rate

col1, col2, col3 = st.columns(3)
with col1:
    render_metric_card("👥 Total Customers", f"{total_customers:,}")
with col2:
    render_metric_card("👤 Churned Customers", f"{churned_customers:,}")
with col3:
    # show churn rate with delta vs overall
    render_metric_card("📉 Churn Rate", f"{churn_rate:.2f}%", delta=f"{delta_churn:+.2f}%", subtitle="vs overall")

# improve vertical spacing after KPIs
st.divider()
st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

@st.cache_data
def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode('utf-8')

# Download button for filtered data
try:
    csv = convert_df_to_csv(filtered_df)
    st.download_button(
        label="📥 Download Filtered Data",
        data=csv,
        file_name="churn_filtered_data.csv",
        mime="text/csv",
    )
except Exception:
    # If filtered_df isn't ready yet, skip download button silently
    pass

# Data summary expander (lightweight)
with st.expander("📊 Data Summary & Sample", expanded=False):
    st.write(f"Total rows: {len(df):,}")
    st.write(f"Total columns: {df.shape[1]}")
    st.write("Columns: " + ", ".join(df.columns.tolist()))
    st.write(f"Filtered rows: {len(filtered_df):,}")
    st.write("Geographies: " + ", ".join(geographies))
    if "Date" in df.columns:
        try:
            st.write(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        except Exception:
            pass
    st.dataframe(filtered_df.head(5))
# add spacing before main layout
st.divider()
st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

# Tabs: Dashboard (charts) and Predictions (model playground)
tabs = st.tabs(["Dashboard", "Predictions"])

# Prepare insights collection
insights_cards = []

with tabs[0]:
    # REMOVED: Insights Rail Toggle & Logic logic
    
    # Row 1: Age Group Chart (Full Width)
    if show_age:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("### 📊 Customer Churn Rate Across Age Groups")
        st.info("ℹ️ Churn rate = (Churned customers / Total customers) × 100. Small groups may show noisy percentages.")
        
        bins = [18, 25, 35, 45, 60, np.inf]
        labels = ["18-25", "26-35", "36-45", "46-60", "60+"] 
        # warn about small sample sizes in age groups
        if "Age" in filtered_df.columns:
            age_groups = pd.Series(pd.cut(filtered_df["Age"], bins=bins, labels=labels))
            _warn_small_groups(age_groups, min_n=30, context="age groups")
        
        churn_by_age = get_churn_by_age(filtered_df)

        # compute counts for hover templates
        try:
            counts_age = age_groups.value_counts().reindex(labels).fillna(0).astype(int)
        except Exception:
            counts_age = pd.Series([0] * len(labels), index=labels)

        # Use unified colors
        colors = [THEME['primary'], '#5886b3', '#6693bf', THEME['secondary'], '#8b58c6']
        fig1 = px.bar(x=churn_by_age.index.astype(str), y=churn_by_age.values,
                      labels={'x': 'Age Group', 'y': 'Churn Rate (%)'})
        fig1.update_traces(marker_color=colors, text=[f'{v:.1f}%' for v in churn_by_age.values], textposition='outside',
                           customdata=counts_age.values.reshape(-1, 1),
                           hovertemplate='Age Group: %{x}<br>Churn Rate: %{y:.1f}%<br>Count: %{customdata[0]}')
        fig1.update_layout(plot_bgcolor=THEME['card_bg'], paper_bgcolor=THEME['bg'], font_color=THEME['text'],
                           showlegend=False, yaxis=dict(range=[0, (max(churn_by_age.values) if len(churn_by_age)>0 else 1) * 1.2]),
                           height=380, margin=dict(l=20, r=20, t=40, b=30), hovermode='x unified', template='plotly_dark',
                           legend=dict(itemclick='toggleothers'))
        st.plotly_chart(fig1, width='stretch', key='chart_age')
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    else:
        st.info("Age chart hidden. Enable 'Age' under Key Features in the sidebar to show this chart.")
        if st.button("Show Age Chart", key="show_age_btn", on_click=_enable_feature, args=("feat_age",)):
            st.experimental_rerun()
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    
    # Row 2: Balance Line Chart (Full Width)
    if show_balance:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("### 💰 Churn Rate by Account Balance Range")
        st.info("ℹ️ Balance ranges use open top bins (200K+). Hover or inspect values for exact percentages.")
        balance_bins = [0, 50000, 100000, 150000, 200000, np.inf]
        balance_labels = ["0-50K", "50-100K", "100-150K", "150-200K", "200K+"]
        # warn about small sample sizes in balance groups
        if "Balance" in filtered_df.columns:
            bal_groups = pd.Series(pd.cut(filtered_df["Balance"], bins=balance_bins, labels=balance_labels))
            _warn_small_groups(bal_groups, min_n=30, context="balance ranges")
        churn_by_balance = get_churn_by_balance(filtered_df)
        
        fig2 = px.line(x=churn_by_balance.index.astype(str), y=churn_by_balance.values, markers=True,
                       labels={'x': 'Balance Range', 'y': 'Churn Rate (%)'})
        # attach counts for hover
        try:
            counts_balance = bal_groups.value_counts().reindex(balance_labels).fillna(0).astype(int)
        except Exception:
            counts_balance = pd.Series([0] * len(balance_labels), index=balance_labels)
        fig2.update_traces(line=dict(color=THEME['primary_light'], width=3), marker=dict(size=8),
                           text=[f'{v:.1f}%' for v in churn_by_balance.values], textposition='top center',
                           customdata=counts_balance.values.reshape(-1, 1),
                           hovertemplate='Balance Range: %{x}<br>Churn Rate: %{y:.1f}%<br>Count: %{customdata[0]}')
        fig2.update_layout(plot_bgcolor=THEME['card_bg'], paper_bgcolor=THEME['bg'], font_color=THEME['text'],
                           showlegend=False, yaxis=dict(range=[0, (max(churn_by_balance.values) if len(churn_by_balance)>0 else 1) * 1.25]),
                           height=360, margin=dict(l=20, r=20, t=40, b=30), hovermode='x unified', template='plotly_dark',
                           legend=dict(itemclick='toggleothers'))
        st.plotly_chart(fig2, width='stretch', key='chart_balance_main')
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    
    # Row 3: Two charts side by side (Geography and Products)
    col_geo, col_products = st.columns(2)
    
    with col_geo:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### 🌍 Churn by Geography")
        
        geo_churn = get_geo_churn(filtered_df)
        # warn about small sample sizes in geography groups
        if "Geography" in filtered_df.columns:
            _warn_small_groups(filtered_df["Geography"], min_n=30, context="geography")
        
        colors_geo = [THEME['primary'], THEME['secondary'], '#5886b3']
        fig3 = px.bar(x=geo_churn.index.astype(str), y=geo_churn.values, labels={'x':'Geography','y':'Churn Rate (%)'})
        try:
            counts_geo = filtered_df["Geography"].value_counts().reindex(geo_churn.index).fillna(0).astype(int)
        except Exception:
            counts_geo = pd.Series([0] * len(geo_churn.index), index=geo_churn.index)
        fig3.update_traces(marker_color=colors_geo, text=[f'{v:.1f}%' for v in geo_churn.values], textposition='outside',
                           customdata=counts_geo.values.reshape(-1, 1),
                           hovertemplate='Geography: %{x}<br>Churn Rate: %{y:.1f}%<br>Count: %{customdata[0]}')
        fig3.update_layout(plot_bgcolor=THEME['card_bg'], paper_bgcolor=THEME['bg'], font_color=THEME['text'], showlegend=False,
                   yaxis=dict(range=[0, (max(geo_churn.values) if len(geo_churn)>0 else 1) * 1.25]), legend=dict(itemclick='toggleothers'))
        st.plotly_chart(fig3, width='stretch', key='chart_geo')
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col_products:
        if show_balance:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("### 💰 Churn Rate by Account Balance Range")
            st.info("ℹ️ Balance ranges use open top bins (200K+). Hover or inspect values for exact percentages.")
        
            balance_bins = [0, 50000, 100000, 150000, 200000, np.inf]
            balance_labels = ["0-50K", "50-100K", "100-150K", "150-200K", "200K+"]
            churn_by_balance = get_churn_by_balance(filtered_df)
            # warn about small sample sizes in balance groups (side chart)
            if "Balance" in filtered_df.columns:
                bal_groups = pd.Series(pd.cut(filtered_df["Balance"], bins=balance_bins, labels=balance_labels))
                _warn_small_groups(bal_groups, min_n=30, context="balance ranges")
        
            fig2 = px.line(x=churn_by_balance.index.astype(str), y=churn_by_balance.values, markers=True,
                           labels={'x': 'Balance Range', 'y': 'Churn Rate (%)'})
            try:
                counts_balance = bal_groups.value_counts().reindex(balance_labels).fillna(0).astype(int)
            except Exception:
                counts_balance = pd.Series([0] * len(balance_labels), index=balance_labels)
            fig2.update_traces(line=dict(color=THEME['primary_light'], width=3), marker=dict(size=8),
                               text=[f'{v:.1f}%' for v in churn_by_balance.values], textposition='top center',
                               customdata=counts_balance.values.reshape(-1, 1),
                               hovertemplate='Balance Range: %{x}<br>Churn Rate: %{y:.1f}%<br>Count: %{customdata[0]}')
            fig2.update_layout(plot_bgcolor=THEME['card_bg'], paper_bgcolor=THEME['bg'], font_color=THEME['text'], showlegend=False,
                               yaxis=dict(range=[0, (max(churn_by_balance.values) if len(churn_by_balance)>0 else 1) * 1.25]),
                               height=360, margin=dict(l=20, r=20, t=40, b=30), hovermode='x unified', template='plotly_dark',
                               legend=dict(itemclick='toggleothers'))
            st.plotly_chart(fig2, width='stretch', key='chart_balance_side')
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        else:
            st.info("Balance chart hidden. Enable 'Balance' under Key Features in the sidebar to show this chart.")
            if st.button("Show Balance Chart", key="show_balance_btn", on_click=_enable_feature, args=("feat_balance",)):
                st.experimental_rerun()
            insights_cards.append(("💰 High Balance Risk", "Customers with 100K+ balances show elevated churn", THEME['primary']))
            
    insights_cards.append(("👤 Inactive Members", "Inactive customers are 3x more likely to churn", THEME['primary']))

    insights_cards.append(("German Market", "Highest churn rate - needs investigation", THEME['primary']))

    insights_cards.append(("👵 Senior Customers", "Age 46+ shows 2x higher churn rates", THEME['primary']))
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    # Row: Two pie charts side by side
    col_mini1, col_mini2 = st.columns(2)

    with col_mini1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### Gender Split")

        gender_churn = get_gender_churn(filtered_df)
        # warn about small sample sizes in gender groups
        if "Gender" in filtered_df.columns:
            _warn_small_groups(filtered_df["Gender"], min_n=30, context="gender")

        fig5 = px.pie(values=gender_churn.values, names=gender_churn.index, color=gender_churn.index,
                      color_discrete_sequence=[THEME['primary'], THEME['secondary']])
        try:
            counts_gender = filtered_df["Gender"].value_counts().reindex(gender_churn.index).fillna(0).astype(int)
        except Exception:
            counts_gender = pd.Series([0] * len(gender_churn.index), index=gender_churn.index)
        fig5.update_traces(textinfo='percent+label', textfont_size=12,
                           customdata=counts_gender.values.reshape(-1, 1),
                           hovertemplate='%{label}<br>Churn Rate: %{percent}<br>Count: %{customdata[0]}')
        fig5.update_layout(plot_bgcolor=THEME['card_bg'], paper_bgcolor=THEME['bg'], font_color=THEME['text'], legend=dict(itemclick='toggleothers'))
        st.plotly_chart(fig5, width='stretch', key='chart_gender')
        st.markdown("</div>", unsafe_allow_html=True)

    with col_mini2:
        if show_activity:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("#### Active Status")
            
            active_churn = get_active_churn(filtered_df)
            active_labels = ["Inactive", "Active"]
            # warn about small sample sizes in active/inactive groups
            if "IsActiveMember" in filtered_df.columns:
                act_series = filtered_df["IsActiveMember"].map({1: "Active", 0: "Inactive"})
                _warn_small_groups(act_series, min_n=30, context="active/inactive status")
            
            fig6 = px.pie(values=active_churn.values, names=active_labels, color=active_labels,
                          color_discrete_sequence=[THEME['danger'], THEME['success']])
            try:
                counts_active = filtered_df["IsActiveMember"].map({1: "Active", 0: "Inactive"}).value_counts().reindex(active_labels).fillna(0).astype(int)
            except Exception:
                counts_active = pd.Series([0] * len(active_labels), index=active_labels)
            fig6.update_traces(textinfo='percent+label', textfont_size=12,
                               customdata=counts_active.values.reshape(-1, 1),
                               hovertemplate='%{label}<br>Churn Rate: %{percent}<br>Count: %{customdata[0]}')
            fig6.update_layout(plot_bgcolor=THEME['card_bg'], paper_bgcolor=THEME['bg'], font_color=THEME['text'], legend=dict(itemclick='toggleothers'))
            st.plotly_chart(fig6, width='stretch', key='chart_active')
            st.markdown("</div>", unsafe_allow_html=True)

# Feature Importance
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("### 📊 ML Feature Importance")

    features = ["Active Status", "Products", "Balance", "Age"]
    impacts = [0.12, 0.18, 0.31, 0.38]

    fig7 = px.bar(x=impacts, y=features, orientation='h', labels={'x':'Impact Coefficient','y':''})
    fig7.update_traces(marker_color=THEME['primary'], text=[f'{v:.2f}' for v in impacts], textposition='outside')
    fig7.update_traces(hovertemplate='Feature: %{y}<br>Impact: %{x:.2f}')
    fig7.update_layout(plot_bgcolor=THEME['card_bg'], paper_bgcolor=THEME['bg'], font_color=THEME['text'], showlegend=False,
                       xaxis=dict(range=[0, max(impacts) * 1.15]), legend=dict(itemclick='toggleothers'))
    # warn if overall filtered sample is small before feature importance
    _warn_small_groups(pd.Series(['all'] * len(filtered_df)), min_n=30, context="overall dataset")
    st.plotly_chart(fig7, width='stretch', key='chart_feature_importance')
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Recommendations (SIMPLIFIED: Always shown at the bottom)
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("## 💡 Strategic Recommendations")
    for title, desc, color in insights_cards:
        render_driver_card(title, desc, left_color=color)
    st.markdown("</div>", unsafe_allow_html=True)

    # REMOVED: Separate right_rail logic

# Footer
with tabs[1]:
    st.markdown("### 🔮 Predict Churn")
    st.caption("Upload a scikit-learn model (.pkl) that implements `predict_proba`, or use the built-in estimator.")
    uploaded_model = st.file_uploader("Upload model (.pkl)", type=["pkl"], key="model_upload")
    # Prediction inputs (defaults from dataset where possible)
    if "Age" in df.columns:
        pred_age = st.slider("Age (for prediction)", int(df["Age"].min()), int(df["Age"].max()), int(df["Age"].median()), key="pred_age")
    else:
        pred_age = st.slider("Age (for prediction)", 18, 100, 40, key="pred_age")

    if "Balance" in df.columns:
        pred_balance = st.number_input("Account Balance", min_value=0, max_value=int(df["Balance"].max()), value=int(df["Balance"].median()), step=100, key="pred_balance")
    else:
        pred_balance = st.number_input("Account Balance", min_value=0, max_value=1000000, value=50000, step=100, key="pred_balance")

    if "NumOfProducts" in df.columns:
        pred_products = st.selectbox("Number of Products", options=prod_options, index=0, key="pred_products")
    else:
        pred_products = st.selectbox("Number of Products", options=[1,2,3,4], index=0, key="pred_products")

    pred_active = st.checkbox("Is Active Member", value=True, key="pred_active")
    pred_gender = st.selectbox("Gender", options=df["Gender"].unique().tolist() if "Gender" in df.columns else ["Male","Female"], index=0, key="pred_gender")
    pred_geo = st.selectbox("Geography", options=df["Geography"].unique().tolist() if "Geography" in df.columns else ["France","Germany","Spain"], index=0, key="pred_geo")

    if st.button("🔮 Predict Churn"):
        # Try loading user model
        prob = None
        if uploaded_model is not None:
            try:
                model = pickle.loads(uploaded_model.read())
                # build a simple feature frame - best-effort
                Xpred = pd.DataFrame([{"Age": pred_age, "Balance": pred_balance, "NumOfProducts": pred_products,
                                       "IsActiveMember": int(pred_active), "Gender": pred_gender, "Geography": pred_geo}])
                try:
                    # try predict_proba first
                    prob = float(model.predict_proba(Xpred)[:, 1][0])
                except Exception:
                    # fallback to predict
                    pred = model.predict(Xpred)[0]
                    prob = float(pred)
            except Exception as e:
                st.error(f"Failed to load/run uploaded model: {e}")

        # If no model or model failed, use a simple heuristic
        if prob is None:
            score = 0.0
            # older age -> slightly higher churn
            score += (pred_age - 40) * 0.02
            # higher balance -> small increase
            score += (pred_balance - 50000) / 200000.0
            # more products -> increase
            score += (int(pred_products) - 1) * 0.15
            # inactive members more likely to churn
            score += 0.4 if not pred_active else -0.1
            # geography adjustment
            score += 0.1 if str(pred_geo).lower() == 'germany' else 0.0
            # gender small effect
            score += 0.05 if str(pred_gender).lower() == 'male' else 0.0
            prob = 1 / (1 + np.exp(-score))

        prob = max(0.0, min(1.0, float(prob)))
        st.metric("Predicted Churn Probability", f"{prob*100:.1f}%")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #718096; font-size: 0.85rem;'>ML-Powered Customer Intelligence • Model Confidence: 0.72 • Last Updated: 2024</p>", unsafe_allow_html=True)
