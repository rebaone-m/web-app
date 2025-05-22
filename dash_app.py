import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# --- Streamlit Config ---
st.set_page_config(page_title="Sales Performance Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- Enhanced CSS for Static Layout with Adjusted Summary Styling ---
st.markdown(
    """
    <style>
    /* Hide Streamlit default elements */
    #MainMenu, footer, header { visibility: hidden; height: 0; }
    
    /* Global styles - Static layout with no scrolling */
    html, body, [data-testid="stAppViewContainer"], main {
        overflow: hidden !important; /* Disable scrolling */
        height: 100vh !important; /* Fixed height */
        font-family: 'Segoe UI', Arial, sans-serif;
        background: #1c2526 !important;
        color: #ffffff !important;
    }
    
    /* Sidebar styling - Enhanced with gradient and no top space */
    section [data-testid="stSidebar"] {
        width: 200px !important; /* Reduced width */
        background: linear-gradient(135deg, #34495e, #2c3e50); /* Softer gradient for better contrast */
        border-right: 2px solid #4b5563;
        padding: 0 !important; /* Remove top padding to eliminate empty space */
        box-shadow: 2px 0 5px rgba(0,0,0,0.5); /* Stronger shadow for definition */
        height: 100vh !important; /* Ensure full height */
    }
    .stSidebar > div:first-child {
        padding-top: 0 !important; /* Remove internal top padding */
    }
    .stSidebar .stMarkdown {
        margin-top: 0 !important; /* Remove margin above header */
    }
    .stSidebar h3 {
        font-size: 11px !important; /* Increased font size */
        color: #ecf0f1 !important; /* Light text for contrast */
        padding: 0.5rem 0.5rem 0.3rem; /* Adjusted padding */
        background: rgba(0,0,0,0.1); /* Subtle background for header */
        margin: 0 !important;
    }
    .stSidebar .stSelectbox, .stSidebar .stMultiSelect, .stSidebar .stDateInput {
        font-size: 11px !important; /* Increased font size */
        margin: 0.3rem 0.5rem;
        color: #ffffff !important;
        background: #465c71 !important; /* Lighter background for inputs */
        border: 1px solid #5d6d7e;
        border-radius: 4px;
    }
    .stSidebar .stSelectbox div[data-baseweb="select"] > div, 
    .stSidebar .stMultiSelect div[data-baseweb="select"] > div {
        background: #465c71 !important;
        color: #ffffff !important;
    }
    .stSidebar button {
        font-size: 11px !important;
        padding: 0.3rem;
        margin: 0.3rem 0.5rem;
        border-radius: 4px;
        background: #3498db !important; /* Brighter blue for buttons */
        color: #ffffff !important;
        border: none;
    }
    .stSidebar button:hover {
        background: #2980b9 !important;
    }
    .stSidebar .stDownloadButton {
        margin: 0.3rem 0.5rem;
    }
    
    /* Main content */
    .block-container {
        padding: 0.3rem !important;
        height: 100vh !important; /* Fixed height */
        overflow: hidden !important; /* Disable scrolling */
        background: #1c2526 !important;
    }
    
    /* Metric boxes with minimal styling */
    .metric-box, .insight-box {
        padding: 3px 4px;
        background-color: #f9f9f9;
        border-radius: 4px;
        font-size: 8px !important; /* Reduced font size */
        margin-bottom: 1px;
        word-wrap: break-word;
        white-space: normal;
        max-width: 100%;
        box-shadow: 0 1px 1px rgba(0,0,0,0.04);
    }
    .metric-box.reached, .metric-box.average, .metric-box.below {
        color: #000000 !important; /* Black text for readability on colored backgrounds */
    }
    .metric-box.reached {
        background-color: #22c55e !important; /* Green background for Reached */
    }
    .metric-box.average {
        background-color: #f97316 !important; /* Orange background for Average */
    }
    .metric-box.below {
        background-color: #ef4444 !important; /* Red background for Below */
    }
    .stMetric {
        font-size: 8px !important; /* Default for other metrics */
    }
    
    /* Target status metrics with larger font and colored values */
    .target-metrics .stMetric {
        font-size: 12px !important; /* Increased for better visibility */
    }
    .target-metrics .reached {
        color: #22c55e !important; /* Green for Reached */
    }
    .target-metrics .average {
        color: #f97316 !important; /* Orange for Average */
    }
    .target-metrics .below {
        color: #ef4444 !important; /* Red for Below */
    }
    .target-metrics .stMetric > div:last-child {
        font-weight: bold;
    }
    
    /* Chart containers - Minimal size */
    .chart-container {
        height: 150px !important; /* Reduced height for minimal charts */
        padding: 0.3rem;
        background: #374151 !important;
        border-radius: 4px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        border: 1px solid #333 !important;
        margin: 2px 0;
        overflow: hidden !important;
    }
    .single-chart {
        margin: 0 auto;
        max-width: 50%;
        height: 150px !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        font-size: 10px !important; /* Reduced font size */
        padding: 0.3rem 0.6rem;
        border-radius: 4px 4px 0 0;
        background: #374151 !important;
        color: #d1d5db !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: #2563eb !important;
        color: #ffffff !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    
    /* Headers and text - Minimal (Increased font above tabs) */
    h1, h2, h3, .stSubheader {
        font-size: 12px !important; /* Increased font size for headers above tabs */
        margin: 1px 0 !important;
        color: #ffffff !important;
    }
    .stMarkdown, .stText, .stSubheader div {
        line-height: 1.1;
        margin: 0 !important;
        padding: 0 !important;
        font-size: 9px !important; /* Reduced font size */
        color: #d1d5db !important;
    }
    
    /* Summary card - Enhanced for better visibility */
    .summary-card {
        background: #4b5563 !important; /* Slightly brighter background for contrast */
        border-left: 3px solid #3b82f6 !important; /* Brighter blue border */
        padding: 0.6rem !important; /* Increased padding */
        border-radius: 6px !important; /* Slightly larger border radius */
        box-shadow: 0 3px 6px rgba(0,0,0,0.4) !important; /* Stronger shadow */
        font-size: 12px !important; /* Increased font size to match headers */
        margin-bottom: 0.5rem !important; /* Slightly more margin */
        color: #e5e7eb !important; /* Lighter text for readability */
        line-height: 1.4 !important; /* Improved line spacing */
    }
    
    /* Key for target status - Increased font to match headers (12px) */
    .target-key {
        background: #2d3536 !important;
        padding: 0.3rem;
        border-radius: 4px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        display: flex;
        justify-content: center;
        gap: 0.3rem;
        font-size: 12px !important; /* Increased to match header font size */
        color: #d1d5db !important;
        margin-bottom: 0.3rem;
    }
    .target-key span {
        display: flex;
        align-items: center;
        gap: 0.1rem;
    }
    .color-box {
        width: 8px;
        height: 8px;
        border-radius: 2px;
        display: inline-block;
    }
    
    /* Responsive layout */
    .stColumn {
        padding: 0 3px !important; /* Reduced padding */
    }
    </style>
    """, unsafe_allow_html=True
)


# --- Load Data ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Dataset.csv', parse_dates=['Date'])
        string_columns = [
            'Country', 'Company Name', 'Subscription Status', 'Returning Customer',
            'Campaign Name', 'Campaign Type', 'Product Name', 'Category',
            'Sale Status', 'Discount Applied', 'Conversion Source'
        ]
        for col in string_columns:
            df[col] = df[col].astype(str).replace('nan', '')
        numeric_columns = [
            'Total Revenue', 'Profit Per Sale', 'Loss from Returns',
            'Revenue Per Product', 'Customer Lifetime Value', 'Average Order Value',
            'Session Duration (min)', 'Pages Visited', 'Bounce Rate (%)'
        ]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['Country', 'Total Revenue', 'Sale Status'])
        return df
    except FileNotFoundError:
        st.error("File is not found. Ensure the file is in the correct directory.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.error("No data loaded. Please check the file and ensure it contains valid data.")
    st.stop()

# --- Sidebar Filters ---
with st.sidebar:
    st.markdown("<h3 style='color: #ffffff;'>Filters</h3>", unsafe_allow_html=True)
    if st.button("Instructions", key="instructions"):
        st.session_state.page = "Instructions"
        st.rerun()

    if "team" not in st.session_state:
        st.session_state.team = "Sales & Marketing Team"
    team_options = [
        "Sales & Marketing Team", "Marketing Analyst", "Marketing Manager",
        "Sales Data Analyst", "Sales Manager"
    ]
    st.session_state.team = st.selectbox("Team Member", team_options, index=0)

    filtered = df.copy()
    for col in ['Country', 'Product Name', 'Company Name', 'Sale Status']:
        options = sorted(df[col].dropna().unique())
        selected = st.multiselect(col, options, default=[], key=f"filter_{col.lower()}")
        if selected:
            filtered = filtered[filtered[col].isin(selected)]

    date_min, date_max = df['Date'].min(), df['Date'].max()
    start_date, end_date = st.date_input(
        "Date Range",
        [date_min.date(), date_max.date()],
        min_value=date_min.date(),
        max_value=date_max.date()
    )
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    filtered = filtered[(filtered['Date'] >= start) & (filtered['Date'] <= end)]

    st.markdown("---")
    @st.cache_data
    def convert_csv(data): return data.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", convert_csv(filtered), "filtered_sales.csv", "text/csv")

    # Debug Toggle
    debug = st.checkbox("Show Debug Info", key="debug")
    if debug:
        st.write(f"Dataset Rows: {len(df)}")
        st.write(f"Filtered Rows: {len(filtered)}")
        st.write(f"Columns: {list(df.columns)}")
        st.write(f"Missing Columns: {set(['Country', 'Total Revenue', 'Sale Status', 'Date']) - set(df.columns)}")
        st.write(f"Data Types:\n{df.dtypes}")


completed = filtered[filtered['Sale Status'].str.lower().str.strip() == 'completed']

# --- Chart Configuration - Minimal Settings ---
chart_config = dict(
    height=150,  # Reduced for minimal charts
    margin=dict(l=10, r=10, t=20, b=10),  # Reduced margins
    title_font_size=10,  # Reduced font size
    xaxis_tickfont_size=8,  # Reduced font size
    yaxis_tickfont_size=8,  # Reduced font size
    xaxis_title_font_size=10,  # Reduced font size
    yaxis_title_font_size=10,  # Reduced font size
    showlegend=True,
    legend=dict(font=dict(size=8, color='#ffffff'), x=1, y=0.5, xanchor="right"),  # Reduced font size
    paper_bgcolor='#374151',
    plot_bgcolor='#374151',
    font=dict(color='#ffffff'),
    xaxis_gridcolor='#4b5563',
    yaxis_gridcolor='#4b5563'
)

# Plotly Fullscreen Configuration
plotly_config = {
    'displayModeBar': True  # Enables the mode bar with fullscreen option
}

# --- Color Definitions ---
GREEN = "#22c55e"  # Reached
ORANGE = "#f97316"  # Average
RED = "#ef4444"  # Below or Loss

# --- Helper Function to Assign Colors Based on Performance ---
def assign_performance_colors(values, threshold_high, threshold_low):
    colors = []
    for val in values:
        if val >= threshold_high:
            colors.append(GREEN)  # Reached
        elif val >= threshold_low:
            colors.append(ORANGE)  # Average
        else:
            colors.append(RED)  # Below
    return colors

# --- Target Status Function ---
def get_target_status(actual, target, reverse=False):
    if reverse:  # For metrics like Bounce Rate where lower is better
        if actual <= target:
            return "Reached", "reached", GREEN
        elif actual <= target * 1.33:  # Up to 33% worse than target (e.g., 40% if target is 30%)
            return "Average", "average", ORANGE
        else:
            return "Below", "below", RED
    else:  # For metrics where higher is better (e.g., Revenue, Profit)
        if actual >= target:
            return "Reached", "reached", GREEN
        elif actual >= target * 0.60:  # 60%-75% of Target
            return "Average", "average", ORANGE
        else:
            return "Below", "below", RED

# --- Main Content ---
if st.session_state.get('page') == 'Instructions':
    st.markdown("<h3 style='color: #ffffff;'>Dashboard Guide</h3>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='summary-card'>
            <b>Navigate:</b> Use sidebar filters to refine data.<br>
            <b>Visuals:</b> Hover on charts for details. Click the fullscreen icon to expand charts.<br>
            <b>Tabs:</b> Overview, Customer Insights, Geographical Insights, Monthly/Quarterly Trends, Losses & Returns.<br>
            <b>Recovery:</b> If "No data", adjust filters.<br>
            <b>Export:</b> Download CSV from sidebar.
        </div>
        """, unsafe_allow_html=True
    )
    if st.button("Back to Dashboard", key="back_to_dashboard"):
        st.session_state.page = "overview"
        st.rerun()

else:
    # --- Target Status Key ---
    st.markdown(
        """
        <div class='target-key'>
            <span><span class='color-box' style='background: #22c55e;'></span>Reached: ≥ Target</span>
            <span><span class='color-box' style='background: #f97316;'></span>Average: 60%-75% of Target</span>
            <span><span class='color-box' style='background: #ef4444;'></span>Below: < 60% of Target</span>
        </div>
        """, unsafe_allow_html=True
    )

    if st.session_state.team == "Sales & Marketing Team":
        tabs = st.tabs([
            "Customer Insights", "Geographical Insights",
            "Monthly/Quarterly Trends", "Losses & Returns"
        ])
       
        with tabs[0]:  # Customer Insights
            st.header("Customer Insights")
            if not filtered.empty:
                # Statistical Summary for Customer Insights
                loyal = completed.groupby('Company Name').size().nlargest(5).reset_index(name='Purchases')
                top_customer = loyal.iloc[0] if not loyal.empty else None
                lifetime_value = completed.groupby('Company Name')['Customer Lifetime Value'].sum().reset_index()
                avg_ltv = lifetime_value['Customer Lifetime Value'].mean() if not lifetime_value.empty else 0
                returning_sales = completed.groupby('Returning Customer').size().reset_index(name='Sales Count')
                return_rate = (returning_sales.set_index('Returning Customer').loc['Yes']['Sales Count'] / returning_sales['Sales Count'].sum()) * 100 if 'Yes' in returning_sales['Returning Customer'].values else 0.0
                top_campaign = completed.groupby('Campaign Name')['Total Revenue'].sum().nlargest(1).reset_index().iloc[0] if not completed.empty else None

                summary_text = (
                    f"<b>Customer Insights Summary:</b><br>"
                    f"Top Customer: {top_customer['Company Name'] if top_customer is not None else 'N/A'} with {top_customer['Purchases'] if top_customer is not None else 0} purchases.<br>"
                    f"Average LTV: ${avg_ltv:,.0f}.<br>"
                    f"Returning Customer Rate: {return_rate:.1f}%. "
                    f"Top Campaign: {top_campaign['Campaign Name'] if top_campaign is not None else 'N/A'} (${top_campaign['Total Revenue']:,.0f})."
                )
                st.markdown(f"<div class='summary-card'>{summary_text}</div>", unsafe_allow_html=True)

                col1, col2, col3 = st.columns(3)
                if not loyal.empty:
                    top_customer = loyal.iloc[0]
                    col1.metric("Top Customer", top_customer['Company Name'], f"{top_customer['Purchases']} Purchases")
                    fig1 = px.bar(loyal, x='Company Name', y='Purchases', title="Top Loyal Customers")
                    fig1.update_layout(**chart_config)
                    col1.plotly_chart(fig1, use_container_width=True, config=plotly_config, key="chart_ci_1")
                else:
                    col1.warning("No data available to show top customer.")
                if not lifetime_value.empty:
                    max_ltv = lifetime_value.sort_values('Customer Lifetime Value', ascending=False).iloc[0]
                    col2.metric("Highest LTV", max_ltv['Company Name'], f"${max_ltv['Customer Lifetime Value']:,.2f}")
                    mean_ltv = lifetime_value['Customer Lifetime Value'].mean()
                    threshold_high_ltv = mean_ltv * 1.1
                    threshold_low_ltv = mean_ltv * 0.9
                    colors = assign_performance_colors(lifetime_value['Customer Lifetime Value'], threshold_high_ltv, threshold_low_ltv)
                    fig2 = px.histogram(lifetime_value, x='Company Name', y='Customer Lifetime Value', title="Lifetime Value by Company",
                                        color=lifetime_value['Customer Lifetime Value'], color_discrete_sequence=colors)
                    fig2.update_layout(**chart_config)
                    col2.plotly_chart(fig2, use_container_width=True, config=plotly_config, key="chart_ci_2")
                else:
                    col2.warning("No data available for Lifetime Value.")
                if not returning_sales.empty:
                    if 'Yes' in returning_sales['Returning Customer'].values:
                        return_pct = (returning_sales.set_index('Returning Customer').loc['Yes']['Sales Count'] / returning_sales['Sales Count'].sum()) * 100
                    else:
                        return_pct = 0.0
                    col3.metric("Returning Customer Rate", f"{return_pct:.1f}%", " ")
                    fig3 = px.pie(returning_sales, names='Returning Customer', values='Sales Count', title="Returning Customers")
                    fig3.update_layout(**chart_config)
                    col3.plotly_chart(fig3, use_container_width=True, config=plotly_config, key="chart_ci_3")

                col4, col5, col6, col7 = st.columns(4)
                with col4:
                    camp_type_rev = completed.groupby('Campaign Type')['Total Revenue'].sum().reset_index()
                    mean_rev = camp_type_rev['Total Revenue'].mean()
                    threshold_high = mean_rev * 1.1
                    threshold_low = mean_rev * 0.9
                    colors = assign_performance_colors(camp_type_rev['Total Revenue'], threshold_high, threshold_low)
                    fig4 = px.bar(camp_type_rev, x='Campaign Type', y='Total Revenue', title="Campaign Revenue by Type",
                                  color=camp_type_rev['Total Revenue'], color_discrete_sequence=colors)
                    fig4.update_layout(**chart_config)
                    col4.plotly_chart(fig4, use_container_width=True, config=plotly_config, key="chart_ci_4")
                with col5:
                    ltv = completed.groupby('Company Name')['Customer Lifetime Value'].sum().reset_index()
                    mean_ltv = ltv['Customer Lifetime Value'].mean()
                    threshold_high_ltv = mean_ltv * 1.1
                    threshold_low_ltv = mean_ltv * 0.9
                    colors = assign_performance_colors(ltv['Customer Lifetime Value'], threshold_high_ltv, threshold_low_ltv)
                    fig5 = px.histogram(ltv, x='Company Name', y='Customer Lifetime Value', title="Customer Lifetime Value",
                                        color=ltv['Customer Lifetime Value'], color_discrete_sequence=colors)
                    fig5.update_layout(**chart_config)
                    col5.plotly_chart(fig5, use_container_width=True, config=plotly_config, key="chart_ci_5")
                with col6:
                    returning = completed.groupby('Returning Customer').size().reset_index(name='Count')
                    fig6 = px.pie(returning, names='Returning Customer', values='Count', title="Returning Customers")
                    fig6.update_layout(**chart_config)
                    col6.plotly_chart(fig6, use_container_width=True, config=plotly_config, key="chart_ci_6")
                with col7:
                    prod_camp = completed.groupby(['Campaign Name', 'Product Name'])['Total Revenue'].sum().nlargest(5).reset_index()
                    mean_rev = prod_camp['Total Revenue'].mean()
                    threshold_high = mean_rev * 1.1
                    threshold_low = mean_rev * 0.9
                    colors = assign_performance_colors(prod_camp['Total Revenue'], threshold_high, threshold_low)
                    fig7 = px.bar(prod_camp, x='Product Name', y='Total Revenue', color='Campaign Name', title="Top Products by Campaign",
                                  color_discrete_sequence=colors)
                    fig7.update_layout(**chart_config)
                    col7.plotly_chart(fig7, use_container_width=True, config=plotly_config, key="chart_ci_7")
            else:
                st.info("No data available.")
        with tabs[1]:  # Geographical Insights
            st.header("Geographical Insights")
            if not filtered.empty:
                # Statistical Summary for Geographical Insights
                geo = completed.groupby('Country')['Total Revenue'].sum().reset_index()
                top_country = geo.sort_values('Total Revenue', ascending=False).iloc[0] if not geo.empty else None
                total_countries = len(geo)
                top_country_share = (top_country['Total Revenue'] / geo['Total Revenue'].sum() * 100) if top_country is not None else 0.0
                avg_profit_per_sale = completed.groupby('Country')['Profit Per Sale'].mean().mean()

                summary_text = (
                    f"<b>Geographical Insights Summary:</b><br>"
                    f"Top Country: {top_country['Country'] if top_country is not None else 'N/A'} (${top_country['Total Revenue']:,.0f}, {top_country_share:.1f}% of total).<br>"
                    f"Total Countries: {total_countries}.<br>"
                    f"Avg Profit/Sale: ${avg_profit_per_sale:,.0f}."
                )
                st.markdown(f"<div class='summary-card'>{summary_text}</div>", unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                top_country_rev = geo.sort_values('Total Revenue', ascending=False).iloc[0]
                col1.metric("Top Country by Revenue", top_country_rev['Country'], f"${top_country_rev['Total Revenue']:,.0f}")
                mean_rev = geo['Total Revenue'].mean()
                threshold_high = mean_rev * 1.1
                threshold_low = mean_rev * 0.9
                colors = assign_performance_colors(geo['Total Revenue'], threshold_high, threshold_low)
                fig1 = px.choropleth(geo, locations='Country', locationmode='country names', color='Total Revenue', title="Revenue by Country",
                                     color_continuous_scale=[RED, ORANGE, GREEN])
                fig1.update_layout(**chart_config, geo=dict(showframe=False, projection_type='equirectangular'))
                col1.plotly_chart(fig1, use_container_width=True, config=plotly_config, key="chart_gi_1")
                profit_per_sale = completed.groupby('Country')['Profit Per Sale'].mean().reset_index()
                top_country_profit = profit_per_sale.sort_values('Profit Per Sale', ascending=False).iloc[0]
                col2.metric("Top Country Profit/Sale", top_country_profit['Country'], f"${top_country_profit['Profit Per Sale']:,.2f}")
                mean_profit = profit_per_sale['Profit Per Sale'].mean()
                threshold_high = mean_profit * 1.1
                threshold_low = mean_profit * 0.9
                colors = assign_performance_colors(profit_per_sale['Profit Per Sale'], threshold_high, threshold_low)
                fig2 = px.bar(profit_per_sale, x='Country', y='Profit Per Sale', title="Profit per Sale by Country",
                              color=profit_per_sale['Profit Per Sale'], color_discrete_sequence=colors)
                fig2.update_layout(**chart_config)
                col2.plotly_chart(fig2, use_container_width=True, config=plotly_config, key="chart_gi_2")

                col3, col4, col5, col6 = st.columns(4)
                with col3:
                    camp = completed.groupby('Campaign Name')['Total Revenue'].sum().nlargest(10).reset_index()
                    mean_rev = camp['Total Revenue'].mean()
                    threshold_high = mean_rev * 1.1
                    threshold_low = mean_rev * 0.9
                    colors = assign_performance_colors(camp['Total Revenue'], threshold_high, threshold_low)
                    fig3 = px.bar(camp, x='Campaign Name', y='Total Revenue', title="Top Campaigns",
                                  color=camp['Total Revenue'], color_discrete_sequence=colors)
                    fig3.update_layout(**chart_config)
                    col3.plotly_chart(fig3, use_container_width=True, config=plotly_config, key="chart_gi_3")
                with col4:
                    camp_trend = completed.groupby([pd.Grouper(key='Date', freq='ME'), 'Campaign Name'])['Total Revenue'].sum().reset_index()
                    mean_rev = camp_trend['Total Revenue'].mean()
                    threshold_high = mean_rev * 1.1
                    threshold_low = mean_rev * 0.9
                    colors = assign_performance_colors(camp_trend['Total Revenue'], threshold_high, threshold_low)
                    fig4 = px.line(camp_trend, x='Date', y='Total Revenue', color='Campaign Name', title="Monthly Campaign Revenue",
                                   color_discrete_sequence=colors)
                    fig4.update_layout(**chart_config)
                    col4.plotly_chart(fig4, use_container_width=True, config=plotly_config, key="chart_gi_4")
                with col5:
                    cat_rev = completed.groupby('Category')['Total Revenue'].sum().reset_index()
                    mean_rev = cat_rev['Total Revenue'].mean()
                    threshold_high = mean_rev * 1.1
                    threshold_low = mean_rev * 0.9
                    colors = assign_performance_colors(cat_rev['Total Revenue'], threshold_high, threshold_low)
                    fig5 = px.pie(cat_rev, names='Category', values='Total Revenue', title="Revenue by Category",
                                  color=cat_rev['Total Revenue'], color_discrete_sequence=colors)
                    fig5.update_layout(**chart_config)
                    col5.plotly_chart(fig5, use_container_width=True, config=plotly_config, key="chart_gi_5")
                with col6:
                    camp = completed.groupby('Campaign Name')['Total Revenue'].sum().reset_index()
                    mean_rev = camp['Total Revenue'].mean()
                    threshold_high = mean_rev * 1.1
                    threshold_low = mean_rev * 0.9
                    colors = assign_performance_colors(camp['Total Revenue'], threshold_high, threshold_low)
                    fig6 = px.bar(camp, x='Campaign Name', y='Total Revenue', title="Top Campaigns (All)",
                                  color=camp['Total Revenue'], color_discrete_sequence=colors)
                    fig6.update_layout(**chart_config)
                    col6.plotly_chart(fig6, use_container_width=True, config=plotly_config, key="chart_gi_6")
            else:
                st.info("No data available.")
        with tabs[2]:  # Monthly/Quarterly Trends
            st.header("Monthly/Quarterly Trends")
            if not filtered.empty:
                # Statistical Summary for Monthly/Quarterly Trends
                monthly = completed.set_index('Date').resample('ME')['Total Revenue'].sum().reset_index()
                revenue_change = ((monthly['Total Revenue'].iloc[-1] - monthly['Total Revenue'].iloc[-2]) / monthly['Total Revenue'].iloc[-2] * 100) if len(monthly) > 1 and monthly['Total Revenue'].iloc[-2] != 0 else 0.0
                monthly_aov = completed.set_index('Date').resample('ME')['Average Order Value'].mean().reset_index()
                aov_trend = ((monthly_aov['Average Order Value'].iloc[-1] - monthly_aov['Average Order Value'].iloc[-2]) / monthly_aov['Average Order Value'].iloc[-2] * 100) if len(monthly_aov) > 1 and monthly_aov['Average Order Value'].iloc[-2] != 0 else 0.0
                monthly_sales = completed.set_index('Date').resample('ME').size().reset_index(name='Sales Count')
                avg_sales_volume = monthly_sales['Sales Count'].mean()

                summary_text = (
                    f"<b>Trends Summary:</b><br>"
                    f"Latest MoM Revenue Change: {revenue_change:+.1f}%. "
                    f"MoM AOV Change: {aov_trend:+.1f}%. "
                    f"Avg Monthly Sales Volume: {avg_sales_volume:.0f} orders."
                )
                st.markdown(f"<div class='summary-card'>{summary_text}</div>", unsafe_allow_html=True)

                col1, col2, col3 = st.columns(3)
                latest_month = monthly.iloc[-1]
                delta = latest_month['Total Revenue'] - monthly.iloc[-2]['Total Revenue'] if len(monthly) > 1 else 0
                col1.metric("Latest Monthly Revenue", f"${latest_month['Total Revenue']:,.0f}", f"${delta:,.0f}")
                trend = 'positive' if delta > 0 else 'negative' if delta < 0 else 'mixed'
                color = GREEN if trend == 'positive' else ORANGE if trend == 'mixed' else RED
                fig1 = px.line(monthly, x='Date', y='Total Revenue', title="Monthly Revenue",
                               color_discrete_sequence=[color])
                fig1.update_layout(**chart_config)
                col1.plotly_chart(fig1, use_container_width=True, config=plotly_config, key="chart_mqt_1")
                quarterly = completed.set_index('Date').resample('QE')['Total Revenue'].sum().reset_index()
                quarterly['Quarter'] = quarterly['Date'].dt.to_period('Q').astype(str)
                q_latest = quarterly.iloc[-1]
                delta_q = q_latest['Total Revenue'] - quarterly.iloc[-2]['Total Revenue'] if len(quarterly) > 1 else 0
                col2.metric("Latest Quarter Revenue", f"${q_latest['Total Revenue']:,.0f}", f"${delta_q:,.0f}")
                trend = 'positive' if delta_q > 0 else 'negative' if delta_q < 0 else 'mixed'
                color = GREEN if trend == 'positive' else ORANGE if trend == 'mixed' else RED
                fig2 = px.line(quarterly, x='Quarter', y='Total Revenue', title="Quarterly Revenue",
                               color_discrete_sequence=[color])
                fig2.update_layout(**chart_config)
                col2.plotly_chart(fig2, use_container_width=True, config=plotly_config, key="chart_mqt_2")
                monthly_avg_order_value = completed.set_index('Date').resample('ME')['Average Order Value'].mean().reset_index()
                latest_avg = monthly_avg_order_value.iloc[-1]
                delta_avg = latest_avg['Average Order Value'] - monthly_avg_order_value.iloc[-2]['Average Order Value'] if len(monthly_avg_order_value) > 1 else 0
                col3.metric("Latest Avg Order Value", f"${latest_avg['Average Order Value']:,.2f}", f"${delta_avg:,.2f}")
                fig3 = px.line(monthly_avg_order_value, x='Date', y='Average Order Value', title="Monthly Avg Order Value")
                fig3.update_layout(**chart_config)
                col3.plotly_chart(fig3, use_container_width=True, config=plotly_config, key="chart_mqt_3")

                col4, col5, col6, col7 = st.columns(4)
                with col4:
                    profit_prod = completed.groupby('Product Name')['Profit Per Sale'].mean().reset_index()
                    mean_profit = profit_prod['Profit Per Sale'].mean()
                    threshold_high = mean_profit * 1.1
                    threshold_low = mean_profit * 0.9
                    colors = assign_performance_colors(profit_prod['Profit Per Sale'], threshold_high, threshold_low)
                    fig4 = px.bar(profit_prod, x='Product Name', y='Profit Per Sale', title="Profit per Sale by Product",
                                  color=profit_prod['Profit Per Sale'], color_discrete_sequence=colors)
                    fig4.update_layout(**chart_config)
                    col4.plotly_chart(fig4, use_container_width=True, config=plotly_config, key="chart_mqt_4")
                with col5:
                    status_counts = filtered.groupby('Sale Status').size().reset_index(name='Count')
                    fig5 = px.pie(status_counts, names='Sale Status', values='Count', title="Sale Status Distribution")
                    fig5.update_layout(**chart_config)
                    col5.plotly_chart(fig5, use_container_width=True, config=plotly_config, key="chart_mqt_5")
                with col6:
                    aov = completed.groupby('Company Name')['Average Order Value'].mean().reset_index()
                    mean_aov = aov['Average Order Value'].mean()
                    threshold_high = mean_aov * 1.1
                    threshold_low = mean_aov * 0.9
                    colors = assign_performance_colors(aov['Average Order Value'], threshold_high, threshold_low)
                    fig6 = px.bar(aov, x='Company Name', y='Average Order Value', title="Avg Order Value by Company",
                                  color=aov['Average Order Value'], color_discrete_sequence=colors)
                    fig6.update_layout(**chart_config)
                    col6.plotly_chart(fig6, use_container_width=True, config=plotly_config, key="chart_mqt_6")
                with col7:
                    monthly_sales = completed.set_index('Date').resample('ME').size().reset_index(name='Sales Count')
                    fig7 = px.line(monthly_sales, x='Date', y='Sales Count', title="Monthly Sales Volume")
                    fig7.update_layout(**chart_config)
                    col7.plotly_chart(fig7, use_container_width=True, config=plotly_config, key="chart_mqt_7")
            else:
                st.info("No data available.")
        with tabs[3]:  # Losses & Returns
            st.header("Losses & Returns")
            if not filtered.empty:
                # Statistical Summary for Losses & Returns
                total_loss = completed['Loss from Returns'].sum()
                top_loss_product = completed.groupby('Product Name')['Loss from Returns'].sum().nlargest(1).reset_index().iloc[0]
                avg_return_rate = (completed.groupby('Product Name')['Loss from Returns'].sum() / completed.groupby('Product Name')['Total Revenue'].sum()).mean() * 100

                summary_text = (
                    f"<b>Losses & Returns Summary:</b><br>"
                    f"Total Loss: ${total_loss:,.0f}. "
                    f"Top Loss Product: {top_loss_product['Product Name']} (${top_loss_product['Loss from Returns']:,.0f}). "
                    f"Avg Return Rate: {avg_return_rate:.1f}%."
                )
                st.markdown(f"<div class='summary-card'>{summary_text}</div>", unsafe_allow_html=True)

                col1, col2, col3 = st.columns(3)
                losses = completed.groupby('Product Name')['Loss from Returns'].sum().nlargest(10).reset_index()
                top_loss_product = losses.iloc[0]
                col1.metric("Biggest Loss Product", top_loss_product['Product Name'], f"${top_loss_product['Loss from Returns']:,.0f}")
                fig1 = px.pie(losses, names='Product Name', values='Loss from Returns', title="Top Loss Products")
                fig1.update_layout(**chart_config)
                col1.plotly_chart(fig1, use_container_width=True, config=plotly_config, key="chart_lr_1")
                campaign_losses = completed.groupby('Campaign Name')['Loss from Returns'].sum().reset_index()
                top_loss_campaign = campaign_losses.sort_values('Loss from Returns', ascending=False).iloc[0]
                col2.metric("Top Loss Campaign", top_loss_campaign['Campaign Name'], f"${top_loss_campaign['Loss from Returns']:,.0f}")
                fig2 = px.bar(campaign_losses, x='Campaign Name', y='Loss from Returns', title="Losses by Campaign",
                              color_discrete_sequence=[RED])
                fig2.update_layout(**chart_config)
                col2.plotly_chart(fig2, use_container_width=True, config=plotly_config, key="chart_lr_2")
                return_rate = completed.groupby('Product Name')['Loss from Returns'].sum() / completed.groupby('Product Name')['Total Revenue'].sum()
                return_rate = return_rate.reset_index().rename(columns={0: 'Return Rate'}).sort_values(by='Return Rate', ascending=False).head(10)
                top_return_rate = return_rate.iloc[0]
                col3.metric("Highest Return Rate", top_return_rate['Product Name'], f"{top_return_rate['Return Rate']*100:.1f}%")
                fig3 = px.bar(return_rate, x='Product Name', y='Return Rate', title="Return Rate by Product")
                fig3.update_layout(**chart_config)
                col3.plotly_chart(fig3, use_container_width=True, config=plotly_config, key="chart_lr_3")

                col4, col5, col6, col7 = st.columns(4)
                with col4:
                    comp_rev = completed.groupby('Company Name')['Total Revenue'].sum().reset_index()
                    mean_rev = comp_rev['Total Revenue'].mean()
                    threshold_high = mean_rev * 1.1
                    threshold_low = mean_rev * 0.9
                    colors = assign_performance_colors(comp_rev['Total Revenue'], threshold_high, threshold_low)
                    fig4 = px.bar(comp_rev, x='Company Name', y='Total Revenue', title="Revenue by Company",
                                  color=comp_rev['Total Revenue'], color_discrete_sequence=colors)
                    fig4.update_layout(**chart_config)
                    col4.plotly_chart(fig4, use_container_width=True, config=plotly_config, key="chart_lr_4")
                with col5:
                    losses = completed.groupby('Product Name')['Loss from Returns'].sum().nlargest(10).reset_index()
                    fig5 = px.pie(losses, names='Product Name', values='Loss from Returns', title="Loss from Returns by Product",
                                  color_discrete_sequence=[RED])
                    fig5.update_layout(**chart_config)
                    col5.plotly_chart(fig5, use_container_width=True, config=plotly_config, key="chart_lr_5")
                with col6:
                    top_cust = completed.groupby('Company Name')['Total Revenue'].sum().nlargest(5).reset_index()
                    mean_rev = top_cust['Total Revenue'].mean()
                    threshold_high = mean_rev * 1.1
                    threshold_low = mean_rev * 0.9
                    colors = assign_performance_colors(top_cust['Total Revenue'], threshold_high, threshold_low)
                    fig6 = px.bar(top_cust, x='Company Name', y='Total Revenue', title="Top Customers by Revenue",
                                  color=top_cust['Total Revenue'], color_discrete_sequence=colors)
                    fig6.update_layout(**chart_config)
                    col6.plotly_chart(fig6, use_container_width=True, config=plotly_config, key="chart_lr_6")
                with col7:
                    return_rate = completed.groupby('Product Name')['Loss from Returns'].sum() / completed.groupby('Product Name')['Total Revenue'].sum()
                    return_rate = return_rate.reset_index().rename(columns={0: 'Return Rate'}).sort_values(by='Return Rate', ascending=False).head(10)
                    fig7 = px.bar(return_rate, x='Product Name', y='Return Rate', title="Return Rate by Product")
                    fig7.update_layout(**chart_config)
                    col7.plotly_chart(fig7, use_container_width=True, config=plotly_config, key="chart_lr_7")
            else:
                st.info("No data available.")
    else:
        st.markdown('<div class="team-charts">', unsafe_allow_html=True)
        st.header(f"{st.session_state.team} Dashboard")
        if not filtered.empty:
            # Statistical Summary for Team-Specific Dashboards
            if st.session_state.team == "Marketing Analyst":
                camp_type_rev = completed.groupby('Campaign Type')['Total Revenue'].sum().reset_index()
                top_campaign_type = camp_type_rev.sort_values('Total Revenue', ascending=False).iloc[0] if not camp_type_rev.empty else None
                avg_ltv = completed['Customer Lifetime Value'].mean()
                return_rate = (completed.groupby('Returning Customer').size().reset_index(name='Count').set_index('Returning Customer').loc['Yes']['Count'] / len(completed) * 100) if 'Yes' in completed['Returning Customer'].values else 0.0
                summary_text = (
                    f"<b>Marketing Analyst Summary:</b><br>"
                    f"Top Campaign Type: {top_campaign_type['Campaign Type'] if top_campaign_type is not None else 'N/A'} (${top_campaign_type['Total Revenue']:,.0f}). "
                    f"Avg LTV: ${avg_ltv:,.0f}. "
                    f"Returning Customer Rate: {return_rate:.1f}%."
                )
            elif st.session_state.team == "Marketing Manager":
                geo = completed.groupby('Country')['Total Revenue'].sum().reset_index()
                top_country = geo.sort_values('Total Revenue', ascending=False).iloc[0] if not geo.empty else None
                camp = completed.groupby('Campaign Name')['Total Revenue'].sum().nlargest(1).reset_index()
                top_campaign = camp.iloc[0] if not camp.empty else None
                summary_text = (
                    f"<b>Marketing Manager Summary:</b><br>"
                    f"Top Country: {top_country['Country'] if top_country is not None else 'N/A'} (${top_country['Total Revenue']:,.0f}). "
                    f"Top Campaign: {top_campaign['Campaign Name'] if top_campaign is not None else 'N/A'} (${top_campaign['Total Revenue']:,.0f})."
                )
            elif st.session_state.team == "Sales Data Analyst":
                avg_profit = completed['Profit Per Sale'].mean()
                avg_aov = completed['Average Order Value'].mean()
                completed_sales = len(completed)
                summary_text = (
                    f"<b>Sales Data Analyst Summary:</b><br>"
                    f"Avg Profit/Sale: ${avg_profit:,.0f}. "
                    f"Avg Order Value: ${avg_aov:,.0f}. "
                    f"Completed Sales: {completed_sales:,}."
                )
            elif st.session_state.team == "Sales Manager":
                total_loss = completed['Loss from Returns'].sum()
                top_cust = completed.groupby('Company Name')['Total Revenue'].sum().nlargest(1).reset_index().iloc[0]
                avg_return_rate = (completed.groupby('Product Name')['Loss from Returns'].sum() / completed.groupby('Product Name')['Total Revenue'].sum()).mean() * 100
                summary_text = (
                    f"<b>Sales Manager Summary:</b><br>"
                    f"Total Loss: ${total_loss:,.0f}. "
                    f"Top Customer: {top_cust['Company Name']} (${top_cust['Total Revenue']:,.0f}). "
                    f"Avg Return Rate: {avg_return_rate:.1f}%."
                )
            st.markdown(f"<div class='summary-card'>{summary_text}</div>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)
            if st.session_state.team == "Marketing Analyst":
                with col1:
                    camp_type_rev = completed.groupby('Campaign Type')['Total Revenue'].sum().reset_index()
                    mean_rev = camp_type_rev['Total Revenue'].mean()
                    threshold_high = mean_rev * 1.1
                    threshold_low = mean_rev * 0.9
                    colors = assign_performance_colors(camp_type_rev['Total Revenue'], threshold_high, threshold_low)
                    fig = px.bar(camp_type_rev, x='Campaign Type', y='Total Revenue', title="Campaign Revenue by Type",
                                 color=camp_type_rev['Total Revenue'], color_discrete_sequence=colors)
                    fig.update_layout(**chart_config)
                    st.plotly_chart(fig, use_container_width=True, config=plotly_config, key="chart_team_ma_1")
                with col2:
                    ltv = completed.groupby('Company Name')['Customer Lifetime Value'].sum().reset_index()
                    mean_ltv = ltv['Customer Lifetime Value'].mean()
                    threshold_high_ltv = mean_ltv * 1.1
                    threshold_low_ltv = mean_ltv * 0.9
                    colors = assign_performance_colors(ltv['Customer Lifetime Value'], threshold_high_ltv, threshold_low_ltv)
                    fig = px.histogram(ltv, x='Company Name', y='Customer Lifetime Value', title="Customer Lifetime Value",
                                       color=ltv['Customer Lifetime Value'], color_discrete_sequence=colors)
                    fig.update_layout(**chart_config)
                    st.plotly_chart(fig, use_container_width=True, config=plotly_config, key="chart_team_ma_2")
                with col3:
                    returning = completed.groupby('Returning Customer').size().reset_index(name='Count')
                    fig = px.pie(returning, names='Returning Customer', values='Count', title="Returning Customers")
                    fig.update_layout(**chart_config)
                    st.plotly_chart(fig, use_container_width=True, config=plotly_config, key="chart_team_ma_3")
                with col4:
                    prod_camp = completed.groupby(['Campaign Name', 'Product Name'])['Total Revenue'].sum().nlargest(5).reset_index()
                    mean_rev = prod_camp['Total Revenue'].mean()
                    threshold_high = mean_rev * 1.1
                    threshold_low = mean_rev * 0.9
                    colors = assign_performance_colors(prod_camp['Total Revenue'], threshold_high, threshold_low)
                    fig = px.bar(prod_camp, x='Product Name', y='Total Revenue', color='Campaign Name', title="Top Products by Campaign",
                                 color_discrete_sequence=colors)
                    fig.update_layout(**chart_config)
                    st.plotly_chart(fig, use_container_width=True, config=plotly_config, key="chart_team_ma_4")
            elif st.session_state.team == "Marketing Manager":
                with col1:
                    geo = completed.groupby('Country')['Total Revenue'].sum().reset_index()
                    mean_rev = geo['Total Revenue'].mean()
                    threshold_high = mean_rev * 1.1
                    threshold_low = mean_rev * 0.9
                    colors = assign_performance_colors(geo['Total Revenue'], threshold_high, threshold_low)
                    fig = px.choropleth(geo, locations='Country', locationmode='country names', color='Total Revenue', title="Revenue by Country",
                                        color_continuous_scale=[RED, ORANGE, GREEN])
                    fig.update_layout(**chart_config, geo=dict(showframe=False, projection_type='equirectangular'))
                    st.plotly_chart(fig, use_container_width=True, config=plotly_config, key="chart_team_mm_1")
                with col2:
                    camp = completed.groupby('Campaign Name')['Total Revenue'].sum().nlargest(10).reset_index()
                    mean_rev = camp['Total Revenue'].mean()
                    threshold_high = mean_rev * 1.1
                    threshold_low = mean_rev * 0.9
                    colors = assign_performance_colors(camp['Total Revenue'], threshold_high, threshold_low)
                    fig = px.bar(camp, x='Campaign Name', y='Total Revenue', title="Top Campaigns",
                                 color=camp['Total Revenue'], color_discrete_sequence=colors)
                    fig.update_layout(**chart_config)
                    st.plotly_chart(fig, use_container_width=True, config=plotly_config, key="chart_team_mm_2")
                with col3:
                    camp_trend = completed.groupby([pd.Grouper(key='Date', freq='ME'), 'Campaign Name'])['Total Revenue'].sum().reset_index()
                    mean_rev = camp_trend['Total Revenue'].mean()
                    threshold_high = mean_rev * 1.1
                    threshold_low = mean_rev * 0.9
                    colors = assign_performance_colors(camp_trend['Total Revenue'], threshold_high, threshold_low)
                    fig = px.line(camp_trend, x='Date', y='Total Revenue', color='Campaign Name', title="Monthly Campaign Revenue",
                                  color_discrete_sequence=colors)
                    fig.update_layout(**chart_config)
                    st.plotly_chart(fig, use_container_width=True, config=plotly_config, key="chart_team_mm_3")
                with col4:
                    cat_rev = completed.groupby('Category')['Total Revenue'].sum().reset_index()
                    mean_rev = cat_rev['Total Revenue'].mean()
                    threshold_high = mean_rev * 1.1
                    threshold_low = mean_rev * 0.9
                    colors = assign_performance_colors(cat_rev['Total Revenue'], threshold_high, threshold_low)
                    fig = px.pie(cat_rev, names='Category', values='Total Revenue', title="Revenue by Category",
                                 color=cat_rev['Total Revenue'], color_discrete_sequence=colors)
                    fig.update_layout(**chart_config)
                    st.plotly_chart(fig, use_container_width=True, config=plotly_config, key="chart_team_mm_4")
            elif st.session_state.team == "Sales Data Analyst":
                with col1:
                    profit_prod = completed.groupby('Product Name')['Profit Per Sale'].mean().reset_index()
                    mean_profit = profit_prod['Profit Per Sale'].mean()
                    threshold_high = mean_profit * 1.1
                    threshold_low = mean_profit * 0.9
                    colors = assign_performance_colors(profit_prod['Profit Per Sale'], threshold_high, threshold_low)
                    fig = px.bar(profit_prod, x='Product Name', y='Profit Per Sale', title="Profit per Sale by Product",
                                 color=profit_prod['Profit Per Sale'], color_discrete_sequence=colors)
                    fig.update_layout(**chart_config)
                    st.plotly_chart(fig, use_container_width=True, config=plotly_config, key="chart_team_sda_1")
                with col2:
                    status_counts = filtered.groupby('Sale Status').size().reset_index(name='Count')
                    fig = px.pie(status_counts, names='Sale Status', values='Count', title="Sale Status Distribution")
                    fig.update_layout(**chart_config)
                    st.plotly_chart(fig, use_container_width=True, config=plotly_config, key="chart_team_sda_2")
                with col3:
                    aov = completed.groupby('Company Name')['Average Order Value'].mean().reset_index()
                    mean_aov = aov['Average Order Value'].mean()
                    threshold_high = mean_aov * 1.1
                    threshold_low = mean_aov * 0.9
                    colors = assign_performance_colors(aov['Average Order Value'], threshold_high, threshold_low)
                    fig = px.bar(aov, x='Company Name', y='Average Order Value', title="Avg Order Value by Company",
                                 color=aov['Average Order Value'], color_discrete_sequence=colors)
                    fig.update_layout(**chart_config)
                    st.plotly_chart(fig, use_container_width=True, config=plotly_config, key="chart_team_sda_3")
                with col4:
                    monthly_sales = completed.set_index('Date').resample('ME').size().reset_index(name='Sales Count')
                    fig = px.line(monthly_sales, x='Date', y='Sales Count', title="Monthly Sales Volume")
                    fig.update_layout(**chart_config)
                    st.plotly_chart(fig, use_container_width=True, config=plotly_config, key="chart_team_sda_4")
            elif st.session_state.team == "Sales Manager":
                with col1:
                    comp_rev = completed.groupby('Company Name')['Total Revenue'].sum().reset_index()
                    mean_rev = comp_rev['Total Revenue'].mean()
                    threshold_high = mean_rev * 1.1
                    threshold_low = mean_rev * 0.9
                    colors = assign_performance_colors(comp_rev['Total Revenue'], threshold_high, threshold_low)
                    fig = px.bar(comp_rev, x='Company Name', y='Total Revenue', title="Revenue by Company",
                                 color=comp_rev['Total Revenue'], color_discrete_sequence=colors)
                    fig.update_layout(**chart_config)
                    st.plotly_chart(fig, use_container_width=True, config=plotly_config, key="chart_team_sm_1")
                with col2:
                    losses = completed.groupby('Product Name')['Loss from Returns'].sum().nlargest(10).reset_index()
                    fig = px.pie(losses, names='Product Name', values='Loss from Returns', title="Loss from Returns by Product",
                                 color_discrete_sequence=[RED])
                    fig.update_layout(**chart_config)
                    st.plotly_chart(fig, use_container_width=True, config=plotly_config, key="chart_team_sm_2")
                with col3:
                    top_cust = completed.groupby('Company Name')['Total Revenue'].sum().nlargest(5).reset_index()
                    mean_rev = top_cust['Total Revenue'].mean()
                    threshold_high = mean_rev * 1.1
                    threshold_low = mean_rev * 0.9
                    colors = assign_performance_colors(top_cust['Total Revenue'], threshold_high, threshold_low)
                    fig = px.bar(top_cust, x='Company Name', y='Total Revenue', title="Top Customers by Revenue",
                                 color=top_cust['Total Revenue'], color_discrete_sequence=colors)
                    fig.update_layout(**chart_config)
                    st.plotly_chart(fig, use_container_width=True, config=plotly_config, key="chart_team_sm_3")
                with col4:
                    return_rate = completed.groupby('Product Name')['Loss from Returns'].sum() / completed.groupby('Product Name')['Total Revenue'].sum()
                    return_rate = return_rate.reset_index().rename(columns={0: 'Return Rate'}).sort_values(by='Return Rate', ascending=False).head(10)
                    fig = px.bar(return_rate, x='Product Name', y='Return Rate', title="Return Rate by Product")
                    fig.update_layout(**chart_config)
                    st.plotly_chart(fig, use_container_width=True, config=plotly_config, key="chart_team_sm_4")

            st.markdown("<h3 style='font-size: 12px; margin: 3px 0;'>Target Performance</h3>", unsafe_allow_html=True)
            target_dict = {
                "Marketing Analyst": 500000,
                "Marketing Manager": 1500000,
                "Sales Data Analyst": 750000,
                "Sales Manager": 2000000
            }
            target = target_dict[st.session_state.team]
            actual = completed['Total Revenue'].sum()
            num_sales = len(completed)
            avg_order_value = completed['Average Order Value'].mean() if not completed.empty else 0
            status, status_class, _ = get_target_status(actual, target)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('<div class="target-metrics">', unsafe_allow_html=True)
                st.metric("Number of Sales", f"{num_sales:,}")
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="target-metrics">', unsafe_allow_html=True)
                st.markdown(f'<div class="{status_class}">', unsafe_allow_html=True)
                st.metric("Target Reached Status", status)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="target-metrics">', unsafe_allow_html=True)
                st.metric("Avg Order Value", f"${avg_order_value:,.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No data available for selected filters.")
        st.markdown('</div>', unsafe_allow_html=True)
