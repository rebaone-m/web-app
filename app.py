import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import random
import numpy as np
from datetime import datetime, timedelta
import pycountry_convert as pc
import pycountry
from faker import Faker

# Initialize Faker
fake = Faker()

# --- Streamlit Config ---
st.set_page_config(page_title="Company Sales Dashboard", layout="wide", initial_sidebar_state="expanded")


# --- Data Generation Functions ---
def get_continent(country_name):
    try:
        country_code = pc.country_name_to_country_alpha2(country_name)
        continent_code = pc.country_alpha2_to_continent_code(country_code)
        continent_name = {
            "AF": "Africa",
            "NA": "North America",
            "OC": "Oceania",
            "AN": "Antarctica",
            "AS": "Asia",
            "EU": "Europe",
            "SA": "South America"
        }
        return continent_name.get(continent_code, "Unknown")
    except:
        return "Unknown"

# Static data
countries = [country.name for country in pycountry.countries]
country_continent_map = {country: get_continent(country) for country in countries}
companies = [
    "TechWave Corp", "InnovateX Solutions", "SkyNet Technologies", "NeoTech Enterprises",
    "DataMinds Inc.", "AI Global", "FutureSoft Systems", "IntelliBots Ltd.",
    "Visionary Dynamics", "OmniAI Technologies"
]
products = [
    (301, 'AI Assistant', 'AI Products'),
    (302, 'Automation Solutions', 'AI Products'),
    (303, 'AI Phishing Detector', 'AI Products'),
    (304, 'Prototyping Tools', 'Prototype Products'),
    (305, 'Digital Solutions', 'Demo Products')
]
request_categories = [
    (101, 'Job Placement'),
    (102, 'Schedule Demo'),
    (103, 'AI Assistant Inquiry'),
    (104, 'Other')
]
campaigns = [
    ("Summer Blast", "Email"),
    ("Innovation Wave", "Social Media"),
    ("AI Awareness", "Paid Ads"),
    ("Prototype Launch", "Referral"),
    ("Global Reach", "Organic")
]
sale_statuses = ['Completed', 'Pending', 'Cancelled']
conversion_sources = ['Campaign Organic', 'Paid Ads', 'Referral', 'Social Media']
http_methods = ['GET', 'POST', 'PUT', 'DELETE']
resources = ['/index.html', '/scheduledemo', '/page54.html', '/event', '/submitjob', '/aiassistant']
status_codes = [200, 304, 404, 500]
device_types = ['Mobile', 'Desktop', 'Tablet']

# Time and date generators
def generate_timestamp(continent):
    if continent == 'North America':
        peak_hour = 13
    elif continent == 'Europe':
        peak_hour = 14
    else:
        peak_hour = 12
    hour = int(np.random.normal(peak_hour, 4)) % 24
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return datetime.strptime(f"{hour:02d}:{minute:02d}:{second:02d}", '%H:%M:%S').time()

def generate_date():
    start_date = datetime(2021, 1, 1)
    end_date = datetime.now()
    month_weights = [0.05, 0.05, 0.06, 0.06, 0.07, 0.07, 0.08, 0.08, 0.09, 0.12, 0.13, 0.14]
    month = random.choices(range(1, 13), weights=month_weights, k=1)[0]
    if month == 2:
        max_day = 28
    elif month in [4, 6, 9, 11]:
        max_day = 30
    else:
        max_day = 31
    day = random.randint(1, max_day)
    year = random.randint(start_date.year, end_date.year)
    date = datetime(year, month, day)
    if date < start_date or date > end_date:
        delta = end_date - start_date
        days_offset = random.randint(0, delta.days)
        date = start_date + timedelta(days=days_offset)
    return date

# --- Load Data ---
@st.cache_data
def load_data():
    data = []
    num_rows = 10_000  #  adjust as needed

    for _ in range(num_rows):
        country = random.choices(
            countries,
            weights=[0.4 if country_continent_map[c] in ['North America', 'Europe'] else 0.1 for c in countries],
            k=1
        )[0]
        continent = country_continent_map.get(country, "Unknown")
        company = random.choice(companies)
        campaign_name, campaign_type = random.choice(campaigns)
        product_id, product_name, category = random.choice(products)
        request_cat_id, request_cat = random.choice(request_categories)
        device = random.choice(device_types)
        session_duration = round(np.random.lognormal(mean=1.5 if device != 'Mobile' else 1.2, sigma=0.8), 2)
        session_duration = min(session_duration, 30)
        pages_visited = max(1, min(int(np.random.normal(session_duration * 0.8, 2)) + 1, 20))
        returning_customer = random.choices(['Yes', 'No'], weights=[0.878, 0.122], k=1)[0]
        bounce_rate = round(max(10, 80 - session_duration * 5 - (10 if returning_customer == 'Yes' else 0) + np.random.normal(0, 10)), 2)
        bounce_rate = min(bounce_rate, 100)
        subscription_status = random.choice(['Active', 'Inactive'])
        number_of_products_viewed = random.randint(1, min(5, pages_visited))
        purchase_intention_base = np.random.exponential(20)
        purchase_intention_score = round(purchase_intention_base * (1.2 if continent in ['North America', 'Europe'] else 0.9), 2)
        purchase_intention_score = min(purchase_intention_score, 100)
        revenue_multiplier = 1.5 if category == 'AI Products' else 1.0
        revenue_per_product = round(np.random.lognormal(mean=8, sigma=0.5) * revenue_multiplier, 2)
        if random.random() < 0.01:
            revenue_per_product *= 5
        sale_status = random.choices(
            sale_statuses,
            weights=[0.8 if purchase_intention_score > 50 else 0.5, 0.15, 0.05],
            k=1
        )[0]
        discount_applied = random.choice(['Yes', 'No'])
        if campaign_type == 'Social Media':
            conversion_source = random.choices(conversion_sources, weights=[0.1, 0.2, 0.2, 0.5], k=1)[0]
        else:
            conversion_source = random.choice(conversion_sources)
        total_revenue = revenue_per_product * random.randint(1, 5)
        if company == 'TechWave Corp':
            total_revenue *= 1.3
        profit_per_sale = round(total_revenue * random.uniform(0.15, 0.35), 2)
        return_rate = 0.03 if product_name == 'Prototyping Tools' else 0.02
        loss_from_returns = round(total_revenue * np.random.exponential(return_rate), 2)
        loss_from_returns = min(loss_from_returns, total_revenue * 0.075)
        revenue_per_customer_segment = round(total_revenue * random.uniform(0.9, 1.1), 2)
        average_order_value = round(total_revenue / random.randint(1, 5), 2)
        customer_lifetime_value = round(average_order_value * np.random.lognormal(mean=2, sigma=0.5), 2)

        row = {
            'Date': generate_date(),
            'Timestamp': generate_timestamp(continent),
            'IP Address': fake.ipv4(),
            'HTTP Method': random.choices(http_methods, weights=[0.7, 0.2, 0.05, 0.05], k=1)[0],
            'Requested Resource': random.choice(resources),
            'Response Code': random.choices(status_codes, weights=[0.85, 0.1, 0.04, 0.01], k=1)[0],
            'Country': country,
            'Continent': continent,
            'Company Name': company,
            'Session Duration (min)': session_duration,
            'Pages Visited': pages_visited,
            'Bounce Rate (%)': bounce_rate,
            'Subscription Status': subscription_status,
            'Returning Customer': returning_customer,
            'Campaign Name': campaign_name,
            'Campaign Type': campaign_type,
            'Product ID': product_id,
            'Product Name': product_name,
            'Category': category,
            'Number of Products Viewed': number_of_products_viewed,
            'Purchase Intention Score': purchase_intention_score,
            'Sale Status': sale_status,
            'Profit Per Sale': profit_per_sale,
            'Loss from Returns': loss_from_returns,
            'Discount Applied': discount_applied,
            'Conversion Source': conversion_source,
            'Total Revenue': total_revenue,
            'Revenue Per Product': revenue_per_product,
            'Revenue Per Customer Segment': revenue_per_customer_segment,
            'Average Order Value': average_order_value,
            'Customer Lifetime Value': customer_lifetime_value
        }
        data.append(row)

    df = pd.DataFrame(data)
    # Ensure data types
    string_columns = [
        'Country', 'Continent', 'Company Name', 'Subscription Status',
        'Returning Customer', 'Campaign Name', 'Campaign Type',
        'Product Name', 'Category', 'Sale Status', 'Discount Applied', 'Conversion Source',
        'IP Address', 'HTTP Method', 'Requested Resource'
    ]
    for col in string_columns:
        df[col] = df[col].astype(str)
    numeric_columns = [
        'Total Revenue', 'Profit Per Sale', 'Loss from Returns',
        'Revenue Per Product', 'Customer Lifetime Value', 'Average Order Value',
        'Session Duration (min)', 'Pages Visited', 'Bounce Rate (%)',
        'Number of Products Viewed', 'Purchase Intention Score'
    ]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'])
    df['Response Code'] = df['Response Code'].astype(int)
    df['Product ID'] = df['Product ID'].astype(int)
    return df

df = load_data()

# CSS to remove top space above sidebar and constrain team member charts
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] div:first-child {
        padding-top: 0rem;
    }
    .team-charts {
        max-height: 550px !important;
        overflow: hidden !important;
        padding: 5px !important;
    }
    .team-charts .stMetric {
        font-size: 12px !important;
    }
    .team-charts .stMetric > div:first-child {
        font-size: 12px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

filtered = df.copy()

# --- Sidebar Filters ---
# Instructions Button (moved to top)
if st.sidebar.button("Instructions"):
    st.session_state.page = "Instructions"
    st.rerun()

# Team member filter
if "team" not in st.session_state:
    st.session_state.team = "Sales & Marketing Team"
team_options = [
    "Sales & Marketing Team", "Marketing Analyst", "Marketing Manager",
    "Sales Data Analyst", "Sales Manager"
]
st.session_state.team = st.sidebar.selectbox("Team Member", team_options, index=0)

# Categorical filters
for col in ['Country', 'Product Name', 'Company Name', 'Sale Status']:
    options = sorted(df[col].dropna().unique())
    selected = st.sidebar.multiselect(col, options, default=[])
    if not selected:
        selected = options
    filtered = filtered[filtered[col].isin(selected)]

# Date range filter
date_min, date_max = df['Date'].min(), df['Date'].max()
start_date, end_date = st.sidebar.date_input(
    "Date Range",
    [date_min.date(), date_max.date()],
    min_value=date_min.date(),
    max_value=date_max.date()
)
start = pd.to_datetime(start_date)
end = pd.to_datetime(end_date)
filtered = filtered[(filtered['Date'] >= start) & (filtered['Date'] <= end)]

# --- Download ---
st.sidebar.markdown("---")
@st.cache_data
def convert_csv(data): return data.to_csv(index=False).encode('utf-8')
st.sidebar.download_button("Download CSV", convert_csv(filtered), "filtered.csv", "text/csv")

# --- Compute Metrics ---
completed = filtered[filtered['Sale Status'].str.lower() == 'completed']


# --- Custom CSS ---
st.markdown(
    """
    <style>
    #MainMenu, footer, header {
        visibility: hidden;
        height: 0;
    }
    section [data-testid="stSidebar"] div:first-child {
        padding-top: 0rem;
        width: 320px !important;
        background-color: #f0f2f6;
        border-right: 1px solid #e6e6e6;
    }
    html, body, [data-testid="stAppViewContainer"], main {
        overflow: hidden !important;
        height: 100% !important;
    }
    .block-container {
        padding-top: 0rem !important;
        margin-top: -1rem !important;
    }
    .element-container {
        padding-top: 0rem !important;
        margin-top: 0rem !important;
    }
    </style>
    """, unsafe_allow_html=True
)

# --- Instructions Page ---
if st.session_state.get('page') == 'Instructions':
    st.markdown("<h4>👋 Welcome to the Dashboard Instructions</h4>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='font-size: 14px'>
            This guide will help you navigate and understand how to use the dashboard effectively.
            <h5>✅ Visual Content</h5>
            - Company <b>logos</b>, <b>charts</b>, and <b>interactive graphs</b> give you insights at a glance.<br>
            - <b>Top customers</b>, <b>campaigns</b>, and <b>returns</b> are color-coded for clarity.
            <h5>✅ System Usage</h5>
            - <b>Sidebar Filters</b> let you slice data by product, country, or time.<br>
            - Use the <i>tabs</i> above to explore different categories like <i>Customer Insights</i>, <i>Geographical Trends</i>, <i>Campaigns</i>, and <i>Losses</i>.
            <h5>✅ Help & Recovery</h5>
            - If a section shows <b>"No data available"</b>, adjust your filters for more results.<br>
            - Hover over graphs for more detail.<br>
            - Most charts and metrics auto-refresh based on the filters.
            <h5>🔁 Quick Tips</h5>
            - Click Download to export filtered data.<br>
            - Charts resize automatically to fit your screen.<br>
            - You can always go back to the main view.
        </div>
        <style>
            .small-back-button button {
                font-size: 12px !important;
                padding: 4px 8px !important;
                line-height: 1.2 !important;
                height: auto !important;
                width: auto !important;
                min-width: 100px !important;
                margin-top: 10px !important;
                background-color: #f0f2f6 !important;
                border: 1px solid #e6e6e6 !important;
                border-radius: 4px !important;
            }
            .small-back-button button:hover {
                background-color: #e0e2e6 !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    if st.button("Back", key="small_back_button", help="Return to Overview"):
        st.session_state.page = "overview"
        st.rerun()

# --- Main Content ---
else:
    if st.session_state.team == "Sales & Marketing Team":
        tabs = st.tabs([
            "Overview", "Customer Insights", "Geographical Insights",
            "Monthly/Quarterly Trends", "Campaign Analysis", "Losses & Returns"
        ])
        with tabs[0]:
            if 'page' not in st.session_state:
                st.session_state.page = 'overview'
            if st.session_state.page == 'overview':
                st.markdown("""
                <style>
                    .metric-box, .insight-box {
                        padding: 4px 6px;
                        background-color: #f9f9f9;
                        border-radius: 6px;
                        font-size: 10px !important;
                        margin-bottom: 2px;
                        word-wrap: break-word;
                        white-space: normal;
                        max-width: 100%;
                        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
                    }
                    .stMetric {
                        font-size: 11px !important;
                    }
                    h1, h2, h3, .stSubheader {
                        font-size: 12px !important;
                        margin: 2px 0 2px 0 !important;
                    }
                    .stMarkdown, .stText, .stSubheader div {
                        line-height: 1.2;
                        margin: 0 !important;
                        padding: 0 !important;
                    }
                    .element-container {
                        margin: 0 !important;
                        padding: 0 !important;
                    }
                    .stColumn {
                        padding: 0 5px !important;
                    }
                </style>
                """, unsafe_allow_html=True)
                st.markdown("### Business Performance")
                if not filtered.empty:
                    total_revenue = filtered['Total Revenue'].sum()
                    total_profit = filtered['Profit Per Sale'].sum()
                    total_loss = filtered['Loss from Returns'].sum()
                    total_orders = len(filtered)
                    avg_order_value = filtered['Average Order Value'].mean()
                    avg_clv = filtered['Customer Lifetime Value'].mean()
                    product_counts = filtered['Product Name'].value_counts()
                    most_purchased = product_counts.head(2)
                    least_purchased = product_counts.tail(2)
                    country_revenue = filtered.groupby('Country')['Total Revenue'].sum().sort_values(ascending=False).head(2)
                    company_revenue = filtered.groupby('Company Name')['Total Revenue'].sum().sort_values(ascending=False).head(2)
                    col1, col2, col3, col4 = st.columns([3, 3, 3, 1])
                    with col1:
                        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                        st.metric("Revenue", f"${total_revenue:,.0f}")
                        st.metric("Profit", f"${total_profit:,.0f}")
                        st.metric("Loss", f"${total_loss:,.0f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                        st.metric("Avg Order", f"${avg_order_value:,.0f}")
                        st.metric("Orders", f"{total_orders:,}")
                        st.metric("Avg CLV", f"${avg_clv:,.0f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col3:
                        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                        st.markdown("**Top Products**")
                        for p, c in most_purchased.items():
                            st.markdown(f"- {p} ({c})")
                        st.markdown("**Low Products**")
                        for p, c in least_purchased.items():
                            st.markdown(f"- {p} ({c})")
                        st.markdown("**Top Country**")
                        for c, r in country_revenue.items():
                            st.markdown(f"- {c} (${r:,.0f})")
                        st.markdown("**Top Company**")
                        for c, r in company_revenue.items():
                            st.markdown(f"- {c} (${r:,.0f})")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col4:
                        st.write("")
                        st.write("")
                        if st.button("Sales Overview"):
                            st.session_state.page = 'sales_data'
                            st.rerun()
                else:
                    st.info("No data for selected filters.")
                if filtered.empty:
                    st.info("No data to display detailed sales status.")
            elif st.session_state.page == 'sales_data':
                st.title("Sales Data Detailed View")
                if not filtered.empty and "Sale Status" in filtered.columns:
                    status_counts = filtered.pivot_table(
                        index="Company Name",
                        columns="Sale Status",
                        values="Total Revenue",
                        aggfunc="count",
                        fill_value=0
                    )
                    status_icon_map = {
                        "Completed": "✅ Completed",
                        "Pending": "⏳ Pending",
                        "Cancelled": "❌ Cancelled",
                        "Returned": "🔄 Returned",
                        "Failed": "⚠️ Failed"
                    }
                    status_counts = status_counts.rename(columns=status_icon_map)
                    st.dataframe(status_counts.style.format("{:,}"), use_container_width=True)
                else:
                    st.info("No sales data available.")
                if st.button("Back to Overview"):
                    st.session_state.page = 'overview'
                    st.rerun()
        with tabs[1]:
            st.header("Customer Insights")
            if not filtered.empty:
                col1, col2, col3 = st.columns(3)
                loyal = completed.groupby('Company Name').size().nlargest(5).reset_index(name='Purchases')
                if not loyal.empty:
                    top_customer = loyal.iloc[0]
                    col1.metric("Top Customer", top_customer['Company Name'], f"{top_customer['Purchases']} Purchases")
                    fig1 = px.bar(loyal, x='Company Name', y='Purchases', title="Top Loyal Customers")
                    col1.plotly_chart(fig1, use_container_width=True)
                else:
                    col1.warning("No data available to show top customer.")
                lifetime_value = completed.groupby('Company Name')['Customer Lifetime Value'].sum().reset_index()
                if not lifetime_value.empty:
                    max_ltv = lifetime_value.sort_values('Customer Lifetime Value', ascending=False).iloc[0]
                    col2.metric("Highest LTV", max_ltv['Company Name'], f"${max_ltv['Customer Lifetime Value']:,.2f}")
                    fig2 = px.histogram(lifetime_value, x='Company Name', y='Customer Lifetime Value', title="Lifetime Value by Company")
                    col2.plotly_chart(fig2, use_container_width=True)
                else:
                    col2.warning("No data available for Lifetime Value.")
                returning_sales = completed.groupby('Returning Customer').size().reset_index(name='Sales Count')
                if not returning_sales.empty:
                    if 'Yes' in returning_sales['Returning Customer'].values:
                        return_pct = (returning_sales.set_index('Returning Customer').loc['Yes']['Sales Count'] / returning_sales['Sales Count'].sum()) * 100
                    else:
                        return_pct = 0.0
                    col3.metric("Returning Customer Rate", f"{return_pct:.1f}%", " ")
                    fig3 = px.pie(returning_sales, names='Returning Customer', values='Sales Count', title="Returning Customers")
                    col3.plotly_chart(fig3, use_container_width=True)
                else:
                    col3.warning("No data available for Returning Customers.")
            else:
                st.info("No data available.")
        with tabs[2]:
            st.header("Geographical Insights")
            if not filtered.empty:
                col1, col2, col3 = st.columns(3)
                geo = completed.groupby('Country')['Total Revenue'].sum().reset_index()
                top_country_rev = geo.sort_values('Total Revenue', ascending=False).iloc[0]
                col1.metric("Top Country by Revenue", top_country_rev['Country'], f"${top_country_rev['Total Revenue']:,.0f}")
                fig1 = px.choropleth(geo, locations='Country', locationmode='country names', color='Total Revenue', title="Revenue by Country")
                col1.plotly_chart(fig1, use_container_width=True)
                profit_per_sale = completed.groupby('Country')['Profit Per Sale'].mean().reset_index()
                top_country_profit = profit_per_sale.sort_values('Profit Per Sale', ascending=False).iloc[0]
                col2.metric("Top Country Profit/Sale", top_country_profit['Country'], f"${top_country_profit['Profit Per Sale']:,.2f}")
                fig2 = px.bar(profit_per_sale, x='Country', y='Profit Per Sale', title="Profit per Sale by Country")
                col2.plotly_chart(fig2, use_container_width=True)
                # Add a placeholder for the third column if needed
                with col3:
                    st.write("")  # Empty placeholder to maintain layout
            else:
                st.info("No data available.")
        with tabs[3]:
            st.header("Monthly/Quarterly Trends")
            if not filtered.empty:
                col1, col2, col3 = st.columns(3)
                monthly = completed.set_index('Date').resample('ME')['Total Revenue'].sum().reset_index()
                latest_month = monthly.iloc[-1]
                delta = latest_month['Total Revenue'] - monthly.iloc[-2]['Total Revenue'] if len(monthly) > 1 else 0
                col1.metric("Latest Monthly Revenue", f"${latest_month['Total Revenue']:,.0f}", f"${delta:,.0f}")
                fig1 = px.line(monthly, x='Date', y='Total Revenue', title="Monthly Revenue")
                col1.plotly_chart(fig1, use_container_width=True)
                quarterly = completed.set_index('Date').resample('QE')['Total Revenue'].sum().reset_index()
                quarterly['Quarter'] = quarterly['Date'].dt.to_period('Q').astype(str)
                q_latest = quarterly.iloc[-1]
                delta_q = q_latest['Total Revenue'] - quarterly.iloc[-2]['Total Revenue'] if len(quarterly) > 1 else 0
                col2.metric("Latest Quarter Revenue", f"${q_latest['Total Revenue']:,.0f}", f"${delta_q:,.0f}")
                fig2 = px.line(quarterly, x='Quarter', y='Total Revenue', title="Quarterly Revenue")
                col2.plotly_chart(fig2, use_container_width=True)
                monthly_avg_order_value = completed.set_index('Date').resample('ME')['Average Order Value'].mean().reset_index()
                latest_avg = monthly_avg_order_value.iloc[-1]
                delta_avg = latest_avg['Average Order Value'] - monthly_avg_order_value.iloc[-2]['Average Order Value'] if len(monthly_avg_order_value) > 1 else 0
                col3.metric("Latest Avg Order Value", f"${latest_avg['Average Order Value']:,.2f}", f"${delta_avg:,.2f}")
                fig3 = px.line(monthly_avg_order_value, x='Date', y='Average Order Value', title="Monthly Avg Order Value")
                col3.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("No data available.")
        with tabs[4]:
            st.header("Campaign Analysis")
            if not filtered.empty:
                col1 = st.columns(1)[0]
                camp = completed.groupby('Campaign Name')['Total Revenue'].sum().reset_index()
                top_campaign = camp.sort_values('Total Revenue', ascending=False).iloc[0]
                col1.metric("Top Campaign", top_campaign['Campaign Name'], f"${top_campaign['Total Revenue']:,.0f}")
                fig1 = px.bar(camp, x='Campaign Name', y='Total Revenue', title="Top Campaigns")
                col1.plotly_chart(fig1, use_container_width=True)
            else:
                st.info("No data available.")
        with tabs[5]:
            st.header("Losses & Returns")
            if not filtered.empty:
                col1, col2, col3 = st.columns(3)
                losses = completed.groupby('Product Name')['Loss from Returns'].sum().nlargest(10).reset_index()
                top_loss_product = losses.iloc[0]
                col1.metric("Biggest Loss Product", top_loss_product['Product Name'], f"${top_loss_product['Loss from Returns']:,.0f}")
                fig1 = px.pie(losses, names='Product Name', values='Loss from Returns', title="Top Loss Products")
                col1.plotly_chart(fig1, use_container_width=True)
                campaign_losses = completed.groupby('Campaign Name')['Loss from Returns'].sum().reset_index()
                top_loss_campaign = campaign_losses.sort_values('Loss from Returns', ascending=False).iloc[0]
                col2.metric("Top Loss Campaign", top_loss_campaign['Campaign Name'], f"${top_loss_campaign['Loss from Returns']:,.0f}")
                fig2 = px.bar(campaign_losses, x='Campaign Name', y='Loss from Returns', title="Losses by Campaign")
                col2.plotly_chart(fig2, use_container_width=True)
                return_rate = completed.groupby('Product Name')['Loss from Returns'].sum() / completed.groupby('Product Name')['Total Revenue'].sum()
                return_rate = return_rate.reset_index().rename(columns={0: 'Return Rate'}).sort_values(by='Return Rate', ascending=False).head(10)
                top_return_rate = return_rate.iloc[0]
                col3.metric("Highest Return Rate", top_return_rate['Product Name'], f"{top_return_rate['Return Rate']*100:.1f}%")
                fig3 = px.bar(return_rate, x='Product Name', y='Return Rate', title="Return Rate by Product")
                col3.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("No data available.")
    else:
        st.markdown('<div class="team-charts">', unsafe_allow_html=True)
        st.header(f"{st.session_state.team} Dashboard")
        if not filtered.empty:
            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)
            chart_config = dict(
                height=200,
                margin=dict(l=20, r=20, t=20, b=10),
                title_font_size=12,
                xaxis_tickfont_size=10,
                yaxis_tickfont_size=10,
                xaxis_title_font_size=12,
                yaxis_title_font_size=12
            )
            if st.session_state.team == "Marketing Analyst":
                with col1:
                    camp_type_rev = completed.groupby('Campaign Type')['Total Revenue'].sum().reset_index()
                    fig = px.bar(camp_type_rev, x='Campaign Type', y='Total Revenue', title="Campaign Revenue by Type")
                    fig.update_layout(**chart_config)
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    ltv = completed.groupby('Company Name')['Customer Lifetime Value'].sum().reset_index()
                    fig = px.histogram(ltv, x='Company Name', y='Customer Lifetime Value', title="Customer Lifetime Value")
                    fig.update_layout(**chart_config)
                    st.plotly_chart(fig, use_container_width=True)
                with col3:
                    returning = completed.groupby('Returning Customer').size().reset_index(name='Count')
                    fig = px.pie(returning, names='Returning Customer', values='Count', title="Returning Customers")
                    fig.update_layout(**chart_config)
                    st.plotly_chart(fig, use_container_width=True)
                with col4:
                    prod_camp = completed.groupby(['Campaign Name', 'Product Name'])['Total Revenue'].sum().nlargest(5).reset_index()
                    fig = px.bar(prod_camp, x='Product Name', y='Total Revenue', color='Campaign Name', title="Top Products by Campaign")
                    fig.update_layout(**chart_config)
                    st.plotly_chart(fig, use_container_width=True)
            elif st.session_state.team == "Marketing Manager":
                with col1:
                    geo = completed.groupby('Country')['Total Revenue'].sum().reset_index()
                    fig = px.choropleth(geo, locations='Country', locationmode='country names', color='Total Revenue', title="Revenue by Country")
                    fig.update_layout(**chart_config)
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    camp = completed.groupby('Campaign Name')['Total Revenue'].sum().nlargest(10).reset_index()
                    fig = px.bar(camp, x='Campaign Name', y='Total Revenue', title="Top Campaigns")
                    fig.update_layout(**chart_config)
                    st.plotly_chart(fig, use_container_width=True)
                with col3:
                    camp_trend = completed.groupby([pd.Grouper(key='Date', freq='ME'), 'Campaign Name'])['Total Revenue'].sum().reset_index()
                    fig = px.line(camp_trend, x='Date', y='Total Revenue', color='Campaign Name', title="Monthly Campaign Revenue")
                    fig.update_layout(**chart_config)
                    st.plotly_chart(fig, use_container_width=True)
                with col4:
                    cat_rev = completed.groupby('Category')['Total Revenue'].sum().reset_index()
                    fig = px.pie(cat_rev, names='Category', values='Total Revenue', title="Revenue by Category")
                    fig.update_layout(**chart_config)
                    st.plotly_chart(fig, use_container_width=True)
            elif st.session_state.team == "Sales Data Analyst":
                with col1:
                    profit_prod = completed.groupby('Product Name')['Profit Per Sale'].mean().reset_index()
                    fig = px.bar(profit_prod, x='Product Name', y='Profit Per Sale', title="Profit per Sale by Product")
                    fig.update_layout(**chart_config)
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    status_counts = filtered.groupby('Sale Status').size().reset_index(name='Count')
                    fig = px.pie(status_counts, names='Sale Status', values='Count', title="Sale Status Distribution")
                    fig.update_layout(**chart_config)
                    st.plotly_chart(fig, use_container_width=True)
                with col3:
                    aov = completed.groupby('Company Name')['Average Order Value'].mean().reset_index()
                    fig = px.bar(aov, x='Company Name', y='Average Order Value', title="Avg Order Value by Company")
                    fig.update_layout(**chart_config)
                    st.plotly_chart(fig, use_container_width=True)
                with col4:
                    monthly_sales = completed.set_index('Date').resample('ME').size().reset_index(name='Sales Count')
                    fig = px.line(monthly_sales, x='Date', y='Sales Count', title="Monthly Sales Volume")
                    fig.update_layout(**chart_config)
                    st.plotly_chart(fig, use_container_width=True)
            elif st.session_state.team == "Sales Manager":
                with col1:
                    comp_rev = completed.groupby('Company Name')['Total Revenue'].sum().reset_index()
                    fig = px.bar(comp_rev, x='Company Name', y='Total Revenue', title="Revenue by Company")
                    fig.update_layout(**chart_config)
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    losses = completed.groupby('Product Name')['Loss from Returns'].sum().nlargest(10).reset_index()
                    fig = px.pie(losses, names='Product Name', values='Loss from Returns', title="Loss from Returns by Product")
                    fig.update_layout(**chart_config)
                    st.plotly_chart(fig, use_container_width=True)
                with col3:
                    top_cust = completed.groupby('Company Name')['Total Revenue'].sum().nlargest(5).reset_index()
                    fig = px.bar(top_cust, x='Company Name', y='Total Revenue', title="Top Customers by Revenue")
                    fig.update_layout(**chart_config)
                    st.plotly_chart(fig, use_container_width=True)
                with col4:
                    return_rate = completed.groupby('Product Name')['Loss from Returns'].sum() / completed.groupby('Product Name')['Total Revenue'].sum()
                    return_rate = return_rate.reset_index().rename(columns={0: 'Return Rate'}).sort_values(by='Return Rate', ascending=False).head(10)
                    fig = px.bar(return_rate, x='Product Name', y='Return Rate', title="Return Rate by Product")
                    fig.update_layout(**chart_config)
                    st.plotly_chart(fig, use_container_width=True)
            st.markdown("<h3 style='font-size: 14px; margin: 5px 0;'>Sales Target</h3>", unsafe_allow_html=True)
            target_dict = {
                "Marketing Analyst": 500000,
                "Marketing Manager": 1500000,
                "Sales Data Analyst": 750000,
                "Sales Manager": 2000000
            }
            target = target_dict[st.session_state.team]
            actual = completed['Total Revenue'].sum()
            status = "Reached" if actual >= target else "Not Reached"
            delta_color = "normal" if actual >= target else "inverse"
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Target", f"${target:,.0f}")
            with col2:
                st.metric("Actual Revenue", f"${actual:,.0f}")
            with col3:
                st.metric("Status", status, delta_color=delta_color)
        else:
            st.info("No data available for selected filters.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        
        
        
        
