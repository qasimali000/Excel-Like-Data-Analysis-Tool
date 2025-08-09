
import streamlit as st
import pandas as pd
import numpy as np
import io
import os
from datetime import datetime

# plotting
import plotly.express as px
import plotly.graph_objects as go

# optional profiling
try:
    from ydata_profiling import ProfileReport
    PROFILING_AVAILABLE = True
except Exception:
    PROFILING_AVAILABLE = False

st.set_page_config(page_title="Excel-like Data Tool", layout="wide")

# ----------------- Helpers -----------------

@st.cache_data
def load_file(uploaded_file):
    """Load CSV or Excel into a pandas DataFrame."""
    fname = uploaded_file.name.lower()
    if fname.endswith('.csv') or fname.endswith('.txt'):
        return pd.read_csv(uploaded_file)
    elif fname.endswith('.xls') or fname.endswith('.xlsx'):
        return pd.read_excel(uploaded_file)
    else:
        # try csv as fallback
        try:
            return pd.read_csv(uploaded_file)
        except Exception:
            st.error('Unsupported file type. Please upload CSV or Excel.')
            return pd.DataFrame()


def download_df_as_csv(df, filename="data.csv"):
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer


def make_profile_report(df):
    if not PROFILING_AVAILABLE:
        st.warning('ydata_profiling not installed — profiling disabled. Install with `pip install ydata-profiling`.')
        return None
    pr = ProfileReport(df, explorative=True)
    tmpfile = f"profile_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.html"
    pr.to_file(tmpfile)
    with open(tmpfile, 'rb') as f:
        content = f.read()
    os.remove(tmpfile)
    return content

# ----------------- UI -----------------

st.title('🧮 Excel‑Like Data Analysis Tool')

col1, col2 = st.columns([1, 3])

with col1:
    st.header('Upload')
    uploaded = st.file_uploader('Upload CSV or Excel', type=['csv', 'xls', 'xlsx', 'txt'])
    sample_data_btn = st.button('Load sample dataset (tips)')

    st.markdown('''
    **Quick actions**
    - Upload a CSV/Excel
    - View and filter the table
    - Generate stats & charts
    - Export cleaned data or a profiling report
    ''')

    if sample_data_btn:
        uploaded = None
        df = px.data.iris()
        st.session_state['df'] = df

with col2:
    if uploaded is None and 'df' not in st.session_state:
        st.info('Upload a CSV/Excel file to get started, or click "Load sample dataset".')
    elif uploaded is not None:
        try:
            df = load_file(uploaded)
            st.session_state['df'] = df
            st.success(f'Loaded {uploaded.name} — {df.shape[0]} rows, {df.shape[1]} columns')
        except Exception as e:
            st.error(f'Failed to load file: {e}')

# require df
if 'df' not in st.session_state:
    st.stop()

df = st.session_state['df']

# show basic info
with st.expander('Dataset summary', expanded=True):
    st.write('Shape:', df.shape)
    st.write('Columns and types:')
    dtypes = pd.DataFrame(df.dtypes, columns=['dtype'])
    st.dataframe(dtypes)

# Column selection and basic cleaning
st.sidebar.header('Column & Cleaning')
cols = df.columns.tolist()
selected_cols = st.sidebar.multiselect('Visible columns', cols, default=cols)

# NA handling
na_action = st.sidebar.selectbox('Missing value handling', ['Leave', 'Drop rows with NA', 'Fill with (constant)', 'Fill with median (numeric)'])
fill_value = None
if na_action == 'Fill with (constant)':
    fill_value = st.sidebar.text_input('Fill value (string)')

# Type conversions
st.sidebar.header('Type conversion')
for c in cols:
    t = df[c].dtype
    # Offer only for object -> numeric/date
    if t == object:
        if st.sidebar.checkbox(f'Try convert {c} to numeric', key=f'num_{c}'):
            df[c] = pd.to_numeric(df[c], errors='coerce')
    if t == object:
        if st.sidebar.checkbox(f'Try convert {c} to datetime', key=f'dt_{c}'):
            df[c] = pd.to_datetime(df[c], errors='coerce')

# Apply NA actions
if na_action == 'Drop rows with NA':
    df = df.dropna()
elif na_action == 'Fill with (constant)' and fill_value is not None and fill_value != '':
    df = df.fillna(fill_value)
elif na_action == 'Fill with median (numeric)':
    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())

# Visible df
st.subheader('Data table')
st.dataframe(df[selected_cols])

# Quick stats
st.subheader('Quick statistics')
with st.expander('Numeric summary'):
    st.write(df.describe())

with st.expander('Categorical summary'):
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        for c in cat_cols:
            st.write(f'**{c}** — {df[c].nunique()} unique')
            st.write(df[c].value_counts().head(10))
    else:
        st.write('No categorical columns detected')

# Interactive filtering
st.subheader('Filter / Query the data (simple)')
filter_col = st.selectbox('Filter column (for equality)', cols)
unique_vals = df[filter_col].dropna().unique().tolist()
if len(unique_vals) > 50:
    st.write('Too many unique values to show selector — using text match.')
    text_match = st.text_input('Text to match in column')
    if text_match:
        df_filtered = df[df[filter_col].astype(str).str.contains(text_match, case=False, na=False)]
    else:
        df_filtered = df
else:
    pick = st.multiselect('Pick values', unique_vals, default=unique_vals[:5])
    if pick:
        df_filtered = df[df[filter_col].isin(pick)]
    else:
        df_filtered = df

st.write('Filtered rows:', df_filtered.shape[0])
if st.checkbox('Show filtered table'):
    st.dataframe(df_filtered[selected_cols])

# Grouping & aggregation
st.subheader('Group & aggregate')
group_cols = st.multiselect('Group by', cols)
agg_col = st.selectbox('Aggregate column (numeric)', df.select_dtypes(include=[np.number]).columns.tolist() if not df.select_dtypes(include=[np.number]).empty else [])
agg_func = st.selectbox('Aggregation', ['sum', 'mean', 'median', 'count', 'min', 'max'])

if group_cols and agg_col:
    grouped = df_filtered.groupby(group_cols)[agg_col].agg(agg_func).reset_index()
    st.dataframe(grouped)
    if st.button('Show bar chart of aggregation'):
        fig = px.bar(grouped, x=group_cols[-1], y=agg_col, color=group_cols[0] if len(group_cols) > 1 else None)
        st.plotly_chart(fig, use_container_width=True)

# Visualizations
st.subheader('Visualizations')
plot_type = st.selectbox('Chart type', ['Histogram', 'Boxplot', 'Scatter', 'Correlation heatmap'])

if plot_type == 'Histogram':
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    col = st.selectbox('Numeric column', num_cols)
    bins = st.slider('Bins', 5, 200, 30)
    fig = px.histogram(df_filtered, x=col, nbins=bins)
    st.plotly_chart(fig, use_container_width=True)

elif plot_type == 'Boxplot':
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    col = st.selectbox('Numeric column', num_cols)
    fig = px.box(df_filtered, y=col)
    st.plotly_chart(fig, use_container_width=True)

elif plot_type == 'Scatter':
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    xcol = st.selectbox('X', num_cols, key='xcol')
    ycol = st.selectbox('Y', num_cols, key='ycol')
    color = st.selectbox('Color (optional)', [None] + cols)
    fig = px.scatter(df_filtered, x=xcol, y=ycol, color=color)
    st.plotly_chart(fig, use_container_width=True)

elif plot_type == 'Correlation heatmap':
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        st.write('Need at least 2 numeric columns for correlation')
    else:
        corr = num.corr()
        fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1))
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

# Profiling report
st.subheader('Profiling report')
if st.button('Generate profiling report (HTML)'):
    content = make_profile_report(df)
    if content:
        st.success('Profile generated — ready to download')
        st.download_button('Download profile (HTML)', data=content, file_name='profile_report.html', mime='text/html')

# Export cleaned data
st.subheader('Export cleaned data')
fn = st.text_input('Filename', value='cleaned_data.csv')
if st.button('Download cleaned CSV'):
    buffer = download_df_as_csv(df)
    st.download_button('Download CSV', data=buffer.getvalue(), file_name=fn, mime='text/csv')

# Save session state data snapshot
if st.button('Save snapshot to session (for reuse)'):
    st.session_state['snapshot'] = df.copy()
    st.success('Snapshot saved to session')

