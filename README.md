



# Excel-Like Data Analysis Tool (Python + Streamlit)

A simple, interactive data analysis web app built with Python and Streamlit — designed to provide Excel-like functionality for data exploration, cleaning, filtering, visualization, and export.

---

## Features

- Upload CSV or Excel files and preview data
- Basic data cleaning: missing value handling, type conversions
- Interactive filtering and querying of datasets
- Grouping and aggregation with common functions (sum, mean, median, count, min, max)
- Dynamic visualizations: histogram, boxplot, scatter plot, correlation heatmap (via Plotly)
- Generate detailed profiling reports using [ydata-profiling](https://github.com/ydataai/ydata-profiling)
- Export cleaned and filtered data as CSV

---

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/excel-like-data-tool.git
   cd excel-like-data-tool

2. Create and activate a Python virtual environment (optional but recommended):

  ```bash
  python -m venv venv
  source venv/bin/activate  # macOS/Linux
  .\\venv\\Scripts\\activate  # Windows
  ```





3. Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

4. Alternatively, install manually:

  ```bash
  pip install streamlit pandas plotly ydata-profiling openpyxl
  ```

5. Usage
  - Run the Streamlit app locally:

```bash
  streamlit run excel_like_tool.py
  ```

  - This will open the app in your default browser. Upload your CSV or Excel file, then explore and analyze your data interactively.

6. Sample Data
You can test the app with the provided sample dataset:

```bash
  sample_sales_data.xlsx
  ```


7. Demo
![Adobe Express - 2025-08-09 15-04-08](https://github.com/user-attachments/assets/36120c53-037b-48c5-bed9-275e20de6c28)

Here is the link to the Medium article as well where I go into detailed step by step: https://medium.com/@capali/build-your-own-excel-like-data-analysis-tool-in-python-12951df0d3bd

8. Contributing
Feel free to fork the repo and submit pull requests. Suggestions and feature requests are welcome!

9. License
This project is open source under the MIT License.



