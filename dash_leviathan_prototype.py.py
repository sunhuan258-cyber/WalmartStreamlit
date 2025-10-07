import dash
import dash_bootstrap_components as dbc
import pandas as pd
import duckdb
import os
from dash import html, dcc, callback, Input, Output
from sqlalchemy import create_engine

# --- 1. 路径与数据库连接 ---
print("--- 正在初始化路径与数据库连接... ---")
# 动态获取当前文件所在目录
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "walmart.db")

# 创建数据库连接 (我们暂时不需要像Streamlit那样缓存它)
try:
    db_uri = f"duckdb:///{DB_PATH}"
    engine = create_engine(db_uri, connect_args={"read_only": True})
    con = engine.raw_connection()
    print(f"--- 数据库连接成功: {DB_PATH} ---")
except Exception as e:
    print(f"XXX 数据库连接失败: {e}")
    con = None

# --- 2. 初始化APP ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# --- 3. 定义页面布局 ---

# 从数据库动态获取店铺列表作为下拉选项
store_list_df = con.execute('SELECT DISTINCT "Store" FROM walmart ORDER BY "Store" ASC').df()
store_options = [{'label': f'店铺 {i}', 'value': i} for i in store_list_df['Store']]

app.layout = dbc.Container(
    [
        html.H1("利维坦 (Dash版) - 沃尔玛销售分析平台", className="my-4 text-center"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H4("控制面板"),
                        html.Hr(),
                        dbc.Label("请选择要分析的店铺:"),
                        dcc.Dropdown(
                            id='store-dropdown',
                            options=store_options,
                            value=store_options[0]['value'] # 默认选中第一个店铺
                        ),
                        # 更多控件将添加于此
                    ],
                    width=3,
                    className="bg-light p-4"
                ),
                dbc.Col(
                    [
                        # 这个Div将作为我们所有图表和KPI的“画布”
                        html.Div(id='store-output-div')
                    ],
                    width=9
                )
            ]
        )
    ],
    fluid=True
)

# --- 4. 定义数据获取与组件渲染函数 ---
def get_store_kpis(selected_store):
    """Queries and calculates the main KPIs for a given store."""
    if not con or selected_store is None: return None
    try:
        sql_query_agg = f'''SELECT SUM("Weekly_Sales") as total_sales, AVG("Weekly_Sales") as avg_sales, COUNT(DISTINCT "Dept") as dept_count FROM walmart WHERE "Store" = {selected_store}'''
        agg_df = con.execute(sql_query_agg).df()
        sql_query_peak = f'''SELECT "Date", "Weekly_Sales" as max_sales FROM walmart WHERE "Store" = {selected_store} ORDER BY "Weekly_Sales" DESC LIMIT 1'''
        peak_df = con.execute(sql_query_peak).df()
        if not agg_df.empty and not peak_df.empty:
            return {
                "total_sales": agg_df['total_sales'].iloc[0],
                "avg_sales": agg_df['avg_sales'].iloc[0],
                "dept_count": agg_df['dept_count'].iloc[0],
                "max_sales": peak_df['max_sales'].iloc[0],
                "max_sales_date": pd.to_datetime(peak_df['Date'].iloc[0]).strftime('%Y-%m-%d')
            }
    except Exception as e:
        print(f"Error getting store KPIs: {e}")
        return None

def get_sales_trend_data(selected_store):
    """Queries data for the sales trend graph."""
    if not con or selected_store is None: return None
    sql_query = f'''SELECT "Date", SUM("Weekly_Sales") as weekly_sales FROM walmart WHERE "Store" = {selected_store} GROUP BY "Date" ORDER BY "Date" ASC'''
    trend_df = con.execute(sql_query).df()
    trend_df['Date'] = pd.to_datetime(trend_df['Date'])
    return trend_df

def create_sales_trend_graph(trend_df, selected_store):
    """Creates the sales trend graph component."""
    if trend_df is None or trend_df.empty: return None
    fig = px.line(trend_df, x='Date', y='weekly_sales', title=f'店铺 {selected_store} 周销售额趋势', labels={'Date': '日期', 'weekly_sales': '周销售额'})
    return dcc.Graph(figure=fig)

# --- 5. 定义交互逻辑 (回调函数) ---
@callback(
    Output('store-output-div', 'children'),
    Input('store-dropdown', 'value')
)
def update_store_output(selected_store):
    if selected_store is None: return dbc.Alert("请选择一个店铺进行分析。", color="info")
    
    # 1. 获取并渲染KPIs
    kpis = get_store_kpis(selected_store)
    if kpis is None: return dbc.Alert(f"无法获取店铺 {selected_store} 的KPI数据。", color="danger")
    kpi_cards = dbc.Row([
        dbc.Col(dbc.Card([dbc.CardBody([html.H4("历史总销售额", className="card-title"), html.P(f"${kpis['total_sales']:,.2f}", className="card-text fs-3 fw-bold")])]), width=3),
        dbc.Col(dbc.Card([dbc.CardBody([html.H4("平均周销售额", className="card-title"), html.P(f"${kpis['avg_sales']:,.2f}", className="card-text fs-3 fw-bold")])]), width=3),
        dbc.Col(dbc.Card([dbc.CardBody([html.H4("历史最高周销售额", className="card-title"), html.P(f"${kpis['max_sales']:,.2f}", className="card-text fs-3 fw-bold"), html.Small(f"发生在 {kpis['max_sales_date']}", className="text-muted")])]), width=3),
        dbc.Col(dbc.Card([dbc.CardBody([html.H4("部门数量", className="card-title"), html.P(f"{kpis['dept_count']}", className="card-text fs-3 fw-bold")])]), width=3)
    ])
    
    # 2. 获取并渲染销售趋势图
    trend_df = get_sales_trend_data(selected_store)
    sales_trend_graph = create_sales_trend_graph(trend_df, selected_store)

    # 3. 将所有组件组合在一个列表中返回
    return [kpi_cards, html.Hr(), sales_trend_graph]

# --- 6. 启动服务器 ---
if __name__ == '__main__':
    if con: 
        app.run(debug=True)
    else:
        print("由于数据库连接失败，Dash应用无法启动。请检查DB_PATH是否正确，以及walmart.db文件是否存在。")
