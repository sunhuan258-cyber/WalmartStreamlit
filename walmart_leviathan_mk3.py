import streamlit as st
import pandas as pd
import duckdb
import os
import plotly.express as px
from io import StringIO
from sqlalchemy import create_engine

# ================== 文件路径定义 ==================
DB_PATH = r"C:\walmart.db"

# ================== 页面基础配置 ==================
st.set_page_config(
    page_title="利维坦 Mk-X | 沃尔玛销售分析平台",
    page_icon="🐳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== 数据库连接 ==================
from langchain_community.utilities import SQLDatabase
from langchain_community.chat_models import ChatOllama
from langchain.chains import create_sql_query_chain

@st.cache_resource
def get_connections():
    """创建唯一的SQLAlchemy引擎，并从中派生出LangChain DB和原生DuckDB连接。"""
    try:
        db_uri = f"duckdb:///{DB_PATH}"
        engine = create_engine(db_uri, connect_args={"read_only": True})
        db = SQLDatabase(engine, include_tables=['walmart'])
        con = engine.raw_connection()
        print("--- Database connections established successfully. ---")
        return db, con
    except Exception as e:
        st.sidebar.error(f"创建数据库连接失败: {e}")
        return None, None

# 初始化所有连接
db, con = get_connections()

# ================== 渲染函数定义 ==================

def render_seer_tab():
    """渲染“洞察者”模块的全部图表。"""
    st.header("洞察者 - 商业洞察仪表盘")
    st.markdown("在这里，我们为您预设了三个核心的商业分析视角，以最高效的方式洞察数据背后的故事。")
    with st.expander("💡 点击查看：‘洞察者’模块功能说明"):
        st.markdown("""
        本模块是一个“专家驾驶舱”，旨在提供最直观的商业洞察力。
        1.  **“销售脉搏”**: 查看整体业务的季节性波动与增长趋势。
        2.  **“王牌对决”**: 识别出“明星店铺”和“黄金部门”。
        3.  **“关系探索”**: 探索外部环境因素对销售额的潜在影响。
        """)
    if not con:
        st.error("数据库连接尚未准备就绪，无法渲染图表。")
        return

    # --- 图表1: “销售脉搏” - 总体销售趋势图 ---
    st.subheader("📈 1. “销售脉搏” - 总体销售趋势图")
    with st.spinner("正在查询并渲染总体销售趋势..."):
        try:
            sql_query_1 = '''
            SELECT "Date", SUM("Weekly_Sales") as total_weekly_sales
            FROM walmart
            GROUP BY "Date"
            ORDER BY "Date" ASC
            '''
            sales_trend_df = con.execute(sql_query_1).df()
            sales_trend_df['Date'] = pd.to_datetime(sales_trend_df['Date'])
            fig1 = px.line(sales_trend_df, x='Date', y='total_weekly_sales', title='沃尔玛总体周销售额趋势', labels={'Date': '日期', 'total_weekly_sales': '总销售额'})
            st.plotly_chart(fig1, use_container_width=True)
        except Exception as e:
            st.error(f"渲染‘销售脉搏’图表失败: {e}")

    st.divider()

    # --- 图表2: “王牌对决” - 带数据条的动态排序表格 ---
    st.subheader("📊 2. “王牌对决” - 店铺/部门销售贡献图")
    col1, col2 = st.columns([1, 3])
    with col1:
        dimension_mapping = {"店铺 (Store)": "Store", "部门 (Dept)": "Dept"}
        display_dim = st.selectbox("选择分析维度:", list(dimension_mapping.keys()), key="seer_group_by")
        group_by_dim = dimension_mapping[display_dim]
        top_n = st.slider("选择查看Top N:", min_value=5, max_value=50, value=10, key="seer_top_n")
        sort_option = st.radio("选择排序方式:", ["按销售额降序", "按编号升序"], key="seer_sort", horizontal=True)
    
    with col2:
        with st.spinner(f"正在查询Top {top_n} {display_dim}..."):
            try:
                if sort_option == "按销售额降序":
                    order_by_clause = "ORDER BY total_sales DESC"
                else:
                    order_by_clause = f'ORDER BY "{group_by_dim}" ASC'

                sql_query_2 = f'''
                SELECT "{group_by_dim}", SUM("Weekly_Sales") as total_sales
                FROM walmart
                GROUP BY "{group_by_dim}"
                {order_by_clause}
                LIMIT {top_n}
                '''
                top_df = con.execute(sql_query_2).df()
                st.dataframe(top_df, 
                             column_config={
                                 "total_sales": st.column_config.ProgressColumn(
                                     "总销售额",
                                     format="$%.2f",
                                     min_value=0,
                                     max_value=top_df['total_sales'].max(),
                                 ),
                                 group_by_dim: st.column_config.TextColumn(display_dim)
                             },
                             use_container_width=True)
            except Exception as e:
                st.error(f"渲染‘王牌对决’图表失败: {e}")

    st.divider()

    # --- 图表3: “关系探索” - 交互式散点图 ---
    st.subheader("🔬 3. “关系探索” - 交互式散点图")
    st.markdown("自由选择两个数值变量，探索它们之间的潜在关系。")
    numeric_vars = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Size']
    col3, col4 = st.columns(2)
    with col3:
        x_var = st.selectbox("选择X轴变量:", numeric_vars, index=1, key="scatter_x")
    with col4:
        y_var = st.selectbox("选择Y轴变量:", numeric_vars, index=0, key="scatter_y")
    st.caption("单位说明: Temperature (华氏度), Fuel_Price (美元/加仑)")
    if x_var and y_var:
        with st.spinner(f"正在查询并渲染 {x_var} 与 {y_var} 的关系..."):
            try:
                sql_query_3 = f'SELECT "{x_var}", "{y_var}" FROM walmart USING SAMPLE 5000 ROWS'
                scatter_df = con.execute(sql_query_3).df()
                fig3 = px.scatter(scatter_df, x=x_var, y=y_var, title=f'{x_var} 与 {y_var} 关系散点图', trendline="ols")
                fig3.update_layout(xaxis_title=x_var, yaxis_title=y_var)
                st.plotly_chart(fig3, use_container_width=True)
            except Exception as e:
                st.error(f"渲染‘关系探索’图表失败: {e}")

def render_deep_dive_tab():
    """渲染“店铺深度剖析”模块。"""
    st.header("🏬 店铺深度剖析")
    st.markdown("请从下方选择一个您想深入分析的店铺，我们将为您生成一份该店专属的分析报告。")
    if not con:
        st.error("数据库连接尚未准备就绪，无法获取店铺列表。")
        return
    try:
        store_list_df = con.execute('SELECT DISTINCT "Store" FROM walmart ORDER BY "Store" ASC').df()
        store_list = store_list_df['Store'].tolist()
        selected_store = st.selectbox("请选择要分析的店铺编号:", store_list)
        if selected_store:
            render_store_kpis(selected_store)
            st.markdown("--- (后续图表功能待开发) ---")
    except Exception as e:
        st.error(f"获取店铺列表或渲染页面时出错: {e}")

def render_store_kpis(selected_store):
    """为选定的店铺渲染顶部的核心KPI。"""
    st.subheader(f"核心指标总览 - 店铺 {selected_store}")
    with st.spinner("正在计算核心KPI..."):
        sql_query = f'''
        SELECT
            SUM("Weekly_Sales") as total_sales,
            AVG("Weekly_Sales") as avg_sales,
            MAX("Weekly_Sales") as max_sales,
            COUNT(DISTINCT "Dept") as dept_count
        FROM walmart
        WHERE "Store" = {selected_store}
        '''
        kpi_df = con.execute(sql_query).df()
        if not kpi_df.empty:
            total_sales = kpi_df['total_sales'].iloc[0]
            avg_sales = kpi_df['avg_sales'].iloc[0]
            max_sales = kpi_df['max_sales'].iloc[0]
            dept_count = kpi_df['dept_count'].iloc[0]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("历史总销售额", f"${total_sales:,.2f}")
            col2.metric("平均周销售额", f"${avg_sales:,.2f}")
            col3.metric("历史最高周销售额", f"${max_sales:,.2f}")
            col4.metric("部门数量", f"{dept_count}")

# ================== 主页面渲染逻辑 ==================
st.sidebar.title("Leviathan Mk-X 控制台")
tab = st.sidebar.radio(
    "选择功能模块",
    ["项目总览", "神谕 (自然语言查询)", "洞察者 (可视化分析)", "店铺深度剖析", "先知 (预测模型)", "解剖者 (AI可解释性)"],
    captions=["关于本项目", "与数据对话", "探索销售趋势", "深入单个店铺", "预测未来销售", "理解模型决策"]
)

if tab == "项目总览":
    st.header("欢迎来到《利维坦 Mk-X》项目")
    st.markdown("""
    ### 项目架构: 《利维坦 Mk-X：最终稳定版》
    我们通过将数据库转移到纯英文路径，规避了底层库的编码Bug，确保了应用的最终稳定运行。
    """")
    if con:
        st.subheader("本地DuckDB数据库连接测试")
        with st.spinner("正在通过派生的原生DuckDB接口读取前5条数据..."):
            try:
                test_df = con.execute("SELECT * FROM walmart LIMIT 5").df()
                st.success("引擎派生连接成功！已成功读取数据。")
                st.dataframe(test_df)
            except Exception as e:
                st.error(f"数据读取失败: {e}")

elif tab == "神谕 (自然语言查询)":
    st.header("神谕 - 自然语言查询")
    st.markdown("向您的沃尔玛数据仓库提问，AI将为您生成SQL并执行查询。")
    if db and con:
        try:
            llm = ChatOllama(model="qwen3:30b")
            st.sidebar.success("Ollama (qwen3:30b) 已连接")
        except Exception as e:
            st.error(f"初始化Ollama LLM失败: {e}")
            st.stop()
        query_chain = create_sql_query_chain(llm, db)
        query = st.text_area("在此输入您的问题:", placeholder="例如：哪个店铺的周销售额总和最高？")
        if st.button("提交问题"):
            if query:
                with st.spinner("AI正在将您的问题转化为SQL..."):
                    try:
                        response_str = query_chain.invoke({"question": query})
                        temp_query = response_str
                        if "</think>" in temp_query:
                            temp_query = temp_query.split("</think>")[-1]
                        if "SQLQuery:" in temp_query:
                            temp_query = temp_query.split("SQLQuery:")[-1]
                        sql_query = temp_query.strip()
                        st.info(f"AI生成的SQL查询语句:\n```sql\n{sql_query}\n```")
                        with st.spinner("正在执行SQL查询..."):
                            result_df = con.execute(sql_query).df()
                        st.subheader("查询结果:")
                        st.dataframe(result_df)
                    except Exception as e:
                        st.error(f"查询过程中出现错误: {e}")
            else:
                st.warning("请输入您的问题。")
    else:
        st.error("数据库连接尚未准备就绪，请检查侧边栏错误信息。")

elif tab == "洞察者 (可视化分析)":
    render_seer_tab()

elif tab == "店铺深度剖析":
    render_deep_dive_tab()

elif tab == "先知 (预测模型)":
    st.header("先知 (预测模型)")
    st.write("此功能正在建设中...")
elif tab == "解剖者 (AI可解释性)":
    st.header("解剖者 (AI可解释性)")
    st.write("此功能正在建设中...")