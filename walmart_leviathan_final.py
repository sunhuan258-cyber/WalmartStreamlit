import streamlit as st
import pandas as pd
import duckdb
import os
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap
import matplotlib.pyplot as plt
from io import StringIO
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_community.chat_models import ChatOllama
from langchain_groq import ChatGroq
from langchain.chains import create_sql_query_chain

# ================== 文件路径定义 (v2.3 - 适配云端部署) ==================
# 适配Streamlit云端部署，路径为相对于仓库根目录
DB_PATH = "walmart.db"
MODEL_PATH = "walmart_model_artifacts.joblib"

# ================== 页面基础配置 ==================
st.set_page_config(
    page_title="利维坦 (最终版) | 沃尔玛销售分析平台",
    page_icon="🐳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== 全局CSS美化 (v2.2 - 动态交互版) ==================
st.markdown("""
<style>
/* General Body and Font */
body {
    color: #E0E0E0;
    background-color: #0F1116;
}

/* Main content area */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 5rem;
    padding-right: 5rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #1A1C24;
}

/* Titles and Headers */
h1, h2, h3 {
    color: #64B5F6;
    font-weight: 600;
}

/* Buttons */
.stButton>button {
    border: 1px solid #64B5F6;
    background-color: transparent;
    color: #64B5F6;
    padding: 10px 24px;
    border-radius: 8px;
    transition: all 0.3s ease-in-out;
    font-weight: 600;
}
.stButton>button:hover {
    background-color: #64B5F6;
    color: #0F1116;
    border-color: #64B5F6;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px 0 rgba(100, 181, 246, 0.3);
}
.stButton>button:focus {
    box-shadow: 0 0 0 0.2rem rgba(100, 181, 246, 0.5) !important;
}

/* Expander */
.st-expander {
    border: 1px solid #333;
    border-radius: 0.5rem;
}
.st-expander header {
    font-size: 1.1rem;
    color: #A9A9A9;
}

/* Metric Label */
[data-testid="stMetricLabel"] {
    color: #A9A9A9;
}

/* Card Containers - This targets the containers with border=True */
div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] > div[data-testid="stContainer"],
div[data-testid="stVerticalBlock"] > div[data-testid="stContainer"] {
    transition: all 0.3s ease-in-out;
}

div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] > div[data-testid="stContainer"]:hover,
div[data-testid="stVerticalBlock"] > div[data-testid="stContainer"]:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 24px 0 rgba(0,0,0,0.4), 0 0 20px rgba(100, 181, 246, 0.4);
}

</style>
""", unsafe_allow_html=True)


# ================== 核心功能加载 ==================


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

@st.cache_resource
def load_artifacts():
    """加载包含模型、scaler、encoder和特征列表的完整工具箱。"""
    try:
        artifacts = joblib.load(MODEL_PATH)
        print("--- Model artifacts loaded successfully. ---")
        # 验证工具箱的完整性
        if isinstance(artifacts, dict) and 'model' in artifacts and 'feature_names' in artifacts and 'scaler' in artifacts and 'encoder' in artifacts:
            print("--- Artifacts validated. Contains 'model', 'scaler', 'encoder', and 'feature_names' keys. ---")
            return artifacts
        else:
            actual_keys = list(artifacts.keys()) if isinstance(artifacts, dict) else "Not a dictionary"
            return f"错误：模型文件格式不正确。预期需要'model', 'scaler', 'encoder', 'feature_names'，但文件中实际的键为: {actual_keys}"
    except FileNotFoundError:
        return f"错误：在路径 {MODEL_PATH} 找不到模型文件。"
    except Exception as e:
        return f"加载模型 artifacts 时出错: {e}"

@st.cache_resource
def get_feature_defaults():
    """计算并缓存用于预测输入界面的默认值（中位数）。"""
    if not con:
        return None
    try:
        sql_query = '''
        SELECT 
            MEDIAN("Temperature") as temp,
            MEDIAN("Fuel_Price") as fuel,
            MEDIAN("CPI") as cpi,
            MEDIAN("Unemployment") as unemployment,
            (SELECT MIN("Weekly_Sales") FROM walmart WHERE "Store" = 1) as min_sales,
            (SELECT MAX("Weekly_Sales") FROM walmart WHERE "Store" = 1) as max_sales
        FROM walmart
        '''
        defaults_df = con.execute(sql_query).df()
        return defaults_df.to_dict('records')[0]
    except Exception as e:
        st.warning(f"无法计算默认值: {e}")
        return None

# 初始化所有连接和“工具箱”
db, con = get_connections()
artifacts = load_artifacts()

# 从“工具箱”中提取所有工具
model = None
model_features = []
scaler = None
encoder = None

if isinstance(artifacts, dict):
    model = artifacts.get('model')
    model_features = artifacts.get('feature_names')
    scaler = artifacts.get('scaler')
    encoder = artifacts.get('encoder')

# ================== 渲染函数定义 ==================

def render_seer_tab():
    st.header("洞察者 - 商业洞察仪表盘")
    with st.expander("💡 点击查看：‘洞察者’模块功能说明"):
        st.markdown("本模块是一个“专家驾驶舱”，旨在提供最直观的商业洞察力...")
    if not con:
        st.error("数据库连接尚未准备就绪，无法渲染图表。")
        return

    # 卡片1: 销售脉搏
    with st.container(border=True):
        st.subheader("📈 1. “销售脉搏” - 总体销售趋势图")
        with st.spinner("正在查询并渲染总体销售趋势..."):
            try:
                sql_query_1 = 'SELECT "Date", SUM("Weekly_Sales") as total_weekly_sales FROM walmart GROUP BY "Date" ORDER BY "Date" ASC'
                sales_trend_df = con.execute(sql_query_1).df()
                sales_trend_df['Date'] = pd.to_datetime(sales_trend_df['Date'])
                fig1 = px.line(sales_trend_df, x='Date', y='total_weekly_sales', title='沃尔玛总体周销售额趋势', labels={'Date': '日期', 'total_weekly_sales': '总销售额'}, template="plotly_dark")
                fig1.update_traces(line=dict(color='#64B5F6')) # 主题蓝
                st.plotly_chart(fig1, use_container_width=True)
            except Exception as e:
                st.error(f"渲染‘销售脉搏’图表失败: {e}")

    # 卡片2: 王牌对决
    with st.container(border=True):
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
                    order_by_clause = "ORDER BY total_sales DESC" if sort_option == "按销售额降序" else f'ORDER BY "{group_by_dim}" ASC'
                    sql_query_2 = f'SELECT "{group_by_dim}", SUM("Weekly_Sales") as total_sales FROM walmart GROUP BY "{group_by_dim}" {order_by_clause} LIMIT {top_n}'
                    top_df = con.execute(sql_query_2).df()
                    st.dataframe(top_df, column_config={
                        "total_sales": st.column_config.ProgressColumn("总销售额", format="$%.2f", min_value=0, max_value=top_df['total_sales'].max()),
                        group_by_dim: st.column_config.TextColumn(display_dim)
                    }, use_container_width=True)
                except Exception as e:
                    st.error(f"渲染‘王牌对决’图表失败: {e}")

    # 卡片3: 关系探索
    with st.container(border=True):
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
                    fig3 = px.scatter(scatter_df, x=x_var, y=y_var, title=f'{x_var} 与 {y_var} 关系散点图', trendline="ols", template="plotly_dark")
                    fig3.update_traces(marker=dict(color='#64B5F6')) # 主题蓝
                    st.plotly_chart(fig3, use_container_width=True)
                except Exception as e:
                    st.error(f"渲染‘关系探索’图表失败: {e}")

def render_deep_dive_tab():
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
            with st.container(border=True):
                render_store_kpis(selected_store)
            with st.container(border=True):
                render_store_sales_trend(selected_store)
            with st.container(border=True):
                render_store_dept_performance(selected_store)
            with st.container(border=True):
                render_store_holiday_impact(selected_store)
    except Exception as e:
        st.error(f"获取店铺列表或渲染页面时出错: {e}")

def render_store_kpis(selected_store):
    st.subheader(f"核心指标总览 - 店铺 {selected_store}")
    with st.spinner("正在计算核心KPI..."):
        try:
            sql_query_agg = f'''SELECT SUM("Weekly_Sales") as total_sales, AVG("Weekly_Sales") as avg_sales, COUNT(DISTINCT "Dept") as dept_count FROM walmart WHERE "Store" = {selected_store}'''
            agg_df = con.execute(sql_query_agg).df()
            sql_query_peak = f'''SELECT "Date", "Weekly_Sales" as max_sales FROM walmart WHERE "Store" = {selected_store} ORDER BY "Weekly_Sales" DESC LIMIT 1'''
            peak_df = con.execute(sql_query_peak).df()
            if not agg_df.empty and not peak_df.empty:
                total_sales = agg_df['total_sales'].iloc[0]
                avg_sales = agg_df['avg_sales'].iloc[0]
                dept_count = agg_df['dept_count'].iloc[0]
                max_sales = peak_df['max_sales'].iloc[0]
                max_sales_date = pd.to_datetime(peak_df['Date'].iloc[0]).strftime('%Y-%m-%d')
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("💰 历史总销售额", f"${total_sales:,.2f}")
                with col2:
                    st.metric("📊 平均周销售额", f"${avg_sales:,.2f}")
                with col3:
                    st.metric("🚀 历史最高周销售额", f"${max_sales:,.2f}", f"发生在 {max_sales_date}")
                with col4:
                    st.metric("📦 部门数量", f"{dept_count}")
        except Exception as e:
            st.error(f"计算KPI时出错: {e}")

def render_store_sales_trend(selected_store):
    st.subheader(f"📈 销售脉搏 - 店铺 {selected_store}")
    with st.spinner("正在查询并渲染销售趋势图..."):
        try:
            sql_query = f'''SELECT "Date", SUM("Weekly_Sales") as weekly_sales FROM walmart WHERE "Store" = {selected_store} GROUP BY "Date" ORDER BY "Date" ASC'''
            trend_df = con.execute(sql_query).df()
            trend_df['Date'] = pd.to_datetime(trend_df['Date'])
            fig = px.line(trend_df, x='Date', y='weekly_sales', title=f'店铺 {selected_store} 周销售额趋势', labels={'Date': '日期', 'weekly_sales': '周销售额'}, template="plotly_dark")
            fig.update_traces(line=dict(color='#64B5F6')) # 主题蓝
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"渲染单店销售趋势图时出错: {e}")

def render_store_dept_performance(selected_store):
    st.subheader(f"📊 部门王牌对决 - 店铺 {selected_store}")
    col1, col2 = st.columns([1, 3])
    with col1:
        top_n = st.slider("选择查看Top N:", min_value=5, max_value=50, value=10, key="deep_dive_top_n")
        sort_option = st.radio("选择排序方式:", ["按销售额降序", "按编号升序"], key="deep_dive_sort", horizontal=True)
    with col2:
        with st.spinner(f"正在查询店铺 {selected_store} 的Top {top_n} 部门..."):
            try:
                order_by_clause = "ORDER BY total_sales DESC" if sort_option == "按销售额降序" else 'ORDER BY "Dept" ASC'
                sql_query = f'''SELECT "Dept", SUM("Weekly_Sales") as total_sales FROM walmart WHERE "Store" = {selected_store} GROUP BY "Dept" {order_by_clause} LIMIT {top_n}'''
                dept_df = con.execute(sql_query).df()
                st.dataframe(dept_df, column_config={
                    "total_sales": st.column_config.ProgressColumn("总销售额", format="$%.2f", min_value=0, max_value=dept_df['total_sales'].max()),
                    "Dept": st.column_config.TextColumn("部门 (Dept)")
                }, use_container_width=True)
            except Exception as e:
                st.error(f"渲染店内部门表现时出错: {e}")

def render_store_holiday_impact(selected_store):
    st.subheader(f"⚖️ 节假日销售影响诊断 - 店铺 {selected_store}")
    with st.spinner("正在查询并渲染节假日影响图..."):
        try:
            sql_query = f'''SELECT "IsHoliday", AVG("Weekly_Sales") as avg_sales FROM walmart WHERE "Store" = {selected_store} GROUP BY "IsHoliday"'''
            impact_df = con.execute(sql_query).df()
            if impact_df.shape[0] == 2:
                holiday_sales = impact_df.loc[impact_df['IsHoliday'] == True, 'avg_sales'].iloc[0]
                non_holiday_sales = impact_df.loc[impact_df['IsHoliday'] == False, 'avg_sales'].iloc[0]
                delta = ((holiday_sales - non_holiday_sales) / non_holiday_sales) * 100
                st.markdown("**核心指标对比:**")
                col1, col2 = st.columns(2)
                col1.metric("节假日周平均销售额", f"${holiday_sales:,.2f}", f"{delta:.2f}%")
                col2.metric("普通周平均销售额", f"${non_holiday_sales:,.2f}")
                st.markdown("**可视化对比:**")
                impact_df['IsHoliday'] = impact_df['IsHoliday'].map({True: '节假日周', False: '普通周'})
                fig = px.bar(impact_df, x="IsHoliday", y="avg_sales", color="IsHoliday", 
                             title=f'店铺 {selected_store} 节假日 vs. 普通周平均销售额对比', 
                             labels={"IsHoliday": "周类型", "avg_sales": "平均周销售额"}, 
                             template="plotly_dark",
                             color_discrete_map={
                                 '节假日周': '#64B5F6', # 主题蓝
                                 '普通周': '#3D5A80'  # 深蓝灰色
                             })
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("该店铺数据不足，无法进行有效的节假日/非节假日对比。")
        except Exception as e:
            st.error(f"渲染节假日影响图时出错: {e}")

def render_prophet_tab():
    st.header("🔮 先知 - 销售预测模块")

    # 检查模型和所有必要的工具是否加载成功
    if isinstance(artifacts, str):
        st.error(artifacts)
        return
    if not model or not model_features or not scaler or not encoder:
        st.error("模型或其必要的工具(scaler, encoder, feature_names)未能成功从工具箱中提取。")
        return
    else:
        st.success(f"✅ 模型及所有必需工具(共{len(model_features)}个特征)已成功加载！")

    defaults = get_feature_defaults()
    if not defaults:
        st.error("无法加载用于填充界面的默认值。")
        return

    # 卡片1: 参数输入
    with st.container(border=True):
        st.subheader("参数输入")
        st.info("ℹ️ **自动填充说明**: 为方便使用，所有输入框已使用**历史数据中位数**进行自动填充。您只需修改关心的参数即可预测。")
        st.selectbox("选择店铺编号:", [1], disabled=True)
        st.info("ℹ️ **关于单店模型**: 当前的预测模型仅针对【1号店铺】的数据进行训练和优化，以作为功能展示。")

        col1, col2, col3 = st.columns(3)
        with col1:
            dept = st.number_input("选择部门 (Dept):", min_value=1, max_value=99, value=1, step=1)
            date = st.date_input("选择日期:")
            is_holiday = st.checkbox("是否为节假日周?", value=False)
        with col2:
            temperature = st.number_input("温度 (Fahrenheit):", value=defaults.get('temp', 68.0))
            fuel_price = st.number_input("燃油价格 (USD/Gallon):", value=defaults.get('fuel', 3.5))
            cpi = st.number_input("消费者物价指数 (CPI):", value=defaults.get('cpi', 192.0))
        with col3:
            unemployment = st.number_input("失业率:", value=defaults.get('unemployment', 8.0))
            size = st.number_input("店铺面积 (Size):", value=151315, disabled=True)
            st.text_input("店铺类型 (Type):", "A", disabled=True)

    if st.button("执行预测", type="primary", use_container_width=True):
        # 卡片2: 预测结果
        with st.container(border=True):
            with st.spinner("正在准备特征并执行预测..."):
                try:
                    # --- 特征工程 (v2.0 - 正确的管道) ---
                    manual_feature_columns = [col for col in model_features if 'auto_feat_' not in col]

                    input_data = {
                        'Store': 1, 'Dept': dept, 'IsHoliday': is_holiday, 'Size': size,
                        'Temperature': temperature, 'Fuel_Price': fuel_price, 'CPI': cpi,
                        'Unemployment': unemployment, 'Type_B': False, 'Type_C': False,
                        'year': date.year, 'month': date.month, 'week_of_year': date.isocalendar().week,
                        'day_of_week': date.weekday(),
                        'sales_lag_1': 0, 'sales_lag_52': 0, 'sales_rolling_mean_4': 0,
                        'MarkDown1': 0, 'MarkDown2': 0, 'MarkDown3': 0, 'MarkDown4': 0, 'MarkDown5': 0
                    }

                    manual_features_df = pd.DataFrame([input_data])[manual_feature_columns]
                    manual_features_scaled = scaler.transform(manual_features_df)
                    encoded_features = encoder.predict(manual_features_scaled)
                    encoded_features_df = pd.DataFrame(encoded_features, columns=[f'auto_feat_{i}' for i in range(8)])
                    
                    final_input_df = pd.concat([manual_features_df.reset_index(drop=True), encoded_features_df], axis=1)
                    final_input_df = final_input_df[model_features]

                    # --- 执行预测 ---
                    prediction = model.predict(final_input_df)[0]

                    # --- 结果展示 (v2.1 - 深色主题) ---
                    st.subheader("🎯 预测结果")
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number", value=prediction,
                        number={'valueformat': ',.2f', 'font': {'color': '#E0E0E0'}},
                        title={'text': "预测周销售额 (USD)", 'font': {'color': '#E0E0E0'}},
                        gauge={
                            'axis': {'range': [defaults.get('min_sales', 0), defaults.get('max_sales', 500000)], 'tickfont': {'color': '#E0E0E0'}},
                            'bar': {'color': "#64B5F6"},
                            'steps': [
                                {'range': [defaults.get('min_sales', 0), defaults.get('max_sales', 500000) * 0.5], 'color': '#3D5A80'},
                                {'range': [defaults.get('max_sales', 500000) * 0.5, defaults.get('max_sales', 500000) * 0.75], 'color': '#2A3B4D'}
                            ]
                        }
                    ))
                    fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20), template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)

                    # --- SHAP 可解释性分析 (v2.1 - 深色主题) ---
                    st.subheader("🔍 模型决策解剖 (SHAP)")
                    with st.spinner("正在计算并绘制SHAP贡献图..."):
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(final_input_df)
                        
                        # 为SHAP图应用自定义深色主题
                        plt.style.use('dark_background')
                        plt.rcParams.update({
                            'figure.facecolor': '#1A1C24',
                            'axes.facecolor': '#1A1C24',
                            'axes.edgecolor': '#E0E0E0',
                            'axes.labelcolor': '#E0E0E0',
                            'text.color': '#E0E0E0',
                            'xtick.color': '#E0E0E0',
                            'ytick.color': '#E0E0E0',
                            'grid.color': '#444444',
                        })

                        fig_shap, ax_shap = plt.subplots(figsize=(10, 4), dpi=150)
                        shap.summary_plot(shap_values, final_input_df, plot_type="bar", show=False)
                        # 手动设置条形颜色以匹配主题
                        for bar in ax_shap.patches:
                            bar.set_color('#64B5F6')
                        
                        st.pyplot(fig_shap, bbox_inches='tight', facecolor=ax_shap.get_facecolor())
                        plt.clf()
                        plt.style.use('default') # 重置matplotlib样式以避免影响其他会话
                        
                        with st.expander("💡 如何解读SHAP贡献图？"):
                            st.markdown("""
                            上图展示了对于本次预测，哪些特征的“贡献度”最大。
                            - **特征（Feature）**: 影响预测的各种因素，如`CPI`, `Size`, `Dept`等。
                            - **贡献度（SHAP value）**: 条形越长，代表该特征对本次预测结果的影响越大。它衡量的是，该特征的存在，将预测结果“推高”或“拉低”了多少。
                            
                            通过此图，您可以直观地看出，模型在做出当前预测时，最“看重”的是哪些信息。
                            """)

                except Exception as e:
                    st.error(f"执行预测时出错: {e}")
# ================== 主页面渲染逻辑 (v2.0 - 封装版) ==================
def render_sidebar():
    """渲染侧边栏的所有组件，并返回用户选择的Tab。"""
    st.sidebar.title("Leviathan 控制台")
    tab = st.sidebar.radio(
        "选择功能模块",
        ["项目总览", "神谕 (自然语言查询)", "洞察者 (可视化分析)", "店铺深度剖析", "先知 (预测模型)"],
        captions=["关于本项目", "与数据对话", "探索销售趋势", "深入单个店铺", "预测未来销售"]
    )
    return tab

# --- 主程序入口 ---
tab = render_sidebar()

if tab == "项目总览":
    st.header("欢迎来到《利维坦》项目")
    st.markdown("本仪表盘是一个集成了描述性分析、诊断性分析、预测性分析与AI驱动的探索性分析于一体的综合性商业智能平台。")
    if con:
        st.subheader("数据库连接测试")
        with st.spinner("正在读取前5条数据..."):
            try:
                test_df = con.execute("SELECT * FROM walmart LIMIT 5").df()
                st.success("数据库连接成功！")
                st.dataframe(test_df)
            except Exception as e:
                st.error(f"数据读取失败: {e}")

elif tab == "神谕 (自然语言查询)":

    @st.cache_resource
    def get_llm():
        """智能LLM连接器，优先云端，备用本地。"""
        model_name = "llama-3.1-8b-instant" # 更新为用户验证的、当前现役的生产模型
        # 优先尝试连接Groq云端模型
        try:
            if 'GROQ_API_KEY' in st.secrets:
                groq_api_key = st.secrets["GROQ_API_KEY"]
                llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_name)
                # 测试连接
                llm.invoke("test") 
                return llm, f"☁️ 已连接到 Groq 云端 ({model_name})"
        except Exception as e:
            st.sidebar.warning(f"连接Groq失败: {e}。正在尝试本地模型...")

        # 如果云端连接失败，则尝试连接本地Ollama
        try:
            llm = ChatOllama(model="qwen3:30b")
            llm.invoke("test")
            return llm, f"💻 已连接到本地 Ollama (qwen3:30b)"
        except Exception as e:
            return None, "❌ 所有AI连接均失败。请检查Groq密钥或本地Ollama服务。"

    st.header("神谕 - 自然语言查询")
    with st.expander("💡 点击查看：如何写出高质量的查询提示词？"):
        st.markdown("""
        **黄金公式**：`作为一名[角色], 我想知道关于[主体]的[指标], [可选的限定条件]。`
        **优秀范例：**
        *   `我们一共有多少个不同的店铺？`
        *   `作为一名数据分析师, 我想知道每个店铺的**周销售额总和**, 请按**总销售额降序**排列。`
        *   `我想知道3号店铺在**2011年7月**的**周销售额总和**是多少？`
        """)

    if db and con:
        llm, message = get_llm()

        if llm:
            st.sidebar.success(message)
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
            st.sidebar.error(message)
            st.error(message)
            st.stop()
    else:
        st.error("数据库连接尚未准备就绪，请检查侧边栏错误信息。")

elif tab == "洞察者 (可视化分析)":
    render_seer_tab()

elif tab == "店铺深度剖析":
    render_deep_dive_tab()

elif tab == "先知 (预测模型)":
    render_prophet_tab()