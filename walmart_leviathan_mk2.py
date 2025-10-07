
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import socks
import socket

# ================== 代理配置 (Proxy Configuration) ==================
# 在所有网络操作之前，设置SOCKS5代理
# 7890是Clash常用的默认SOCKS5端口，如果您的端口不同，请修改这里
PROXY_HOST = '127.0.0.1'
PROXY_PORT = 7890
socks.set_default_proxy(socks.SOCKS5, PROXY_HOST, PROXY_PORT)
socket.socket = socks.socksocket
print(f"--- SOCKS5 Proxy enabled through {PROXY_HOST}:{PROXY_PORT} ---")


# ================== 页面基础配置 ==================
st.set_page_config(
    page_title="利维坦 Mk-II | 沃尔玛销售分析平台",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== 数据库连接 ==================
# 使用Streamlit的Secrets Management来安全地管理凭证

def get_db_connection():
    try:
        engine = create_engine(
            st.secrets["postgres_url"],
            connect_args={'sslmode': 'require'}
        )
        return engine
    except Exception as e:
        st.error(f"数据库连接失败: {e}")
        return None

engine = get_db_connection()

# ================== 侧边栏与多Tab导航 ==================
st.sidebar.title("Leviathan Mk-II 控制台")

tab = st.sidebar.radio(
    "选择功能模块",
    ["项目总览", "神谕 (自然语言查询)", "洞察者 (可视化分析)", "先知 (预测模型)", "解剖者 (AI可解释性)"],
    captions=["关于本项目", "与数据对话", "探索销售趋势", "预测未来销售", "理解模型决策"]
)

# ================== 主页面内容 ==================

if tab == "项目总览":
    st.header("欢迎来到《利维坦 Mk-II》项目")
    st.markdown("""
    ### 项目背景
    本项目旨在将沃尔玛销售数据，从静态分析，升级为一个动态、可交互、并由AI驱动的智能数据平台。
    它体现了从数据处理、云端部署、到模型服务化和自然语言交互的全栈数据科学能力。

    ### 使用指南
    请通过左侧的侧边栏，在不同的功能模块间进行切换。

    - **神谕**: 与数据进行自然语言对话。
    - **洞察者**: 探索预设的深度可视化分析。
    - **先知**: 使用机器学习模型预测未来销售。
    - **解剖者**: 查看模型为什么会做出这样的预测。
    """)
    
    # 测试数据库连接与数据读取
    if engine:
        st.subheader("数据库连接测试")
        with st.spinner("正在从云端数据库读取前5条数据..."):
            try:
                test_df = pd.read_sql("SELECT * FROM public.walmart LIMIT 5", con=engine)
                st.success("数据库连接成功！已成功读取数据。")
                st.dataframe(test_df)
            except Exception as e:
                st.error(f"数据读取失败: {e}")

elif tab == "神谕 (自然语言查询)":
    st.header("神谕 (自然语言查询)")
    st.write("此功能正在建设中...")

elif tab == "洞察者 (可视化分析)":
    st.header("洞察者 (可视化分析)")
    st.write("此功能正在建设中...")

elif tab == "先知 (预测模型)":
    st.header("先知 (预测模型)")
    st.write("此功能正在建设中...")

elif tab == "解剖者 (AI可解释性)":
    st.header("解剖者 (AI可解释性)")
    st.write("此功能正在建设中...")

