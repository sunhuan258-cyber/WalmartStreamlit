
import pandas as pd
from sqlalchemy import create_engine
import time

# =====================================================================================
# 数据库凭证配置
# 请在这里填入您自己的数据库信息
# =====================================================================================
DB_USER = "postgres"
DB_PASSWORD = "1378701asEW"  # <--- 在这里替换为您保存的真实密码
DB_HOST = "cpcxwxszecknkovlhbmy.supabase.co" # <--- 这是我们确认的主机地址
DB_PORT = "5432"
DB_NAME = "postgres"

# 构建数据库连接URL
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# =====================================================================================
# 数据加载与处理函数
# =====================================================================================
def load_and_prepare_data():
    print("--- 正在加载本地CSV文件...")
    # 定义文件路径
    train_path = r'C:/Users/Administrator/Desktop/沃尔玛项目/train.csv'
    features_path = r'C:/Users/Administrator/Desktop/沃尔玛项目/features.csv'
    stores_path = r'C:/Users/Administrator/Desktop/沃尔玛项目/stores.csv'

    # 加载数据
    train_df = pd.read_csv(train_path)
    features_df = pd.read_csv(features_path)
    stores_df = pd.read_csv(stores_path)
    print("--- CSV文件加载完毕。")

    print("--- 正在进行数据预处理与合并...")
    # 数据预处理
    features_df['Date'] = pd.to_datetime(features_df['Date'], format='%Y-%m-%d')
    train_df['Date'] = pd.to_datetime(train_df['Date'], format='%Y-%m-%d')

    # 合并数据
    df_merged = pd.merge(train_df, stores_df, on='Store', how='left')
    final_df = pd.merge(df_merged, features_df, on=['Store', 'Date', 'IsHoliday'], how='left')

    # 清理负销售额
    final_df = final_df[final_df['Weekly_Sales'] >= 0]

    # 填充缺失值
    markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    final_df[markdown_cols] = final_df[markdown_cols].fillna(0)
    
    # CPI和Unemployment的缺失值用中位数填充，更稳健
    final_df['CPI'] = final_df['CPI'].fillna(final_df['CPI'].median())
    final_df['Unemployment'] = final_df['Unemployment'].fillna(final_df['Unemployment'].median())

    print("--- 数据处理完毕。")
    return final_df

# =====================================================================================
# 主执行逻辑
# =====================================================================================
if __name__ == "__main__":
    # 1. 加载和处理数据
    final_df = load_and_prepare_data()
    
    # 2. 定义输出路径
    output_path = "C:/Users/Administrator/Desktop/沃尔玛项目STREAMLIT/walmart_cleaned_for_upload.csv"
    print(f"--- 准备将 {len(final_df)} 条清洗后的数据保存到本地CSV文件...")
    print(f"--- 文件路径: {output_path}")

    # 3. 保存到CSV
    try:
        final_df.to_csv(output_path, index=False)
        print(f"\n*** 本地CSV文件生成成功！***")
        print("--- 下一步，我们将在Supabase UI中手动上传这个文件。")
    except Exception as e:
        print(f"\nXXX 文件保存失败。错误信息: {e}")
