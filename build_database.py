
import pandas as pd
import duckdb
import os

# 定义文件路径
# 核心修正：将所有路径转移到不含中文字符的简单路径，以规避DuckDB在Windows上的路径编码BUG
PROJECT_DIR = r"C:\Users\Administrator\Desktop\沃尔玛项目STREAMLIT" # 原始CSV的路径依然不变
CSV_PATH = os.path.join(PROJECT_DIR, "walmart_cleaned_for_upload.csv")

# 数据库和临时文件将被创建在C盘根目录
CLEAN_CSV_PATH = r"C:\__temp_clean_walmart_for_duckdb.csv"
DB_PATH = r"C:\walmart.db"

def build_database_via_airlock():
    """
    通过“数据气闸”方法，创建最终的数据库，以解决顽固的编码问题。
    """
    try:
        print(f"--- 正在以'latin1'编码读取源CSV文件: {CSV_PATH}")
        df = pd.read_csv(CSV_PATH, encoding='latin1')
        print("--- CSV文件读取成功。")

        print("--- 正在对'Type'列进行One-Hot编码...")
        df = pd.get_dummies(df, columns=['Type'], drop_first=True)
        df.rename(columns={'Type_B': 'type_b', 'Type_C': 'type_c'}, inplace=True)
        print("--- One-Hot编码完成。")

        print("--- 正在将'Date'列转换为日期格式...")
        df['Date'] = pd.to_datetime(df['Date'])
        print("--- 日期格式转换完成。")

        print(f"--- 正在创建UTF-8净化文件: {CLEAN_CSV_PATH}")
        df.to_csv(CLEAN_CSV_PATH, index=False, encoding='utf-8')
        print("--- 净化文件创建成功。")

        print(f"--- 准备从净化文件创建/覆盖DuckDB数据库: {DB_PATH}")
        con = duckdb.connect(database=DB_PATH, read_only=False)
        
        table_name = "walmart"
        sql_query = f"""CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_csv_auto('{CLEAN_CSV_PATH.replace('\\', '/')}', header=true);"""
        con.execute(sql_query)
        
        con.close()
        print(f"\n*** DuckDB数据库 '{os.path.basename(DB_PATH)}' 创建/更新成功！***")

    finally:
        if os.path.exists(CLEAN_CSV_PATH):
            os.remove(CLEAN_CSV_PATH)
            print(f"--- 临时净化文件 {CLEAN_CSV_PATH} 已删除。")

if __name__ == "__main__":
    build_database_via_airlock()
