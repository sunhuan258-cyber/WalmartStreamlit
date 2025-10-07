import pandas as pd
import numpy as np
import os
import joblib
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

def create_artifacts():
    """
    一个完整的、独立的脚本，用于加载原始数据，执行完整的特征工程，
    训练模型，并最终保存一个包含所有必需工具的、完整的artifact文件。
    """
    print("--- 《凤凰计划》启动 ---")

    # --- 步骤 1: 定义文件路径 ---
    print("\n--- 步骤 1: 定义文件路径 ---")
    base_project_dir = r"C:\Users\Administrator\Desktop\沃尔玛项目"
    train_path = os.path.join(base_project_dir, 'train.csv')
    stores_path = os.path.join(base_project_dir, 'stores.csv')
    features_path = os.path.join(base_project_dir, 'features.csv')
    
    output_dir = r"C:\Users\Administrator\Desktop\沃尔玛项目STREAMLIT"
    artifact_path = os.path.join(output_dir, 'walmart_model_artifacts.joblib')
    print(f"原始数据将从: {base_project_dir} 读取")
    print(f"最终的军备包将保存至: {artifact_path}")

    # --- 步骤 2: 加载并合并原始数据 ---
    print("\n--- 步骤 2: 加载并合并原始数据 ---")
    try:
        train_df = pd.read_csv(train_path)
        stores_df = pd.read_csv(stores_path)
        features_df = pd.read_csv(features_path)
        train_df['Date'] = pd.to_datetime(train_df['Date'], format='%Y-%m-%d')
        features_df['Date'] = pd.to_datetime(features_df['Date'], format='%Y-%m-%d')
        df_merged = pd.merge(train_df, stores_df, on='Store', how='left')
        final_df = pd.merge(df_merged, features_df, on=['Store', 'Date', 'IsHoliday'], how='left')
        print("数据加载与合并成功！")
    except Exception as e:
        print(f"XXX 错误：加载或合并原始数据时失败: {e}")
        return

    # --- 步骤 3: 数据清洗与预处理 ---
    print("\n--- 步骤 3: 数据清洗与预处理 ---")
    final_df = final_df[final_df['Weekly_Sales'] >= 0]
    markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    final_df[markdown_cols] = final_df[markdown_cols].fillna(0)
    final_df_encoded = pd.get_dummies(final_df, columns=['Type'], drop_first=True)
    print("数据清洗与独热编码完成。")

    # --- 步骤 4: 手工特征工程 ---
    print("\n--- 步骤 4: 执行手工特征工程 ---")
    df_for_ml = final_df_encoded.copy()
    df_for_ml['year'] = df_for_ml['Date'].dt.year
    df_for_ml['month'] = df_for_ml['Date'].dt.month
    df_for_ml['week_of_year'] = df_for_ml['Date'].dt.isocalendar().week.astype(int)
    df_for_ml['day_of_week'] = df_for_ml['Date'].dt.dayofweek
    df_for_ml.sort_values(by=['Store', 'Dept', 'Date'], inplace=True)
    df_for_ml['sales_lag_1'] = df_for_ml.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(1)
    df_for_ml['sales_lag_52'] = df_for_ml.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(52)
    df_for_ml['sales_rolling_mean_4'] = df_for_ml.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(lambda x: x.shift(1).rolling(window=4, min_periods=1).mean())
    df_for_ml.dropna(inplace=True)
    print("时间特征、滞后特征创造完毕。")

    # --- 步骤 5: 准备训练/测试数据 ---
    print("\n--- 步骤 5: 准备训练/测试数据与特征列表 ---")
    manual_feature_columns = [col for col in df_for_ml.columns if col not in ['Weekly_Sales', 'Date']]
    X = df_for_ml[manual_feature_columns]
    y = df_for_ml['Weekly_Sales']
    split_date = '2012-01-01'
    train_indices = df_for_ml['Date'] < split_date
    test_indices = df_for_ml['Date'] >= split_date
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")

    # --- 步骤 6: 训练“炼金炉” (Autoencoder) ---
    print("\n--- 步骤 6: 训练“炼金炉” (Autoencoder) ---")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) # 使用在训练集上fit的scaler

    input_dim = X_train_scaled.shape[1]
    encoding_dim = 8
    input_layer = Input(shape=(input_dim,))
    encoder_layer_1 = Dense(16, activation='relu')(input_layer)
    bottleneck_layer = Dense(encoding_dim, activation='relu')(encoder_layer_1)
    encoder = Model(inputs=input_layer, outputs=bottleneck_layer, name="Encoder")
    decoder_layer_1 = Dense(16, activation='relu')(bottleneck_layer)
    output_layer = Dense(input_dim, activation='sigmoid')(decoder_layer_1)
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    print("自编码器架构构建完毕，开始训练...")
    autoencoder.fit(X_train_scaled, X_train_scaled, epochs=15, batch_size=64, shuffle=True, verbose=0) # 静默训练
    print("“炼金炉”训练完成！")

    # --- 步骤 7: 创造“混合特征联军” ---
    print("\n--- 步骤 7: 创造“混合特征联军” ---")
    X_train_encoded = encoder.predict(X_train_scaled)
    X_test_encoded = encoder.predict(X_test_scaled)
    auto_feature_names = [f'auto_feat_{i}' for i in range(encoding_dim)]
    hybrid_feature_names = manual_feature_columns + auto_feature_names
    X_train_hybrid = pd.DataFrame(np.concatenate([X_train.values, X_train_encoded], axis=1), columns=hybrid_feature_names, dtype=float)
    X_test_hybrid = pd.DataFrame(np.concatenate([X_test.values, X_test_encoded], axis=1), columns=hybrid_feature_names, dtype=float)
    print(f"混合特征联军组建完毕，共 {len(hybrid_feature_names)} 个特征。")

    # --- 步骤 8: 训练并评估最终的LightGBM模型 ---
    print("\n--- 步骤 8: 使用最佳参数训练并评估最终模型 ---")
    best_params_hybrid = {
        'learning_rate': 0.08557, 'num_leaves': 59, 'max_depth': 7, 
        'min_child_samples': 78, 'feature_fraction': 0.9842, 'bagging_fraction': 0.9570, 
        'bagging_freq': 4, 'lambda_l1': 1.979e-07, 'lambda_l2': 1.634e-08,
        'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 5000,
        'random_state': 42, 'n_jobs': -1
    }

    final_lgb_model_hybrid = lgb.LGBMRegressor(**best_params_hybrid)
    final_lgb_model_hybrid.fit(X_train_hybrid, y_train, 
                               eval_set=[(X_test_hybrid, y_test)],
                               eval_metric='rmse',
                               callbacks=[lgb.early_stopping(100, verbose=False)])
    print("最终模型训练完成！")

    # --- 步骤 9: 最终性能验证 ---
    print("\n--- 步骤 9: 最终性能验证 ---")
    y_pred_final = final_lgb_model_hybrid.predict(X_test_hybrid)
    final_rmse = np.sqrt(mean_squared_error(y_test, y_pred_final))
    final_mae = mean_absolute_error(y_test, y_pred_final)
    print("\n" + "="*20 + " 《凤凰计划》最终战报 " + "="*20)
    print(f"最终模型在测试集上的 RMSE (均方根误差): ${final_rmse:,.2f}")
    print(f"最终模型在测试集上的 MAE (平均绝对误差): ${final_mae:,.2f}")
    print("="*65)

    # --- 步骤 10: 封装并保存所有军备 ---
    print("\n--- 步骤 10: 正在封装并保存所有军备... ---")
    artifacts = {
        'model': final_lgb_model_hybrid,
        'scaler': scaler,
        'encoder': encoder,
        'feature_names': hybrid_feature_names
    }
    joblib.dump(artifacts, artifact_path)
    print(f"--- 军备封装完成！---")
    print(f"最终的“工具箱”已成功保存至: {artifact_path}")
    print("\n--- 《凤凰计划》执行完毕 ---")

if __name__ == "__main__":
    create_artifacts()