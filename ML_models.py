import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
import utils
import models
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold, train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.preprocessing import StandardScaler, PowerTransformer, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge
import time
from typhoon_transformer import TransformerRegressor
import torch

# 配置 GPU 内存增长
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)

def clear_session():
    """清理 Keras 后端会话"""
    tf.keras.backend.clear_session()

# 读取CSV数据
df = pd.read_csv('imputed_data.csv')

# 检查并排除完全缺失的特征列
null_columns = df.columns[df.isnull().all()].tolist()
if null_columns:
    print("\n发现完全缺失的特征列:")
    for col in null_columns:
        print(f"- {col}")
    # 从数据中删除这些列
    df = df.drop(columns=null_columns)

# 排除非特征列和目标变量列
exclude_columns = ['storm_id', 'basin', 'storm_name', 'init_time', 
                  'intensity_truth_006h', 'intensity_truth_012h', 
                  'intensity_truth_018h', 'intensity_truth_024h',
                  'intensity_truth_036h', 'intensity_truth_048h',
                  'intensity_truth_072h', 'intensity_truth_096h',
                  'intensity_truth_120h']

# 获取所有可用特征（排除非特征列、目标变量列和完全缺失的列）
feature_columns = [col for col in df.columns if col not in exclude_columns and col not in null_columns]
print(f"\n使用的特征总数: {len(feature_columns)}")

X = df[feature_columns]

# 目标变量列表
targets = ['intensity_truth_006h', 'intensity_truth_012h', 'intensity_truth_018h', 'intensity_truth_024h']

# 初始化结果字典
all_results = {}

# 各时效最佳参数
best_params_dict = {
    '24h': {'hidden_dim': 128, 'num_layers': 4, 'num_heads': 8, 'dropout': 0.1, 'epochs': 100, 'batch_size': 64, 'patience': 10, 'learning_rate': 0.0005, 'weight_decay': 0.0001, 'verbose': 1},
    '18h': {'hidden_dim': 128, 'num_layers': 4, 'num_heads': 8, 'dropout': 0.2, 'epochs': 100, 'batch_size': 64, 'patience': 10, 'learning_rate': 0.0005, 'weight_decay': 0.0001, 'verbose': 1},
    '12h': {'hidden_dim': 128, 'num_layers': 2, 'num_heads': 4, 'dropout': 0.2, 'epochs': 100, 'batch_size': 64, 'patience': 10, 'learning_rate': 0.0005, 'weight_decay': 0.0001, 'verbose': 1},
    '6h':  {'hidden_dim': 128, 'num_layers': 2, 'num_heads': 8, 'dropout': 0.1, 'epochs': 100, 'batch_size': 64, 'patience': 10, 'learning_rate': 0.0005, 'weight_decay': 0.0001, 'verbose': 1}
}

def split_data(df, target_cols):
    data = df.copy()
    # 输出basin为EP、AL、CP的样本数和占比
    for basin in ['EP', 'AL', 'CP']:
        basin_count = (data['basin'] == basin).sum()
        total_count = len(data)
        print(f"basin为{basin}的样本数: {basin_count}")
        print(f"basin为{basin}的样本占比: {basin_count / total_count:.2%}")
    null_columns = data.columns[data.isnull().all()].tolist()
    data = data.drop(columns=null_columns)
    exclude_columns = [
        'storm_id', 'basin', 'DELV_t0', 'storm_name', 'init_time',
        'intensity_truth_006h', 'intensity_truth_012h', 'intensity_truth_018h', 'intensity_truth_024h',
        'intensity_truth_036h', 'intensity_truth_048h', 'intensity_truth_072h', 'intensity_truth_096h', 'intensity_truth_120h'
    ]
    feature_cols = [col for col in data.columns if col not in exclude_columns and col not in null_columns]
    print(f"使用的特征总数: {len(feature_cols)}")
    if 'lat' in data.columns and 'lon' in data.columns:
        original_len = len(data)
        data = data.dropna(subset=['lat', 'lon'])
        removed_rows = original_len - len(data)
        if removed_rows > 0:
            print(f"\n已删除 {removed_rows} 行包含 lat 或 lon NaN值的数据 ({(removed_rows/original_len)*100:.2f}%)")
    target_col_list = list(target_cols.values())
    non_numeric_cols = [col for col in feature_cols if not pd.api.types.is_numeric_dtype(data[col])]
    if non_numeric_cols:
        raise ValueError(f"以下特征不是数值型：{non_numeric_cols}")
    data = data.sort_values(['storm_id', 'init_time'])
    X_all = data[feature_cols].values
    splits = {}
    test_data_info = {}
    
    for hour, col in target_cols.items():
        print(f"\n===== 预测 {hour} ({col}) =====")
        valid_data = data.dropna(subset=[col]).copy()
        X = valid_data[feature_cols].values
        y = valid_data[col].values
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        unique_storms = valid_data['storm_id'].unique()
        storm_basins = valid_data.groupby('storm_id')['basin'].first()
        train_storms, calib_storms, val_storms, test_storms = [], [], [], []
        for basin in ['EP', 'AL', 'CP']:
            basin_storms = storm_basins[storm_basins == basin].index.tolist()
            # 按风暴最早的init_time排序
            storm_first_time = valid_data[valid_data['storm_id'].isin(basin_storms)].groupby('storm_id')['init_time'].min()
            sorted_storms = storm_first_time.sort_values().index.tolist()
            n = len(sorted_storms)
            n_train = int(n * 0.6)
            n_calib = int(n * 0.1)
            n_val = int(n * 0.1)
            n_test = n - n_train - n_calib - n_val
            train_storms_basin = sorted_storms[:n_train]
            calib_storms_basin = sorted_storms[n_train:n_train + n_calib]
            val_storms_basin = sorted_storms[n_train + n_calib:n_train + n_calib + n_val]
            test_storms_basin = sorted_storms[n_train + n_calib + n_val:]
            train_storms.extend(train_storms_basin)
            calib_storms.extend(calib_storms_basin)
            val_storms.extend(val_storms_basin)
            test_storms.extend(test_storms_basin)
        train_mask = valid_data['storm_id'].isin(train_storms)
        calib_mask = valid_data['storm_id'].isin(calib_storms)
        val_mask = valid_data['storm_id'].isin(val_storms)
        test_mask = valid_data['storm_id'].isin(test_storms)
        
        # 保存测试集数据信息，包括风暴ID、风暴名称和初始时间
        test_data = valid_data[test_mask].copy()
        test_data_info[hour] = {
            'storm_ids': test_data['storm_id'].values,
            'storm_names': test_data['storm_name'].values,
            'init_times': test_data['init_time'].values,
            'basin': test_data['basin'].values,
            'test_mask': test_mask
        }
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_calib, y_calib = X[calib_mask], y[calib_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        print(f"训练集: {X_train.shape}, 校准集: {X_calib.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
        print("训练集各basin分布:")
        print(valid_data[train_mask]['basin'].value_counts(normalize=True))
        print("校准集各basin分布:")
        print(valid_data[calib_mask]['basin'].value_counts(normalize=True))
        print("验证集各basin分布:")
        print(valid_data[val_mask]['basin'].value_counts(normalize=True))
        print("测试集各basin分布:")
        print(valid_data[test_mask]['basin'].value_counts(normalize=True))
        splits[hour] = (X_train, X_calib, X_val, X_test, y_train, y_calib, y_val, y_test, test_data_info[hour])
    feature_cols_out = feature_cols
    return splits, feature_cols_out

def get_optimized_models(X, y, feature_pipeline=None, target_hour=None):
    """获取固定参数的模型"""
    
    # 定义固定参数
    fixed_params = {
        'XGBoost': {
            'n_estimators': 1000,
            'learning_rate': 0.01,
            'max_depth': 9,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.9
        },
        'LightGBM': {
            'n_estimators': 1500,
            'learning_rate': 0.01,
            'max_depth': 9,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.9,
            'num_leaves': 31,
            'boosting_type': 'gbdt'
        },
        'CatBoost': {
            'iterations': 1500,
            'learning_rate': 0.01,
            'depth': 9,
            'verbose': False
        },
        'RandomForest': {
            'n_estimators': 1500,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'n_jobs': -1
        }
    }
    
    # 基础估计器
    base_models = {
        'XGBoost': xgb.XGBRegressor(random_state=42, **fixed_params['XGBoost']),
        'LightGBM': lgb.LGBMRegressor(random_state=42, **fixed_params['LightGBM']),
        'CatBoost': cb.CatBoostRegressor(random_state=42, **fixed_params['CatBoost']),
        'RandomForest': RandomForestRegressor(random_state=42, **fixed_params['RandomForest']),
        'MLP': models.mlp(input_shape=X.shape[1:]),
        'TyphoonTransformer': TransformerRegressor(
            **best_params_dict.get(target_hour, best_params_dict['24h']),
            target_hour=target_hour
        )
    }
    
    optimized_models = {}
    
    # 初始化每个模型
    for name, model in base_models.items():
        print(f"\n初始化 {name} 模型...")
        
        # 如果有特征管道，创建完整的管道
        if feature_pipeline is not None and name not in ['MLP']:
            pipeline = Pipeline([
                ('features', feature_pipeline),
                ('regressor', model)
            ])
            optimized_models[name] = pipeline
        else:
            optimized_models[name] = model
    
    return optimized_models

def evaluate_model(model, X_train, X_calib, X_val, X_test, y_train, y_calib, y_val, y_test, model_name, target_hour=None, storm_ids=None, storm_names=None, init_times=None, basins=None):
    """评估模型性能并保存预测结果"""
    try:
        print(f"\n评估 {model_name}...")
        # 对于 Keras 模型特殊处理
        if model_name in ['MLP']:
            # 标准化数据
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            
            start_time = time.time()
            history = model.fit(
                X_train_scaled, y_train,
                epochs=100,
                batch_size=32,
                validation_data=(X_val_scaled, y_val),
                callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
                verbose=1
            )
            training_time = time.time() - start_time
            
            val_loss = min(history.history['val_loss'])
            epochs = len(history.history['loss'])
            y_pred = model.predict(X_test_scaled).ravel()
            best_params = {}
            
        elif model_name == 'TyphoonTransformer':
            # 自动加载最优权重
            model.model = model._build_model(X_test.shape[1])
            weight_path = f"best_transformer_model_{target_hour}.pth"
            model.model.load_state_dict(torch.load(weight_path, map_location=model.device, weights_only=True))
            model.model.eval()
            y_pred = model.predict(X_test)
            best_params = model.get_params()
            training_time = np.nan
            val_loss = np.nan
            epochs = np.nan
        else:
            start_time = time.time()
            if isinstance(model, GridSearchCV):
                model.fit(X_train, y_train)
                print(f"\n最佳参数组合:")
                for param, value in model.best_params_.items():
                    print(f"- {param}: {value}")
                print(f"最佳交叉验证得分 (MSE): {-model.best_score_:.4f}")
                y_pred = model.predict(X_test)
                best_params = model.best_params_
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                best_params = {}
            
            training_time = time.time() - start_time
            val_loss = np.nan
            epochs = np.nan
        
        # 计算评估指标
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # 保存预测结果到CSV文件
        predictions_df = pd.DataFrame({
            'True_Value': y_test,
            'Predicted_Value': y_pred,
            'Error': y_test - y_pred
        })
        
        # 添加风暴相关信息到预测结果中
        if storm_ids is not None:
            predictions_df['storm_id'] = storm_ids
        if storm_names is not None:
            predictions_df['storm_name'] = storm_names
        if init_times is not None:
            predictions_df['init_time'] = init_times
        if basins is not None:
            predictions_df['basin'] = basins
        
        # 创建目录（如果不存在）
        os.makedirs('model_predictions', exist_ok=True)
        
        # 保存预测结果
        predictions_file = os.path.join('model_predictions', f'{target_hour}_{model_name}_predictions.csv')
        predictions_df.to_csv(predictions_file, index=False)
        print(f"预测结果已保存到 {predictions_file}")
        
        return {
            'model_name': model_name,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'training_time': training_time,
            'val_loss': val_loss,
            'epochs': epochs,
            'best_params': best_params,
            'predictions_file': predictions_file
        }
        
    except Exception as e:
        print(f"评估 {model_name} 时发生错误: {str(e)}")
        return None

def main():
    print("加载数据...")
    
    # ----------- 配置目标列 -------------
    target_cols = {
        '24h': 'intensity_truth_024h',
        '18h': 'intensity_truth_018h',
        '12h': 'intensity_truth_012h',
        '6h': 'intensity_truth_006h'
    }
    splits, feature_cols = split_data(df, target_cols)
    
    # 创建预测结果目录
    os.makedirs('model_predictions', exist_ok=True)
    
    for hour, col in target_cols.items():
        print(f"\n=== 训练目标: {hour} ({col}) ===")
        X_train, X_calib, X_val, X_test, y_train, y_calib, y_val, y_test, test_info = splits[hour]
        
        # 获取测试集对应的风暴信息
        test_storm_ids = test_info['storm_ids']
        test_storm_names = test_info['storm_names']
        test_init_times = test_info['init_times']
        test_basins = test_info['basin']
        
        optimized_models = get_optimized_models(X_train, y_train, target_hour=hour)
        results = []
        for model_name, model in optimized_models.items():
            result = evaluate_model(
                model, X_train, X_calib, X_val, X_test, 
                y_train, y_calib, y_val, y_test, 
                model_name, target_hour=hour, 
                storm_ids=test_storm_ids,
                storm_names=test_storm_names,
                init_times=test_init_times,
                basins=test_basins
            )
            if result is not None:
                results.append(result)
                print(f"\n{model_name} Results:")
                print(f"MAE: {result['mae']:.4f}")
                print(f"RMSE: {result['rmse']:.4f}")
                print(f"R²: {result['r2']:.4f}")
                if not np.isnan(result['val_loss']):
                    print(f"Val Loss: {result['val_loss']:.4f}")
                if not np.isnan(result['epochs']):
                    print(f"Epochs: {result['epochs']}")
                print(f"Training Time: {result['training_time']:.4f}")
                print(f"预测结果已保存到: {result['predictions_file']}")
                if result['best_params']:
                    print("最佳参数组合:")
                    for param, value in result['best_params'].items():
                        print(f"- {param}: {value}")
        
        # 合并所有模型的预测结果到一个文件
        all_predictions = []
        for result in results:
            model_name = result['model_name']
            pred_file = result['predictions_file']
            pred_df = pd.read_csv(pred_file)
            pred_df['Model'] = model_name
            all_predictions.append(pred_df)
        
        if all_predictions:
            combined_predictions = pd.concat(all_predictions, ignore_index=True)
            # 添加额外的测试集信息
            if 'storm_id' not in combined_predictions.columns:
                combined_predictions['storm_id'] = np.tile(test_storm_ids, len(results))
            if 'storm_name' not in combined_predictions.columns:
                combined_predictions['storm_name'] = np.tile(test_storm_names, len(results))
            if 'init_time' not in combined_predictions.columns:
                combined_predictions['init_time'] = np.tile(test_init_times, len(results))
            if 'basin' not in combined_predictions.columns:
                combined_predictions['basin'] = np.tile(test_basins, len(results))
            
            # 保存到以目标变量命名的CSV文件
            combined_file = os.path.join('model_predictions', f'{col}_all_models_predictions.csv')
            combined_predictions.to_csv(combined_file, index=False)
            print(f"\n所有模型的预测结果已合并保存到: {combined_file}")
        
        # 保存评估结果
        results_df = pd.DataFrame([{
            'Model': r['model_name'],
            'MAE': r['mae'],
            'RMSE': r['rmse'],
            'R²': r['r2'],
            'Training Time': r['training_time'],
            'Best Parameters': str(r['best_params'])
        } for r in results])
        os.makedirs('evaluation_results', exist_ok=True)
        output_file = os.path.join('evaluation_results', f'model_evaluation_results_{hour}.csv')
        results_df.to_csv(output_file, index=False)
        print(f"\n评估结果已保存到: {output_file}")
        print("\n所有结果 (按MAE排序):")
        print(results_df.sort_values('MAE'))
        best_models = results_df.nsmallest(3, 'MAE')
        print("\n前3名模型:")
        print(best_models)
        all_results[hour] = results_df
    
    # 汇总所有时间点的评估结果
    summary_results = pd.concat(
        [df.assign(Target=hour) for hour, df in all_results.items()]
    )
    summary_file = os.path.join('evaluation_results', 'all_models_summary.csv')
    summary_results.to_csv(summary_file, index=False)
    print(f"\n汇总评估结果已保存到: {summary_file}")
    
    print("\n所有模型的预测结果已保存到 'model_predictions' 目录中，文件名格式为:")
    print("1. 单模型预测: <时间点>_<模型名称>_predictions.csv")
    print("2. 所有模型合并: <目标变量名>_all_models_predictions.csv")
    print("\n每个预测结果文件包含真实值、预测值、误差、风暴ID、初始时间和海盘信息")

if __name__ == '__main__':
    main()
