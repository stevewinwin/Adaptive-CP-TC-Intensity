import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from typhoon_transformer import TransformerRegressor
import matplotlib.pyplot as plt
from tabulate import tabulate
import torch

def process_and_evaluate(data_file, label):
    """处理数据并评估模型性能"""
    print(f"\n===== 【{label}】({data_file}) 预测性能评估 =====")
    
    target_cols = {
        '24h': 'intensity_truth_024h',
        '18h': 'intensity_truth_018h',
        '12h': 'intensity_truth_012h',
        '6h': 'intensity_truth_006h'
    }
    
    # 读取数据
    data = pd.read_csv(data_file)
    
    # 打印各个basin的分布情况
    for basin in ['EP', 'AL', 'CP']:
        basin_count = (data['basin'] == basin).sum()
        total_count = len(data)
        print(f"basin为{basin}的样本数: {basin_count}")
        print(f"basin为{basin}的样本占比: {basin_count / total_count:.2%}")
    
    # 数据清理
    null_columns = data.columns[data.isnull().all()].tolist()
    data = data.drop(columns=null_columns)
    exclude_columns = [
        'storm_id', 'basin', 'DELV_t0', 'storm_name', 'init_time',
        'intensity_truth_006h', 'intensity_truth_012h', 'intensity_truth_018h', 'intensity_truth_024h',
        'intensity_truth_036h', 'intensity_truth_048h', 'intensity_truth_072h', 'intensity_truth_096h', 'intensity_truth_120h'
    ]
    feature_cols = [col for col in data.columns if col not in exclude_columns and col not in null_columns]
    print(f"使用的特征总数: {len(feature_cols)}")
    
    # 处理缺失值
    if 'lat' in data.columns and 'lon' in data.columns:
        original_len = len(data)
        data = data.dropna(subset=['lat', 'lon'])
        removed_rows = original_len - len(data)
        if removed_rows > 0:
            print(f"\n已删除 {removed_rows} 行包含 lat 或 lon NaN值的数据 ({(removed_rows/original_len)*100:.2f}%)")
    
    # 检查数值型列
    non_numeric_cols = [col for col in feature_cols if not pd.api.types.is_numeric_dtype(data[col])]
    if non_numeric_cols:
        raise ValueError(f"以下特征不是数值型：{non_numeric_cols}")
    
    # 按风暴ID和时间排序
    data = data.sort_values(['storm_id', 'init_time'])
    X = data[feature_cols].values
    
    results = {}
    
    results = {}
    # 已知最佳参数
    best_params_dict = {
        '24h': {'hidden_dim': 128, 'num_layers': 4, 'num_heads': 8, 'dropout': 0.1, 'epochs': 100, 'patience': 10, 'batch_size': 64, 'learning_rate': 0.0005, 'weight_decay': 0.0001, 'verbose': 1, 'target_hour': '24h'},
        '18h': {'hidden_dim': 128, 'num_layers': 4, 'num_heads': 8, 'dropout': 0.2, 'epochs': 100, 'patience': 10, 'batch_size': 64, 'learning_rate': 0.0005, 'weight_decay': 0.0001, 'verbose': 1, 'target_hour': '18h'},
        '12h': {'hidden_dim': 128, 'num_layers': 2, 'num_heads': 4, 'dropout': 0.2, 'epochs': 100, 'patience': 10, 'batch_size': 64, 'learning_rate': 0.0005, 'weight_decay': 0.0001, 'verbose': 1, 'target_hour': '12h'},
        '6h':  {'hidden_dim': 128, 'num_layers': 2, 'num_heads': 8, 'dropout': 0.1, 'epochs': 100, 'patience': 10, 'batch_size': 64, 'learning_rate': 0.0005, 'weight_decay': 0.0001, 'verbose': 1, 'target_hour': '6h'}
    }
    
    for hour, col in target_cols.items():
        print(f"\n===== 预测 {hour} ({col}) =====")
        # 只保留当前预测时效的有效数据
        valid_data = data.dropna(subset=[col]).copy()
        feature_cols = [col for col in valid_data.columns if col not in exclude_columns and col not in null_columns]
        X_hour = valid_data[feature_cols].values
        y = valid_data[col].values
        
        # 针对原始数据，去除X_hour和y中有缺失值的样本
        if label == "原始数据":
            valid_rows = (~np.isnan(X_hour).any(axis=1)) & (~np.isnan(y))
            if valid_rows.sum() < len(X_hour):
                print(f"\n原始数据中移除了 {len(X_hour) - valid_rows.sum()} 行包含NaN的样本 ({(len(X_hour) - valid_rows.sum())/len(X_hour)*100:.2f}%)")
                X_hour = X_hour[valid_rows]
                y = y[valid_rows]
                valid_data = valid_data.iloc[valid_rows].reset_index(drop=True)
        
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        # 按storm_id进行分层抽样
        storm_basins = valid_data.groupby('storm_id')['basin'].first()
        train_storms, calib_storms, val_storms, test_storms = [], [], [], []
        
        # 为每个basin分别进行storm级别的划分
        for basin in ['EP', 'AL', 'CP']:
            basin_storms = storm_basins[storm_basins == basin].index.tolist()
            if not basin_storms:  # 如果这个basin没有风暴，跳过
                continue
                
            # 按风暴最早的init_time排序
            storm_first_time = valid_data[valid_data['storm_id'].isin(basin_storms)].groupby('storm_id')['init_time'].min()
            sorted_storms = storm_first_time.sort_values().index.tolist()
            n = len(sorted_storms)
            n_train = int(n * 0.6)
            n_calib = int(n * 0.1)
            n_val = int(n * 0.1)
            n_test = n - n_train - n_calib - n_val
            
            train_storms.extend(sorted_storms[:n_train])
            calib_storms.extend(sorted_storms[n_train:n_train + n_calib])
            val_storms.extend(sorted_storms[n_train + n_calib:n_train + n_calib + n_val])
            test_storms.extend(sorted_storms[n_train + n_calib + n_val:])
        train_mask = valid_data['storm_id'].isin(train_storms)
        calib_mask = valid_data['storm_id'].isin(calib_storms)
        val_mask = valid_data['storm_id'].isin(val_storms)
        test_mask = valid_data['storm_id'].isin(test_storms)

        X_train, y_train = X_hour[train_mask], y[train_mask]
        X_val, y_val = X_hour[val_mask], y[val_mask]
        X_test, y_test = X_hour[test_mask], y[test_mask]
        X_calib, y_calib = X_hour[calib_mask], y[calib_mask]
        
        print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}, 校准集: {X_calib.shape}")
        
        print("训练集各basin分布:")
        print(valid_data[train_mask]['basin'].value_counts(normalize=True))
        print("校准集各basin分布:")
        print(valid_data[calib_mask]['basin'].value_counts(normalize=True))
        print("验证集各basin分布:")
        print(valid_data[val_mask]['basin'].value_counts(normalize=True))
        print("测试集各basin分布:")
        print(valid_data[test_mask]['basin'].value_counts(normalize=True))
        
        print("\n各数据集的风暴数量:")
        print(f"训练集: {len(set(valid_data[train_mask]['storm_id']))}个风暴")
        print(f"校准集: {len(set(valid_data[calib_mask]['storm_id']))}个风暴")
        print(f"验证集: {len(set(valid_data[val_mask]['storm_id']))}个风暴")
        print(f"测试集: {len(set(valid_data[test_mask]['storm_id']))}个风暴")
        

        
        params = best_params_dict[hour]
        model = TransformerRegressor(**params)
        # 先初始化模型结构
        model.model = model._build_model(X_test.shape[1])
        # 加载最佳权重
        weight_path = f"best_transformer_model_{hour}.pth"
        model.model.load_state_dict(torch.load(weight_path, map_location=model.device, weights_only=True))
        model.model.eval()
        y_pred = model.predict(X_test)
        
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f'复现测试集 RMSE: {rmse:.4f}')
        print(f'复现测试集 MAE: {mae:.4f}')
        print(f'复现测试集 R2: {r2:.4f}')
        
        # 创建预测结果DataFrame，只包含关键信息
        test_results = pd.DataFrame({
            'storm_id': valid_data[test_mask]['storm_id'],
            'storm_name': valid_data[test_mask]['storm_name'],
            'init_time': valid_data[test_mask]['init_time'],
            'true_intensity': y_test,
            'predicted_intensity': y_pred,
            'error': y_pred - y_test  # 预测值减真实值
        })
        
        # 按风暴ID和时间排序
        test_results = test_results.sort_values(['storm_id', 'init_time'])
        
        # 保存结果到CSV文件
        output_file = f'prediction_results_{hour}.csv'
        test_results.to_csv(output_file, index=False)
        print(f"\n预测结果已保存到: {output_file}")
        
        results[hour] = {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'BestParams': params}
    
    return results

def main():
    # 数据文件
    data_files = [
        ('原始数据', 'combined_ships_dataset_1.csv'),
        ('插补后数据', 'imputed_data.csv')
    ]
    
    # 存储所有结果
    all_results = {}
    
    # 处理每个数据文件
    for label, data_file in data_files:
        try:
            all_results[label] = process_and_evaluate(data_file, label)
        except Exception as e:
            print(f"处理 {data_file} 时出错: {str(e)}")
    
    # 输出对比表格
    if len(all_results) == 2:
        print("\n\n===== 数据处理前后性能对比 =====")
        
        # 准备表格数据
        table_data = []
        headers = ["预测时效", "指标", "处理前", "处理后", "提升百分比"]
        
        # 获取所有时效
        all_hours = set()
        for results in all_results.values():
            all_hours.update(results.keys())
        
        # 按时效排序
        for hour in sorted(all_hours, key=lambda x: int(x.rstrip('h'))):
            # 检查两组数据是否都有该时效的结果
            if hour not in all_results['原始数据'] or hour not in all_results['插补后数据']:
                continue
            
            orig = all_results['原始数据'][hour]
            proc = all_results['插补后数据'][hour]
            
            # 添加RMSE行
            rmse_improve = (orig['RMSE'] - proc['RMSE']) / orig['RMSE'] * 100
            table_data.append([f"{hour}", "RMSE", f"{orig['RMSE']:.4f}", f"{proc['RMSE']:.4f}", f"{rmse_improve:.2f}%"])
            
            # 添加MAE行
            mae_improve = (orig['MAE'] - proc['MAE']) / orig['MAE'] * 100
            table_data.append(["", "MAE", f"{orig['MAE']:.4f}", f"{proc['MAE']:.4f}", f"{mae_improve:.2f}%"])
            
            # 添加R2行
            r2_improve = (proc['R2'] - orig['R2']) / abs(orig['R2']) * 100 if orig['R2'] != 0 else float('inf')
            table_data.append(["", "R2", f"{orig['R2']:.4f}", f"{proc['R2']:.4f}", f"{r2_improve:.2f}%"])
        
        # 输出表格
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # 保存结果到CSV
        results_df = pd.DataFrame(table_data, columns=headers)
        results_df.to_csv('data_processing_comparison.csv', index=False)
        print("\n结果已保存到 data_processing_comparison.csv")

if __name__ == '__main__':
    main()
