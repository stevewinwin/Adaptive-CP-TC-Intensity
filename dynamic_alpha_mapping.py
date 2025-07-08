import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from mapie.regression import MapieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')
from scipy.optimize import minimize
from typhoon_transformer import TransformerRegressor
import torch
import random
from math import exp
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

class DynamicAlphaMapping:
    def __init__(self):
        self.xgb_model = None
        self.mapie_model = None
        self.wind_speed_alpha_map = {}  # 存储不同风速类别对应的alpha值
        self.wind_change_alpha_map = {}  # 存储不同风速变化率类别对应的alpha值
        self.weights = None

    def categorize_wind_speed(self, wind_speed):
        if wind_speed < 34:
            return 'TD'
        elif wind_speed < 64:
            return 'TS'
        elif wind_speed < 83:
            return 'C1'
        elif wind_speed < 96:
            return 'C2'
        elif wind_speed < 113:
            return 'C3'
        else:
            return 'C4'

    def categorize_wind_change(self, delta_v):
        """根据风速变化率进行分类"""
        if delta_v >= 15:
            return 'RI'  # Rapid Intensification
        elif 0 < delta_v < 15:
            return 'SI'  # Slow Intensification
        elif delta_v == 0:
            return 'ST'  # Stable
        elif -15 < delta_v < 0:
            return 'SW'  # Slow Weakening
        else:
            return 'RW'  # Rapid Weakening

    def train_and_predict(self, X_train, X_val, X_test, y_train, y_val, y_test, target_hour):
        """加载各时效最佳权重并预测"""
        best_params_dict = {
            '24h': {'hidden_dim': 128, 'num_layers': 4, 'num_heads': 8, 'dropout': 0.1, 'epochs': 100, 'batch_size': 64, 'patience': 10, 'learning_rate': 0.0005, 'weight_decay': 0.0001, 'verbose': 1},
            '18h': {'hidden_dim': 128, 'num_layers': 4, 'num_heads': 8, 'dropout': 0.2, 'epochs': 100, 'batch_size': 64, 'patience': 10, 'learning_rate': 0.0005, 'weight_decay': 0.0001, 'verbose': 1},
            '12h': {'hidden_dim': 128, 'num_layers': 2, 'num_heads': 4, 'dropout': 0.2, 'epochs': 100, 'batch_size': 64, 'patience': 10, 'learning_rate': 0.0005, 'weight_decay': 0.0001, 'verbose': 1},
            '6h':  {'hidden_dim': 128, 'num_layers': 2, 'num_heads': 8, 'dropout': 0.1, 'epochs': 100, 'batch_size': 64, 'patience': 10, 'learning_rate': 0.0005, 'weight_decay': 0.0001, 'verbose': 1}
        }
        params = best_params_dict.get(target_hour, best_params_dict['24h'])
        regressor = TransformerRegressor(**params)
        regressor.model = regressor._build_model(X_test.shape[1])
        weight_path = f"best_transformer_model_{target_hour}.pth"
        regressor.model.load_state_dict(torch.load(weight_path, map_location=regressor.device, weights_only=True))
        regressor.model.eval()
        y_pred_train = regressor.predict(X_train)
        y_pred_val = regressor.predict(X_val)
        y_pred_test = regressor.predict(X_test)
        return regressor, y_pred_train, y_pred_val, y_pred_test

    def calculate_bin_risks(self, X_val, y_val_dict, y_pred_dict, wind_speeds, delta_vs, bin_func):
        # Step 1: 按类别和目标分别计算绝对误差
        bin_mae = {}
        bin_counts = {}
        targets = list(y_val_dict.keys())
        for i in range(len(X_val)):
            category = bin_func(wind_speeds[i] if bin_func==self.categorize_wind_speed else delta_vs[i])
            if category not in bin_mae:
                bin_mae[category] = {t: [] for t in targets}
                bin_counts[category] = 0
            for t in targets:
                bin_mae[category][t].append(abs(y_pred_dict[t][i] - y_val_dict[t][i]))
            bin_counts[category] += 1
        # Step 2: 计算每类别每目标的MAE和全局MAE，标准化误差
        mae_c_t = {cat: {} for cat in bin_mae}
        mae_t = {}
        for t in targets:
            # 全局MAE（所有类别合并）
            all_errors = []
            for cat in bin_mae:
                all_errors.extend(bin_mae[cat][t])
            mae_t[t] = np.mean(all_errors) if all_errors else 1.0
        error_c_t_std = {cat: {} for cat in bin_mae}
        for cat in bin_mae:
            for t in targets:
                mae = np.mean(bin_mae[cat][t]) if bin_mae[cat][t] else 0
                mae_c_t[cat][t] = mae
                error_c_t_std[cat][t] = mae / (mae_t[t] + 1e-8)
        # Step 3: 等权重汇总（所有权重设为1）
        wt_map = {'intensity_truth_006h': 1.0, 'intensity_truth_012h': 1.0, 'intensity_truth_018h': 1.0, 'intensity_truth_024h': 1.0}
        error_c_weighted = {}
        for cat in bin_mae:
            error_c_weighted[cat] = sum(wt_map[t] * error_c_t_std[cat][t] for t in targets)
        # Step 4: 归一化
        weighted_errors = list(error_c_weighted.values())
        err_min, err_max = min(weighted_errors), max(weighted_errors)
        error_c_norm = {}
        for cat in error_c_weighted:
            if err_max > err_min:
                error_c_norm[cat] = (error_c_weighted[cat] - err_min) / (err_max - err_min)
            else:
                error_c_norm[cat] = 0.5
        # Step 5: alpha映射
        alpha_map = {}
        for cat in error_c_norm:
            # 使用sigmoid函数映射alpha
            a = 0.3  # 最大alpha
            b = 0.05  # 最小alpha
            k = 10    # 陡峭度参数
            x = error_c_norm[cat]
            alpha = b + (a - b) / (1 + np.exp(k * (x - 0.5)))
            alpha = min(max(alpha, b), a)
            alpha_map[cat] = alpha
        # 详细信息用于打印
        bin_details = {}
        for cat in bin_mae:
            bin_details[cat] = {}
            for t in targets:
                bin_details[cat][t] = (mae_c_t[cat][t], error_c_t_std[cat][t])
        return error_c_norm, bin_details, alpha_map, bin_counts

    def bin_mapping_wind_speed(self, X_val, y_val_dict, y_pred_dict, wind_speeds):
        """基于风速分箱的alpha映射"""
        # 计算每个分箱的综合标准化风险
        bin_risks, bin_details, alpha_map, bin_counts = self.calculate_bin_risks(
            X_val, y_val_dict, y_pred_dict, wind_speeds, None, self.categorize_wind_speed)

        targets = list(y_val_dict.keys())
        # 提取目标简称用于打印
        target_short = [t[-4:] if t.startswith('intensity_truth_') else t for t in targets]
        print("\n风速分箱的风险统计和alpha值:")
        # 新表头，增加MAE列
        header = f"{'类别':<15}{'样本数':<8}{'综合风险':>10}{'Alpha值':>10}{'MAE':>10}{'标准化误差':>10}"
        print(header)
        print('-' * len(header))
        self.wind_speed_alpha_map = {}
        for cat in sorted(bin_risks.keys()):
            risk = bin_risks[cat]
            alpha = alpha_map[cat]
            # 只输出第一个目标的MAE和标准化误差
            mae = bin_details[cat][targets[0]][0]
            std_risk = bin_details[cat][targets[0]][1]
            print(f"{cat:<15}{bin_counts.get(cat, 0):<8}{risk:>10.4f}{alpha:>10.4f}{mae:>10.2f}{std_risk:>10.2f}")
            self.wind_speed_alpha_map[cat] = alpha

    def bin_mapping_wind_change(self, X_val, y_val_dict, y_pred_dict, delta_vs):
        """基于风速变化率分箱的alpha映射"""
        # 计算每个分箱的综合标准化风险
        bin_risks, bin_details, alpha_map, bin_counts = self.calculate_bin_risks(
            X_val, y_val_dict, y_pred_dict, None, delta_vs, self.categorize_wind_change)

        targets = list(y_val_dict.keys())
        target_short = [t[-4:] if t.startswith('intensity_truth_') else t for t in targets]
        print("\n风速变化率分箱的风险统计和alpha值:")
        # 新表头，增加MAE列
        header = f"{'变化率类别':<15}{'样本数':<8}{'综合风险':>10}{'Alpha值':>10}{'MAE':>10}{'标准化误差':>10}"
        print(header)
        print('-' * len(header))
        self.wind_change_alpha_map = {}
        for cat in sorted(bin_risks.keys()):
            risk = bin_risks[cat]
            alpha = alpha_map[cat]
            mae = bin_details[cat][targets[0]][0]
            std_risk = bin_details[cat][targets[0]][1]
            print(f"{cat:<15}{bin_counts.get(cat, 0):<8}{risk:>10.4f}{alpha:>10.4f}{mae:>10.2f}{std_risk:>10.2f}")
            self.wind_change_alpha_map[cat] = alpha

    def print_weighted_risk_table(self, all_bin_mae, all_bin_counts, all_weighted_risk, all_alpha, targets, targets_short):
        """打印所有时效下的分箱加权风险和alpha表"""
        # 已废弃：不再输出加权风险和alpha表
        pass

    def calculate_and_print_all_weighted_risk(self, X_val_dict, y_val_dict, y_pred_val_dict, feature_cols_dict, target_cols, hours):
        # 已废弃：不再进行加权风险和alpha计算
        pass

    def calculate_and_print_all_weighted_risk_windchange(self, X_val_dict, y_val_dict, y_pred_val_dict, feature_cols_dict, target_cols, hours):
        # 已废弃：不再进行加权风险和alpha计算
        pass

def prepare_data(target_hour):
    """准备训练和测试数据，为Transformer模型创建特征集，并输出各basin分布和特征信息"""
    data = pd.read_csv('imputed_data.csv')
    target_cols = {
        '24h': 'intensity_truth_024h',
        '18h': 'intensity_truth_018h',
        '12h': 'intensity_truth_012h',
        '6h': 'intensity_truth_006h'
    }
    
    # 输出各basin样本数和占比
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
    
    # 按storm_id进行分层抽样
    unique_storms = data['storm_id'].unique()
    storm_basins = data.groupby('storm_id')['basin'].first()
    
    # 为每个basin分别进行storm级别的划分
    train_storms = []
    val_storms = []
    test_storms = []
    
    for basin in ['EP', 'AL', 'CP']:
        basin_storms = storm_basins[storm_basins == basin].index.tolist()
        # 按风暴最早的init_time排序
        storm_first_time = data[data['storm_id'].isin(basin_storms)].groupby('storm_id')['init_time'].min()
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
        val_storms.extend(val_storms_basin)
        test_storms.extend(test_storms_basin)
    
    # 只保留当前预测时效的有效数据
    valid_data = data.dropna(subset=[target_cols[target_hour]]).copy()
    
    # 创建训练集、验证集和测试集的掩码
    train_mask = valid_data['storm_id'].isin(train_storms)
    val_mask = valid_data['storm_id'].isin(val_storms)
    test_mask = valid_data['storm_id'].isin(test_storms)
    
    # 提取特征和目标变量
    X_hour = valid_data[feature_cols].values
    y = valid_data[target_cols[target_hour]].values
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 使用掩码来分割数据
    X_train, y_train = X_hour[train_mask], y[train_mask]
    X_val, y_val = X_hour[val_mask], y[val_mask]
    X_test, y_test = X_hour[test_mask], y[test_mask]
    
    print(f"\n===== 预测 {target_hour} ({target_cols[target_hour]}) =====")
    print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
    
    print("训练集各basin分布:")
    print(valid_data[train_mask]['basin'].value_counts(normalize=True))
    print("验证集各basin分布:")
    print(valid_data[val_mask]['basin'].value_counts(normalize=True))
    print("测试集各basin分布:")
    print(valid_data[test_mask]['basin'].value_counts(normalize=True))
    
    print("\n各数据集的风暴数量:")
    print(f"训练集: {len(set(valid_data[train_mask]['storm_id']))}个风暴")
    print(f"验证集: {len(set(valid_data[val_mask]['storm_id']))}个风暴")
    print(f"测试集: {len(set(valid_data[test_mask]['storm_id']))}个风暴")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols

def main():
    target_cols = {
        '24h': 'intensity_truth_024h',
        '18h': 'intensity_truth_018h',
        '12h': 'intensity_truth_012h',
        '6h': 'intensity_truth_006h'
    }
    
    # 初始化模型
    model = DynamicAlphaMapping()
    
    # 对每个预测时效分别处理
    for hour in ['6h', '12h', '18h', '24h']:
        print(f"\n===== 处理 {hour} 预测时效 =====")
        
        # 使用prepare_data获取数据
        X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = prepare_data(hour)
        
        # 训练和预测
        print("\n训练基础模型...")
        regressor, y_pred_train, y_pred_val, y_pred_test = model.train_and_predict(
            X_train, X_val, X_test, y_train, y_val, y_test, hour
        )
        
        # 获取风速和风速变化率
        wind_speeds_val = X_val[:, feature_cols.index('intensity')]
        delta_vs_val = X_val[:, feature_cols.index('DELV-12')]
        
        # 计算alpha映射
        print(f"\n—— {hour} 时效的alpha映射 ——")
        target = target_cols[hour]
        
        # 风速分箱的alpha映射
        model.bin_mapping_wind_speed(
            X_val, 
            {target: y_val}, 
            {target: y_pred_val}, 
            wind_speeds_val
        )
        
        # 风速变化率分箱的alpha映射
        model.bin_mapping_wind_change(
            X_val, 
            {target: y_val}, 
            {target: y_pred_val}, 
            delta_vs_val
        )
        
        # 收集alpha值
        alpha_rows = []
        for cat, alpha in model.wind_speed_alpha_map.items():
            alpha_rows.append({'hour': hour, 'type': 'wind', 'category': cat, 'alpha': alpha})
        for cat, alpha in model.wind_change_alpha_map.items():
            alpha_rows.append({'hour': hour, 'type': 'windchg', 'category': cat, 'alpha': alpha})
        
        # 保存当前时效的alpha映射
        df_alpha = pd.DataFrame(alpha_rows)
        df_alpha = df_alpha.sort_values(['hour', 'type', 'category'])
        df_alpha['alpha'] = df_alpha['alpha'].round(2)
        
        if hour == '6h':  # 第一次运行时创建文件
            df_alpha.to_csv('alpha_mapping.csv', index=False, mode='w')
        else:  # 后续追加到文件
            df_alpha.to_csv('alpha_mapping.csv', index=False, mode='a', header=False)
    
    print('\n已将所有alpha值输出到 alpha_mapping.csv')

    # 绘制sigmoid曲线
    plt.figure(figsize=(12, 5))
    
    # First subplot: Sigmoid curves
    plt.subplot(1, 2, 1)
    a, b = 0.3, 0.05  # alpha range
    x = np.linspace(0, 1, 200)  # normalized risk
    for k in [2, 4, 6, 8, 10]:
        alpha = b + (a - b) / (1 + np.exp(k * (x - 0.5)))
        plt.plot(x, alpha, label=f'k={k}')
    plt.xlabel('Normalized Risk')
    plt.ylabel('Alpha')
    plt.title('Sigmoid Mapping')
    plt.legend()
    plt.grid(True)

    # Second subplot: Comparison with other functions
    plt.subplot(1, 2, 2)
    # Sigmoid (k=6)
    alpha_sigmoid = b + (a - b) / (1 + np.exp(6 * (x - 0.5)))
    plt.plot(x, alpha_sigmoid, label='Sigmoid')
    # Linear
    alpha_linear = b + (a - b) * (1 - x)
    plt.plot(x, alpha_linear, label='Linear')
    # Exponential
    alpha_exp = b + (a - b) * np.exp(-3 * x)
    plt.plot(x, alpha_exp, label='Exponential')
    # Step function
    alpha_step = np.where(x < 0.5, a, b)
    plt.plot(x, alpha_step, label='Step')
    
    plt.xlabel('Normalized Risk')
    plt.ylabel('Alpha')
    plt.title('Different Mapping Functions')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('sigmoid_mapping.png')
    plt.close()

if __name__ == "__main__":
    main()
