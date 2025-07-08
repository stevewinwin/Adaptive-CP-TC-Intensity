import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from typhoon_transformer import TransformerRegressor
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import torch

# Load alpha mapping at module level
alpha_df = pd.read_csv('alpha_mapping.csv')

def conformal_interval(y_val, y_pred_val, y_pred_test, alpha=0.1):
    """标准 Conformal Prediction 区间生成
    
    Args:
        y_val: 验证集真实值
        y_pred_val: 验证集预测值
        y_pred_test: 测试集预测值
        alpha: 显著性水平 (可以是标量或数组)
    
    Returns:
        lower: 预测区间下界
        upper: 预测区间上界
        alphas: 使用的alpha值
    """
    residuals = y_val - y_pred_val  # 使用验证集计算残差分布
    
    if isinstance(alpha, (float, int)):
        # 计算上下分位数
        q_lower = np.quantile(residuals, alpha/2)
        q_upper = np.quantile(residuals, 1 - alpha/2)
        
        # 构建预测区间
        lower = y_pred_test + q_lower  # 加上下分位数
        upper = y_pred_test + q_upper  # 加上上分位数
        alphas = np.full_like(y_pred_test, alpha, dtype=float)
    else:
        lower, upper, alphas = [], [], []
        for i, a in enumerate(alpha):
            # 对每个样本分别计算分位数
            q_lower = np.quantile(residuals, a/2)
            q_upper = np.quantile(residuals, 1 - a/2)
            
            # 构建预测区间
            lower.append(y_pred_test[i] + q_lower)
            upper.append(y_pred_test[i] + q_upper)
            alphas.append(a)
            
        lower = np.array(lower)
        upper = np.array(upper)
        alphas = np.array(alphas)
    
    return lower, upper, alphas

def dynamic_alpha(intensity, delv, hour, w_intensity, w_delv):
    """
    Dynamically generate alpha based on intensity and change rate using values from alpha_mapping.csv
    
    Args:
        intensity: Current intensity value
        delv: Intensity change rate
        hour: Forecast hour ('6h', '12h', '18h', '24h')
        w_intensity: Weight for intensity
        w_delv: Weight for change rate
    """
    # Get wind speed category
    if intensity >= 113:
        wind_cat = 'C4'
    elif intensity >= 96:
        wind_cat = 'C3'
    elif intensity >= 83:
        wind_cat = 'C2'
    elif intensity >= 64:
        wind_cat = 'C1'
    elif intensity >= 34:
        wind_cat = 'TS'
    else:
        wind_cat = 'TD'

    # Get wind change category
    if delv == 0:
        change_cat = 'ST'  # Stable
    elif delv >= 15:
        change_cat = 'RI'  # Rapid Intensification
    elif delv > 0:
        change_cat = 'SI'  # Slow Intensification
    elif delv > -15:
        change_cat = 'SW'  # Slow Weakening
    else:
        change_cat = 'RW'  # Rapid Weakening

    # Get alpha values from the DataFrame for the specific hour
    alpha_i = alpha_df[(alpha_df['hour'] == hour) & 
                      (alpha_df['type'] == 'wind') & 
                      (alpha_df['category'] == wind_cat)]['alpha'].iloc[0]
    
    alpha_d = alpha_df[(alpha_df['hour'] == hour) & 
                      (alpha_df['type'] == 'windchg') & 
                      (alpha_df['category'] == change_cat)]['alpha'].iloc[0]

    # Fixed weights combination
    alpha = w_intensity * alpha_i + w_delv * alpha_d
    return alpha

def winkler_score_func(y_true, lower, upper, alpha=0.1):
    """计算标准 Winkler Score
    
    Wt,h = {
        Bt,h                      for yt,h ∈ [Lt,h, Ut,h]
        Bt,h + 2/α(Lt,h - yt,h)  for yt,h < Lt,h
        Bt,h + 2/α(yt,h - Ut,h)  for yt,h > Ut,h
    }
    
    其中：
    - Bt,h 是区间宽度 (Ut,h - Lt,h)
    - Lt,h 是下界
    - Ut,h 是上界
    - yt,h 是真实值
    - α 是置信水平
    
    Args:
        y_true: 真实值
        lower: 预测区间下界
        upper: 预测区间上界
        alpha: 置信水平 (可以是标量或数组)
    
    Returns:
        float: 平均 Winkler Score
    """
    y_true = np.array(y_true)
    lower = np.array(lower)
    upper = np.array(upper)
    alpha = np.array(alpha) if isinstance(alpha, (list, np.ndarray)) else np.full_like(y_true, alpha, dtype=float)
    
    # 计算区间宽度 Bt,h
    interval_width = upper - lower
    
    # 初始化所有点的分数为区间宽度
    scores = interval_width.copy()
    
    # 对区间外的点添加惩罚项
    below = y_true < lower
    above = y_true > upper
    
    # yt,h < Lt,h 的情况
    scores[below] += 2/alpha[below] * (lower[below] - y_true[below])
    
    # yt,h > Ut,h 的情况
    scores[above] += 2/alpha[above] * (y_true[above] - upper[above])
    
    return np.mean(scores)

def optimize_weight(X_train, y_train, X_val, y_val, val_mask, valid_data, y_pred_train, y_pred_val, current_hour):
    """使用网格搜索找到最优权重组合（以区间宽度/覆盖率为主目标）"""
    w_range = np.arange(0.1, 1.0, 0.1)  # 权重从0.1到0.9，步长0.1
    best_score = float('inf')
    best_w_intensity = None
    best_w_delv = None
    best_ratio = float('inf')
    best_ratio_w_intensity = None
    best_ratio_w_delv = None
    # 存储所有组合的结果
    all_results = []
    # 获取验证集数据
    val_data = valid_data[val_mask]
    for w_intensity in w_range:
        w_delv = 1 - w_intensity
        alphas = []
        # 对验证集中的每个样本计算alpha
        for _, row in val_data.iterrows():
            intensity = row['intensity'] if 'intensity' in row else row['intensity_knots']
            delv = row['DELV-12'] if 'DELV-12' in row else row['delv_knots']
            alphas.append(dynamic_alpha(intensity, delv, current_hour, w_intensity, w_delv))
        alphas = np.array(alphas)
        # 使用验证集的误差分布来构建conformal区间
        lower, upper, _ = conformal_interval(y_val, y_pred_val, y_pred_val, alpha=alphas)
        # 计算评估指标
        score = winkler_score_func(y_val, lower, upper, alpha=alphas)
        coverage = np.mean((y_val >= lower) & (y_val <= upper))
        mean_width = np.mean(upper - lower)
        # 新增：区间宽度与覆盖率的比值
        width_coverage_ratio = mean_width / coverage if coverage > 0 else float('inf')
        all_results.append({
            'w_intensity': w_intensity,
            'w_delv': w_delv,
            'winkler_score': score,
            'coverage': coverage,
            'mean_width': mean_width,
            'mean_alpha': np.mean(alphas),
            'width_coverage_ratio': width_coverage_ratio
        })
        # 以width_coverage_ratio为主目标
        if width_coverage_ratio < best_ratio:
            best_ratio = width_coverage_ratio
            best_ratio_w_intensity = w_intensity
            best_ratio_w_delv = w_delv
        # 兼容原有winkler最优
        if score < best_score:
            best_score = score
            best_w_intensity = w_intensity
            best_w_delv = w_delv
    # 转换为DataFrame并排序
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('width_coverage_ratio')
    # 打印详细结果
    print(f"\n验证集权重优化结果（{current_hour}）：")
    print(f"最优组合（区间宽度/覆盖率最小）：")
    print(f"  w_intensity = {best_ratio_w_intensity:.1f}")
    print(f"  w_delv = {best_ratio_w_delv:.1f}")
    print(f"  Width/Coverage Ratio = {best_ratio:.4f}")
    print(f"  Coverage = {results_df.iloc[0]['coverage']:.3f}")
    print(f"  Mean Width = {results_df.iloc[0]['mean_width']:.3f}")
    print(f"  Mean Alpha = {results_df.iloc[0]['mean_alpha']:.3f}")
    print(f"  Winkler Score = {results_df.iloc[0]['winkler_score']:.4f}")
    print("\n所有权重组合结果：")
    print(results_df.to_string(index=False))
    # 将权重优化结果存储到全局字典中，以便最后绘制单个大图
    if not hasattr(optimize_weight, 'all_results'):
        optimize_weight.all_results = {}
    
    optimize_weight.all_results[current_hour] = {
        'w_intensity': results_df['w_intensity'].values,
        'winkler_score': results_df['winkler_score'].values,
        'coverage': results_df['coverage'].values,
        'mean_width': results_df['mean_width'].values,
        'width_coverage_ratio': results_df['width_coverage_ratio'].values,
        'best_ratio_w_intensity': best_ratio_w_intensity
    }
    
    # 如果这是最后一个时效，则绘制合并图像
    if len(optimize_weight.all_results) == 4:  # 收集了6h、12h、18h和24h的数据
        hours = ['6h', '12h', '18h', '24h']
        colors = {
            '6h': '#41cef8',
            '12h': 'orange',
            '18h': 'green',
            '24h': '#f8ed41'
        }
        
        # 创建一个4x4的大图
        fig, axs = plt.subplots(4, 4, figsize=(20, 20))
        fig.subplots_adjust(hspace=0.4, wspace=0.3)
        label_fontsize = 18
        title_fontsize = 20
        tick_fontsize = 16
        for i, hour in enumerate(hours):
            data = optimize_weight.all_results[hour]
            # 按w_intensity升序排序
            sort_idx = np.argsort(data['w_intensity'])
            w_intensity = np.array(data['w_intensity'])[sort_idx]
            winkler_score = np.array(data['winkler_score'])[sort_idx]
            coverage = np.array(data['coverage'])[sort_idx]
            mean_width = np.array(data['mean_width'])[sort_idx]
            width_coverage_ratio = np.array(data['width_coverage_ratio'])[sort_idx]
            # 第一列: Winkler Score
            axs[i, 0].plot(w_intensity, winkler_score, '-', marker='o', color=colors[hour])
            axs[i, 0].axvline(x=data['best_ratio_w_intensity'], color='r', linestyle='--', alpha=0.5)
            axs[i, 0].set_xlabel('Intensity Weight', fontsize=label_fontsize)
            axs[i, 0].set_ylabel('Winkler Score', fontsize=label_fontsize)
            axs[i, 0].grid(True)
            axs[i, 0].set_title(f'{hour} - Winkler Score vs Weight', fontsize=title_fontsize)
            axs[i, 0].tick_params(axis='both', labelsize=tick_fontsize)
            # 第二列: Coverage
            axs[i, 1].plot(w_intensity, coverage, '-', marker='o', color=colors[hour])
            axs[i, 1].axvline(x=data['best_ratio_w_intensity'], color='r', linestyle='--', alpha=0.5)
            axs[i, 1].set_xlabel('Intensity Weight', fontsize=label_fontsize)
            axs[i, 1].set_ylabel('Coverage Rate', fontsize=label_fontsize)
            axs[i, 1].grid(True)
            axs[i, 1].set_title(f'{hour} - Coverage vs Weight', fontsize=title_fontsize)
            axs[i, 1].tick_params(axis='both', labelsize=tick_fontsize)
            # 第三列: Mean Width
            axs[i, 2].plot(w_intensity, mean_width, '-', marker='o', color=colors[hour])
            axs[i, 2].axvline(x=data['best_ratio_w_intensity'], color='r', linestyle='--', alpha=0.5)
            axs[i, 2].set_xlabel('Intensity Weight', fontsize=label_fontsize)
            axs[i, 2].set_ylabel('Mean Interval Width', fontsize=label_fontsize)
            axs[i, 2].grid(True)
            axs[i, 2].set_title(f'{hour} - Interval Width vs Weight', fontsize=title_fontsize)
            axs[i, 2].tick_params(axis='both', labelsize=tick_fontsize)
            # 第四列: Width/Coverage Ratio
            axs[i, 3].plot(w_intensity, width_coverage_ratio, '-', marker='o', color=colors[hour])
            axs[i, 3].axvline(x=data['best_ratio_w_intensity'], color='r', linestyle='--', alpha=0.5)
            axs[i, 3].set_xlabel('Intensity Weight', fontsize=label_fontsize)
            axs[i, 3].set_ylabel('Width/Coverage Ratio', fontsize=label_fontsize)
            axs[i, 3].grid(True)
            axs[i, 3].set_title(f'{hour} - Width/Coverage Ratio vs Weight', fontsize=title_fontsize)
            axs[i, 3].tick_params(axis='both', labelsize=tick_fontsize)
        plt.suptitle('Weight Optimization Results for All Horizons', fontsize=24)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig('weight_optimization_combined.png', dpi=600, bbox_inches='tight')
        plt.close()
        print("\n所有时效的权重优化结果已保存到: weight_optimization_combined.png")
    
    # 仍然单独保存当前时效的图像
    plt.figure(figsize=(20, 5))
    plt.subplot(141)
    plt.plot(results_df['w_intensity'], results_df['winkler_score'], 'b-', marker='o')
    plt.axvline(x=best_ratio_w_intensity, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Intensity Weight')
    plt.ylabel('Winkler Score')
    plt.grid(True)
    plt.title('Winkler Score vs Weight')
    plt.subplot(142)
    plt.plot(results_df['w_intensity'], results_df['coverage'], 'g-', marker='o')
    plt.axvline(x=best_ratio_w_intensity, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Intensity Weight')
    plt.ylabel('Coverage Rate')
    plt.grid(True)
    plt.title('Coverage vs Weight')
    plt.subplot(143)
    plt.plot(results_df['w_intensity'], results_df['mean_width'], 'r-', marker='o')
    plt.axvline(x=best_ratio_w_intensity, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Intensity Weight')
    plt.ylabel('Mean Interval Width')
    plt.grid(True)
    plt.title('Interval Width vs Weight')
    plt.subplot(144)
    plt.plot(results_df['w_intensity'], results_df['width_coverage_ratio'], 'm-', marker='o')
    plt.axvline(x=best_ratio_w_intensity, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Intensity Weight')
    plt.ylabel('Width/Coverage Ratio')
    plt.grid(True)
    plt.title('Width/Coverage Ratio vs Weight')
    plt.suptitle(f'Weight Optimization Results ({current_hour})')
    plt.tight_layout()
    plt.savefig(f'weight_optimization_{current_hour}.png', dpi=300, bbox_inches='tight')
    plt.close()
    return best_ratio_w_intensity, best_ratio_w_delv

def main():
    target_cols = {
        '24h': 'intensity_truth_024h',
        '18h': 'intensity_truth_018h',
        '12h': 'intensity_truth_012h',
        '6h': 'intensity_truth_006h'
    }
    data = pd.read_csv('imputed_data.csv')
    
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

    # 为每个basin分别进行storm级别的划分（与typhoon_transformer.py一致，按时间排序）
    train_storms, calib_storms, val_storms, test_storms = [], [], [], []
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
        calib_storms.extend(calib_storms_basin)
        val_storms.extend(val_storms_basin)
        test_storms.extend(test_storms_basin)

    results = {}
    # 已知最佳参数
    best_params_dict = {
        '24h': {'hidden_dim': 128, 'num_layers': 4, 'num_heads': 8, 'dropout': 0.1, 'epochs': 100, 'patience': 10, 'batch_size': 64, 'learning_rate': 0.0005, 'weight_decay': 0.0001, 'verbose': 1},
        '18h': {'hidden_dim': 128, 'num_layers': 4, 'num_heads': 8, 'dropout': 0.2, 'epochs': 100, 'patience': 10, 'batch_size': 64, 'learning_rate': 0.0005, 'weight_decay': 0.0001, 'verbose': 1},
        '12h': {'hidden_dim': 128, 'num_layers': 2, 'num_heads': 4, 'dropout': 0.2, 'epochs': 100, 'patience': 10, 'batch_size': 64, 'learning_rate': 0.0005, 'weight_decay': 0.0001, 'verbose': 1},
        '6h':  {'hidden_dim': 128, 'num_layers': 2, 'num_heads': 8, 'dropout': 0.1, 'epochs': 100, 'patience': 10, 'batch_size': 64, 'learning_rate': 0.0005, 'weight_decay': 0.0001, 'verbose': 1}
    }
    
    for hour, col in target_cols.items():
        print(f"\n===== 预测 {hour} ({col}) =====")
        
        # 只保留当前预测时效的有效数据
        valid_data = data.dropna(subset=[col]).copy()
        
        # 掩码
        train_mask = valid_data['storm_id'].isin(train_storms)
        calib_mask = valid_data['storm_id'].isin(calib_storms)
        val_mask = valid_data['storm_id'].isin(val_storms)
        test_mask = valid_data['storm_id'].isin(test_storms)
        
        # 特征和目标
        X_hour = valid_data[feature_cols].values
        y = valid_data[col].values
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 使用掩码分割
        X_train, y_train = X_hour[train_mask], y[train_mask]
        X_calib, y_calib = X_hour[calib_mask], y[calib_mask]
        X_val, y_val = X_hour[val_mask], y[val_mask]
        X_test, y_test = X_hour[test_mask], y[test_mask]
        
        print(f"训练集: {X_train.shape}, 校准集: {X_calib.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
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
        
        y_pred_train = model.predict(X_train)
        y_pred_calib = model.predict(X_calib)
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)

        # === 权重优化（验证集） ===
        best_w_intensity, best_w_delv = optimize_weight(
            X_train, y_train, X_val, y_val, val_mask, valid_data, 
            y_pred_train, y_pred_val, hour
        )
        
        # 使用最优权重生成动态alpha（测试集）
        alphas = []
        test_data = valid_data[test_mask]
        for _, row in test_data.iterrows():
            intensity = row['intensity'] if 'intensity' in row else row['intensity_knots']
            delv = row['DELV-12'] if 'DELV-12' in row else row['delv_knots']
            alphas.append(dynamic_alpha(intensity, delv, hour, best_w_intensity, best_w_delv))
        alphas = np.array(alphas)
        
        # 使用校准集的误差分布来构建conformal区间
        lower, upper, used_alphas = conformal_interval(y_calib, y_pred_calib, y_pred_test, alpha=alphas)
        
        # 创建详细的预测结果DataFrame
        results_df = pd.DataFrame({
            'storm_id': test_data['storm_id'],
            'storm_name': test_data['storm_name'],
            'basin': test_data['basin'],
            'init_time': test_data['init_time'],
            'current_intensity': test_data['intensity'] if 'intensity' in test_data else test_data['intensity_knots'],
            'intensity_change': test_data['DELV-12'] if 'DELV-12' in test_data else test_data['delv_knots'],
            'true_value': y_test,
            'predicted_value': y_pred_test,
            'prediction_error': y_test - y_pred_test,
            'lower_bound': lower,
            'upper_bound': upper,
            'interval_width': upper - lower,
            'alpha': alphas,
            'is_covered': (y_test >= lower) & (y_test <= upper),
            'deviation_if_not_covered': np.where(
                (y_test >= lower) & (y_test <= upper),
                0,
                np.where(
                    y_test < lower,
                    lower - y_test,
                    y_test - upper
                )
            )
        })
        
        # 添加额外的统计信息
        results_df['abs_error'] = np.abs(results_df['prediction_error'])
        results_df['relative_error'] = np.abs(results_df['prediction_error'] / results_df['true_value'])
        results_df['relative_interval_width'] = results_df['interval_width'] / results_df['true_value']
        
        # 按storm_id和init_time排序
        results_df = results_df.sort_values(['storm_id', 'init_time'])
        
        # 保存到CSV文件
        output_file = f'prediction_results_CP_{hour}.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\n详细预测结果已保存到: {output_file}")
        
        # 计算评估指标
        rmse = np.sqrt(mean_squared_error(results_df['true_value'], results_df['predicted_value']))
        mae = mean_absolute_error(results_df['true_value'], results_df['predicted_value'])
        r2 = r2_score(results_df['true_value'], results_df['predicted_value'])
        coverage = results_df['is_covered'].mean()
        mean_width = results_df['interval_width'].mean()
        winkler = winkler_score_func(results_df['true_value'], results_df['lower_bound'], 
                              results_df['upper_bound'], alpha=results_df['alpha'])
        
        # 打印汇总统计
        print("\n预测结果汇总:")
        print(f"总样本数: {len(results_df)}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R2: {r2:.4f}")
        print(f"覆盖率: {coverage:.3f}")
        print(f"平均区间宽度: {mean_width:.3f}")
        print(f"平均alpha: {results_df['alpha'].mean():.3f}")
        print(f"Winkler Score: {winkler:.4f}")
        
        # 按风暴强度分类统计
        intensity_bins = [(0, 34, 'TD'), (34, 64, 'TS'), (64, 83, 'C1'),
                         (83, 96, 'C2'), (96, 113, 'C3'), (113, float('inf'), 'C4')]
        print("\n按强度分类的统计:")
        for low, high, cat in intensity_bins:
            mask = (results_df['current_intensity'] >= low) & (results_df['current_intensity'] < high)
            if mask.any():
                cat_df = results_df[mask]
                print(f"\n{cat}类别 (样本数: {len(cat_df)}):")
                print(f"覆盖率: {cat_df['is_covered'].mean():.3f}")
                print(f"平均区间宽度: {cat_df['interval_width'].mean():.3f}")
                print(f"平均alpha: {cat_df['alpha'].mean():.3f}")
                print(f"平均绝对误差: {cat_df['abs_error'].mean():.3f}")
        
        # 更新results字典
        results[hour] = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'coverage': coverage,
            'mean_width': mean_width,
            'winkler': winkler,
            'mean_alpha': results_df['alpha'].mean()
        }
    
    print("\n===== 所有时效预测结果 =====")
    for hour, metrics in results.items():
        print(f"\n{hour}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

    # === 年度动态方法统计 ===
    try:
        df = pd.read_csv('detailed_comparison_24h.csv')
        df['year'] = df['init_time'].str[:4]
        summary = df.groupby('year').agg(
            coverage_rate=('dyn_covered', 'mean'),
            mean_width=('dyn_width', 'mean'),
            mean_alpha=('dyn_alpha', 'mean'),
            count=('dyn_covered', 'count')
        ).reset_index()
        summary['coverage_rate'] = summary['coverage_rate'] * 100
        print('\n===== 动态方法年度统计（24h） =====')
        print(summary[['year', 'coverage_rate', 'mean_width', 'mean_alpha', 'count']])
        summary.to_csv('yearly_dynamic_summary_24h.csv', index=False)
        print('\n已输出 yearly_dynamic_summary_24h.csv')
    except Exception as e:
        print(f'年度动态方法统计失败: {e}')

if __name__ == '__main__':
    main()
