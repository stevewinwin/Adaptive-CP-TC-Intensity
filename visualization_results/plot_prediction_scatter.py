import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.collections import LineCollection
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error
from scipy import stats
from scipy.ndimage import gaussian_filter
from sklearn.feature_selection import SelectKBest, f_regression
from typhoon_transformer import TransformerRegressor
import torch
import random

# 设置全局随机种子，保证实验可复现
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

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
    
    # 按storm_id进行分层抽样
    unique_storms = data['storm_id'].unique()
    storm_basins = data.groupby('storm_id')['basin'].first()
    
    # 为每个basin分别进行storm级别的划分
    train_storms = []
    val_storms = []
    test_storms = []
    calib_storms = []

    for basin in ['EP', 'AL', 'CP']:
        basin_storms = storm_basins[storm_basins == basin].index.tolist()
        np.random.seed(42)
        # 1. 先分出20%为"测试+验证"集
        n_testval = int(len(basin_storms) * 0.2)
        testval_storms_basin = np.random.choice(basin_storms, size=n_testval, replace=False)
        remaining_storms = [s for s in basin_storms if s not in testval_storms_basin]
        # 2. 从剩余的80%中分25%为校准集
        n_calib = int(len(remaining_storms) * 0.25)
        calib_storms_basin = np.random.choice(remaining_storms, size=n_calib, replace=False)
        train_storms_basin = [s for s in remaining_storms if s not in calib_storms_basin]
        # 3. 从testval_storms_basin中分一半为验证集，一半为测试集
        n_val = int(len(testval_storms_basin) * 0.5)
        val_storms_basin = np.random.choice(testval_storms_basin, size=n_val, replace=False)
        test_storms_basin = [s for s in testval_storms_basin if s not in val_storms_basin]
        train_storms.extend(train_storms_basin)
        val_storms.extend(val_storms_basin)
        test_storms.extend(test_storms_basin)
        calib_storms.extend(calib_storms_basin)

    print(f"\n===== 预测 {target_hour} ({target_cols[target_hour]}) =====")
    
    # 只保留当前预测时效的有效数据
    valid_data = data.dropna(subset=[target_cols[target_hour]]).copy()
    
    # 创建训练集、校准集、验证集和测试集的掩码
    train_mask = valid_data['storm_id'].isin(train_storms)
    calib_mask = valid_data['storm_id'].isin(calib_storms)
    val_mask = valid_data['storm_id'].isin(val_storms)
    test_mask = valid_data['storm_id'].isin(test_storms)
    
    # 提取特征和目标变量
    X_hour = valid_data[feature_cols].values
    y = valid_data[target_cols[target_hour]].values
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 使用掩码来分割数据
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
    
    return X_train, X_calib, X_val, X_test, y_train, y_calib, y_val, y_test, feature_cols

def train_and_predict(X_train, X_calib, X_val, X_test, y_train, y_calib, y_val, y_test, target_hour):
    """加载各时效最佳权重并预测"""
    # 各时效最佳参数
    best_params_dict = {
        '24h': {'hidden_dim': 128, 'num_layers': 4, 'num_heads': 8, 'dropout': 0.1, 'epochs': 100, 'batch_size': 64, 'patience': 10, 'learning_rate': 0.0005, 'weight_decay': 0.0001, 'verbose': 1},
        '18h': {'hidden_dim': 128, 'num_layers': 4, 'num_heads': 8, 'dropout': 0.2, 'epochs': 100, 'batch_size': 64, 'patience': 10, 'learning_rate': 0.0005, 'weight_decay': 0.0001, 'verbose': 1},
        '12h': {'hidden_dim': 128, 'num_layers': 2, 'num_heads': 4, 'dropout': 0.2, 'epochs': 100, 'batch_size': 64, 'patience': 10, 'learning_rate': 0.0005, 'weight_decay': 0.0001, 'verbose': 1},
        '6h':  {'hidden_dim': 128, 'num_layers': 2, 'num_heads': 8, 'dropout': 0.1, 'epochs': 100, 'batch_size': 64, 'patience': 10, 'learning_rate': 0.0005, 'weight_decay': 0.0001, 'verbose': 1}
    }
    params = best_params_dict[target_hour]
    regressor = TransformerRegressor(**params)
    # 加载权重复现最优结果
    regressor.model = regressor._build_model(X_test.shape[1])
    weight_path = f'best_transformer_model_{target_hour}.pth'
    regressor.model.load_state_dict(torch.load(weight_path, map_location=regressor.device, weights_only=True))
    regressor.model.eval()
    y_pred_train = regressor.predict(X_train)
    y_pred_calib = regressor.predict(X_calib)
    y_pred_val = regressor.predict(X_val)
    y_pred_test = regressor.predict(X_test)
    print(f"\n预测结果形状:")
    print(f"训练集: {y_pred_train.shape}, 校准集: {y_pred_calib.shape}, 验证集: {y_pred_val.shape}, 测试集: {y_pred_test.shape}")
    # 计算评估指标
    r2_train = r2_score(y_train, y_pred_train)
    r2_calib = r2_score(y_calib, y_pred_calib)
    r2_val = r2_score(y_val, y_pred_val)
    r2_test = r2_score(y_test, y_pred_test)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_calib = mean_absolute_error(y_calib, y_pred_calib)
    mae_val = mean_absolute_error(y_val, y_pred_val)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    print(f"\nR² 分数 - 训练集: {r2_train:.4f}, 校准集: {r2_calib:.4f}, 验证集: {r2_val:.4f}, 测试集: {r2_test:.4f}")
    print(f"MAE 分数 - 训练集: {mae_train:.4f}, 校准集: {mae_calib:.4f}, 验证集: {mae_val:.4f}, 测试集: {mae_test:.4f}")
    return y_train, y_calib, y_val, y_test, y_pred_train, y_pred_calib, y_pred_val, y_pred_test, r2_train, r2_calib, r2_val, r2_test

def plot_comprehensive_analysis():
    """创建综合分析图，包含散点图、热力图、残差分析、残差分布和箱型图"""
    # 创建保存目录
    os.makedirs('visualization_results', exist_ok=True)
    
    # 为每个时间点创建图，使用与数据文件一致的时间点格式
    target_hours = ['24h', '18h', '12h', '6h']
    for hour in target_hours:
        print(f"\nProcessing {hour} forecast...")
        
        # 准备数据
        X_train, X_calib, X_val, X_test, y_train, y_calib, y_val, y_test, feature_cols = prepare_data(hour)
        
        # 训练模型并得到预测
        y_train, y_calib, y_val, y_test, y_pred_train, y_pred_calib, y_pred_val, y_pred_test, r2_train, r2_calib, r2_val, r2_test = train_and_predict(
            X_train, X_calib, X_val, X_test, y_train, y_calib, y_val, y_test, hour
        )
        
        # 计算残差
        residuals_calib = y_calib - y_pred_calib
        residuals_test = y_test - y_pred_test
        all_residuals = np.concatenate([residuals_calib, residuals_test])
        
        # 创建图形布局
        fig = plt.figure(figsize=(14, 18))
        
        # 使用GridSpec来创建不规则的网格
        # 设置height_ratios使所有子图高度一致
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1], figure=fig)
        
        # 调整图形之间的间距
        gs.update(wspace=0.2, hspace=0.3)
        
        # 调色板
        palette = {'Calibration': '#a1e3a1', 'Test': '#f4ba8a','Validation':'#b4d4e1'}
        
        # (a) 散点图 - 左上
        ax1 = fig.add_subplot(gs[0, 0])
        data_val = pd.DataFrame({
            'True': y_val,
            'Predicted': y_pred_val,
            'Data Set': 'Validation'
        })
        
        data_test = pd.DataFrame({
            'True': y_test,
            'Predicted': y_pred_test,
            'Data Set': 'Test'
        })
        
        data = pd.concat([data_val, data_test])
        
        sns.scatterplot(data=data, x='True', y='Predicted', hue='Data Set', 
                       palette=palette, alpha=0.5, ax=ax1, s=80)
        
        # 添加回归线
        sns.regplot(data=data_val, x='True', y='Predicted', scatter=False,
                   ax=ax1, color='#b4d4e1', label='Validation Regression Line')
        sns.regplot(data=data_test, x='True', y='Predicted', scatter=False,
                   ax=ax1, color='#f4ba8a', label='Test Regression Line')
        
        # 添加对角线
        lims = [np.min([ax1.get_xlim(), ax1.get_ylim()]), np.max([ax1.get_xlim(), ax1.get_ylim()])]
        ax1.plot(lims, lims, 'k--', alpha=0.5, label='Perfect Prediction')
        
        ax1.set_aspect('equal')
        ax1.grid(True, linestyle='--', alpha=0.3)
        
        # 增加子图边框宽度
        for spine in ax1.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color('black')
        
        # 设置图例
        legend = ax1.legend(fontsize=8, loc='upper left', 
                          frameon=True, fancybox=True, shadow=True,
                          bbox_to_anchor=(0.01, 0.99))
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_linewidth(1)
        
        ax1.set_title(r'$\mathbf{(a)}$ Scatter Plot', fontsize=16, pad=15)
        
        # 添加R²分数
        ax1.text(0.95, 0.15, f'Validation $R^2$ = {r2_val:.3f}', transform=ax1.transAxes,
                fontsize=14, ha='right', va='bottom',
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
        ax1.text(0.95, 0.05, f'Test $R^2$ = {r2_test:.3f}', transform=ax1.transAxes,
                fontsize=14, ha='right', va='bottom',
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
        
        # (b) 热力图 - 右上
        ax2 = fig.add_subplot(gs[0, 1])
        
        # 计算2D直方图
        x_range = [min(min(y_calib), min(y_test)), max(max(y_calib), max(y_test))]
        y_range = [min(min(y_pred_calib), min(y_pred_test)), max(max(y_pred_calib), max(y_pred_test))]
        
        H, xedges, yedges = np.histogram2d(np.concatenate([y_calib, y_test]),
                                          np.concatenate([y_pred_calib, y_pred_test]),
                                          bins=50, range=[x_range, y_range])
        
        # 对数化处理以突出细节
        H = np.log1p(H)
        H = gaussian_filter(H, sigma=1)  # 平滑处理
        
        # 使用pcolormesh绘制热力图
        X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
        pcm = ax2.pcolormesh(X, Y, H.T, cmap='YlOrRd', shading='gouraud')
        
        # 添加对角线
        lims = [np.min([ax1.get_xlim(), ax1.get_ylim()]), np.max([ax1.get_xlim(), ax1.get_ylim()])]
        ax2.plot(lims, lims, 'k--', alpha=0.5, label='Perfect Prediction')
        
        # 增加子图边框宽度
        for spine in ax2.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color('black')
        
        # 设置标题和标签
        ax2.set_title(r'$\mathbf{(b)}$ Heat Map', fontsize=16, pad=15)
        ax2.set_xlabel('Actual Intensity (knots)', fontsize=14)
        ax2.set_ylabel('Predicted Intensity (knots)', fontsize=14)
        
        # 调整刻度标签和网格
        ax2.tick_params(axis='x', rotation=0)
        ax2.tick_params(axis='y', rotation=0)
        ax2.grid(True, linestyle='--', alpha=0.3)
        ax2.set_aspect('equal')
        
        # 添加colorbar
        cbar = plt.colorbar(pcm, ax=ax2, orientation='vertical', pad=0.02)
        # 添加R²分数
        ax2.text(0.95, 0.15, f'Calibration $R^2$ = {r2_calib:.3f}', transform=ax2.transAxes,
                fontsize=14, ha='right', va='bottom',
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
        ax2.text(0.95, 0.05, f'Test $R^2$ = {r2_test:.3f}', transform=ax2.transAxes,
                fontsize=14, ha='right', va='bottom',
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
        
        # (c) Conformal Prediction - 中间
        ax3 = fig.add_subplot(gs[1, 0:2])
        
        # 设置子图的位置，使其居中且宽度更窄
        pos = ax3.get_position()
        new_width = pos.width * 0.8  # 减小宽度到原来的70%（原来是80%）
        new_left = pos.x0 + (pos.width - new_width) / 2  # 居中对齐
        ax3.set_position([new_left, pos.y0, new_width, pos.height])
        
        xy = np.vstack([y_test, residuals_test])
        kde = stats.gaussian_kde(xy)
        x_grid = np.linspace(10, 140, 100)  
        y_grid = np.linspace(residuals_test.min(), residuals_test.max(), 100)
        X, Y = np.meshgrid(x_grid, y_grid)
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = kde(positions).reshape(X.shape)
        
        # 绘制KDE热力图
        im = ax3.imshow(Z, cmap='YlOrRd', aspect='auto', 
                  extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()],
                  origin='lower', alpha=0.6)
        
        # 添加散点
        ax3.scatter(y_test, residuals_test, c='#f4ba8a', alpha=0.3, s=30)
        
        # 添加预测区间
        alphas = [0.3,  0.05]  # 70%, dynamic alpha, 95% 置信水平
        colors = [ '#90AADC', '#a1e3a1']  # 从浅蓝到深蓝
        
        # 为每个x值计算条件预测区间（使用校准集计算区间，在测试集上可视化）
        x_points = np.linspace(10, 140, 50)  
        for alpha, color in zip(alphas, colors):
            intervals_lower = []
            intervals_upper = []
            
            for x in x_points:
                nearby_mask = np.abs(y_calib - x) < np.std(y_calib) * 0.5
                if np.sum(nearby_mask) > 10:  # 确保有足够的点
                    local_residuals = residuals_calib[nearby_mask]
                    q_lower = np.quantile(local_residuals, alpha/2)
                    q_upper = np.quantile(local_residuals, 1-alpha/2)
                else:
                    # 如果局部点太少，使用全局校准集残差
                    q_lower = np.quantile(residuals_calib, alpha/2)
                    q_upper = np.quantile(residuals_calib, 1-alpha/2)
                
                intervals_lower.append(q_lower)
                intervals_upper.append(q_upper)
            
            # 绘制预测区间
            ax3.plot(x_points, intervals_upper, color=color, linestyle='-', 
                    label=f'{(1-alpha)*100:.0f}% Confidence Interval', linewidth=3)
            ax3.plot(x_points, intervals_lower, color=color, linestyle='-', linewidth=3)
        
        # 添加零线
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3, label='Zero Line')
        ax3.set_xlim(10, 140)  
        
        # 增加子图边框宽度
        for spine in ax3.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color('black')
        
        # 设置网格和图例
        ax3.grid(True, linestyle='--', alpha=0.3)
        legend = ax3.legend(fontsize=12, loc='lower right', frameon=True, fancybox=True, shadow=True)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_linewidth(1)
        
        # 设置标题和标签
        ax3.set_title(r'$\mathbf{(c)}$ Residual Analysis with Conformal Prediction', fontsize=16, pad=15)
        ax3.set_xlabel('Actual Intensity (knots)', fontsize=14)
        ax3.set_ylabel('Residual (knots)', fontsize=14)
        
        # (d) 残差分布 - 左下
        ax4 = fig.add_subplot(gs[2, 0])
        sns.histplot(data=pd.DataFrame({
            'Residuals': np.concatenate([residuals_calib, residuals_test]),
            'Dataset': ['Calibration']*len(residuals_calib) + ['Test']*len(residuals_test)
        }), x='Residuals', hue='Dataset', palette=palette, alpha=0.6, ax=ax4, binwidth=8.0, multiple='dodge',
           edgecolor='black', linewidth=0.5)
        
        ax4.grid(True, linestyle='--', alpha=0.3)
        ax4.set_title(r'$\mathbf{(d)}$ Residual Distribution', fontsize=16, pad=15)
        
        # 增加子图边框宽度
        for spine in ax4.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color('black')
        
        # 设置图例
        labels = ['Calibration', 'Test']  # Match the order in the data
        legend = ax4.legend(labels, fontsize=12, loc='upper left', 
                          frameon=True, fancybox=True, shadow=True,
                          bbox_to_anchor=(0.02, 0.98))
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_linewidth(1)
        
        # (e) 箱型图 - 右下
        ax5 = fig.add_subplot(gs[2, 1])
        
        # 定义热带气旋种类（使用缩写，不包含风速范围）
        categories = [
            'TD',
            'TS',
            'C1',
            'C2',
            'C3',
            'C4+'
        ]
        
        # 根据风速范围生成残差数据
        residuals_categories = []
        
        # 基于风速的分类
        # 1. Tropical Depression: V < 34
        mask = y_test < 34
        residuals_categories.append(residuals_test[mask])
        
        # 2. Tropical Storm: 34 ≤ V < 64
        mask = (y_test >= 34) & (y_test < 64)
        residuals_categories.append(residuals_test[mask])
        
        # 3. Category 1 Hurricane: 64 ≤ V < 83
        mask = (y_test >= 64) & (y_test < 83)
        residuals_categories.append(residuals_test[mask])
        
        # 4. Category 2 Hurricane: 83 ≤ V < 96
        mask = (y_test >= 83) & (y_test < 96)
        residuals_categories.append(residuals_test[mask])
        
        # 5. Category 3 Hurricane: 96 ≤ V < 113
        mask = (y_test >= 96) & (y_test < 113)
        residuals_categories.append(residuals_test[mask])
        
        # 6. Category 4+ Hurricane: V ≥ 113
        mask = y_test >= 113
        residuals_categories.append(residuals_test[mask])
        
        # 绘制箱型图
        boxprops = dict(linewidth=1.5, color='#81D8F8', facecolor='#81D8F8', alpha=0.7)
        whiskerprops = dict(linewidth=1.5)
        capprops = dict(linewidth=1.5)
        medianprops = dict(linewidth=1.5, color='#F8CBAD')
        flierprops = dict(marker='o', markerfacecolor='none', markeredgecolor='#F8CBAD', markersize=2, linestyle='none')
        
        ax5.boxplot(residuals_categories, labels=categories, patch_artist=True, 
                   boxprops=boxprops, whiskerprops=whiskerprops, 
                   capprops=capprops, medianprops=medianprops,
                   flierprops=flierprops, showfliers=True)
        
        # 增加子图边框宽度
        for spine in ax5.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color('black')
        
        # 设置柜子旋转，以便能够清楚地看到所有标签
        ax5.set_xticklabels(categories, rotation=0, fontsize=10)
        
        ax5.set_title(r'$\mathbf{(e)}$ Box Plot of Residuals by Tropical Cyclone Type', fontsize=16, pad=15)
        ax5.set_ylabel('Residuals (knots)', fontsize=14)
        ax5.grid(True, linestyle='--', alpha=0.3)
        
        # 设置所有子图的标签字体大小
        for ax in [ax1, ax2, ax3, ax4, ax5]:
            ax.tick_params(axis='both', which='major', labelsize=12)
            if ax != ax5:  # 箱型图有自己的标签
                ax.set_xlabel('Actual Intensity (knots)', fontsize=14)
                if ax in [ax1, ax2]:
                    ax.set_ylabel('Predicted Intensity (knots)', fontsize=14)
                elif ax == ax3:
                    ax.set_ylabel('Residual (knots)', fontsize=14)
                elif ax == ax4:
                    ax.set_xlabel('Residual (knots)', fontsize=14)
                    ax.set_ylabel('Count', fontsize=14)
        
        # 保存图片
        save_path = os.path.join('visualization_results', f'comprehensive_analysis_{hour}.png')
        plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Saved comprehensive analysis to: {save_path}")
if __name__ == '__main__':
    plot_comprehensive_analysis()