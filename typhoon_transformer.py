import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from itertools import product
from sklearn.metrics import mean_absolute_error

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class TabularDataset(Dataset):
    def __init__(self, features, targets=None):
        self.features = torch.FloatTensor(features)
        self.targets = None if targets is None else torch.FloatTensor(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.targets is not None:
            return self.features[idx], self.targets[idx]
        return self.features[idx]

class TransformerRegressor(BaseEstimator, RegressorMixin):
    """
    封装式Transformer回归器，sklearn风格API
    包含：
    - 自动标准化
    - 支持GPU
    - 早停、学习率调度、梯度裁剪
    - 残差和GELU激活
    - 三层输出FC结构
    - Xavier权重初始化
    """

    def __init__(self, hidden_dim=512, num_layers=6, num_heads=8, dropout=0.2,
                 batch_size=64, epochs=50, learning_rate=3e-4, weight_decay=1e-4,
                 patience=10, device=None, verbose=1, target_hour=None):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.history = {'train_loss': [], 'val_loss': []}
        self.input_dim = None
        self.target_hour = target_hour

    def _build_model(self, input_dim):
        class _TransformerModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout):
                super().__init__()
                self.input_proj = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout)
                )
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    activation='gelu',
                    batch_first=True
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=num_layers,
                    norm=nn.LayerNorm(hidden_dim)
                )
                self.residual = nn.Linear(hidden_dim, hidden_dim)
                self.output = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, hidden_dim // 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 4, 1)
                )
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)

            def forward(self, x):
                x = self.input_proj(x).unsqueeze(1)
                z = self.encoder(x)
                z = z + self.residual(x)
                z = z.squeeze(1)
                return self.output(z).squeeze(-1)

        return _TransformerModel(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout
        ).to(self.device)

    def fit(self, X, y, validation_split=0.2, patience=10, scheduler_mode='plateau', scheduler_factor=0.5, scheduler_patience=5, min_lr=1e-6):
        """
        拟合模型。支持early stopping和ReduceLROnPlateau学习率调度。
        """
        self.input_dim = X.shape[1]
        self.model = self._build_model(self.input_dim)
        n_val = int(len(X) * validation_split)
        indices = np.random.permutation(len(X))
        val_idx, train_idx = indices[:n_val], indices[n_val:]
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        train_loader = DataLoader(TabularDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(TabularDataset(X_val, y_val), batch_size=self.batch_size)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        import copy
        criterion = nn.MSELoss()
        self.history = {'train_loss': [], 'val_loss': [], 'lr': []}
        best_val_loss = float('inf')
        best_state = None
        epochs_no_improve = 0
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, min_lr=min_lr, verbose=self.verbose)
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            # 验证集loss
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['lr'].append(optimizer.param_groups[0]['lr'])
            if self.verbose:
                print(f'Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch+1}. Best Val Loss: {best_val_loss:.4f}")
                    break
            scheduler.step(val_loss)
        # 恢复为最佳模型
        if best_state is not None:
            self.model.load_state_dict(best_state)
            filename = f"best_transformer_model_{self.target_hour}.pth" if self.target_hour else "best_transformer_model.pth"
            torch.save(best_state, filename)
            if self.verbose:
                print(f"已保存最佳模型参数到 {filename}")
        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("模型尚未训练，请先调用fit方法")
        mask = (~np.isnan(X).any(axis=1)) & (~np.isinf(X).any(axis=1))
        X = X[mask]
        loader = DataLoader(TabularDataset(X), batch_size=self.batch_size)
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch_X in loader:
                if isinstance(batch_X, list):
                    batch_X = batch_X[0]
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                predictions.append(outputs.cpu().numpy())
        predictions = np.concatenate(predictions)
        return predictions

    def score(self, X, y):
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X))

def search_best_params(X_train, y_train, X_val, y_val, param_grid, fixed_params=None, verbose=True):
    """
    网格搜索TransformerRegressor的最佳参数
    param_grid: dict, e.g. {'hidden_dim': [64,128], 'num_layers': [2,4]}
    fixed_params: dict, 其它固定参数（如epochs、patience等）
    """
    keys = list(param_grid.keys())
    best_score = float('inf')
    best_params = None
    all_results = []
    for values in product(*[param_grid[k] for k in keys]):
        params = dict(zip(keys, values))
        if fixed_params:
            params.update(fixed_params)
        if verbose:
            print(f"尝试参数: {params}")
        model = TransformerRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        score = mean_absolute_error(y_val, y_pred)
        all_results.append((params.copy(), score))
        if verbose:
            print(f"MAE: {score:.4f}")
        if score < best_score:
            best_score = score
            best_params = params.copy()
    if verbose:
        print(f"最佳参数: {best_params}, 最佳MAE: {best_score:.4f}")
    return best_params, best_score, all_results

if __name__ == '__main__':
    # ----------- 配置目标列，数据清理与循环预测 -------------
    target_cols = {
        '24h': 'intensity_truth_024h',
        '18h': 'intensity_truth_018h',
        '12h': 'intensity_truth_012h',
        '6h': 'intensity_truth_006h'
    }
    data = pd.read_csv('imputed_data.csv')
    
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
    
    # 按storm_id进行分层抽样
    unique_storms = data['storm_id'].unique()
    storm_basins = data.groupby('storm_id')['basin'].first()
    
    # 为每个basin分别进行storm级别的划分
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
        n_test = n - n_train - n_calib - n_val  # 剩下的都给测试集

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
        '24h': {'hidden_dim': 128, 'num_layers': 4, 'num_heads': 8, 'dropout': 0.1, 'epochs': 100, 'patience': 10, 'batch_size': 64, 'learning_rate': 0.0005, 'weight_decay': 0.0001, 'verbose': 1, 'target_hour': '24h'},
        '18h': {'hidden_dim': 128, 'num_layers': 4, 'num_heads': 8, 'dropout': 0.2, 'epochs': 100, 'patience': 10, 'batch_size': 64, 'learning_rate': 0.0005, 'weight_decay': 0.0001, 'verbose': 1, 'target_hour': '18h'},
        '12h': {'hidden_dim': 128, 'num_layers': 2, 'num_heads': 4, 'dropout': 0.2, 'epochs': 100, 'patience': 10, 'batch_size': 64, 'learning_rate': 0.0005, 'weight_decay': 0.0001, 'verbose': 1, 'target_hour': '12h'},
        '6h':  {'hidden_dim': 128, 'num_layers': 2, 'num_heads': 8, 'dropout': 0.1, 'epochs': 100, 'patience': 10, 'batch_size': 64, 'learning_rate': 0.0005, 'weight_decay': 0.0001, 'verbose': 1, 'target_hour': '6h'}
    }
    
    # 1. 读取 common_storm_ids（只需读取一次，放在循环外）
    common_storms_df = pd.read_csv('common_storms.csv')
    common_storm_ids = set(common_storms_df['storm_id'])

    def get_year(storm_id):
        try:
            return int(str(storm_id)[-4:])
        except:
            return 0

    for hour, col in target_cols.items():
        print(f"\n===== 预测 {hour} ({col}) =====")
        
        # 只保留当前预测时效的有效数据
        valid_data = data.dropna(subset=[col]).copy()
        
        # 创建训练集、验证集和测试集的掩码
        train_mask = valid_data['storm_id'].isin(train_storms)
        calib_mask = valid_data['storm_id'].isin(calib_storms)
        val_mask = valid_data['storm_id'].isin(val_storms)
        test_mask = valid_data['storm_id'].isin(test_storms)
        
        # 提取特征和目标变量
        X_hour = valid_data[feature_cols].values
        y = valid_data[col].values
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 使用之前创建的掩码来分割数据
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
        # 只保留common_storm_ids
        test_results = test_results[test_results['storm_id'].isin(common_storm_ids)]
        # 跳过unnamed风暴
        test_results = test_results[test_results['storm_name'].str.strip().str.lower() != 'unnamed']
        # 按风暴ID和时间排序
        test_results = test_results.sort_values(['storm_id', 'init_time'])

        # 按年份分组输出
        test_results['year'] = test_results['storm_id'].apply(get_year)
        for year, group in test_results.groupby('year'):
            output_file = f'prediction_results_{hour}_{year}.csv'
            group.drop(columns=['year']).to_csv(output_file, index=False)
            print(f"\n{year}年预测结果已保存到: {output_file}，风暴数: {group['storm_id'].nunique()}")
        
        results[hour] = {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'BestParams': params}
    
    print("\n===== 所有时效预测结果（复现） =====")
    for hour, metrics in results.items():
        print(f"{hour}: {metrics}")