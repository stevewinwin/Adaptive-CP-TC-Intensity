import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import math
from xgboost import XGBRegressor
import lightgbm as lgb
from scipy.stats import norm
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
from ngboost import NGBRegressor
from ngboost.distns import Normal
import random
from sklearn.metrics import r2_score

class MCDropoutNet(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.1):
        super(MCDropoutNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(p=dropout_rate)
        
    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = F.gelu(self.fc2(x))
        x = self.dropout(x)
        x = F.gelu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

class MCDropoutPredictor:
    def __init__(self, input_dim, dropout_rate=0.1, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = MCDropoutNet(input_dim, dropout_rate).to(device)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def train(self, X, y, epochs=100, batch_size=64, lr=0.001):
        X_train = torch.FloatTensor(self.scaler_X.fit_transform(X)).to(self.device)
        y_train = torch.FloatTensor(self.scaler_y.fit_transform(y.reshape(-1, 1))).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(X_train):.4f}')
    
    def predict_with_uncertainty(self, X, n_samples=100, alpha=0.05, return_std=False):
        self.model.train()  # Enable dropout during prediction
        X = torch.FloatTensor(self.scaler_X.transform(X)).to(self.device)
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.model(X)
                predictions.append(pred.cpu().numpy())
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        mean_pred = self.scaler_y.inverse_transform(mean_pred)
        std_pred = std_pred * self.scaler_y.scale_
        z_score = 1.96  # 95% confidence interval
        interval_width = z_score * std_pred
        lower_bound = mean_pred - interval_width
        upper_bound = mean_pred + interval_width
        if return_std:
            return mean_pred, lower_bound, upper_bound, std_pred
        return mean_pred, lower_bound, upper_bound

def process_data(data_path, target_col):
    """
    处理数据并返回分组后的特征和目标变量，分为train/calib/val/test四组，比例为0.6/0.1/0.1/0.2，按风暴最早init_time排序。
    target_col: 目标变量列名，如 'intensity_truth_024h'
    """
    data = pd.read_csv(data_path)
    null_columns = data.columns[data.isnull().all()].tolist()
    data = data.drop(columns=null_columns)
    if 'lat' in data.columns and 'lon' in data.columns:
        data = data.dropna(subset=['lat', 'lon'])
    data = data.sort_values(['storm_id', 'init_time'])
    exclude_columns = [
        'storm_id', 'basin', 'DELV_t0', 'storm_name', 'init_time',
        'intensity_truth_006h', 'intensity_truth_012h', 'intensity_truth_018h', 'intensity_truth_024h',
        'intensity_truth_036h', 'intensity_truth_048h', 'intensity_truth_072h', 'intensity_truth_096h', 'intensity_truth_120h'
    ]
    feature_cols = [col for col in data.columns if col not in exclude_columns and col not in null_columns]
    non_numeric_cols = [col for col in feature_cols if not pd.api.types.is_numeric_dtype(data[col])]
    if non_numeric_cols:
        raise ValueError(f"以下特征不是数值型：{non_numeric_cols}")
    valid_data = data.dropna(subset=[target_col]).copy()
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
    X = valid_data[feature_cols].values
    y = valid_data[target_col].values
    return {
        'X': X,
        'y': y,
        'feature_cols': feature_cols,
        'train_mask': train_mask.values,
        'calib_mask': calib_mask.values,
        'val_mask': val_mask.values,
        'test_mask': test_mask.values,
        'valid_data': valid_data
    }

def evaluate_predictions(y_true, y_pred, lower_bound, upper_bound, std_pred=None):
    """评估预测结果"""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    # 计算覆盖率
    coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
    # 计算平均区间宽度
    interval_width = np.mean(upper_bound - lower_bound)
    # 计算Winkler Score
    winkler = winkler_score(y_true, lower_bound, upper_bound, alpha=0.05)
    # 计算CRPS和NLL
    if std_pred is not None:
        crps = crps_gaussian(y_true, y_pred, std_pred)
        nll = nll_gaussian(y_true, y_pred, std_pred)
    else:
        crps = None
        nll = None
    # 计算R2分数
    r2 = r2_score(y_true, y_pred)
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'Coverage': coverage,
        'Interval Width': interval_width,
        'Winkler Score': winkler,
        'CRPS': crps,
        'NLL': nll,
        'R2': r2
    }

# ===== Winkler Score for interval prediction =====
def winkler_score(y_true, lower, upper, alpha=0.05):
    # Winkler Score for prediction intervals
    # https://en.wikipedia.org/wiki/Prediction_interval#Winkler_score
    width = upper - lower
    score = np.zeros_like(y_true, dtype=float)
    penalty = 2 / alpha
    # inside interval
    inside = (y_true >= lower) & (y_true <= upper)
    score[inside] = width[inside]
    # below lower
    below = y_true < lower
    score[below] = width[below] + penalty * (lower[below] - y_true[below])
    # above upper
    above = y_true > upper
    score[above] = width[above] + penalty * (y_true[above] - upper[above])
    return np.mean(score)

# ===== CRPS for normal distribution (closed form) =====
def crps_gaussian(y_true, mu, sigma):
    z = (y_true - mu) / sigma
    crps = sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1/np.sqrt(np.pi))
    return np.mean(crps)

# ===== NLL for normal distribution =====
def nll_gaussian(y_true, mu, sigma):
    nll = 0.5 * np.log(2 * np.pi * sigma ** 2) + ((y_true - mu) ** 2) / (2 * sigma ** 2)
    return np.mean(nll)

# ====== 深度集成 ======
class DeepEnsemble:
    def __init__(self, input_dim, n_models=5, dropout_rate=0.1, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.n_models = n_models
        self.models = [MCDropoutPredictor(input_dim, dropout_rate, device) for _ in range(n_models)]
        self.device = device

    def train(self, X, y, epochs=100, save_prefix=None):
        for i, model in enumerate(self.models):
            print(f"Training ensemble model {i+1}/{self.n_models}")
            torch.manual_seed(42 + i)
            np.random.seed(42 + i)
            model.train(X, y, epochs=epochs)
            if save_prefix is not None:
                torch.save(model.model.state_dict(), f"{save_prefix}_ensemble_model_{i+1}.pth")

    def load_weights(self, feature_cols, load_prefix):
        for i, model in enumerate(self.models):
            model.model = MCDropoutNet(len(feature_cols)).to(model.device)
            model.model.load_state_dict(torch.load(f"{load_prefix}_ensemble_model_{i+1}.pth", map_location=model.device))
            model.model.eval()

    def predict_with_uncertainty(self, X, return_std=False):
        preds = []
        for model in self.models:
            model.model.eval()
            X_tensor = torch.FloatTensor(model.scaler_X.transform(X)).to(self.device)
            with torch.no_grad():
                pred = model.model(X_tensor).cpu().numpy()
                pred = model.scaler_y.inverse_transform(pred)
                preds.append(pred)
        preds = np.array(preds)  # shape: (n_models, n_samples, 1)
        mean_pred = np.mean(preds, axis=0).flatten()
        std_pred = np.std(preds, axis=0).flatten()
        z_score = 1.96
        lower = mean_pred - z_score * std_pred
        upper = mean_pred + z_score * std_pred
        if return_std:
            return mean_pred, lower, upper, std_pred
        return mean_pred, lower, upper

# ====== DropConnect网络 ======
class DropConnectLinear(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_rate = dropout_rate
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight_mask = None
        self.bias_mask = None
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            
    def create_masks(self):
        self.weight_mask = torch.bernoulli(torch.ones_like(self.weight) * (1 - self.dropout_rate))
        self.bias_mask = torch.bernoulli(torch.ones_like(self.bias) * (1 - self.dropout_rate))
        
    def forward(self, input):
        if self.training:
            self.create_masks()
            w = self.weight * self.weight_mask.to(self.weight.device)
            b = self.bias * self.bias_mask.to(self.bias.device)
        else:
            w = self.weight * (1 - self.dropout_rate)
            b = self.bias * (1 - self.dropout_rate)
        return F.linear(input, w, b)

class DropConnectNet(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.1):
        super().__init__()
        self.fc1 = DropConnectLinear(input_dim, 256, dropout_rate)
        self.fc2 = DropConnectLinear(256, 256, dropout_rate)
        self.fc3 = DropConnectLinear(256, 256, dropout_rate)
        self.fc4 = DropConnectLinear(256, 1, dropout_rate)
        
    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = F.gelu(self.fc3(x))
        x = self.fc4(x)
        return x

class DropConnectPredictor:
    def __init__(self, input_dim, dropout_rate=0.1, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = DropConnectNet(input_dim, dropout_rate).to(device)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def train(self, X, y, epochs=100, batch_size=64, lr=0.001):
        X_train = torch.FloatTensor(self.scaler_X.fit_transform(X)).to(self.device)
        y_train = torch.FloatTensor(self.scaler_y.fit_transform(y.reshape(-1, 1))).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(X_train):.4f}')
        
    def predict_with_uncertainty(self, X, n_samples=100, alpha=0.05, return_std=False):
        self.model.train()  # Enable DropConnect during prediction
        X = torch.FloatTensor(self.scaler_X.transform(X)).to(self.device)
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.model(X)
                predictions.append(pred.cpu().numpy())
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        mean_pred = self.scaler_y.inverse_transform(mean_pred)
        std_pred = std_pred * self.scaler_y.scale_
        z_score = 1.96  # 95% confidence interval
        interval_width = z_score * std_pred
        lower_bound = mean_pred - interval_width
        upper_bound = mean_pred + interval_width
        if return_std:
            return mean_pred, lower_bound, upper_bound, std_pred
        return mean_pred, lower_bound, upper_bound

class XGBoostCQR:
    """
    基于XGBoost的Conformalized Quantile Regression (CQR)
    """
    def __init__(self, input_dim, alpha=0.05, n_estimators=1500, learning_rate=0.01, max_depth=9, min_child_weight=3, subsample=0.8, colsample_bytree=0.9, **xgb_params):
        self.alpha = alpha
        self.lower_q = alpha / 2
        self.upper_q = 1 - alpha / 2
        self.model_lower = XGBRegressor(objective='reg:quantileerror', quantile_alpha=self.lower_q,
            n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, min_child_weight=min_child_weight,
            subsample=subsample, colsample_bytree=colsample_bytree, **xgb_params)
        self.model_upper = XGBRegressor(objective='reg:quantileerror', quantile_alpha=self.upper_q,
            n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, min_child_weight=min_child_weight,
            subsample=subsample, colsample_bytree=colsample_bytree, **xgb_params)
        self.model_mean = XGBRegressor(objective='reg:squarederror',
            n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, min_child_weight=min_child_weight,
            subsample=subsample, colsample_bytree=colsample_bytree, **xgb_params)

    def train(self, X, y):
        self.model_lower.fit(X, y)
        self.model_upper.fit(X, y)
        self.model_mean.fit(X, y)

    def predict_with_interval(self, X, return_std=False):
        lower = self.model_lower.predict(X)
        upper = self.model_upper.predict(X)
        mean_pred = self.model_mean.predict(X)
        return mean_pred, lower, upper

# --- MLP区间预测（残差法） ---
class MLPRegressorInterval:
    """
    普通神经网络，区间预测采用残差法：用训练集残差std，预测时mean±z*std。
    可调参数：hidden_dim, n_layers, dropout, lr, epochs。
    """
    def __init__(self, input_dim, hidden_dim=128, n_layers=3, dropout=0.1, lr=0.001, epochs=100):
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        self.epochs = epochs
        self.lr = lr
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.residual_std = None

    def train(self, X, y):
        X = torch.FloatTensor(self.scaler_X.fit_transform(X))
        y = torch.FloatTensor(self.scaler_y.fit_transform(y.reshape(-1, 1)))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        for epoch in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            pred = self.model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
        # 计算残差std
        self.model.eval()
        with torch.no_grad():
            pred = self.model(X)
            pred = self.scaler_y.inverse_transform(pred.numpy())
            y_true = self.scaler_y.inverse_transform(y.numpy())
            self.residual_std = np.std(pred.flatten() - y_true.flatten())

    def predict_with_interval(self, X, alpha=0.05):
        X = torch.FloatTensor(self.scaler_X.transform(X))
        self.model.eval()
        with torch.no_grad():
            pred = self.model(X)
            mean_pred = self.scaler_y.inverse_transform(pred.numpy()).flatten()
        z_score = 1.96 if alpha == 0.05 else norm.ppf(1 - alpha / 2)
        lower = mean_pred - z_score * self.residual_std
        upper = mean_pred + z_score * self.residual_std
        return mean_pred, lower, upper

# --- LightGBM CQR ---
class LightGBMCQR:
    """
    基于LightGBM的Conformalized Quantile Regression (CQR)。
    区间预测分别用下分位数、上分位数和均值模型。
    可调参数：n_estimators, learning_rate, max_depth, min_child_weight, subsample, colsample_bytree等。
    """
    def __init__(self, input_dim, alpha=0.05, n_estimators=1500, learning_rate=0.01, max_depth=9, min_child_weight=3, subsample=0.8, colsample_bytree=0.9, **lgb_params):
        self.alpha = alpha
        self.lower_q = alpha / 2
        self.upper_q = 1 - alpha / 2
        self.model_lower = lgb.LGBMRegressor(objective='quantile', alpha=self.lower_q,
            n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, min_child_weight=min_child_weight,
            subsample=subsample, colsample_bytree=colsample_bytree, **lgb_params)
        self.model_upper = lgb.LGBMRegressor(objective='quantile', alpha=self.upper_q,
            n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, min_child_weight=min_child_weight,
            subsample=subsample, colsample_bytree=colsample_bytree, **lgb_params)
        self.model_mean = lgb.LGBMRegressor(objective='mse',
            n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, min_child_weight=min_child_weight,
            subsample=subsample, colsample_bytree=colsample_bytree, **lgb_params)
    def train(self, X, y):
        self.model_lower.fit(X, y)
        self.model_upper.fit(X, y)
        self.model_mean.fit(X, y)
    def predict_with_interval(self, X, return_std=False):
        lower = self.model_lower.predict(X)
        upper = self.model_upper.predict(X)
        mean_pred = self.model_mean.predict(X)
        return mean_pred, lower, upper

class NGBoostPredictor:
    def __init__(self, input_dim, n_estimators=1000, learning_rate=0.01, verbose=False, random_state=42):
        self.model = NGBRegressor(Dist=Normal, n_estimators=n_estimators, learning_rate=learning_rate, verbose=verbose, random_state=random_state)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

    def train(self, X, y):
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        self.model.fit(X_scaled, y_scaled)

    def predict_with_interval(self, X, alpha=0.05, return_std=False):
        X_scaled = self.scaler_X.transform(X)
        preds = self.model.pred_dist(X_scaled)
        mean_pred = self.scaler_y.inverse_transform(preds.loc.reshape(-1, 1)).flatten()
        std_pred = preds.scale * self.scaler_y.scale_[0]
        z_score = 1.96 if alpha == 0.05 else norm.ppf(1 - alpha / 2)
        lower = mean_pred - z_score * std_pred
        upper = mean_pred + z_score * std_pred
        if return_std:
            return mean_pred, lower, upper, std_pred
        return mean_pred, lower, upper

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    data_path = 'imputed_data.csv'
    target_cols = [
        'intensity_truth_006h',
        'intensity_truth_012h',
        'intensity_truth_018h',
        'intensity_truth_024h'
    ]
    results = {}
    
    for target_col in target_cols:
        print(f"\nTraining models for {target_col}")
        # 处理数据
        splits = process_data(data_path, target_col)
        X = splits['X']
        y = splits['y']
        feature_cols = splits['feature_cols']
        
        # 获取训练集和测试集
        X_train = X[splits['train_mask']]
        y_train = y[splits['train_mask']]
        X_calib = X[splits['calib_mask']]
        y_calib = y[splits['calib_mask']]
        X_val = X[splits['val_mask']]
        y_val = y[splits['val_mask']]
        X_test = X[splits['test_mask']]
        y_test = y[splits['test_mask']]
        
        results[target_col] = {}
        
        # ===== LightGBM CQR =====
        print("\n[LightGBM CQR]")
        cqr = LightGBMCQR(
            input_dim=X_train.shape[1],
            alpha=0.05,
            n_estimators=1500,
            learning_rate=0.01,
            max_depth=9,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.9
        )
        cqr.train(X_train, y_train)
        mean_pred, lower, upper = cqr.predict_with_interval(X_test)
        z_score = 1.96
        std_pred = (upper - lower) / (2 * z_score)
        metrics = evaluate_predictions(y_test, mean_pred, lower, upper, std_pred=std_pred)
        results[target_col]['LightGBMCQR'] = metrics
        print(f"Results for {target_col} (LightGBM CQR):")
        for metric, value in metrics.items():
            if value is not None:
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: None")
        # 保存区间详情
        os.makedirs('interval_details', exist_ok=True)
        interval_width = upper - lower
        coverage = ((y_test >= lower) & (y_test <= upper)).astype(int)
        sample_crps = crps_gaussian(y_test, mean_pred, std_pred)
        sample_nll = nll_gaussian(y_test, mean_pred, std_pred)
        interval_df = pd.DataFrame({
            'storm_id': splits['valid_data'].iloc[splits['test_mask']]['storm_id'].values,
            'storm_name': splits['valid_data'].iloc[splits['test_mask']]['storm_name'].values,
            'init_time': splits['valid_data'].iloc[splits['test_mask']]['init_time'].values,
            'y_true': y_test,
            'y_pred': mean_pred,
            'lower': lower,
            'upper': upper,
            'interval_width': interval_width,
            'coverage': coverage,
            'crps': sample_crps,
            'nll': sample_nll
        })
        # 计算每个storm的覆盖率和平均区间宽度
        storm_group = interval_df.groupby('storm_id')
        storm_coverage_map = storm_group['coverage'].mean().to_dict()
        storm_width_map = storm_group['interval_width'].mean().to_dict()
        interval_df['storm_coverage'] = interval_df['storm_id'].map(storm_coverage_map)
        interval_df['storm_avg_interval_width'] = interval_df['storm_id'].map(storm_width_map)
        interval_file = os.path.join('interval_details', f'intervals_LightGBMCQR_{target_col}.csv')
        interval_df.to_csv(interval_file, index=False)
        print(f"区间详情已保存到 {interval_file}")
        
        # ===== Deep Ensemble =====
        print("\n[Deep Ensemble]")
        ens_prefix = f"best_deepensemble_{target_col}"
        ensemble = DeepEnsemble(input_dim=len(feature_cols), n_models=5)
        ensemble.train(X_train, y_train, epochs=100, save_prefix=ens_prefix)
        y_pred, lower, upper, std_pred = ensemble.predict_with_uncertainty(X_test, return_std=True)
        metrics = evaluate_predictions(y_test, y_pred, lower, upper, std_pred=std_pred)
        results[target_col]['DeepEnsemble'] = metrics
        print(f"Results for {target_col} (Deep Ensemble):")
        for metric, value in metrics.items():
            if value is not None:
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: None")
        # 保存区间详情
        os.makedirs('interval_details', exist_ok=True)
        interval_width = upper - lower
        coverage = ((y_test >= lower) & (y_test <= upper)).astype(int)
        sample_crps = crps_gaussian(y_test, y_pred, std_pred)
        sample_nll = nll_gaussian(y_test, y_pred, std_pred)
        interval_df = pd.DataFrame({
            'storm_id': splits['valid_data'].iloc[splits['test_mask']]['storm_id'].values,
            'storm_name': splits['valid_data'].iloc[splits['test_mask']]['storm_name'].values,
            'init_time': splits['valid_data'].iloc[splits['test_mask']]['init_time'].values,
            'y_true': y_test,
            'y_pred': y_pred,
            'lower': lower,
            'upper': upper,
            'interval_width': interval_width,
            'coverage': coverage,
            'crps': sample_crps,
            'nll': sample_nll
        })
        storm_group = interval_df.groupby('storm_id')
        interval_df['storm_coverage'] = interval_df['storm_id'].map(storm_group['coverage'].mean())
        interval_df['storm_avg_interval_width'] = interval_df['storm_id'].map(storm_group['interval_width'].mean())
        interval_file = os.path.join('interval_details', f'intervals_DeepEnsemble_{target_col}.csv')
        interval_df.to_csv(interval_file, index=False)
        print(f"区间详情已保存到 {interval_file}")
        
        # ===== DropConnect =====
        print("\n[DropConnect]")
        dropconnect_weight_path = f"best_dropconnect_{target_col}.pth"
        dropconnect = DropConnectPredictor(input_dim=len(feature_cols), dropout_rate=0.1)
        dropconnect.train(X_train, y_train, epochs=100)
        torch.save(dropconnect.model.state_dict(), dropconnect_weight_path)
        y_pred, lower, upper, std_pred = dropconnect.predict_with_uncertainty(X_test, return_std=True)
        metrics = evaluate_predictions(y_test, y_pred.flatten(), lower.flatten(), upper.flatten(), std_pred=std_pred.flatten())
        results[target_col]['DropConnect'] = metrics
        print(f"Results for {target_col} (DropConnect):")
        for metric, value in metrics.items():
            if value is not None:
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: None")
        # 保存区间详情
        interval_width = upper.flatten() - lower.flatten()
        coverage = ((y_test >= lower.flatten()) & (y_test <= upper.flatten())).astype(int)
        sample_crps = crps_gaussian(y_test, y_pred.flatten(), std_pred.flatten())
        sample_nll = nll_gaussian(y_test, y_pred.flatten(), std_pred.flatten())
        interval_df = pd.DataFrame({
            'storm_id': splits['valid_data'].iloc[splits['test_mask']]['storm_id'].values,
            'storm_name': splits['valid_data'].iloc[splits['test_mask']]['storm_name'].values,
            'init_time': splits['valid_data'].iloc[splits['test_mask']]['init_time'].values,
            'y_true': y_test,
            'y_pred': y_pred.flatten(),
            'lower': lower.flatten(),
            'upper': upper.flatten(),
            'interval_width': interval_width,
            'coverage': coverage,
            'crps': sample_crps,
            'nll': sample_nll
        })
        storm_group = interval_df.groupby('storm_id')
        interval_df['storm_coverage'] = interval_df['storm_id'].map(storm_group['coverage'].mean())
        interval_df['storm_avg_interval_width'] = interval_df['storm_id'].map(storm_group['interval_width'].mean())
        interval_file = os.path.join('interval_details', f'intervals_DropConnect_{target_col}.csv')
        interval_df.to_csv(interval_file, index=False)
        print(f"区间详情已保存到 {interval_file}")
        
        # ===== MC Dropout =====
        print("\n[MC Dropout]")
        mc_weight_path = f"best_mc_dropout_{target_col}.pth"
        mc_dropout = MCDropoutPredictor(input_dim=len(feature_cols))
        mc_dropout.train(X_train, y_train, epochs=100)
        torch.save(mc_dropout.model.state_dict(), mc_weight_path)
        y_pred, lower, upper, std_pred = mc_dropout.predict_with_uncertainty(X_test, return_std=True)
        metrics = evaluate_predictions(y_test, y_pred.flatten(), lower.flatten(), upper.flatten(), std_pred=std_pred.flatten())
        results[target_col]['MCDropout'] = metrics
        print(f"Results for {target_col} (MC Dropout):")
        for metric, value in metrics.items():
            if value is not None:
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: None")
        # 保存区间详情
        interval_width = upper.flatten() - lower.flatten()
        coverage = ((y_test >= lower.flatten()) & (y_test <= upper.flatten())).astype(int)
        sample_crps = crps_gaussian(y_test, y_pred.flatten(), std_pred.flatten())
        sample_nll = nll_gaussian(y_test, y_pred.flatten(), std_pred.flatten())
        interval_df = pd.DataFrame({
            'storm_id': splits['valid_data'].iloc[splits['test_mask']]['storm_id'].values,
            'storm_name': splits['valid_data'].iloc[splits['test_mask']]['storm_name'].values,
            'init_time': splits['valid_data'].iloc[splits['test_mask']]['init_time'].values,
            'y_true': y_test,
            'y_pred': y_pred.flatten(),
            'lower': lower.flatten(),
            'upper': upper.flatten(),
            'interval_width': interval_width,
            'coverage': coverage,
            'crps': sample_crps,
            'nll': sample_nll
        })
        storm_group = interval_df.groupby('storm_id')
        interval_df['storm_coverage'] = interval_df['storm_id'].map(storm_group['coverage'].mean())
        interval_df['storm_avg_interval_width'] = interval_df['storm_id'].map(storm_group['interval_width'].mean())
        interval_file = os.path.join('interval_details', f'intervals_MCDropout_{target_col}.csv')
        interval_df.to_csv(interval_file, index=False)
        print(f"区间详情已保存到 {interval_file}")
        
        # ===== NGBoost =====
        print("\n[NGBoost]")
        ngb = NGBoostPredictor(input_dim=len(feature_cols), n_estimators=1000, learning_rate=0.05)
        ngb.train(X_train, y_train)
        mean_pred, lower, upper, std_pred = ngb.predict_with_interval(X_test, return_std=True)
        metrics = evaluate_predictions(y_test, mean_pred, lower, upper, std_pred=std_pred)
        results[target_col]['NGBoost'] = metrics
        print(f"Results for {target_col} (NGBoost):")
        for metric, value in metrics.items():
            if value is not None:
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: None")
        # 保存区间详情
        interval_width = upper - lower
        coverage = ((y_test >= lower) & (y_test <= upper)).astype(int)
        sample_crps = crps_gaussian(y_test, mean_pred, std_pred)
        sample_nll = nll_gaussian(y_test, mean_pred, std_pred)
        interval_df = pd.DataFrame({
            'storm_id': splits['valid_data'].iloc[splits['test_mask']]['storm_id'].values,
            'storm_name': splits['valid_data'].iloc[splits['test_mask']]['storm_name'].values,
            'init_time': splits['valid_data'].iloc[splits['test_mask']]['init_time'].values,
            'y_true': y_test,
            'y_pred': mean_pred,
            'lower': lower,
            'upper': upper,
            'interval_width': interval_width,
            'coverage': coverage,
            'crps': sample_crps,
            'nll': sample_nll
        })
        storm_group = interval_df.groupby('storm_id')
        interval_df['storm_coverage'] = interval_df['storm_id'].map(storm_group['coverage'].mean())
        interval_df['storm_avg_interval_width'] = interval_df['storm_id'].map(storm_group['interval_width'].mean())
        interval_file = os.path.join('interval_details', f'intervals_NGBoost_{target_col}.csv')
        interval_df.to_csv(interval_file, index=False)
        print(f"区间详情已保存到 {interval_file}")
        
    # ===== 保存所有结果为CSV =====
    records = []
    for target_col, model_dict in results.items():
        for model_name, metrics in model_dict.items():
            row = {'target_col': target_col, 'model': model_name}
            row.update(metrics)
            records.append(row)
    df_results = pd.DataFrame(records)
    df_results.to_csv('all_model_results.csv', index=False)
    print("All results saved to all_model_results.csv")

    return results

if __name__ == "__main__":
    main()
