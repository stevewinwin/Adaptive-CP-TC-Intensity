import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import os
import matplotlib.gridspec as gridspec
import glob

def load_data():
    """Load all necessary data files."""
    # 读取四个 detailed_comparison_xx.csv 文件，找出都出现的storm_id
    detailed_dfs = {}
    storm_id_sets = []
    for hour in ['6h', '12h', '18h', '24h']:
        df = pd.read_csv(f'detailed_comparison_{hour}.csv')
        detailed_dfs[hour] = df
        storm_id_sets.append(set(df['storm_id'].unique()))
    # 取四个文件都出现的storm_id交集
    common_storm_ids = set.intersection(*storm_id_sets)
    # 读取原始数据
    df = pd.read_csv('imputed_data.csv')
    # 只保留common_storm_ids的观测点
    test_storms = df[df['storm_id'].isin(common_storm_ids)][['storm_id', 'storm_name', 'lat', 'lon', 'intensity']].copy()
    # storm_info也只保留这些ID
    storm_info = df[df['storm_id'].isin(common_storm_ids)][['storm_id', 'storm_name']].drop_duplicates()
    # 组装data字典
    data = {
        'test_storms': test_storms,
        'storm_info': storm_info
    }
    for hour in ['6h', '12h', '18h', '24h']:
        data[f'detailed_{hour}'] = detailed_dfs[hour]
        data[f'grouped_{hour}'] = pd.read_csv(f'group/grouped_stats_{hour}.csv')
    return data

# 风速等级定义 (单位: knots)
wind_levels = {
    'Non-Tropical': (-999, 0),  # 非热带系统
    'Subtropical': (0, 17),     # 亚热带低压
    'Tropical Depression': (17, 34),  # 热带低压
    'Tropical Storm': (34, 64),     # 热带风暴
    'Category 1': (64, 83),   # 1级飓风
    'Category 2': (83, 96),   # 2级飓风
    'Category 3': (96, 113),  # 3级飓风
    'Category 4': (113, 137), # 4级飓风
    'Category 5': (137, 999)  # 5级飓风
}

# 颜色映射
color_map = {
    'Non-Tropical': 'lightgray',
    'Subtropical': 'gray',
    'Tropical Depression': 'deepskyblue',
    'Tropical Storm': 'dodgerblue',
    'Category 1': 'yellow',
    'Category 2': 'orange',
    'Category 3': 'red',
    'Category 4': 'darkred',
    'Category 5': 'purple'
}

def parse_init_time(time_str):
    """解析init_time格式的时间字符串"""
    time_str = time_str.replace('Z', '').replace('_', ' ')
    return pd.to_datetime(time_str, format='%Y-%m-%d %H')

def plot_storm_track(ax, storm_data, storm_name):
    """在给定的axes上绘制台风路径图"""
    # ax已经在创建时设置了PlateCarree投影
    
    # 获取有效的经纬度数据
    valid_data = storm_data.dropna(subset=['lat', 'lon'])
    
    if len(valid_data) == 0:
        print(f"台风 {storm_name} 没有有效的经纬度数据")
        return
    
    # 设置地图范围
    lat_min, lat_max = valid_data['lat'].min() - 5, valid_data['lat'].max() + 5
    lon_min, lon_max = valid_data['lon'].min() - 5, valid_data['lon'].max() + 5
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
    # 添加地图要素
    ax.add_feature(cfeature.LAND, facecolor='lightyellow', alpha=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.5)
    ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, edgecolor='gray', linewidth=0.5)
    
    # 添加网格线
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    
    # 确定每个点的颜色
    colors = []
    for wind in valid_data['intensity']:
        matched = False
        for level, (min_speed, max_speed) in wind_levels.items():
            if min_speed <= wind < max_speed:
                colors.append(color_map[level])
                matched = True
                break
        if not matched:
            colors.append(color_map['Non-Tropical'])
    
    # 绘制散点和路径线
    ax.scatter(valid_data['lon'], valid_data['lat'],
              c=colors, marker='o', s=80,
              transform=ccrs.PlateCarree(),
              edgecolor='black', linewidth=0.5,
              alpha=0.8, zorder=3)
    
    ax.plot(valid_data['lon'], valid_data['lat'],
            color='blue', linestyle='-', linewidth=1.5,
            transform=ccrs.PlateCarree(),
            alpha=0.6, zorder=2)
    
    # 添加图例
    handles = [plt.Line2D([0], [0], marker='o', linestyle='-', linewidth=1, color='blue',
                        markerfacecolor=color_map[level], markersize=6, markeredgecolor='black')
              for level in wind_levels.keys()]
    ax.legend(handles, wind_levels.keys(), title='', fontsize=8, loc='best',
             frameon=True, framealpha=0.9, ncol=1, handlelength=1,
             columnspacing=0.8, handletextpad=0.5, borderpad=0.3)
    
    ax.set_title(f'Hurricane {storm_name.upper()}', fontsize=14, fontweight='bold', pad=10)

def load_storm_coverage_dict():
    """批量读取group/storm_coverage_*_*.csv，返回(storm_id, hour, method)为key的字典。"""
    files = glob.glob('group/storm_coverage_*_*.csv')
    cov_dict = {}
    for file in files:
        df = pd.read_csv(file)
        for _, row in df.iterrows():
            key = (row['storm_id'], str(row['hour']), row['method'])
            cov_dict[key] = (row['coverage'], row['mean_width'])
    return cov_dict

def create_storm_overview_figure(storm_id, storm_name, data):
    # Only plot storms with all 6h/12h/18h/24h data
    storms_6h = set(data['detailed_6h']['storm_id'].unique())
    storms_12h = set(data['detailed_12h']['storm_id'].unique())
    storms_18h = set(data['detailed_18h']['storm_id'].unique())
    storms_24h = set(data['detailed_24h']['storm_id'].unique())
    common_storms = storms_6h & storms_12h & storms_18h & storms_24h
    # Print all common storms only once
    if not hasattr(create_storm_overview_figure, '_printed_common_storms'):
        storm_info = data['storm_info'][data['storm_info']['storm_id'].isin(common_storms)]
        print('Storms with all 6h/12h/18h/24h data:')
        print(storm_info[['storm_id', 'storm_name']].to_string(index=False))
        storm_info[['storm_id', 'storm_name']].to_csv('common_storms.csv', index=False)
        create_storm_overview_figure._printed_common_storms = True
    if storm_id not in common_storms:
        return  # Skip incomplete storms
    # Read track data
    storm_data = data['test_storms'][data['test_storms']['storm_id'] == storm_id]
    # Read grouped bar chart data
    grouped_stats = [data[f'grouped_{h}'] for h in ['6h', '12h', '18h', '24h']]
    # Interval method info
    method_info = [
        ('α=0.3', 'cp_lower_03', 'cp_upper_03', '#C5E7E8'),
        ('Dynamic α', 'dyn_lower', 'dyn_upper', '#f4ba8a'),
        ('α=0.05', 'cp_lower_005', 'cp_upper_005', '#a1e3a1'),
    ]
    hours = ['6h', '12h', '18h', '24h']

    fig = plt.figure(figsize=(20,15))  # Proportionally reduced canvas
    gs = gridspec.GridSpec(5, 6, height_ratios=[1, 1, 1, 1, 1])  # 左2列宽，右4列窄       )

    # Track plot (top left, 2x2)
    ax_path = fig.add_subplot(gs[0:2, 0:2], projection=ccrs.PlateCarree())
    plot_storm_track(ax_path, storm_data, storm_name)

    # Grouped bar charts (top right, 2x2)
    bar_gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0:2, 2:6], wspace=0.3, hspace=0.45)
    methods = ['CP_0.3', 'Dynamic', 'CP_0.05']
    method_labels = ['α=0.3', 'Dynamic α', 'α=0.05']
    hour_map = ['6h', '12h', '18h', '24h']
    cov_dict = load_storm_coverage_dict()
    bar_titles = ['6h Coverage & Width', '12h Coverage & Width', '18h Coverage & Width', '24h Coverage & Width']

    for i, hour in enumerate(hour_map):
        ax_bar = fig.add_subplot(bar_gs[i//2, i%2])
        ax_bar.set_title(bar_titles[i], fontsize=14, fontweight='bold', pad=15)
        coverage_data = []
        width_data = []
        for method in methods:
            key = (storm_id, hour, method)
            if key in cov_dict:
                coverage, mean_width = cov_dict[key]
                coverage_data.append(coverage * 100)  # percent
                width_data.append(mean_width)
            else:
                coverage_data.append(0)
                width_data.append(0)
        x = np.arange(len(methods))
        width = 0.18

        # Left y-axis: Coverage
        bars1 = ax_bar.bar(x - width/2, coverage_data, width, label='Coverage (%)', color='#a1e3a1', edgecolor='black', alpha=0.7)
        # 在左侧bar上添加覆盖率数值
        for rect, value in zip(bars1, coverage_data):
            ax_bar.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 2, f'{value:.1f}',
                        ha='center', va='bottom', fontsize=10, color='#73b475', fontweight='bold')

        ax_bar.set_ylabel('Coverage (%)', fontsize=14, fontweight='bold', color='#73b475')
        ax_bar.set_ylim(0, 100)
        ax_bar.tick_params(axis='y', labelcolor='#73b475', labelsize=12)
        # x-axis
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(method_labels, rotation=0, fontsize=13, fontweight='bold')

        # Right y-axis: Interval width
        ax_bar2 = ax_bar.twinx()
        bars2 = ax_bar2.bar(x + width/2, width_data, width, label='Interval width (kt)', color='#ff9e5c', edgecolor='black', alpha=0.7)
        # 在右侧bar上添加区间宽度数值
        for rect, value in zip(bars2, width_data):
            ax_bar2.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 1, f'{value:.1f}',
                         ha='center', va='bottom', fontsize=10, color='#ff9e5c', fontweight='bold')
        ax_bar2.set_ylabel('Interval width (kt)', fontsize=14, fontweight='bold', color='#ff9e5c')
        ax_bar2.tick_params(axis='y', labelcolor='#ff9e5c', labelsize=12)
        ax_bar2.set_ylim(0, max(width_data) * 1.2 if max(width_data) > 0 else 1)

        # Only show legend once
        if i == 1:
            bars = [bars1[0], bars2[0]]
            labels = ['Coverage (%)', 'Interval width (kt)']
            ax_bar.legend(bars, labels, fontsize=9, loc='upper left', frameon=True)

        ax_bar.grid(True, alpha=0.3, axis='y')

    # Interval grid plot (bottom 3x4), now wider
    bottom_gs = gridspec.GridSpecFromSubplotSpec(3, 4, subplot_spec=gs[2:5, :], wspace=0.25, hspace=0.3)
    for row, (method, lower, upper, fill_color) in enumerate(method_info):
        for col, hour in enumerate(hours):
            ax = fig.add_subplot(bottom_gs[row, col])
            df = data[f'detailed_{hour}']
            storm_df = df[df['storm_id'] == storm_id].sort_values('init_time')
            x_seq = np.arange(1, len(storm_df) + 1)
            if lower in storm_df.columns and upper in storm_df.columns:
                ax.fill_between(x_seq, storm_df[lower], storm_df[upper], color=fill_color, alpha=0.5, label='Interval')
            if 'predicted_value' in storm_df.columns:
                ax.plot(x_seq, storm_df['predicted_value'], '-^', color=(0/255, 112/255, 192/255),
                        label='Predicted', linewidth=2.5, markersize=8, markerfacecolor=(0/255, 112/255, 192/255))
            if 'true_value' in storm_df.columns:
                ax.plot(x_seq, storm_df['true_value'], '-o', color=(255/255, 189/255, 0/255),
                        label='Actual', linewidth=2.5, markersize=8, markerfacecolor=(255/255, 189/255, 0/255))
            ax.grid(True, linestyle='--', alpha=0.3)
            if row == 0:
                ax.set_title(f'{hour} Forecast', fontsize=14, pad=8)
            if col == 0:
                ax.set_ylabel(f'{method}\nIntensity (kt)', fontsize=14, fontweight='bold')
            if row == 2:
                ax.set_xlabel('Timepoint', fontsize=12)
            # Only show legend in the first plot of each method (first column)
            if col == 0:
                ax.legend(loc='upper right', fontsize=10, frameon=True, framealpha=0.8, edgecolor='none')
            ax.tick_params(axis='both', labelsize=10)


    plt.suptitle(f'Hurricane {storm_name.upper()} ({storm_id}) Overview', fontsize=24, y=0.98, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_dir = 'storm_overview'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'storm_overview_{storm_name}_{storm_id}.png'), dpi=600, bbox_inches='tight')
    plt.close()
    print(f'已生成 {storm_name} ({storm_id}) 的总览大图')

def main():
    # Load all data
    data = load_data()
    
    # 检查所有预测时间点的数据中包含哪些basin
    all_basins = set()
    all_storms = set()
    hours = ['6h', '12h', '18h', '24h']
    
    # 输出每个预测时间点的basin信息
    print("\n=== Basin information for each forecast horizon ===")
    for hour in hours:
        if f'detailed_{hour}' in data:
            basins = data[f'detailed_{hour}']['basin'].unique()
            all_basins.update(basins)
            all_storms.update(data[f'detailed_{hour}']['storm_id'].unique())
            print(f"{hour} forecast basins: {sorted(basins)}")
            
            # 输出每个basin的风暴数量
            basin_counts = data[f'detailed_{hour}'].drop_duplicates('storm_id')['basin'].value_counts()
            for basin, count in basin_counts.items():
                print(f"  {hour} - {basin}: {count} storms")
    
    print(f"\nAll available basins in data: {sorted(all_basins)}")
    
    # 输出每个basin的风暴ID和名称
    print("\n=== Storm IDs and names by basin ===")
    for basin in sorted(all_basins):
        print(f"\nStorms in {basin} basin:")
        basin_storms = []
        
        # 收集所有时间点中该basin的风暴
        for hour in hours:
            if f'detailed_{hour}' in data:
                basin_data = data[f'detailed_{hour}'][data[f'detailed_{hour}']['basin'] == basin]
                for storm_id in basin_data['storm_id'].unique():
                    storm_name = basin_data[basin_data['storm_id'] == storm_id]['storm_name'].iloc[0] if 'storm_name' in basin_data.columns else 'Unknown'
                    basin_storms.append((storm_id, storm_name))
        
        # 去重并排序
        basin_storms = sorted(set(basin_storms), key=lambda x: x[0])
        for storm_id, storm_name in basin_storms:
            print(f"  {storm_id}: {storm_name}")
    
    print("\n=== End of basin information ===")
    
    # 按basin分组统计风暴数量
    basin_storm_counts = {}
    for hour in hours:
        if f'detailed_{hour}' in data:
            for basin in data[f'detailed_{hour}']['basin'].unique():
                basin_storms = set(data[f'detailed_{hour}'][data[f'detailed_{hour}']['basin'] == basin]['storm_id'].unique())
                if basin not in basin_storm_counts:
                    basin_storm_counts[basin] = basin_storms
                else:
                    basin_storm_counts[basin].update(basin_storms)

    # 只保留basin为'EP'的分组
    ep_storms = basin_storm_counts.get('EP', set())
    print("\n[EP basin] Storm count:")
    print(f"  EP: {len(ep_storms)} storms")
    print("  Storm IDs:", sorted(ep_storms))
    
    # 输出所有测试集风暴的数量和ID
    all_test_storms = set(data['storm_info']['storm_id'])
    print("\n[All test storms] Storm count:")
    print(f"  Total: {len(all_test_storms)} storms")
    print("  Storm IDs:", sorted(all_test_storms))
    
    # 遍历所有测试集风暴进行可视化
    print("\n[DEBUG] all_test_storms:", all_test_storms)
    for storm_id in sorted(all_test_storms):
        print(f"[DEBUG] Processing storm_id: {storm_id}")
        # 获取风暴名称
        storm_row = data['storm_info'][data['storm_info']['storm_id'] == storm_id]
        print(f"[DEBUG] storm_row for {storm_id}:\n", storm_row)
        if storm_row.empty:
            print(f"[DEBUG] storm_row is empty for storm_id: {storm_id}, skipping.")
            continue
        storm_name = storm_row.iloc[0]['storm_name']
        print(f"[DEBUG] storm_name for {storm_id}: {storm_name}")
        # 跳过unnamed风暴
        if str(storm_name).strip().lower() == 'unnamed':
            print(f"[DEBUG] storm_name is 'unnamed' for storm_id: {storm_id}, skipping.")
            continue
        # 只处理2020年及以后的台风
        try:
            year = int(str(storm_id)[-4:])
            print(f"[DEBUG] year for {storm_id}: {year}")
        except Exception as e:
            print(f"[DEBUG] Failed to extract year from storm_id: {storm_id}, error: {e}")
            continue
        if year < 2020:
            print(f"[DEBUG] year < 2020 for storm_id: {storm_id}, skipping.")
            continue
        print(f"Creating visualization for storm {storm_id} ({storm_name})")
        create_storm_overview_figure(storm_id, storm_name, data)

if __name__ == "__main__":
    main()
