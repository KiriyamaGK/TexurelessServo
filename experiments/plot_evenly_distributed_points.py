import plotly.express as px
import pandas as pd
import numpy as np
from itertools import product
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

# 生成示例数据
translations = list(product(
    np.arange(-0.005, 0.006, 0.005),  # x
    np.arange(-0.005, 0.006, 0.005),  # y
    np.arange(0, 0.051, 0.005)  # z
))
rotations = list(product(
    np.arange(-5, 6, 2),  # rx
    np.arange(-5, 6, 2),  # ry
    np.arange(-30, 31, 5)  # rz
))

# 创建嵌套DataFrame
print("正在生成数据...")
data = []
for tx, ty, tz in translations:
    for rx, ry, rz in rotations:
        data.append({
            'x': tx, 'y': ty, 'z': tz,
            'rx': rx, 'ry': ry, 'rz': rz,
            'success': np.random.uniform(0, 100)
        })
# data=np.load("/home/kiriyamagk/桌面/AlignAnything/trained_models/trial/2025-05-23_21-23-39/eval_results/2025-06-02_23-47-26(epoch280)/even_distributed_successrate.npy",allow_pickle=True)
# data=data.tolist()
df = pd.DataFrame(data)

# 计算平均成功率
df_avg = df.groupby(['x', 'y', 'z'], as_index=False)['success'].mean()
df_avg['point_id'] = df_avg.index


# 创建3D图表函数
def create_3d_fig(dataframe, x_col, y_col, z_col, color_col, title):
    fig = px.scatter_3d(
        dataframe,
        x=x_col, y=y_col, z=z_col,
        color=color_col,
        color_continuous_scale='Viridis',
        title=title,
        opacity=0.7,
        size_max=10
    )
    fig.update_layout(
        scene=dict(
            xaxis_title=x_col.upper(),
            yaxis_title=y_col.upper(),
            zaxis_title=z_col.upper(),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=700  # 固定高度
    )
    return fig


# 初始化应用
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# 初始平移空间图表
translation_fig = create_3d_fig(
    df_avg, 'x', 'y', 'z', 'success',
    '平移空间 - 点击任意点查看对应旋转空间'
)

# 应用布局
app.layout = html.Div(
    style={
        'display': 'flex',
        'height': '100vh',
        'width': '100vw',
        'margin': '0',
        'padding': '0',
        'overflow': 'hidden'
    },
    children=[
        html.Div(
            dcc.Graph(
                id='translation-scatter',
                figure=translation_fig,
                style={'height': '100%', 'width': '100%'}
            ),
            style={
                'width': '50%',
                'padding': '10px',
                'height': '100%',
                'boxSizing': 'border-box'
            }
        ),
        html.Div(
            dcc.Graph(
                id='rotation-3d-scatter',
                style={
                    'height': '100%',
                    'width': '100%',
                    'display': 'none'
                }
            ),
            style={
                'width': '50%',
                'padding': '10px',
                'height': '100%',
                'boxSizing': 'border-box'
            }
        )
    ]
)


# 回调函数
@app.callback(
    Output('rotation-3d-scatter', 'figure'),
    Output('rotation-3d-scatter', 'style'),
    Input('translation-scatter', 'clickData')
)
def update_rotation_3d(click_data):
    if not click_data:
        return {}, {'display': 'none'}

    point_info = click_data['points'][0]
    point_id = point_info.get('customdata', point_info['pointNumber'])
    tx, ty, tz = df_avg.loc[point_id, ['x', 'y', 'z']]
    rotation_df = df[(df['x'] == tx) & (df['y'] == ty) & (df['z'] == tz)]

    rotation_fig = create_3d_fig(
        rotation_df, 'rx', 'ry', 'rz', 'success',
        f'旋转空间 (平移: x={tx:.3f}, y={ty:.3f}, z={tz:.3f})'
    )

    return rotation_fig, {'display': 'block'}


if __name__ == '__main__':
    app.run(debug=True)