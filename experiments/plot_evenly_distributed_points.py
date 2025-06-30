import plotly.express as px
import pandas as pd
import numpy as np
from itertools import product
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

# 创建3D图表函数
def create_3d_fig(dataframe, x_col, y_col, z_col, color_col, title,THRESHOLD):
    # Create a single figure with all points
    fig = px.scatter_3d(
        dataframe,
        x=x_col, y=y_col, z=z_col,
        color=color_col,
        color_continuous_scale='Viridis',
        title=title,
        opacity=0.7,
        size_max=10
    )

    # 如果有超过阈值的点，将它们设为红色
    if (dataframe[color_col] > THRESHOLD).any():
        # 获取超过阈值的点的位置索引（从0开始的连续索引）
        above_mask = (dataframe[color_col] > THRESHOLD).values
        above_indices = np.where(above_mask)[0]

        # 创建一个颜色数组的副本
        colors = fig.data[0]['marker']['color']

        # # 为每个超过阈值的点设置颜色
        # for idx in above_indices:
        #     colors[idx] = 2

        # 更新图形的颜色
        fig.data[0]['marker']['color'] = colors

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


def init_translation_graph(x_col,y_col,z_col,df_avg,object_col,THRESHOLD):
    # 初始化应用
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    # 初始平移空间图表
    translation_fig = create_3d_fig(
        df_avg, x_col, y_col, z_col, object_col,
        '平移空间 - 点击任意点查看对应旋转空间',THRESHOLD
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
    return app

if __name__ == '__main__':
    data = np.load("F://alignanything//even_distributed_successrate.npy", allow_pickle=True)
    data = data.tolist()
    # for i in range(len(data)):
    #     if data[i]["rx"]==5 and data[i]["ry"]==5 and data[i]["rz"]==10:
    #         m=np.array(data[i]["error_rot"]).mean()
    #         print(m)
    df = pd.DataFrame(data)
    obj_col="error_trans"
    THRESHOLD = 1  # 可以根据需要调整

    real_obj_col_key=obj_col
    if isinstance(df[obj_col][0],list):
        real_obj_col_key = obj_col + "_mean"
        df[real_obj_col_key] = df[obj_col].apply(np.mean)

    # 计算平均成功率
    df_avg = df.groupby(['x', 'y', 'z'], as_index=False)[real_obj_col_key].mean()
    df_avg['point_id'] = df_avg.index

    app=init_translation_graph(x_col='x', y_col='y', z_col='z',df_avg=df_avg,object_col=real_obj_col_key,THRESHOLD=THRESHOLD)
    @app.callback(             # 回调函数
        Output('rotation-3d-scatter', 'figure'),
        Output('rotation-3d-scatter', 'style',),
        Input('translation-scatter', 'clickData')
    )
    def update_rotation_3d(click_data):
        if not click_data:
            return {}, {'display': 'none'}

        point_info = click_data['points'][0]
        # print(point_info)
        point_id = point_info.get('customdata', point_info['pointNumber'])
        # print(point_id)
        tx, ty, tz = df_avg.loc[point_id, ['x', 'y', 'z']]
        rotation_df = df[(df['x'] == tx) & (df['y'] == ty) & (df['z'] == tz)]

        if obj_col!=real_obj_col_key:
            rotation_df[real_obj_col_key] = rotation_df[obj_col].apply(np.mean)
        rotation_fig = create_3d_fig(
            rotation_df, 'rx', 'ry', 'rz', real_obj_col_key,
            f'旋转空间 (平移: x={tx:.3f}, y={ty:.3f}, z={tz:.3f})',THRESHOLD
        )

        return rotation_fig, {'display': 'block'}

    app.run(debug=True)
# import plotly.express as px
# import pandas as pd
# import numpy as np
# from itertools import product
# from dash import Dash, dcc, html, Input, Output
# import dash_bootstrap_components as dbc
#
#
# # 创建3D图表函数
# def create_3d_fig(dataframe, x_col, y_col, z_col, color_col, title, THRESHOLD):
#     # 1. 创建颜色数组：超过阈值的点设为红色，其他点映射到 Viridis 色标
#     colors = []
#     color_min = dataframe[color_col].min()
#     color_max = dataframe[color_col].max()
#
#     for val in dataframe[color_col]:
#         if val > THRESHOLD:
#             colors.append('#FF0000')  # 红色（十六进制）
#         else:
#             # 将数值归一化到 [0, 1]，用于 Viridis 色标
#             normalized_val = (val - color_min) / (color_max - color_min)
#             colors.append(normalized_val)
#
#     # 2. 准备 hover_data，只包含 DataFrame 中存在的列
#     hover_columns = [x_col, y_col, z_col, color_col]
#     hover_data = {col: True for col in hover_columns if col in dataframe.columns}
#
#     # 3. 创建图形
#     fig = px.scatter_3d(
#         dataframe,
#         x=x_col, y=y_col, z=z_col,
#         color=None,  # 禁用自动颜色映射
#         title=title,
#         opacity=0.7,
#         size_max=10,
#         hover_data=hover_data  # 仅包含存在的列
#     )
#
#     # 4. 手动设置颜色
#     fig.update_traces(
#         marker=dict(
#             color=colors,
#             colorscale=[
#                 [0.0, 'rgb(26, 42, 108)'],  # 深蓝 (Dark Blue)
#                 [0.17, 'rgb(0, 147, 146)'],  # 蓝绿色 (Teal)
#                 [0.34, 'rgb(84, 199, 89)'],  # 鲜绿色 (Vivid Green)
#                 [0.5, 'rgb(255, 230, 50)'],  # 亮黄色
#                 [1.0, 'rgb(255, 150, 30)']  # 橙色
#             ],
#             cmin=color_min,
#             cmax=color_max,
#             colorbar=dict(title=color_col),
#             showscale=True  # 显示色标
#         )
#     )
#
#     # 5. 更新布局
#     fig.update_layout(
#         scene=dict(
#             xaxis_title=x_col.upper(),
#             yaxis_title=y_col.upper(),
#             zaxis_title=z_col.upper(),
#             aspectmode='cube'
#         ),
#         margin=dict(l=0, r=0, b=0, t=30),
#         height=700
#     )
#     return fig
#
#
# def init_translation_graph(x_col, y_col, z_col, df_avg, object_col, THRESHOLD):
#     # 初始化应用
#     app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
#
#     # 初始平移空间图表
#     translation_fig = create_3d_fig(
#         df_avg, x_col, y_col, z_col, object_col,
#         '平移空间 - 点击任意点查看对应旋转空间', THRESHOLD
#     )
#
#     # 应用布局
#     app.layout = html.Div(
#         style={
#             'display': 'flex',
#             'height': '100vh',
#             'width': '100vw',
#             'margin': '0',
#             'padding': '0',
#             'overflow': 'hidden'
#         },
#         children=[
#             html.Div(
#                 dcc.Graph(
#                     id='translation-scatter',
#                     figure=translation_fig,
#                     style={'height': '100%', 'width': '100%'}
#                 ),
#                 style={
#                     'width': '50%',
#                     'padding': '10px',
#                     'height': '100%',
#                     'boxSizing': 'border-box'
#                 }
#             ),
#             html.Div(
#                 dcc.Graph(
#                     id='rotation-3d-scatter',
#                     style={
#                         'height': '100%',
#                         'width': '100%',
#                         'display': 'none'
#                     }
#                 ),
#                 style={
#                     'width': '50%',
#                     'padding': '10px',
#                     'height': '100%',
#                     'boxSizing': 'border-box'
#                 }
#             )
#         ]
#     )
#     return app
#
#
# if __name__ == '__main__':
#     data = np.load("F://alignanything//even_distributed_successrate.npy", allow_pickle=True)
#     obj_col = "error_transz"
#     THRESHOLD = 1# 可以根据需要调整
#
#     data = data.tolist()
#     for i in range(len(data)):
#         if data[i]["rx"] == 5 and data[i]["ry"] == 5 and data[i]["rz"] == 10:
#             m = np.array(data[i]["error_rot"]).mean()
#             print(m)
#     df = pd.DataFrame(data)
#
#     real_obj_col_key = obj_col
#     if isinstance(df[obj_col][0], list):
#         real_obj_col_key = obj_col + "_mean"
#         df[real_obj_col_key] = df[obj_col].apply(np.mean)
#
#     # 计算平均成功率
#     df_avg = df.groupby(['x', 'y', 'z'], as_index=False)[real_obj_col_key].mean()
#     df_avg['point_id'] = df_avg.index
#
#     app = init_translation_graph(x_col='x', y_col='y', z_col='z', df_avg=df_avg, object_col=real_obj_col_key,
#                                  THRESHOLD=THRESHOLD)
#
#
#     @app.callback(  # 回调函数
#         Output('rotation-3d-scatter', 'figure'),
#         Output('rotation-3d-scatter', 'style', ),
#         Input('translation-scatter', 'clickData')
#     )
#     def update_rotation_3d(click_data):
#         if not click_data:
#             return {}, {'display': 'none'}
#
#         point_info = click_data['points'][0]
#         point_id = point_info.get('customdata', point_info['pointNumber'])
#         tx, ty, tz = df_avg.loc[point_id, ['x', 'y', 'z']]
#         rotation_df = df[(df['x'] == tx) & (df['y'] == ty) & (df['z'] == tz)]
#
#         if obj_col != real_obj_col_key:
#             rotation_df[real_obj_col_key] = rotation_df[obj_col].apply(np.mean)
#         rotation_fig = create_3d_fig(
#             rotation_df, 'rx', 'ry', 'rz', real_obj_col_key,
#             f'旋转空间 (平移: x={tx:.3f}, y={ty:.3f}, z={tz:.3f})', THRESHOLD
#         )
#
#         return rotation_fig, {'display': 'block'}
#
#
#     app.run(debug=True)
