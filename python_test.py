# import pandas as pd


# df: pd.DataFrame = pd.DataFrame([
#     [1, 1, 1, 1, 1],
#     [2, 2, None,2,2],
#     [3,3,3,3,3]
# ], columns=['one', 'two', 'three', 'four', 'five'])

# titles = [column for column in df]
# ramris_ids = df[titles[0]].values
# print(ramris_ids)
# print(type(ramris_ids))


# print(df.iloc[1, 1:2].values)
# print(type(df.iloc[1, 1:2].values))


# from thop import profile
# from models.model import ModelClass
# import torch
# from models.vit import ViT
# from models.mobilevit import mobilevit_s, mobilevit_xs, mobilevit_xxs

# # model = ModelClass(img_ch=40, num_classes=2)
# model = ViT(image_size=(512, 512), patch_size=(4, 4), num_classes=2, 
#                dim=256, depth=6, heads=8, mlp_dim=512, pool='cls', channels=40, dropout=0.2, emb_dropout=0.2)
# arr = torch.randn((6, 40, 512, 512))

# flops, params = profile(model, inputs=(arr, ))
# print(f'FLOPS: {flops}, params: {params}')

# from sklearn.model_selection import KFold
# import numpy as np

# x = np.array([[1,2], [3,4], [5,6]])
# y = np.array([1, 2, 3])


# kf = KFold(n_splits=3, shuffle=True)
# for train_index, test_index in kf.split(x):
#     print('train:', train_index, 'val:', test_index)
#     x_t, x_v = x[train_index], x[test_index]
#     y_t, y_v = y[train_index], y[test_index]
#     print(x_t)
#     print(y_t)
#     print(1)

a = {'A':[1,2], 'B':[2,3]}

print(a.keys())
print(list(a.keys())[0])