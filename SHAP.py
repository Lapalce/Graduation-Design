import shap
import numpy as np
import torch
import Model

model_path = ['']

# 重新定义模型结构
input_size = [[12, 5, 5, 7], [24, 10, 10, 14]]  # 特征数量
hidden_size = 64
output_size = 5  # 输出特征数量
sparse_layer_size = 128  # 稀疏层的大小
dropout_rate = 0.5  # Dropout率

models = []
for path in model_path:
    tmp_model = Model.PSRegionBiLSTM(input_size, hidden_size, output_size, sparse_layer_size=sparse_layer_size,
                                     dropout_rate=dropout_rate)
    tmp_model.load_state_dict(torch.load('model_weights.pth'))

    # 设置模型为评估模式
    tmp_model.eval()

    # 加载模型权重
    models.append(tmp_model)

print("模型已加载")

X_datasets = [X1, X2, X3]  # 每个数据集的列表

# 创建一个空的列表，用于存储每个模型的特征贡献
all_shap_values = []

# 计算每个模型的 SHAP 值
for i, model in enumerate(models):
    X_test = X_datasets[i]  # 获取第 i 个数据集

    # 使用SHAP来创建解释器（解释器需要模型的输出）
    explainer = shap.Explainer(model, X_test)

    # 计算 SHAP 值
    shap_values = explainer(X_test)

    # 将当前模型的 SHAP 值保存到 all_shap_values 列表
    all_shap_values.append(shap_values)

# 假设每个模型的 SHAP 值是一个列表，每个元素代表一个特征的 SHAP 值
# 现在我们要计算每个特征的平均贡献

# 首先，提取每个模型中的特征贡献（取平均值）
average_shap_values = []

# 对每个特征计算平均 SHAP 值
num_features = X_test.shape[2]  # 假设 X_test 的形状为 [batch_size, sequence_length, features]
for feature_idx in range(num_features):
    # 获取每个模型中当前特征的 SHAP 值
    shap_feature_values = [shap_values[:, feature_idx] for shap_values in all_shap_values]

    # 将所有模型中该特征的 SHAP 值取平均
    average_shap_values.append(np.mean(shap_feature_values, axis=0))

# 输出每个特征的平均贡献
for feature_idx, avg_shap in enumerate(average_shap_values):
    print(f"Feature {feature_idx} - Average SHAP value: {np.mean(avg_shap)}")
