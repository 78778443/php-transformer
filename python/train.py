# 导入PyTorch库和其他必要模块
import torch
from dataset import MNIST
from vit import ViT
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os

# 根据CUDA是否可用设置设备
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 初始化MNIST数据集
dataset = MNIST()

# 初始化ViT模型并将其转移到指定设备上
model = ViT().to(DEVICE)

# 尝试加载预训练模型参数
try:
    model.load_state_dict(torch.load('model.pth'))
except:
    pass

# 初始化Adam优化器，用于模型参数更新
optimzer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 训练模型
EPOCH = 50
BATCH_SIZE = 64

# 创建DataLoader用于数据加载，设置多线程加速
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=10, persistent_workers=True)

# 初始化迭代计数器
iter_count = 0

# 遍历每个epoch
for epoch in range(EPOCH):
    # 遍历每个数据加载器中的图片和标签
    for imgs, labels in dataloader:
        # 前向传播，计算图片的logits
        logits = model(imgs.to(DEVICE))

        # 计算交叉熵损失
        loss = F.cross_entropy(logits, labels.to(DEVICE))

        # 梯度清零
        optimzer.zero_grad()
        # 反向传播计算梯度
        loss.backward()
        # 更新模型参数
        optimzer.step()

        # 每1000次迭代打印损失并保存模型
        if iter_count % 1000 == 0:
            print('epoch:{} iter:{},loss:{}'.format(epoch, iter_count, loss))
            torch.save(model.state_dict(), '.model.pth')
            os.replace('.model.pth', 'model.pth')

        # 更新迭代计数器
        iter_count += 1
