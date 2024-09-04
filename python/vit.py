from torch import nn
import torch

class ViT(nn.Module):
    """
    Vision Transformer (ViT) 类，用于图像分类任务。

    参数:
    - emb_size: int，嵌入层的尺寸，默认为16。

    输入:
    - x: 图像张量，形状为(batch_size, channel=1, width=28, height=28)。

    输出:
    - 输出张量，形状为(batch_size, num_classes)。
    """
    def __init__(self, emb_size=16):
        super().__init__()
        # 定义patch的大小和数量
        self.patch_size = 4
        self.patch_count = 28 // self.patch_size  # 7

        # 定义将图像转换为patch的卷积层
        self.conv = nn.Conv2d(in_channels=1, out_channels=self.patch_size ** 2, kernel_size=self.patch_size, padding=0,
                              stride=self.patch_size)
        # 定义patch嵌入层
        self.patch_emb = nn.Linear(in_features=self.patch_size ** 2, out_features=emb_size)
        # 定义分类用的可学习cls_token
        self.cls_token = nn.Parameter(torch.rand(1, 1, emb_size))
        # 定义位置嵌入
        self.pos_emb = nn.Parameter(torch.rand(1, self.patch_count ** 2 + 1, emb_size))
        # 定义Transformer编码器
        self.tranformer_enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=emb_size, nhead=2, batch_first=True), num_layers=3)
        # 定义分类线性层
        self.cls_linear = nn.Linear(in_features=emb_size, out_features=10)

    def forward(self, x):
        # 将图像转换为patch
        x = self.conv(x)
        # 展平patch
        x = x.view(x.size(0), x.size(1), self.patch_count ** 2)
        # 调整维度顺序
        x = x.permute(0, 2, 1)
        # patch嵌入
        x = self.patch_emb(x)
        # 扩展cls_token并拼接到patch序列前
        cls_token = self.cls_token.expand(x.size(0), 1, x.size(2))
        x = torch.cat((cls_token, x), dim=1)
        # 加上位置嵌入
        x = self.pos_emb + x
        # 通过Transformer编码器
        y = self.tranformer_enc(x)
        # 对[CLS] token输出做分类
        return self.cls_linear(y[:, 0, :])

if __name__ == '__main__':
    vit = ViT()
    x = torch.rand(5, 1, 28, 28)
    y = vit(x)
    print(y.shape)
