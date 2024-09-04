# 导入PyTorch的数据集基础类
from torch.utils.data import Dataset
# 导入将PIL图像转换为张量的工具
from torchvision.transforms.v2 import PILToTensor, Compose
# 导入torchvision库用于处理图像数据
import torchvision

# 定义MNIST数据集类，继承自Dataset类
# 该类用于加载和处理MNIST手写数字数据集
class MNIST(Dataset):
    def __init__(self, is_train=True):
        """
        初始化MNIST数据集类
        参数:
            is_train (bool): 是否为训练集，默认为True
        """
        super().__init__()
        # 下载并加载MNIST数据集，存储在本地目录'mnist/'下
        self.ds = torchvision.datasets.MNIST('./mnist/', train=is_train, download=True)
        # 定义图像转换流程，将PIL图像转换为张量
        self.img_convert = Compose([
            PILToTensor(),
        ])

    def __len__(self):
        """
        返回数据集的样本数量
        返回:
            int: 数据集的样本数量
        """
        return len(self.ds)

    def __getitem__(self, index):
        """
        获取数据集中的指定索引的样本
        参数:
            index (int): 样本索引
        返回:
            tuple: 包含图像张量和标签的元组
        """
        img, label = self.ds[index]
        # 对图像进行转换，并将其像素值归一化到0-1之间
        return self.img_convert(img) / 255.0, label


# 当作为主程序运行时
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # 实例化MNIST数据集对象
    ds = MNIST()
    # 获取第一个样本的图像和标签
    img, label = ds[0]
    # 打印图像对应的标签
    print(label)
    # 显示图像，注意图像通道顺序需调整
    plt.imshow(img.permute(1, 2, 0))
    plt.show()
