<?php
require_once __DIR__ . '/bootstrap.php';
// 定义MNIST数据集类，继承自Dataset类
// 该类用于加载和处理MNIST手写数字数据集

/**
 * @property mixed $img_convert
 * @property $ds
 * @method img_convert(mixed $img)
 */
#[parent('Dataset', 'torch.utils.data')]
class MNIST extends \phpy\PyClass
{
    public function __construct($is_train = true)
    {
        parent::__construct();
        // 导入必要的模块
        $torch = PyCore::import('torch');
        $v2 = PyCore::import('torchvision.transforms.v2');
        $PILToTensor = PyCore::import('torchvision.transforms.v2');
        $torchvision = PyCore::import('torchvision');

        // 下载并加载MNIST数据集，存储在本地目录 'mnist/' 下
        $this->ds = $torchvision->datasets->MNIST('./mnist/', train: $is_train, download: true);
        // 定义图像转换流程，将PIL图像转换为张量
        $this->img_convert = $v2->Compose([$PILToTensor->PILToTensor()]);
    }

    public function __len__()
    {
        // 返回数据集的样本数量
        return $this->ds->__len__();
    }

    public function __getitem__($index)
    {
        // 获取数据集中的指定索引的样本
        list($img, $label) = $this->ds[$index];
        // 对图像进行转换，并将其像素值归一化到0-1之间
        $convert = $this->img_convert($img);
        return PyCore::tuple([PyCore::import('operator')->__truediv__($convert, 255.0), $label]);
    }
}

// 当作为主程序运行时
if (basename(__FILE__) == basename($_SERVER['PHP_SELF'])) {
    $plt = PyCore::import('matplotlib.pyplot');
    // 实例化MNIST数据集对象
    $ds = new MNIST();
    // 获取第一个样本的图像和标签
    list($img, $label) = $ds->__getitem__(0);
    // 打印图像对应的标签
    PyCore::print($img);
    PyCore::print($label);
}

