<?php

require __DIR__ . '/bootstrap.php';

use phpy\PyClass;

/**
 * @property int|mixed $emb_size
 * @property int $patch_size
 * @property $conv
 * @property $patch_emb
 * @property $cls_token
 * @property $pos_emb
 * @property $tranformer_enc
 * @property $cls_linear
 */
#[parent('Module', 'torch.nn')]
class ViT extends PyClass
{
    private $torch; // 存储导入的torch模块
    private $nn;    // 學儲导入的nn模块
    private int $patch_count;

    public function __construct($emb_size = 16)
    {
        parent::__construct();
        $this->super()->__init__();
        $this->torch = PyCore::import('torch');
        $this->nn = PyCore::import('torch.nn');
        $this->emb_size = $emb_size;
        $this->patch_size = 4;
        $this->patch_count = intdiv(28, $this->patch_size);  // 7
        // 定义将图像转换为patch的卷积层
        $this->conv = $this->nn->Conv2d(
            in_channels: 1,
            out_channels: pow($this->patch_size, 2),
            kernel_size: $this->patch_size,
            padding: 0,
            stride: $this->patch_size,
        );
        // 定义patch嵌入层
        $this->patch_emb = $this->nn->Linear(pow($this->patch_size, 2), $this->emb_size);
        // 定义分类用的可学习cls_token
        $this->cls_token = $this->torch->randn([1, 1, $this->emb_size]);
        // 定义位置嵌入
        $this->pos_emb = $this->torch->randn([1, pow($this->patch_count, 2) + 1, $this->emb_size]);
        $encoder_layer = $this->nn->TransformerEncoderLayer($this->emb_size, 2,
            dim_feedforward: 2 * $this->emb_size,
            dropout: 0.1,
            activation: 'relu',
            layer_norm_eps: 1e-5,
            batch_first: true
        );
        $this->tranformer_enc = $this->nn->TransformerEncoder($encoder_layer, 3);
        // 定义分类线性层
        $this->cls_linear = $this->nn->Linear($this->emb_size, 10);
    }

    public function forward($x)
    {
        $operator = PyCore::import('operator');
        // 将图像转换为patch
        $x = $this->conv->forward($x);
        // 展平patch 确保展平操作不会导致非法的步幅
        $batch_size = $x->size(0);
        $out_channels = $x->size(1);
        $height = $x->size(2);
        $width = $x->size(3);
        $x = $x->view($batch_size, $out_channels, $height * $width);

        // 调整维度顺序
        $x = $x->permute([0, 2, 1]);

        // patch嵌入
        $x = $this->patch_emb->forward($x);

        // 扩展cls_token并拼接到patch序列前
        $cls_token = $this->cls_token->expand([$x->size(0), 1, $x->size(2)]);
        $x = $this->torch->cat([$cls_token, $x], 1);

        // 加上位置嵌入
        $x = $operator->__add__($x, $this->pos_emb);

        // 通过Transformer编码器
        $x = $this->tranformer_enc->forward($x);

        // 对[CLS] token输出做分类
        return $this->cls_linear->forward($x->select(1, 0));
    }
}

//$torch = PyCore::import('torch');
//// 示例使用
//$vit = new ViT();
//$x = $torch->rand(5, 1, 28, 28); // 假设输入是一个灰度图像
//$y = $vit->forward($x);
//PyCore::print($y);
