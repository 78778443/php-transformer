<?php

require_once 'vit.php';
require_once 'dataset.php';

// 导入必要的模块
$torch = PyCore::import('torch');
$F = PyCore::import('torch.nn.functional');

$os = PyCore::import('os');
$sys = PyCore::import('sys');
$info = $sys->version;
PyCore::print($info) ;
//die;
$DataLoader = PyCore::import('torch.utils.data');

// 检查CUDA是否可用，设置设备
$DEVICE = $torch->cuda->is_available() ? 'cuda' : 'cpu';

// 初始化MNIST数据集
$dataset = new MNIST();

// 初始化ViT模型并将其转移到指定设备上
$model = new ViT();
//$model = $model->to->forward($DEVICE);

// 尝试加载预训练模型参数
try {
    $model->load_state_dict($torch->load('model.pth'));
} catch (Exception $e) {
    // 如果加载失败，可以在这里处理异常
}

// 初始化Adam优化器，用于模型参数更新
$optimizer = $torch->optim->Adam($model->parameters(), ['lr' => 1e-3]);

// 训练模型
$EPOCH = 50;
$BATCH_SIZE = 64;

// 创建DataLoader用于数据加载，设置多线程加速
$dataloader = new $DataLoader->DataLoader($dataset, ['batch_size' => $BATCH_SIZE, 'shuffle' => true, 'num_workers' => 10, 'persistent_workers' => true]);

// 初始化迭代计数器
$iter_count = 0;

// 遍历每个epoch
for ($epoch = 0; $epoch < $EPOCH; $epoch++) {
    // 遍历每个数据加载器中的图片和标签
    foreach ($dataloader as $batch) {
        list($imgs, $labels) = $batch;

        // 前向传播，计算图片的logits
        $logits = $model->forward($imgs->to($DEVICE));

        // 计算交叉熵损失
        $loss = $F->cross_entropy($logits, $labels->to($DEVICE));

        // 梯度清零
        $optimizer->zero_grad();
        // 反向传播计算梯度
        $loss->backward();
        // 更新模型参数
        $optimizer->step();

        // 每1000次迭代打印损失并保存模型
        if ($iter_count % 1000 == 0) {
            echo sprintf("epoch:%d iter:%d,loss:%s\n", $epoch, $iter_count, $loss);
            $torch->save($model->state_dict(), '.model.pth');
            $os->replace('.model.pth', 'model.pth');
        }

        // 更新迭代计数器
        $iter_count++;
    }
}

