# 花朵分类任务 - ResNet50模型配置 (兼容MMPretrain标准格式)

_base_ = [
    '/content/mmpretrain/configs/_base_/models/resnet50.py',
    '/content/mmpretrain/configs/_base_/datasets/imagenet_bs32.py',
    '/content/mmpretrain/configs/_base_/schedules/imagenet_bs256.py',
    '/content/mmpretrain/configs/_base_/default_runtime.py'
]

# 模型配置 (必须使用标准键名'model')
model = dict(
    head=dict(
        num_classes=5,
        topk=(1,),
    ))

# 数据集路径
data_root = '/content/flower_dataset'
dataset_type = 'ImageNet'

# 训练数据配置
train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix='train/',
        ann_file='train.txt',
        classes=['daisy', 'dandelion', 'rose', 'sunflower', 'tulip'],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='RandomResizedCrop', scale=224),
            dict(type='RandomFlip', prob=0.5),
            dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
            dict(type='PackInputs'),
        ]))

# 验证数据配置
val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix='val/',
        ann_file='val.txt',
        classes=['daisy', 'dandelion', 'rose', 'sunflower', 'tulip'],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='ResizeEdge', scale=256, edge='short'),
            dict(type='CenterCrop', crop_size=224),
            dict(type='PackInputs'),
        ]))

test_dataloader = val_dataloader

# 评估配置
val_evaluator = dict(type='Accuracy', topk=(1,))
test_evaluator = val_evaluator

# 优化器配置
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0001),
    clip_grad=None)

# 学习率调度
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=50),
    dict(type='MultiStepLR', by_epoch=True, milestones=[20, 25], gamma=0.1)
]

# 训练配置
train_cfg = dict(by_epoch=True, max_epochs=30, val_interval=1)

# 预训练模型
load_from = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth'