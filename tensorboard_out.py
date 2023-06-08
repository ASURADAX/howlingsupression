import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# 定义Tensorboard日志路径
log_dir = './logs'

# 创建EventAccumulator对象
event_acc = EventAccumulator(log_dir)
event_acc.Reload()

# 获取所有标量摘要
tags = event_acc.Tags()['scalars']

# 读取训练损失
train_loss = []
for tag in tags:
    if 'train_loss' in tag:
        train_loss = event_acc.Scalars(tag)

# 绘制训练损失图像
import matplotlib.pyplot as plt

steps = [global_step for global_step in train_loss]
losses = [post_mel_loss.value for post_mel_loss in train_loss]
plt.plot(steps, losses)

# 添加坐标轴说明
plt.xlabel('Steps')
plt.ylabel('Loss')

plt.show()

