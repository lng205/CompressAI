# CompressAI

[CompressAI](https://github.com/InterDigitalInc/CompressAI)提供了一个基于pytorch的数据压缩库

## 训练

### 官方模型

1. 参考[官方文档](https://interdigitalinc.github.io/CompressAI/index.html)安装配置

2. 下载数据集(e.g. Kodak)，将图片分至train和test文件夹

3. 选择官方模型(e.g. mbt2018)进行训练：`python3 compressai/examples/train.py -m mbt2018 -d ~/compressAI/Kodak --batch-size 16 -lr 1e-4 --save --cuda`

4. 更新模型：`python3 -m compressai.utils.update_model --architecture mbt2018 checkpoint_best_loss.pth.tar`

5. 测试：`python3 -m compressai.utils.eval_model checkpoint ~/compressAI/Kodak/test -a mbt2018 -p checkpoint_best_loss*.pth.tar`
  
6. 结果：
```bash
Using trained model checkpoint_best_loss-7c8032a4-ans
{
"name": "mbt2018-mse",
"description": "Inference (ans)",
"results": {
    "psnr": [
    18.01904296875
    ],
    "ms-ssim": [
    0.5968405246734619
    ],
    "bpp": [
    0.3708170572916667
    ],
    "encoding_time": [
    2.298441982269287
    ],
    "decoding_time": [
    4.089500999450683
    ]
}
```

- 参数解释：
    1. [bpp](https://www.quora.com/What-is-the-meaning-of-bpp-0-025-of-an-image)(bit per pixel)-压缩后单个像素的体积

    2. [psnr](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio)(peak signal to noise ratio)-以解压缩的图像与原图的差值作为噪声，再假设每个像素均为最大值作为信号，计算信噪比

    3. [ms-ssim]结构相似度，考虑了人的感知来衡量失真，通常对明度轴进行计算

### 自定义模型

1. 网络结构：Pytorch.nn.Module类可用于描述网络结构，compressai在此基础上提供了一个便于优化的模型类CompressionModel:
   
```python
from compressai.models import CompressionModel
from compressai.models.utils import conv, deconv

class Network(CompressionModel):
    def __init__(self, N=128):
        super().__init__()
        self.encode = nn.Sequential(
            conv(3, N),
            GDN(N)
            conv(N, N),
            GDN(N)
            conv(N, N),
        )

        self.decode = nn.Sequential(
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

def forward(self, x):
    y = self.encode(x)
    y_hat, y_likelihoods = self.entropy_bottleneck(y)
    x_hat = self.decode(y_hat)
    return x_hat, y_likelihoods
```


访问模型参数：

```python
import torch.optim as optim

parameters = set(p for n, p in net.named_parameters() if not n.endswith(".quantiles"))
aux_parameters = set(p for n, p in net.named_parameters() if n.endswith(".quantiles"))
optimizer = optim.Adam(parameters, lr=1e-4)
aux_optimizer = optim.Adam(aux_parameters, lr=1e-3)
```

1. 

- 损失函数：我们至少需要从压缩率和失真度两个方面衡量一个有损压缩模型的性能（率失真理论），我们可用bpp和mse加权求和作为损失函数：

```python
import math
import torch.nn as nn
import torch.nn.functional as F

x = torch.rand(1, 3, 64, 64)
net = Network()
x_hat, y_likelihoods = net(x)

# bitrate of the quantized latent
N, _, H, W = x.size()
num_pixels = N * H * W
bpp_loss = torch.log(y_likelihoods).sum() / (-math.log(2) * num_pixels)

# mean square error
mse_loss = F.mse_loss(x, x_hat)

# final loss term
loss = mse_loss + lmbda * bpp_loss
```


- 由于我们选择的模型结构中间有[信息瓶颈层](https://en.wikipedia.org/wiki/Information_bottleneck_method#:~:text=The%20information%20bottleneck%20can%20also,its%20direct%20prediction%20from%20X.)（？不太理解，需要进一步学习），还需要优化辅助损失函数：`aux_loss = net.entropy_bottleneck.loss()`

3. 训练
```python
x = torch.rand(1, 3, 64, 64)
for i in range(10):
    optimizer.zero_grad()
    aux_optimizer.zero_grad()

    x_hat, y_likelihoods = net(x)

    # ...
    # compute loss as before
    # ...

    loss.backward()
    optimizer.step()

    aux_loss = net.aux_loss()
    aux_loss.backward()
    aux_optimizer.step()
```
