# LOG

# 8/30

- 通过VPN连接校内网，通过跳板机登录服务器(*.60)并尝试搭建环境
  
- 由于服务器校外连接延迟较高，配置与本地差距不大，以及配置代理时没有root权限等问题，决定改用本地电脑学习

# 9/1

- 安装Ubuntu LTS 22.0403

# 9/2

- 配置代理，安装VSCode，安装git，安装[pip](https://pip.pypa.io/en/stable/installation/)

- 安装[compressAI](https://interdigitalinc.github.io/CompressAI/installation.html)

- 下载数据集（Kodak），将图片分至train和test(20~24)文件夹。

- （H栋校园网连红花岭比A栋流畅许多）

- 使用mbt2018训练：`python3 examples/train.py -m mbt2018 -d ~/compressAI/Kodak --batch-size 16 -lr 1e-4 --save --cuda`

    - 默认的epoch数为100。batch-size指一个训练批次中的样本数量，一个批次更新一次模型参数，抽样多个批次直至遍历训练集后完成一个epoch。

- 更新模型：`python3 -m compressai.utils.update_model --architecture mbt2018 checkpoint_best_loss.pth.tar`

- 测试：`python3 -m compressai.utils.eval_model checkpoint ~/compressAI/Kodak/test -a mbt2018 -p checkpoint_best_loss*.pth.tar`
  
    结果：
    ```
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
        2.313329792022705
        ],
        "decoding_time": [
        4.11532244682312
        ]
    }
    }
    ```

- 将batch-size改为19（数据集大小）后，模型的性能出现了明显的下降

    ```
    Using trained model checkpoint_best_loss-6f928e71-ans
    {
    "name": "mbt2018-mse",
    "description": "Inference (ans)",
    "results": {
        "psnr": [
        15.485929489135742
        ],
        "ms-ssim": [
        0.5198103129863739
        ],
        "bpp": [
        1.2278483072916666
        ],
        "encoding_time": [
        2.2816078662872314
        ],
        "decoding_time": [
        4.085854768753052
        ]
    }
    }
    ```

- 参数解释（[编码概念介绍](https://www.bilibili.com/video/BV14v41137pK/?spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=591c0e4b8c78ff465989e1f643717175)）：
    1. [bpp](https://www.quora.com/What-is-the-meaning-of-bpp-0-025-of-an-image)(bit per pixel)-压缩后单个像素的体积
    2. [psnr](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio)(peak signal to noise ratio)-以解压缩的图像与原图的差值作为噪声，再假设每个像素均为最大值作为信号，计算信噪比
    3. [ms-ssim]结构相似度，考虑了人的感知来衡量失真，通常对明度轴进行计算。