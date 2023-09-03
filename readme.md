# CompressAI

[link](https://github.com/InterDigitalInc/CompressAI)

该项目提供了一个基于pytorch的数据压缩库

## 使用过程

- 参考官方文档安装配置（pip会安装所有相关依赖，包括pytorch）

- 下载数据集，将图片分至train和test文件夹。

- 使用mbt2018训练：`python3 compressai/examples/train.py -m mbt2018 -d ~/compressAI/Kodak --batch-size 16 -lr 1e-4 --save --cuda`

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
        2.298441982269287
        ],
        "decoding_time": [
        4.089500999450683
        ]
    }
    }
  ```

- 参数解释：
    1. [bpp](https://www.quora.com/What-is-the-meaning-of-bpp-0-025-of-an-image)(bit per pixel)-压缩后单个像素的体积
    2. [psnr](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio)(peak signal to noise ratio)-以解压缩的图像与原图的差值作为噪声，再假设每个像素均为最大值作为信号，计算信噪比
    3. [ms-ssim]结构相似度，考虑了人的感知来衡量失真，通常对明度轴进行计算。