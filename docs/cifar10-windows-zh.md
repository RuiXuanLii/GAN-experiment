# 在 Windows 准备 CIFAR-10

这些命令需要在你的 Windows 电脑上执行。当前 Linux 工作环境无法访问你的 `D:` 盘，因此这里提供下载和校验脚本；并不表示数据已经下载到该盘。

在 Windows PowerShell 中进入本仓库根目录，运行：

```powershell
py -3 scripts/prepare_cifar10.py
```

默认下载目录就是 `D:\ESTR3108\project\CIFAR-10\dataset`。也可以明确指定同一路径：

```powershell
py -3 scripts/prepare_cifar10.py --root "D:\ESTR3108\project\CIFAR-10\dataset"
```

脚本只使用 Python 标准库，准备数据时不需要安装 PyTorch。请预留至少 1 GiB 可用空间，用于约 163 MiB 的压缩包、约 178 MiB 的解压数据及临时文件。完整压缩包会保留在数据目录；中断后重新运行同一命令即可在服务器支持 HTTP Range 时续传，服务器不支持时会重新下载。

完成后，目录中应包含 `cifar-10-python.tar.gz` 和 `cifar-10-batches-py` 文件夹。文件夹内的必要文件为 `data_batch_1` 至 `data_batch_5`、`test_batch`、`batches.meta`。脚本先检查压缩包 MD5，再检查这七个文件的官方 MD5；与官方字节内容一致即可确认其对应的 50,000 张训练图像和 10,000 张测试图像，无需反序列化 NumPy 数据。

只检查已有数据、不访问网络：

```powershell
py -3 scripts/prepare_cifar10.py --verify-only
```

如果校验失败，按错误提示处理后重新运行准备命令。`--verify-only` 不会补下载或解压文件。Linux、macOS 等系统必须显式传入本机的 `--root` 路径；默认 Windows 路径会在创建目录之前被拒绝，避免生成名为 `D:` 的相对目录。

## 连接本项目训练

已提供 [Windows pilot 配置](../configs/windows/cifar10_pilot.json)，其中：

```json
{
  "data_root": "D:\\ESTR3108\\project\\CIFAR-10\\dataset",
  "download": false
}
```

`data_root` 指向 `cifar-10-batches-py` 的父目录，不要再追加该文件夹名称。`download: false` 使训练直接使用已经准备并校验的数据。

按仓库安装说明配置匹配的 PyTorch、torchvision 和 CUDA 环境后，可在仓库根目录运行：

```powershell
py -3 -m scripts.run_suite --config configs/windows/cifar10_pilot.json --output runs/cifar10_pilot_windows
```

该配置使用 `device: cuda`，需要所用 Python 环境中的 `torch.cuda.is_available()` 为 `True`。如果项目依赖安装在已激活的虚拟环境中，可将命令的 `py -3` 换成该环境的 `python`。

项目从官方 50,000 张训练图像中，以 `split_seed: 2026` 固定划分 45,000 张训练图像和 5,000 张验证图像。官方测试批次虽然随压缩包一并准备，但应留给协议冻结后的正式实验最终评估，不用于 pilot 或调参。

## 官方来源

- [多伦多大学 CIFAR-10 页面](https://www.cs.toronto.edu/~kriz/cifar.html)：数据规模、Python 压缩包和 MD5。
- [torchvision 0.20 CIFAR-10 实现](https://docs.pytorch.org/vision/0.20/_modules/torchvision/datasets/cifar.html)：压缩包 URL、解压目录名及各批次、元数据文件的 MD5。
