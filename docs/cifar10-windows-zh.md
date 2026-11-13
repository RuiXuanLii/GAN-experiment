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

已提供三个 Windows 配置；除数据目录外，其参数与对应的通用配置相同：

| 阶段 | 配置 | 用途 |
| --- | --- | --- |
| pilot | [cifar10_pilot.json](../configs/windows/cifar10_pilot.json) | 四法各运行 200 次联合梯度调用，测量 CUDA 耗时和显存，并验证图像评估流程 |
| tuning | [cifar10_tuning.json](../configs/windows/cifar10_tuning.json) | 四法各尝试三种学习率，每次 2,000 次联合梯度调用 |
| main | [cifar10_main.json](../configs/windows/cifar10_main.json) | 待与冻结结果对齐的正式实验模板；需要有效的 `freeze.json` |

三个配置均设置：

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

三个配置均使用 `device: cuda`，需要所用 Python 环境中的 `torch.cuda.is_available()` 为 `True`。如果项目依赖安装在已激活的虚拟环境中，可将命令的 `py -3` 换成该环境的 `python`。

随后使用三种学习率进行调参：

```powershell
py -3 -m scripts.run_tuning --config configs/windows/cifar10_tuning.json --output runs/cifar10_tuning_windows
```

训练完成后，还需要按仓库 README 的 `scripts.export_images`、`scripts.evaluate_images` 流程，对每个 pilot 和成功完成的调参运行，在最终 checkpoint 生成 2,000 张图像的验证集 FID/KID 记录。导出时将 `--data-root` 设置为 `"D:\ESTR3108\project\CIFAR-10\dataset"`，并使用各运行目录中的真实 checkpoint。

正式实验必须先有四法在实际 GPU 上完成的 pilot、测得的训练耗时与峰值显存、完整调参尝试，以及有效的验证指标记录。`scripts.freeze_protocol` 根据这些证据和实际可用 GPU 小时检查统一预算，生成 `freeze.json`；脚本给出的运行时间是基于实测 pilot 的估算，预算与指标应以实际运行记录为依据。

在 PowerShell 中，先将 `$AvailableGpuHours` 设为实际剩余可用 GPU 小时，将 `$MainOracleBudget` 设为根据实测结果选择的四法统一联合梯度调用预算（10 的正整数倍），再运行：

```powershell
py -3 -m scripts.freeze_protocol --pilot runs/cifar10_pilot_windows --tuning runs/cifar10_tuning_windows --output freeze.json --gpu-hours $AvailableGpuHours --budget $MainOracleBudget
```

`configs/windows/cifar10_main.json` 保留 `"freeze_file": "freeze.json"`。这是待与冻结结果对齐的模板，预置的 `lr: 0.0001` 和 `oracle_budget: 20000` 不是调参或资源测量的结论。正式实验中各方法的学习率与统一预算都必须符合实际 `freeze.json`：下方的 `scripts.run_suite` 命令会按冻结结果覆盖各方法学习率，但仍需先将模板中的 `oracle_budget` 改为冻结预算，并确保日志和 checkpoint 间隔与所选预算及方法周期兼容。冻结后保持训练与评估源码不变，在仓库根目录运行正式实验：

```powershell
py -3 -m scripts.run_suite --config configs/windows/cifar10_main.json --output runs/cifar10_main_windows
```

正式实验会采用冻结文件中每种方法选出的学习率和报告种子 `11、22、33`；缺失或不匹配的冻结证据会阻止启动。

项目从官方 50,000 张训练图像中，以 `split_seed: 2026` 固定划分 45,000 张训练图像和 5,000 张验证图像。官方测试批次虽然随压缩包一并准备，但应留给协议冻结后的正式实验最终评估，不用于 pilot 或调参。

## 官方来源

- [多伦多大学 CIFAR-10 页面](https://www.cs.toronto.edu/~kriz/cifar.html)：数据规模、Python 压缩包和 MD5。
- [torchvision 0.20 CIFAR-10 实现](https://docs.pytorch.org/vision/0.20/_modules/torchvision/datasets/cifar.html)：压缩包 URL、解压目录名及各批次、元数据文件的 MD5。
