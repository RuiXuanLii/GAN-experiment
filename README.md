# ESTR3114 · GAN Optimization

比较四种优化算法在同一 WGAN-GP 下的训练稳定性、生成质量和计算成本。CIFAR-10 图像生成是主实验。双线性博弈和八高斯生成用于检查更新行为和模式覆盖。

| 方法 | 一次完整更新消耗的联合梯度计算 |
| --- | ---: |
| Adam | 1 |
| ExtraAdam | 2 |
| Optimistic Adam | 1 |
| Lookahead-Minmax（内层 Adam） | 每个内步 1 |

一次联合梯度计算在同一组生成器、critic 权重上求两边的梯度，包含 critic 的梯度惩罚。四法按实际调用次数计费。

## 安装

创建独立 Python 环境，再按设备安装一对匹配的 PyTorch 和 torchvision。首轮实测使用 Python 3.12 和 PyTorch 2.5.1 CPU。

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pytest -q
```

使用 CUDA 时，先在对应设备的 Python 环境确认 `torch.cuda.is_available()` 返回 `True`，再执行测试。精确的 CPU 版本记录在 `requirements-tested-cpu.txt` 和各次运行的 `metadata.json`。

## 复现第一轮

从仓库根目录运行以下命令。`runs/` 里的 checkpoint 可用于同配置恢复训练，新的运行需要新的输出目录。

```bash
python -m scripts.bilinear_analysis --output runs/bilinear_reference
python -m experiments.bilinear_optimizers --output runs/bilinear_adaptive
python -m scripts.run_suite --config configs/synthetic_pilot_lr0.0001.json --output runs/round1/lr1e-4
python -m scripts.run_suite --config configs/synthetic_pilot_lr0.0003.json --output runs/round1/lr3e-4
python -m scripts.run_suite --config configs/synthetic_pilot_lr0.001.json --output runs/round1/lr1e-3
python -m scripts.run_suite --config configs/synthetic_pilot_extended.json --output runs/round1/extended
python -m experiments.collect_synthetic_round --runs runs/round1 --output runs/round1-summary
python -m scripts.plot_runs --runs runs/round1/extended --output runs/gaussian_extended_figures
```

第一轮结果、配置和运行记录见 [results/round1](results/round1/README.md)。

## CIFAR-10 pilot

在可用的 CUDA 环境中准备 CIFAR-10，并在 `configs/cifar10_pilot.json` 设置 `data_root`。需要由 torchvision 下载数据时，将 `download` 设为 `true`。

```bash
python -m scripts.run_suite --config configs/cifar10_pilot.json --output runs/cifar10_pilot
python -m pip install -r requirements-image-metrics.txt
python -m scripts.export_images --checkpoint RUN_DIR/checkpoint.pt --output RUN_DIR/evaluation --data-root data --split validation --count 2000 --device cuda
python -m scripts.evaluate_images --folder RUN_DIR/evaluation --cuda
```

用实际输出目录替换 `RUN_DIR`。四法都要测耗时与峰值显存，并跑通预训练 Inception 特征的验证 FID/KID，再确定正式预算。`configs/cifar10_main.json` 需要用 `scripts.freeze_protocol` 从实际 pilot 和调参结果生成冻结配置。

算法公式和双线性推导见 [数学笔记](docs/mathematical-notes-zh.md)。论文和第三方测试代码的出处见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
