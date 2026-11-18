# CIFAR-10 的 128×128 GAN

配置使用 `image_size=128`、`batch_size=8`、`base_channels=64`，生成器和判别器总计 1,451,748 个参数。训练真实图与评估参考图均由 32×32 CIFAR-10 双线性插值至 128×128，生成器直接输出 128×128。

与 64×64、batch 32 相比，每批图像像素数相同。模型增加一层上采样和一层下采样，显存峰值与梯度计算耗时以 CUDA 实测为准。同样 20,000 次梯度调用时，batch 8 每轮计入的训练样本数是 batch 32 的四分之一，制定训练预算时要一起记录。

## 在 AutoDL 跑短测

在服务器执行以下命令。它会复用原实验的 CIFAR-10 数据和 CUDA Python 环境。首次运行会建立独立工作目录，后续运行在其中创建新的短测结果目录。

```bash
(
set -eu
GAN_OLD=/root/autodl-tmp/ESTR3114/GAN-experiment
GAN_NEW=/root/autodl-tmp/ESTR3114/GAN-experiment-cifar128
GAN_PY="$GAN_OLD/.venv/bin/python"
if [ ! -d "$GAN_NEW" ]; then
  git -C "$GAN_OLD" fetch origin cifar10-128px
  git -C "$GAN_OLD" worktree add --detach "$GAN_NEW" FETCH_HEAD
  ln -s "$GAN_OLD/data" "$GAN_NEW/data"
fi
cd "$GAN_NEW"
test -f configs/cifar10_128/pilot.json
test -f data/cifar-10-batches-py/data_batch_1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_HOME=/root/autodl-tmp/ESTR3114/torch-cache
"$GAN_PY" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))'
mkdir -p runs
GAN_RUN="$(mktemp -d "$PWD/runs/cifar10_128_pilot_XXXXXX")"
nohup "$GAN_PY" -u -m scripts.run_suite --config configs/cifar10_128/pilot.json \
  --output "$GAN_RUN" > "$GAN_RUN/console.log" 2>&1 </dev/null &
echo "PID=$!  结果目录=$GAN_RUN"
tail -f "$GAN_RUN/console.log"
)
```

每种优化器先跑 200 次联合梯度调用。各运行目录的 `metrics.csv` 记录训练耗时和 PyTorch 显存峰值。`samples.npy` 保存 64 张固定噪声的生成图，生成时按 batch 8 分组。Ctrl+C 退出日志查看。

## 调参与正式训练

沿用 pilot、tuning、freeze、main 流程，分别使用 `configs/cifar10_128/` 的对应配置。为每个 pilot 和 tuning 点生成 2,000 张验证参考图和同样数量的生成图。先根据短测结果确定可用预算。正式配置的起始预算为 20,000 次联合梯度调用，间隔 5,000 保存检查点，生成新 freeze 时传入 `--budget 20000 --checkpoint-every 5000`。

这一设置下四种方法、三个种子共 48 个验证评估点和 12 个终点测试评估。保留全部 128×128 PNG 预计需要约 36 GiB 的保守磁盘预算，另留检查点和现有结果的空间。运行 `experiments/evaluate_cifar10_main.py` 时传入本轮 `--root`，以训练时的尺寸、固定划分和相同样本数计算 FID/KID。比较 64×64 与 128×128 的指标时，需要统一评估尺寸和参考图预处理。
