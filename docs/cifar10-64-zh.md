# CIFAR-10 的 64×64 GAN

训练和评估继续使用 CIFAR-10。真实图用 PIL 双线性插值从 32×32 转为 64×64。新版生成器会直接输出 64×64，训练真实图和评估参考图采用相同的插值。图像原始信息来自 32×32 CIFAR-10。

配置默认 `base_channels=64`、`batch_size=32`。生成器由 4×4 特征逐级上采样并卷积，判别器逐级下采样。生成器和判别器合计 1,402,820 个参数，旧 32×32 基线合计 466,436 个。四种优化器继续使用同一个 WGAN-GP 联合梯度训练循环。

## 在 AutoDL 跑短测

以下命令用独立工作目录、原有 CUDA Python 环境和已有数据。假设旧仓库位于 `/root/autodl-tmp/ESTR3114/GAN-experiment`，且 Git 的 `origin` 指向项目仓库。将分支检出后，可以重复使用第二段命令，每次短测会新建结果目录。

```bash
GAN_OLD=/root/autodl-tmp/ESTR3114/GAN-experiment
GAN_NEW=/root/autodl-tmp/ESTR3114/GAN-experiment-cifar64
git -C "$GAN_OLD" fetch origin cifar10-64px
git -C "$GAN_OLD" worktree add --detach "$GAN_NEW" FETCH_HEAD
ln -s "$GAN_OLD/data" "$GAN_NEW/data"
```

```bash
cd /root/autodl-tmp/ESTR3114/GAN-experiment-cifar64 || exit
GAN_PY=/root/autodl-tmp/ESTR3114/GAN-experiment/.venv/bin/python
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_HOME=/root/autodl-tmp/ESTR3114/torch-cache
mkdir -p runs
GAN_RUN="$(mktemp -d "$PWD/runs/cifar10_64_pilot_XXXXXX")"
nohup "$GAN_PY" -u -m scripts.run_suite --config configs/cifar10_64/pilot.json \
  --output "$GAN_RUN" > "$GAN_RUN/console.log" 2>&1 </dev/null &
echo "PID=$!  结果目录=$GAN_RUN"
tail -f "$GAN_RUN/console.log"
```

四种方法短测各 200 次联合梯度调用。每组最后一行 `metrics.csv` 记录训练耗时与 PyTorch 显存峰值，`samples.npy` 保存 64 张固定噪声的生成图。短测用于检查运行和估算资源。

## 比较模型容量

在同一个 64×64 版本上，用 Adam 对照 `base_channels=32` 与 `base_channels=64`。两组使用相同的 batch、seed、学习率、训练划分与调用预算。宽度较小的 64×64 模型合计 419,812 参数。先用短测估算可承担的预算，再将 `oracle_budget` 设为两组相同的值。以下例子使用 20,000 次调用，每 2,000 次保存检查点，终点评估使用各 2,000 张验证参考图和生成图。

```bash
mkdir -p runs
GAN_PROBE="$(mktemp -d "$PWD/runs/cifar10_64_width_XXXXXX")"
"$GAN_PY" - "$GAN_PROBE" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
base = json.loads(Path("configs/cifar10_64/pilot.json").read_text())
for width in (32, 64):
    config = dict(base, base_channels=width, oracle_budget=20000,
                  log_every=2000, checkpoint_every=2000)
    (root / f"width{width}.json").write_text(json.dumps(config, indent=2))
PY
for GAN_WIDTH in 32 64; do
  "$GAN_PY" -u -m ganopt.train --config "$GAN_PROBE/width$GAN_WIDTH.json" \
    --output "$GAN_PROBE/width$GAN_WIDTH"
done
"$GAN_PY" -m scripts.plot_runs --runs "$GAN_PROBE" --output "$GAN_PROBE/figures"
```

把两组固定噪声生成图和同一批 64×64 验证真实图并排观察。在同尺寸、同参考样本数下用 `scripts.export_images` 和 `scripts.evaluate_images` 评分。历史 32×32 指标属于原评估设置。直接比较历史和新版的指标时，应固定同一种参考预处理与生成图尺寸。

## 四优化器比较

若采用宽版，`configs/cifar10_64/tuning.json` 和 `configs/cifar10_64/main.json` 与短测的分辨率、通道数和 batch 一致。复用仓库的 pilot → tuning → freeze → main 流程。为 pilot 和 tuning 候选生成 2,000 张验证参考图和同数生成图，评估结果保存为各目录的 `evaluation/metrics.json`。使用本轮实测的资源和指标生成新 freeze，main 的预算须与 freeze 一致，并用 `--root` 指向新轮目录来调度评估。先测一次完整 64×64 图片评估的显存和吞吐量，再设置并行数。每个验证点留存 10,000 张 PNG，需把数据盘空间计入预算。

历史运行可使用提交 `8dcaf6b87a1ec28719d0eb0189d0e7d06918fa33` 在独立工作目录复现。检查点与冻结协议会验证训练版本的源码哈希。
