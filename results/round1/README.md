# 第一轮实验（2026-11-05）

CPU 环境执行了双线性博弈和八高斯 WGAN-GP。CIFAR-10 主实验待完成 CUDA pilot。运行依赖与命令见[仓库首页](../../README.md)。

## 双线性博弈

在 \(\min_x\max_y xy\) 上，固定步长 0.3 的理论参考计算得到同步 GDA 的谱半径 1.0440，普通 EG 的谱半径 0.9582。[理论曲线与数据](bilinear_reference/analysis.json)

四种实际优化器从 \((1,1)\) 出发，以 float64 执行六档步长，每次 2,000 次联合梯度计算。Adam、Optimistic Adam 和 Lookahead 各完成 2,000 个内步，ExtraAdam 完成 1,000 次预测校正。步长 0.1 时，各法到原点的终点距离如下。

| Adam | ExtraAdam | Optimistic Adam | Lookahead-Minmax |
| ---: | ---: | ---: | ---: |
| 1.6959 | 0.4043 | 0.9684 | 0.2323 |

初始距离是 \(\sqrt{2}\)。在这个初值、步长和预算下，后三种方法的终点更接近原点。六档结果与完整轨迹复现命令见 [endpoints.csv](bilinear_adaptive/endpoints.csv) 和 [距离曲线](bilinear_adaptive/distance_curves.png)。这组测量描述有限步行为。

## 八高斯 WGAN-GP

四法共享模型、初始权重、seed 9001、batch size 64、梯度惩罚系数 10 和同步 1:1 更新。每法每个学习率（0.0001、0.0003、0.001）运行 3,000 次联合梯度计算，共 12 次。四组随后在 0.001 下从相同起点各重跑 10,000 次，检查延长预算的影响。

每个检查点用固定噪声产生 2,048 个样本。样本到最近高斯中心小于 0.15 时计为合格。每团合格样本达到全部生成样本的 1% 才计为覆盖。用同一规则评估真实分布采样，覆盖 8/8，合格率为 99.02%。

| 方法 | 10,000 次内检查点最高覆盖 | 10,000 次终点覆盖 | 终点合格率 |
| --- | ---: | ---: | ---: |
| Adam | 0/8 | 0/8 | 0.00% |
| ExtraAdam | 1/8 | 0/8 | 0.05% |
| Optimistic Adam | 1/8 | 0/8 | 0.00% |
| Lookahead-Minmax | 1/8 | 0/8 | 0.00% |

16 次训练都完成预算。12 次短跑的终点覆盖均为 0/8。延长运行中，后三法曾短暂达到 1/8，终点均未达到稳定覆盖。本轮只使用一个探索种子，结果反映这组共享配置下的学习情况。下一步先建立可稳定覆盖八团的共同基线，再做成对多种子比较。

- [16 次运行汇总](synthetic_summary.csv)与[逐次配置及日志](synthetic_runs/)
- [检查点覆盖数据](synthetic_checkpoints.csv)、[训练进展图](synthetic_progress.png)和[终点散点图](synthetic_figures/generated_samples.png)
- [真实数据对照](synthetic_target_control.json)和[测试记录](tests.txt)
