"""Produce a compact report from completed, verified round-two metrics."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def report(root):
    rows=list(csv.DictReader((root/'summary.csv').open()))
    traces=list(csv.DictReader((root/'checkpoints.csv').open()))
    by_case={r['case']:r for r in rows}
    final=[by_case[f'adam_alternating_d5_h128_s{s}_120k'] for s in [9001,9002,9003]]
    assert all(int(r['oracle_calls'])==120000 for r in final)
    tails={r['seed']:sorted([x for x in traces if x['case']==r['case']],key=lambda x:int(x['oracle_calls']))[-5:] for r in final}
    tail_stats={seed:{'calls':[int(x['oracle_calls']) for x in tail],
        'minimum_covered_modes':min(int(x['covered_modes']) for x in tail),
        'minimum_quality_fraction':min(float(x['quality_fraction']) for x in tail),
        'maximum_quality_fraction':max(float(x['quality_fraction']) for x in tail)} for seed,tail in tails.items()}
    stats={'seeds':[int(r['seed']) for r in final],'total_calls_per_trajectory':120000,
        'g_updates':20000,'d_updates':100000,
        'mean_quality_fraction':float(np.mean([float(r['quality_fraction']) for r in final])),
        'std_quality_fraction':float(np.std([float(r['quality_fraction']) for r in final],ddof=1)),
        'mean_nearest_distance':float(np.mean([float(r['mean_nearest_distance']) for r in final])),
        'all_final_modes_covered':all(int(r['covered_modes'])==8 for r in final),
        'all_final_quality_at_least_50pct':all(float(r['quality_fraction'])>=.5 for r in final),
        'tail_per_seed':tail_stats,
        'all_last_five_checkpoints_meet_target':all(t['minimum_covered_modes']==8 and t['minimum_quality_fraction']>=.5 for t in tail_stats.values()),
        'unique_training_field_calls_this_round':570000,'frozen_g_probe_field_calls':2000,
        'independent_initialized_training_runs':9,'continuation_folders':3}
    (root/'final_validation.json').write_text(json.dumps(stats,indent=2)+'\n')
    fig,axes=plt.subplots(2,3,figsize=(11,7),layout='constrained')
    angles=np.arange(8)*2*np.pi/8;centers=2*np.stack([np.cos(angles),np.sin(angles)],axis=1)
    rng=np.random.default_rng(123456);real=centers[rng.integers(8,size=2048)]+rng.normal(0,.05,(2048,2))
    panels=[('Real target',real),('Alternating 1:1 / 60k',np.load(root/'runs/adam_alternating_d1_h128/samples.npy'))]
    for row in final:
        panels.append((f"D:G = 5:1 / seed {row['seed']}\n8 modes; quality {float(row['quality_fraction'])*100:.1f}%",np.load(root/'runs'/row['case']/'samples.npy')))
    for ax,(label,samples) in zip(axes.flat,panels):
        ax.scatter(real[:,0],real[:,1],s=2,c='#b1b1b1',alpha=.22)
        ax.scatter(samples[:,0],samples[:,1],s=2,c='#2776b5',alpha=.4)
        ax.scatter(centers[:,0],centers[:,1],c='#cc3333',marker='x',s=20)
        together=np.vstack([samples,real]);low=together.min(0)-.3;high=together.max(0)+.3
        ax.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),title=label,aspect='equal')
        ax.tick_params(labelsize=8)
    ax=axes.flat[5]
    for row in final:
        # Stitch each parent trajectory and its continuation; avoid double counting 60k.
        seed=row['seed'];parent='adam_alternating_d5_h128'+('' if seed=='9001' else f'_s{seed}')
        series=[x for x in traces if x['case']==parent or (x['case']==row['case'] and int(x['oracle_calls'])>60000)]
        series.sort(key=lambda x:int(x['oracle_calls']))
        ax.plot([int(x['oracle_calls'])/1000 for x in series],[100*float(x['quality_fraction']) for x in series],label=f'seed {seed}')
    ax.axhline(50,color='gray',ls='--',lw=1,label='50% target')
    ax.set(xlabel='Full joint field calls (thousands)',ylabel='Samples within radius 0.15 (%)',ylim=(0,100),title='Quality over training')
    ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(root/'final_comparison.png',dpi=145);plt.close(fig)
    table='\n'.join(f"| {r['seed']} | {int(r['covered_modes'])}/8 | {100*float(r['quality_fraction']):.2f}% | {float(r['mean_nearest_distance']):.4f} | {tail_stats[r['seed']]['minimum_covered_modes']}/8 | {100*tail_stats[r['seed']]['minimum_quality_fraction']:.2f}% |" for r in final)
    ablations='\n'.join(f"| {name} | {int(by_case[name]['oracle_calls']):,} | {int(by_case[name]['g_updates']):,} / {int(by_case[name]['d_updates']):,} | {int(by_case[name]['covered_modes'])}/8 | {100*float(by_case[name]['quality_fraction']):.2f}% |" for name in ['adam_duration','adam_gp01','adam_relu_gp10','adam_relu_gp01','adam_relu_h128_gp01','adam_alternating_d1_h128','adam_alternating_d5_h128'])
    text=f'''# 第二轮八高斯实验（2026-11-11—2026-11-13）

本轮已经找到能学到八个目标小团的 Adam 训练配置。单独增加训练次数、降低梯度惩罚、更换激活函数、增加网络宽度或使用交替1:1，均没有解决终点覆盖失败；在宽度128、ReLU、lambda0.1的相同条件下，将判别器与生成器更新比改成5:1后，三个随机种子都能覆盖八团。它说明当前配置的训练配合很重要，不能把此前失败直接归因于四种优化算法本身。

## 三种子最终结果

每条轨迹总计120,000次完整联合场计算，G更新20,000次、D更新100,000次。三个120k目录分别延续已有60k父运行，是三条轨迹的延长，不是新增三个独立初始化。

| seed | 终点覆盖 | 终点合格率 | 最近目标中心平均距离 | 最后5个检查点最低覆盖 | 最后5个检查点最低合格率 |
|---|---:|---:|---:|---:|---:|
{table}

终点合格率均值为 **{stats['mean_quality_fraction']*100:.2f}%**，种子间样本标准差为 **{stats['std_quality_fraction']*100:.2f}个百分点**。真实目标采样的合格率约99%，平均距离约0.062。覆盖8团并不等于完全学到目标分布，小团间仍可能残留连接点；该指标也不能排除团间比例或团内形状误差。

![终点与学习曲线](final_comparison.png)

## 诊断过程

所有训练使用 Adam，下面不是四种优化器排名。每一次完整联合场计算仍然求两边梯度；交替训练只应用当前玩家的梯度，未应用的计算也计入预算。

| 配置目录 | 完整场计算 | G / D 实际更新数 | 终点覆盖 | 终点合格率 |
|---|---:|---:|---:|---:|
{ablations}

原Softplus/宽度32是`adam_duration`；`adam_gp01`只降低lambda；`adam_relu_gp10`只改D激活函数；`adam_relu_gp01`组合这两项；`adam_relu_h128_gp01`在后者基础上增加G、D宽度。交替1:1和5:1均固定宽度128、ReLU、lambda0.1、lr0.0001、batch64。

冻结失败检查点的G，只让D再训练2,000次后，独立评估流上的`D(real)-D(fake)`从-0.2036变为31.5041，G权重哈希未改变。这证明D能够重新排序这组真实／生成样本，支持检查critic追踪的诊断方向；分数幅度受惩罚强度影响，不作为生成质量或真实Wasserstein距离。

## 结论的范围

- 此轮建立的是 **Adam交替训练参考基线**。共同的四优化器同步训练器保持原有定义，四法的新一轮公平比较尚未完成。
- 同步30k和交替1:1的60k具有相同玩家更新数，但后者的两边使用相继的独立新批次，不能单独把差异归因于更新顺序。交替1:1与5:1的60k具有相同完整场预算，但更新次数分配不同。
- 60k三种子终点合格率为46.53%、49.27%、47.12%，尚未达到预定50%目标。由于曲线持续改善，三者均固定参数延长到120k；延长决策用过这三条曲线，因此120k结果属于探索性重复，不是未见种子的最终确认。
- 这是二维合成数据实验。CIFAR-10图像训练及FID/KID尚未执行；二维改善不能直接当作图像质量结论。CPU并行时的耗时不用于优化器速度排名。

## 文件与复现

- [实验协议及事前决策](PROTOCOL.md)、[汇总CSV](summary.csv)、[检查点CSV](checkpoints.csv)、[末段稳定性核验](final_validation.json)。
- [逐次配置、训练日志、生成点和延续关系](runs/)、[判别器冻结诊断](critic_probes/)、[测试记录](tests.txt)。
- [全部终点散点图](samples.png)、[全部检查点曲线](progress.png)、[文件哈希](artifact_hashes.json)、[源代码版本](source_versions.json)。

从仓库根目录执行：

```bash
python -m pytest -q
# 五个同步对照，逐一运行（各30k完整场计算）。
python -m ganopt.train --config configs/round2/adam_duration.json --output runs/round2/adam_duration
python -m ganopt.train --config configs/round2/adam_gp01.json --output runs/round2/adam_gp01
python -m ganopt.train --config configs/round2/adam_relu_gp10.json --output runs/round2/adam_relu_gp10
python -m ganopt.train --config configs/round2/adam_relu_gp01.json --output runs/round2/adam_relu_gp01
python -m ganopt.train --config configs/round2/adam_relu_h128_gp01.json --output runs/round2/adam_relu_h128_gp01
python -m experiments.critic_tracking_probe --checkpoint runs/round2/adam_relu_gp01/checkpoint.pt --output results/round2/critic_probes/adam_relu_gp01
python -m experiments.alternating_reference --config configs/round2/adam_alternating_d1_h128.json --output runs/round2/adam_alternating_d1_h128
python -m experiments.alternating_reference --config configs/round2/adam_alternating_d5_h128.json --output runs/round2/adam_alternating_d5_h128
python -m experiments.alternating_reference --config configs/round2/adam_alternating_d5_h128_s9002.json --output runs/round2/adam_alternating_d5_h128_s9002
python -m experiments.alternating_reference --config configs/round2/adam_alternating_d5_h128_s9003.json --output runs/round2/adam_alternating_d5_h128_s9003
python -m experiments.extend_alternating --parent runs/round2/adam_alternating_d5_h128 --output runs/round2/adam_alternating_d5_h128_s9001_120k --budget 120000
python -m experiments.extend_alternating --parent runs/round2/adam_alternating_d5_h128_s9002 --output runs/round2/adam_alternating_d5_h128_s9002_120k --budget 120000
python -m experiments.extend_alternating --parent runs/round2/adam_alternating_d5_h128_s9003 --output runs/round2/adam_alternating_d5_h128_s9003_120k --budget 120000
python -m experiments.collect_round2 --runs runs/round2 --output results/round2
python -m experiments.report_round2 --results results/round2
```

重新训练时使用空输出目录。checkpoint保存在本地`runs/`，仓库保留配置、CSV和小型生成点文件。首批四个30k对照的10k临时输出未永久保留，原始轨迹使用实际保留的每3k检查点。初始交替运行使用的驱动已保存为`evidence/round2/alternating_reference_at_run.py`；后续版本只修复断点恢复时CSV可能超前的问题，不改变正常连续运行的数值更新。
'''
    (root/'README.md').write_text(text)
    print(json.dumps(stats,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--results',type=Path,required=True)
    report(parser.parse_args().results)
