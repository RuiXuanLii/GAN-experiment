# 文献核对与数学分析 v1.0

阅读顺序：WGAN-GP背景 → Gidel → Daskalakis → Chavdarova。以下算法公式为定义转述，双线性推导是本项目可直接检验的代数分析；不把图像训练经验当定理。

## 1. 三篇主要论文

| 论文 | 阅读位置与核心定义 | 使用方式与限制 |
|---|---|---|
| Gidel et al., *A Variational Inequality Perspective on Generative Adversarial Networks*, ICLR 2019，arXiv:1802.10551 | §2联合博弈场；§3 EG；§5 Algorithm 4 ExtraAdam | prediction与correction均更新矩；参数校正基点是原位置。单调VI结果需要相应假设，论文未证明ExtraAdam在深度GAN收敛 |
| Daskalakis, Ilyas, Syrgkanis & Zeng, *Training GANs with Optimism*, ICLR 2018，arXiv:1711.00141 | §6 Algorithm 1 Optimistic Adam | 乐观项作用于偏差修正自适应方向；不是先对原始梯度外推再Adam。OGDA简化理论不等于OAdam非凸GAN保证 |
| Chavdarova, Pagliardini, Stich, Fleuret & Jaggi, *Taming GANs with Lookahead-Minmax*, ICLR 2021，arXiv:2006.14567 | Algorithm 1；§4博弈分析；附录adaptive optimizers | 对G/D联合fast轨迹与slow起点插值。允许不同内优化器；本研究固定同步Adam、保留moment。内法不稳定时不能无条件保证LA稳定 |

辅助阅读：Kingma & Ba, *Adam: A Method for Stochastic Optimization*, ICLR 2015, arXiv:1412.6980；Gulrajani et al., *Improved Training of Wasserstein GANs*, NeurIPS 2017, arXiv:1704.00028；Heusel et al., *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium*, NeurIPS 2017, arXiv:1706.08500（FID来源）；Bińkowski et al., *Demystifying MMD GANs*, ICLR 2018, arXiv:1801.01401（KID来源）。辅助论文用于正式report前进一步阅读，不声称本次逐条完整复现其代码。

原文与代码入口：

- https://arxiv.org/pdf/1802.10551 与 https://github.com/GauthierGidel/Variational-Inequality-GAN
- https://arxiv.org/pdf/1711.00141
- https://arxiv.org/pdf/2006.14567 与 https://github.com/Chavdarova/LAGAN-Lookahead_Minimax
- https://arxiv.org/abs/1704.00028

作者参考ExtraAdam已下载，完整MIT许可保留在 `tests/reference/gidel_extragradient.py`；Git blob SHA `d1aabae4e22b9b5ad80051a81a4cf0e260f42e14`；文件SHA256见 `evidence/reference_sha256.txt`。读取了同仓库 `optim/omd.py` 与LA仓库 `lagan/optim/lookahead.py`进行定义审查。只将MIT许可明确的Extra文件随测试包保留；其他参考核对记录为URL与hash，未直接复制为产品实现。

**epsilon约定差异必须说明**：2018作者ExtraAdam参考实现使用
\(\eta\sqrt{1-\beta_2^n}\,m/[(1-\beta_1^n)(\sqrt v+\epsilon)]\)，相当于corrected方向分母里epsilon随时钟变化。本项目使用论文公式与现代PyTorch约定
\(\eta\hat m/(\sqrt{\hat v}+\epsilon)\)。两者epsilon=0时相同；参考数值对比使用非零梯度、epsilon=0排除该实现约定，再单独与PyTorch在epsilon=1e-8核对。不是声称所有epsilon下逐位完全相同。

## 2. 零和问题与实际WGAN-GP

零和目标：
\[
V(\theta,\phi)=\mathbb E_xD_\phi(x)-\mathbb E_zD_\phi(G_\theta(z)),\qquad
\min_\theta\max_\phi V(\theta,\phi).
\]

实际损失：
\[
L_G=-\mathbb E_zD_\phi(G_\theta(z)),\quad
L_D=\mathbb E_zD_\phi(G_\theta(z))-\mathbb E_xD_\phi(x)+\lambda P(\theta,\phi),
\]
\[
P=\mathbb E_{x,z,a}\left[(\|\nabla_{\hat x}D_\phi(\hat x)\|_2-1)^2\right],\quad
\hat x=ax+(1-a)G_\theta(z).
\]

L_D依赖G且GP只加入D，因此一般不存在使两个梯度完全一致的单一零和目标。`fake.detach()`表示求D损失关于phi的偏导时固定theta；不意味着数学上GP独立于theta。G的期望真实数据项对theta为常数，因此LG与V的theta偏导相同。GP是插值点软惩罚，不能证明critic全域1-Lipschitz。

令w=(theta,phi)，场
\[
F(w)=\begin{pmatrix}\nabla_\theta L_G\\\nabla_\phi L_D\end{pmatrix}.
\]
两个模块都执行w←w-eta F。平稳条件F(w*)=0是必要一阶条件，不足以证明Nash equilibrium（每个玩家仍需局部最优条件）。若研究约束VI，还需给出约束集合与投影，本项目网络训练不引入人为投影。

## 3. 四种自适应更新的完整状态

定义一次方向核，n表示**实际moment调用次数**：
\[
m_n=\beta_1m_{n-1}+(1-\beta_1)g_n,\quad
v_n=\beta_2v_{n-1}+(1-\beta_2)g_n^{\odot2},\quad
u_n=\frac{m_n/(1-\beta_1^n)}{\sqrt{v_n/(1-\beta_2^n)}+\epsilon}.
\]
初值m0=v0=0，平方/开根逐元素。状态不是参数的可微函数，不反向传播through optimizer。

Adam：\(w_{t+1}=w_t-\eta u_{t+1}\)。所有块梯度同一joint point计算后再更新。

ExtraAdam：设t从0开始，第t个完整步使用n=2t+1与2t+2，
\[
g^p=F(w_t;\xi_{2t+1}),\quad
(u^p,s^p)=A(g^p,s_t),\quad w^p=w_t-\eta u^p,
\]
\[
g^c=F(w^p;\xi_{2t+2}),\quad
(u^c,s_{t+1})=A(g^c,s^p),\quad w_{t+1}=w_t-\eta u^c.
\]
两块一起预测，两块一起重新求梯度；校正不是w^p-eta u^c。保留s^p的矩影响，与丢弃预测矩的另一种variant不同。独立xi主设置；相同xi敏感性实验改变的是随机场相关性，不改变参数恢复定义。

Optimistic Adam：
\[
(u_t,s_t)=A(F(w_t;\xi_t),s_{t-1}),\qquad
w_{t+1}=w_t-\eta(2u_t-u_{t-1}),\quad u_{-1}=0.
\]
这里t的方向编号可从0开始；等价实现首调用previous=0。历史保存u，不能保存g或已经乘2的更新。学习率固定；若后续变eta，必须重新定义历史项乘当前还是旧eta，本协议不使用schedule以避免歧义。

LA-Minmax(Adam)：对周期r，slow=s_r，fast_0=s_r，k次同步Adam后fast_k；
\[
s_{r+1}=(1-\alpha)s_r+\alpha\,fast_k,\qquad fast\leftarrow s_{r+1}.
\]
G/D使用相同r、k、alpha；Adam矩沿fast序列保留。slow仅参数，不是额外moment平均。周期内checkpoint需包含fast、slow、inner_steps及moment；排名端点选择周期边界以避免快慢参数混合报告。

## 4. 可严格证明的双线性结果

取f(x,y)=xy，min_x max_y，w=(x,y)，
\[
F(w)=Aw,\quad A=\begin{pmatrix}0&1\\-1&0\end{pmatrix},\quad A^T=-A,\ A^2=-I.
\]
Jacobian的对称部分为零，特征值为±i。这是旋转场，不是普通标量最小化的对称Hessian。

**结论1，纯同步GDA发散。** M=I-eta A，有
\[
M^TM=(I+\eta A)(I-\eta A)=(1+\eta^2)I.
\]
所以非零初值的平方范数每步乘1+eta²，任意eta>0几何发散。该证明针对原始GDA，不能用于断言Adam也同样发散。

**结论2，固定步长EG收缩。** 预测w^p=(I-eta A)w，校正
\[
w^+=(I-\eta A+\eta^2 A^2)w=((1-\eta^2)I-\eta A)w.
\]
于是
\[
\|w^+\|_2^2=(1-\eta^2+\eta^4)\|w\|_2^2.
\]
0<eta<1时乘数严格小于1，故收敛到原点。eta=1时不收缩，eta>1时发散。有限精度脚本验证等式误差，并画轨迹。这里是EG，不是ExtraAdam。

**结论3，OGDA的一段充分稳定区间。** 纯OGDA满足w_{t+1}=(I-2eta A)w_t+eta A w_{t-1}。在A特征值i的复数子空间，特征方程
\[
\lambda^2-(1-2i\eta)\lambda-i\eta=0,\qquad
\lambda_\pm=\frac{1-2i\eta\pm\sqrt{1-4\eta^2}}2.
\]
0<eta<1/2时r=sqrt(1-4eta²)∈(0,1)，|lambda_±|²=(1±r)/2<1；共轭子空间同样，故所有初值在此充分区间渐近收敛。没有主张这是最大区间，没有把OGDA直接当成OAdam。

**结论4，LA不会自动修复任意发散内法。** 用纯GDA内法M，周期映射B=(1-alpha)I+alpha M^k。稳定性由rho(B)<1决定。k=2时
\[
B=(1-\alpha\eta^2)I-2\alpha\eta A,\quad
\|Bw\|^2=[(1-\alpha\eta^2)^2+4\alpha^2\eta^2]\|w\|^2.
\]
alpha=1/2时乘数=1+eta⁴/4>1，仍发散。这是对“lookahead必然稳定”的直接反例。对0<alpha<2/(4+eta²)可得到该k=2设置收缩。k=5/alpha=.5只是待验证的实验选择；Adam拥有额外moment状态，不能把此二维M替代整个自适应动态Jacobian。

## 5. 证据与推断界限

| 结论类别 | 当前可支持内容 | 不能推出 |
|---|---|---|
| 证明 | 上述线性矩阵恒等式与条件稳定性 | CIFAR非凸博弈收敛或图像质量提高 |
| 算法正确性 | 独立多步手算、PyTorch parity、参考Extra parity、恢复/符号/计算图测试 | 某法科学上更好 |
| CPU smoke | 无NaN、日志/checkpoint可用、resume参数一致 | GPU行为或最终稳定性，短run排名 |
| 待运行实证 | 冻结后多seed FID/KID与效率比较 | 不运行就填论文分数 |

局部线性化可解释rotation、implicit-step近似等直觉；自适应预条件、随机噪声、非凸网络、GP与不同点的数据共同作用，必须依靠受控实验判断。正文、最终report与presentation使用“在此假设下证明”“数值验证”“经验观察”“待检验假设”等准确措辞。
