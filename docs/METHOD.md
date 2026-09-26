# 当前正式方法与证据边界

模型：冻结 OLMo3-7B Base，在最后一个 prompt token 的 L11/12/13/14/15 block 输出施加干预。seed42 五层均启用；标量预测器宽度32、总可训练参数655685。

方向和尺度冻结后逐层计算 `alpha_l = rho_l * tanh(f_l(h_l))`，输出 `h_l + alpha_l * u_l`。后层读取前层已更新状态。输出层零初始化；rho 为训练双意图坐标差绝对值的中位数乘以1。损失为完整答案序列 NLL、保护答案所有前缀的 KL（系数1）和未归一化幅度平方和（系数0.01），幅度按来源/保护样本比例加权。

方向1200步，校准2000步，每200步验证。校准在最坏保护 NLL 增量与前缀 KL 都不超过0.05的候选中选来源序列 NLL 最小者，无改善则 identity fallback。seed42 选中1800步；最坏保护 NLL 增量0.0145733、KL 0.000688647。以上均为验证指标。

实际方向筛选要求 prior/context 两侧 signed margin gain 为正，保护指标保留但未在方向阶段用0.05阈值淘汰。源码中的“Pareto protection selection deferred to calibration”是历史标签；校准实际执行上文的阈值约束下 NLL 最小化，不是新的 Pareto 优化器。早期提纲描述的方向阶段保护筛选不适用于此 checkpoint。

保护基为逐答案 token 的 NLL 梯度构造。五层记录 retained_rank=4096，等于隐藏维数，且 `basis_dot_u_max` 约0.068–0.086。因此不能把该实现或 checkpoint 描述为经过数值证明的严格正交保护，也不能以局部保护指标推导全局能力无损。发布保留原算法，不悄悄修改后冒称同一训练模型。

训练738事实、验证168事实、定位71事实、bundle evaluation106事实；保护 train/validation/evaluation 分别500/500/1000。外部 MR/MC417 包含197/220条，其中13条标记 canonical answer overlap，评估另报 strict_conflict。所有数量以 `data/freeze.json` 为准。人工审核仍待完成。

训练凭据记录的五个源码文件哈希全部与迁移时源码一致。核心文件原样保留；发布入口只移除历史目录依赖、使用冻结数据、增加可配置路径与输入检查。推理不依赖旧仓库。无全规模 GPU 复跑，不能把发布检查称作实验复现。
