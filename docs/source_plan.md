# ICLR 论文提纲与补实验计划：OLMo3-7B 历史有界校准主线

更新：2026-09-23。本版替代此前以 Qwen3-0.6B 软校准为中心的规划。仅修改文档，不表示实验已启动。旧版保存在 archive_before_olmo_rewrite_20260923/。

## 1. 研究定位与模型选择

工作标题：**EpiSetter: Intent-Conditioned Source Reliance via Protected Layerwise Calibration**。

主模型为历史 OLMo3-7B Base，沿用其受保护方向和有界标量校准方法。Qwen3-0.6B 的无界软校准作为独立探索记录，不能与 OLMo 历史结果合并；跨模型复验须匹配方法后另行运行。

选择 OLMo 的理由是它已有值得复核的 MRQA EM/F1 增益和可复用 checkpoint，而不是已经证明它更接近录用。模型规模、训练数据、方法及解码不同，不能从两套历史结果因果归因“7B 优于 0.6B”。本项目内部验收标准不等于 ICLR 官方录用门槛。

主研究问题：固定主模型后，受保护方向上的输入依赖有界校准能否提高同事实双意图的自由生成正确率，并在匹配来源收益时减少独立能力损害？

## 2. 历史证据与复用边界

证据以 ../../RESULTS_AUDIT.md 为准；以下 artifacts 路径相对 episetter_iclr/。

| 产物 | 已完成证据 | 限制 |
|---|---|---|
| artifacts/olmo3_7b_disjoint_seed43/ | 四层 7/15/23/27；校准器 524548 参数 | 训练仅5事实、验证2、定位3、评估4；不是正式规模 |
| artifacts/olmo3_7b_strict_all32_seed44/ | 32层候选，实际启用除0/26外的30层；3934110参数 | 与四层配置不同，不能作为同配置种子复验 |
| 四事实评估 | 序列NLL 5.2624→2.6992/2.4260；margin gain 0.0712/0.4225 | 严格来源EM为0/8，生成字符串未改变；PairAcc以4个事实为分母，不能写0/8 |
| 历史MRQA共同ID | HotpotQA四层 F1 43.34 vs base 22.19；ACC 58.55 vs 57.42 | EM/F1增益可能含格式/冗余文本效应；不能直接解释为来源仲裁改善 |
| 历史MRQA覆盖 | 预测及统一重评分可复用 | base缺SQuAD/SearchQA/TriviaQA，NewsQA不完整；30层TriviaQA不完整 |
| artifacts/olmo3_7b_geometry_audit_20260923/audit.json | 抽查四层正交误差小，跨run方向cosine约0.990–0.993 | 同5个训练事实；不能证明独立泛化或保护任务覆盖充分 |
| 单prompt缓存核验 | 四层checkpoint、16新token，两路径文本相同 | 仅smoke，不算多prompt/两checkpoint一致性验收 |

历史保护样本各split只有4条，主要为算术。保护rank=8不能支持通用能力保持。旧源码未完整冻结；当前重跑是历史checkpoint的复验，不得冒称恢复了全部训练时实现。

## 3. 固定的主方法

单位方向由保护梯度子空间正交补中的双向坐标交换学习。历史方向选择要求验证集两侧signed margin gain为正，并满足历史保护验证条件。当前软分支已修改部分筛选逻辑，新训练前必须建立隔离的历史策略配置及审计，不能直接使用现有默认值。

冻结主干、方向及层尺度，仅学习逐层标量预测器。后层读取前层更新后的状态，干预位置为最后一个prompt token的block输出：

$$
\alpha_\ell(x)=\rho_\ell\tanh f_{\theta,\ell}(\widetilde h_\ell(x)),\qquad
h'_\ell=\widetilde h_\ell+\alpha_\ell(x)u_\ell.
$$

$$
\rho_\ell=m\,\operatorname{median}_{(r,d)\in D_{\mathrm{train}}}
\left|u_\ell^\top h_\ell^d-u_\ell^\top h_\ell^r\right|.
$$

尺度仅由训练数据确定；m在验证集选取，测试不参与。使用历史损失：

$$
\mathcal L=\mathcal L_{\mathrm{source}}
+\lambda_{\mathrm{KL}}\mathcal L_{\mathrm{keep}}
+\lambda_a\mathbb E_x\sum_\ell\alpha_\ell(x)^2.
$$

source为完整目标答案token NLL之和；keep为保护参考答案所有前缀的teacher-to-calibrated KL；幅度项为未按rho归一化的平方和，按来源/保护样本比例加权。输出头零初始化。历史校准选点是在保护验证阈值内取source序列NLL最低者，无改善则identity fallback。该规则作为主方法固定复验；任何margin/Pareto选点、无界输出或归一化正则均单列消融，不能混入旧方法。

## 4. 正文提纲

1. **Introduction**：同问题同材料不同指令的来源需求；提出可检验的来源控制与保护权衡问题。历史MRQA仅作动机。
2. **Task and protocol**：模型prior、独立事实gold、材料答案分开；事实/问题/文档/实体分组；主指标为自由生成PairAcc和分意图EM/F1。
3. **Method**：保护梯度子空间、双向方向学习、有界闭环逐层校准、历史验证选点、局部一阶保护的适用边界。
4. **Main results**：先报独立双意图生成，再报外部冲突与未见意图；MRQA回答质量及独立能力保持另表。
5. **Ablations and mechanism**：保护×KL、固定/自适应、单层/四层/多层、非零随机方向对照；原生组件中介仅在专门实验支持时写入贡献。
6. **Related work / limitations**：核实相关工作后限定创新；披露小数据、格式、筛选、局部保护与历史测试暴露。整合/拒答没有专门任务时不作为贡献。

## 5. 补实验队列与验收

所有新运行使用独立OLMo目录和run ID，保留旧预测和checkpoint。当前状态仅R0部分完成，以下均未因本次文档改写自动启动。

| 顺序 | 实验 | 执行内容 | 产物与验收 |
|---|---|---|---|
| R0 / P0 | 历史复验与冻结 | 绑定权重/tokenizer、bundle、checkpoint及当前源码哈希；核对旧策略；两checkpoint在开发prompt核验cache/参考路径 | 多prompt一致性报告、差异token和logit；已有单例仅smoke |
| R1 / P0 | 补全MRQA对照 | 新目录复制历史预测；从response统一重评分后续跑base缺失项及30层缺失项；旧设置greedy/BF16/2048总预算/32新token | 全覆盖和共同qid两表，重复/引用答案一致性检查；EM/F1/ACC及配对CI |
| R2 / P0 | MRQA增益归因 | 对冻结预测报告格式合规、长度、截断、答案包含率；固定解析器，分层抽样盲审 | 区分内容纠正、冗余文本变化和纯格式改变；审计不用于回选checkpoint |
| R3 / P0 | OLMo独立双意图数据 | 使用 ConFiQA-QA；先按事实组划分，以冻结 OLMo 的稳定闭卷答案作为 prior，不要求与材料答案或 gold 相同；保留失败组，冲突与一致样本分开 | 先100–300事实开发pilot估计配对方差，再决定确认集规模；新确认集不包含旧已见测试；执行细则及状态见 [R3_PROTOCOL.md](../../R3_PROTOCOL.md) |
| R4 / P0 | 冻结旧checkpoint迁移 | base+相同意图prompt、四层seed43、30层seed44在相同冻结数据上评估；先开发诊断，固定协议后确认 | PairAcc、每侧EM/F1、纠正/破坏、both/other、同答案率；候选margin仅辅助 |
| R5 / P1 | 旧方法扩数据重训 | 保持有界公式与历史选择规则，在新的train/validation上重学方向及校准；冻结base、四层及验证最佳单层配置 | 先验证稳定生成收益再开更多层/种子；旧5事实checkpoint结果单列 |
| R6 / P1 | 保护与自适应归因 | 保护/无保护方向×keep-KL有/无；自然、零、常数、打乱与交换意图幅度；匹配reader/预算 | 相同来源收益下保护代价，证明输入依赖与保护的增量 |
| R7 / P1 | 非零压力与层消融 | 匹配非零位移下protected/unconstrained/random方向；最佳单层、7/15/23/27、验证合格多层 | 幅度—收益—损害曲线及失败层；超过rho的强制扫描仅诊断，不称部署策略 |
| R8 / P1 | 外部泛化与保持 | 固定外部冲突集协议及未见意图；独立QA/指令遵循/推理任务，按Base模型适配协议 | 外部PairAcc、官方任务指标及预定非劣性容差；能力测试不以保护KL代替 |
| R9 / P1 | 多种子与机制 | 主配置至少3种子起步，数量随功效审计调整；独立定位后patch/坐标恢复/仅写坐标 | 事实簇CI与种子方差分开；若只有控制证据，不主张唯一原生setter |
| R10 / P2 | 扩展 | 匹配rank-r、无界软约束、LoRA/SFT及第二模型 | 每项独立协议/预算；Qwen旧软约束结果不算匹配跨模型复现 |

SHIFT采用已有OLMo本地移植checkpoint作历史对照，明确12训练例/150 iterations等预算差异，不能写成官方同backbone checkpoint。正式比较须补同数据、同输入可见性及搜索预算匹配；在R4/R5协议固定后运行。

## 6. 统计与决策规则

- PairAcc按事实组计算，双意图必须同时正确；不得以8条意图行当8个独立事实。
- 用开发数据预定最小有意义收益和逐任务退化容差，再按配对方差确定样本量；报告配对事实簇bootstrap 95%区间及多重比较处理。
- 历史MRQA和四事实评估已被查看，仅作探索；新方法基于其调整后，最终结论需要未触碰确认集。
- 历史NLL选点规则保持不变；若其不改善生成，将失败完整报告。改用margin/生成选点属于新消融，不能在测试上反复挑点。
- 主终点若仍无稳定收益，先诊断知识可用性、意图依赖及生成目标错配，暂停扩大层组合和完整多种子矩阵。
- 只有在独立生成收益、匹配保护收益及复现均成立后，才在摘要提出对应主张。MRQA改善本身不足以宣称意图控制成功。

## 7. 图表及执行记录

正文：双意图PairAcc主表；分意图纠正/破坏；匹配收益—保持曲线；保护×KL及controller消融；成本表。附录：历史MRQA完整表、格式归因、所有种子与失败层、筛选流图和数据哈希。

每个阶段记录命令、GPU、退出码、数据/源码/checkpoint哈希、逐例输出及完成标记，并同步 results_gpu3.ipynb。本次仅改写计划和LaTeX提纲；不修改训练代码、不创建tmux任务。
