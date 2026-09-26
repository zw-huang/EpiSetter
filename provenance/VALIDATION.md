# 整理验证 · 2026-09-25

- 训练时记录的五个源码 SHA-256 均与迁入前源码一致，详见 `training_source_audit.json`。
- seed42 checkpoint、MR/MC417 与 MRQA 排除 bundle 均绑定原评估 inputs.json 指纹。
- Base/seed42 共11784条 HotpotQA 原始响应重新解析和评分，逐条 EM/F1/ACC、汇总值、顺序及唯一 qid 完全匹配。
- CPU 恢复真实 checkpoint：L11–15、655685参数、step1800、validation_selected；五层干预输出有限且幅度满足各自 rho 上界。
- 原有8项 tiny OLMo3 CPU 集成测试全部通过，覆盖投影、完整答案评分、hook位置及清理、BF16梯度、有限差分保护梯度、数据泄漏拒绝与方向学习流程。
- `prepare` 与训练、推理、MRQA CLI 帮助入口均通过；训练输入从新仓库冻结数据读取。

测试使用 `/home/zhenwei/miniconda3/envs/DisentQA/bin/python`。Transformers 在 tiny 配置上给出 rope_parameters 未识别字段警告，8项测试仍通过。

未重新训练7B模型或复跑完整GPU评测，未验证论文编译，未执行外部发布。`SHA256SUMS.json` 覆盖发布文件（排除自身、Git、缓存与新运行目录）。

复核：`python scripts/verify_release.py`；`OMP_NUM_THREADS=2 python -m unittest discover -s tests -v`。
