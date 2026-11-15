# 测评集自检（mutation test）

被测 SKILL：`ascend_llm_cli`；执行解释器：`python.exe`。

做法：把 SKILL 复制一份，逐条注入「真实可能发生的退化」，再跑对应用例。
用例必须由 PASS 变成 FAIL，否则说明断言太松，测评集本身不可信。

**检出 5/5。**

| 编号 | 注入的退化 | 期望失败用例 | 实际 | 结论 |
|---|---|---|---|---|
| M1 | 去掉「部署输入不得嵌入 evaluation」的拦截 | P05 | P05=FAIL | 检出 |
| M2 | 评测任务错误地继承 release 权限（ownership 语义被破坏） | P07 | P07=FAIL | 检出 |
| M3 | 去掉「只部署必须保留服务」的策略校验 | P06 | P06=FAIL | 检出 |
| M4 | 去掉目标目录冲突检查（允许覆盖上一次审阅产物） | P03 | P03=FAIL | 检出 |
| M5 | 允许 service 与 endpoint 同时出现（XOR 校验失效） | P08 | P08=FAIL | 检出 |

复现：`python tools/mutation_check.py --python <py> --deps <deps> --md reports/mutation.md`
