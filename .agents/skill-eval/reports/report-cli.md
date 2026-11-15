# SKILL 测评报告：ascend-llm-cli

| 项 | 值 |
|---|---|
| 被测 SKILL | `AISBench/benchmark::.agents/ascend_llm_cli` |
| 版本锚点 | ref `feat/ascend-llm-skills` / commit `25463f377deb761a29a240184e2786faddf9a4ec` |
| 测评集版本 | `0.1.0` |
| 执行时间 | 2026-11-15 15:13:52 中国标准时间 |
| 被测解释器 | `python.exe 3.11.0`（执行器 Python 3.14.6 / Windows） |
| 执行方式 | 离线黑盒：只调用 SKILL 自带入口，不需要 SSH、Docker、NPU |
| 依赖路径 | `.deps` |

## 总评

**综合得分 100.0 / 100**（11/11 用例通过，权重 43/43）

| 维度 | 用例 | 通过 | 权重 | 得分 |
|---|---:|---:|---:|---:|
| D1 部署输入契约 | 6 | 6 | 25 | 100.0 |
| D2 评测输入契约 | 5 | 5 | 18 | 100.0 |

## 用例明细

| 用例 | 名称 | 维度 | 权重 | 结果 | 失败首因 |
|---|---|---|---:|---|---|
| P01 | 部署输入通过本地校验并落盘 plan/bindings | D1 部署输入契约 | 6 | PASS |  |
| P02 | stage 非法时明确拒绝 | D1 部署输入契约 | 3 | PASS |  |
| P03 | 目标目录已存在时拒绝覆盖 | D1 部署输入契约 | 4 | PASS |  |
| P04 | 部署输入出现多余顶层字段时拒绝 | D1 部署输入契约 | 4 | PASS |  |
| P05 | 部署输入里夹带 evaluation 时拒绝 | D1 部署输入契约 | 4 | PASS |  |
| P06 | 部署输入要求空闲即释放时拒绝 | D1 部署输入契约 | 4 | PASS |  |
| P07 | 评测输入归一化为 external_eval，且不继承释放权限 | D2 评测输入契约 | 6 | PASS |  |
| P08 | 同时给 service 与 endpoint 时拒绝 | D2 评测输入契约 | 3 | PASS |  |
| P09 | 给了 service 又单独给 model 时拒绝 | D2 评测输入契约 | 3 | PASS |  |
| P10 | 评测输入出现未知字段时拒绝（只断言错误码） | D2 评测输入契约 | 3 | PASS |  |
| P11 | 缺少 tester 时拒绝（只断言错误码） | D2 评测输入契约 | 3 | PASS |  |

## 失败详情

无失败用例。

## 覆盖边界（未纳入自动判定的部分）

- 只覆盖 prepare-inputs 的本地校验与落盘契约（deploy/eval 两阶段），不覆盖 deploy/eval/release 的实际执行、SSH、Docker、NPU、AISBench 实跑。
- 本 CLI 的输入是单个请求 JSON，占用 fixture 的 job_spec 槽位；bindings 槽位不使用（空对象）。
- 五个新 SKILL（router/plan/deployment/eval/release）只做静态检查（frontmatter、链接、导航登记），不评其对话质量。
- 已知两处报错信息质量问题（多余字段与缺 tester）在用例中只断言错误码，不断言措辞，详见报告备注。

## 复现

```sh
python -m skill_eval --suite suites/ascend-llm-cli --skill-dir ../ascend_llm_cli --python python.exe --deps .deps --work .eval-work/cli --md reports/report-cli.md --json reports/result.json
```

用例输入（job_spec / bindings）与断言全部在测评集内，执行不修改被测 SKILL 目录。
