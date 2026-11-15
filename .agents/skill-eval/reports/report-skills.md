# SKILL 测评报告：.agents

| 项 | 值 |
|---|---|
| 被测 SKILL | `AISBench/benchmark::.agents` |
| 版本锚点 | ref `feat/ascend-llm-skills` / commit `25463f377deb761a29a240184e2786faddf9a4ec` |
| 测评集版本 | `0.1.0` |
| 执行时间 | 2026-11-15 15:13:31 中国标准时间 |
| 被测解释器 | `python.exe 3.14.6`（执行器 Python 3.14.6 / Windows） |
| 执行方式 | 离线黑盒：只调用 SKILL 自带入口，不需要 SSH、Docker、NPU |
| 依赖路径 | `无（使用解释器内置环境）` |

## 总评

**综合得分 100.0 / 100**（3/3 用例通过，权重 11/11）

| 维度 | 用例 | 通过 | 权重 | 得分 |
|---|---:|---:|---:|---:|
| 文档规范 | 3 | 3 | 11 | 100.0 |

## 用例明细

| 用例 | 名称 | 维度 | 权重 | 结果 | 失败首因 |
|---|---|---|---:|---|---|
| K01 | 五个新技能的 frontmatter 完整 | 文档规范 | 4 | PASS |  |
| K02 | 五个新技能引用的相对文件都存在 | 文档规范 | 4 | PASS |  |
| K03 | README 导航登记了五个新技能 | 文档规范 | 3 | PASS |  |

## 失败详情

无失败用例。

## 覆盖边界（未纳入自动判定的部分）

- 只评文档规范（frontmatter、链接、导航登记），不评五个技能的对话质量与阶段性行为；那些要靠 CLI 契约层的 ascend-llm-cli 套件与实机验收。

## 复现

```sh
python -m skill_eval --suite suites/ascend-llm-skills --skill-dir .. --python 'C:\Python314\python.exe' --work .eval-work/skills --md reports/report-skills.md --json reports/result.json
```

用例输入（job_spec / bindings）与断言全部在测评集内，执行不修改被测 SKILL 目录。
