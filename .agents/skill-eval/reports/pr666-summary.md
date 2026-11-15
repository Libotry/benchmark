# PR #666 分支测评结果（feat/ascend-llm-skills）

- 被测分支：`AISBench/benchmark@feat/ascend-llm-skills`，commit `25463f3`
- 本地个人分支：`Libotry/benchmark@feature/skill-eval-pr666`
- 执行方式：离线，无 SSH / Docker / NPU；`python -m skill_eval` 三套依次跑
- 执行时间：2026-11-15

## 总览

| 测评集 | 被测对象 | 用例 | 结果 | 用时 |
|---|---|---:|---|---:|
| `suites/ascend-llm-cli` | `.agents/ascend_llm_cli`（五技能共用的 CLI，测 `prepare-inputs`） | 11 | **11/11 通过** | ~20s |
| `suites/ascend-llm-skills` | `.agents/skills/` 五个新技能的文档规范 | 3 | **3/3 通过** | <1s |
| `suites/ascend-llm-deploy-eval` | 稳定版 `ascend-llm-deploy-eval`（回归，确认工具改动没有误伤） | 27 | **27/27 通过** | ~30s |
| 自检（mutation） | 对 CLI 注入 5 条退化 | 5 | **5/5 检出** | ~40s |

这套东西不是「跑一遍全绿」：CLI 套件做了 5 条注入退化验证，全部被对应用例抓到。

## 新 CLI 套件测了什么

**部署阶段（P01–P06）**

| 用例 | 断言 |
|---|---|
| P01 | 合法部署输入 → `status=EVENTS`、落盘 `plan.json`/`bindings.json`、`review_required=true`、`remote_checks=NOT_RUN`、`derived=[]` |
| P02 | `stage` 非法 → `INVALID_INPUT: stage must be deploy or eval` |
| P03 | 目标目录已存在 → `REQUEST_CONFLICT: preparation destination exists` |
| P04 | 多余顶层字段 → `deployment input fields` |
| P05 | 部署输入夹带 `evaluation` → `cannot embed evaluation` |
| P06 | `policy.on_idle=release_owned` → `must be retained` |

**评测阶段（P07–P11）**

| 用例 | 断言 |
|---|---|
| P07 | 端点输入归一化：`intent=external_eval`、`ownership=external`、`permissions.{release,reset_cache,restart}=false`、`on_idle=keep_external_service`、`on_failure=stop_owned_clients`、tester 绑定落到 bindings |
| P08 | `service` 与 `endpoint` 同时给 → `provide service or endpoint, exclusively` |
| P09 | 给了 `service` 又单独给 `model` → `model must come from supplied service` |
| P10 | 评测输入出现未知字段 → 拒绝（只断言错误码，原因见下） |
| P11 | 缺 `tester` → 拒绝（只断言错误码） |

**五个新技能（K01–K03）**：frontmatter 完整、相对链接可解析、`.agents/README.md` 已登记。

## 测出来的问题

1. **评测阶段的多余字段报错文案误导**（P10）。`prepare()` 里把「未知字段」和「service/endpoint 二选一」合并成同一个判断，于是传一个 `bogus` 字段会得到 `provide service or endpoint, exclusively`，与实际原因无关。建议拆成两个判断。
2. **缺 tester 时错误信息没有细节**（P11）。只返回 `INVALID_INPUT`，没有指到具体字段，排障要靠读源码。
3. 这两条都只影响报错可读性，不影响拒绝行为本身，所以用例里只断言错误码，不断言措辞。

## 顺带修掉的一个自测 bug

第一次跑 K02 时报 `references/cli-contract.md` 断链，差点当成分支缺陷。实际是**我的检查器把相对链接按 skill 根目录解析**，而正确语义是按 Markdown 文件自身所在目录解析。已修正（`harness.py`），修正后 5 个技能全部通过——这正好说明「测评集自身也要能出错并被发现」。

## 覆盖边界

- 只覆盖 `prepare-inputs`（文档明确「仅本地校验，不授权、不连接现场、不执行部署」）。
- 不覆盖 `deploy` / `eval` / `release` 的实际执行、SSH、Docker、NPU、AISBench 实跑；那些属于现场验收（PR 描述里也写了只在 Ascend950DT 上做过一次功能验收）。
- 五个新技能只评文档规范，不评对话质量与阶段交接的语义。

## 复现

```sh
cd .agents/skill-eval
python -m skill_eval --suite suites/ascend-llm-cli  --python python3.11 --deps .deps --md reports/report-cli.md
python -m skill_eval --suite suites/ascend-llm-skills --md reports/report-skills.md
python tools/mutation_check.py --suite suites/ascend-llm-cli --python python3.11 --deps .deps \
    --mutations suites/ascend-llm-cli/mutations.json --md reports/mutation-cli.md
```
