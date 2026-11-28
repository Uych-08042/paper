# IdeaSpark / ScoopCheck 检索运行修复

日期：2026-11-27。本次修复自动流程的运行故障；没有开展新 SGLA 训练或 GPU 实验，也没有完成或伪造新的科研评审。

## 修复内容

1. 默认 Python 3.13 缺少检索依赖。建立独立 Python 3.12.4 环境，安装并锁定 requests、feedparser、openreview-py、beautifulsoup4、pymupdf、python-dotenv 及其依赖，不修改服务器跟踪环境。
2. IdeaSpark 父进程按 Windows 默认编码读取 UTF-8 子进程输出，触发解码异常，随后对空 stderr 调用 `.strip()` 再次异常。统一子进程编码及 Python 输出环境，防御空 stderr，并给主流程的文本和缓存读写显式指定 UTF-8。
3. Paper Search 说明要求 `--json`，但入口没有这个参数。补齐参数和原子 JSON 导出，保留摘要、各来源原始记录、去重后的未过滤集合及显示集合；显式分数过滤不会丢失导出中的原始证据。
4. IdeaSpark OpenAlex 使用 `type:!posted-content`，接口返回 400，明确提示这是 Crossref 原始标签而非 OpenAlex work type。关键词与语义入口均改为 `type:!preprint`，保留排除预印本的原意；相同查询新参数实测返回 200。这不等于保证结果均经过同行评审。

## 固定入口与备份

本机目录：`D:/Users/23920/Documents/Codex/research-tools`。

- Python：`.venv/Scripts/python.exe`。
- IdeaSpark：`idea-spark.ps1`。
- ScoopCheck 的检索引擎：`paper-search.ps1`。ScoopCheck 仍是需要模型执行的查新工作流，并非独立完成所有判断的 Python 脚本。
- 依赖锁：`requirements.lock.txt`。
- 回归验证：`test_utf8_fix.py`；验证记录：`validation/summary.json`；原始日志和结果：`validation/live/`。
- 原文件备份与哈希：`backups/`、`deployment*.json`；最终六个目标文件哈希已校验。

```powershell
& 'D:\Users\23920\Documents\Codex\research-tools\idea-spark.ps1' check_connectors

& 'D:\Users\23920\Documents\Codex\research-tools\paper-search.ps1' `
  --query 'adaptive visual tracking' --start-year 2020 --end-year 2026 `
  --max-papers 10 --json 'E:\Program Files\paper\paper\output\research_audit\papers.json'
```

启动器传递原始参数，固定解释器和 UTF-8，保留退出码。IdeaSpark 后续发出的 `python3` 命令也必须使用该解释器，并保留原脚本、参数和工作目录。正式科研评审仍须使用全新运行目录，不能把检查目录改称正式评审。

两份入口脚本、OpenAlex 连接器及三个相关 SKILL.md 已实际修复；技能说明只增加本机运行入口，不改变评审标准或模型阶段隔离。所有修改均先备份并核哈希。技能更新可能覆盖本机补丁，届时需重新核对，不能盲目回放旧补丁。

## 实际验证

- 5 项回归检查通过：中文子进程、缓存及空 stderr；依赖和六来源模块导入；显式编码约束；JSON 中文摘要与过滤保留；两个 OpenAlex 路由的过滤参数。
- Paper Search 默认六来源并发调用完成，退出码 0，JSON 有 6 篇去重记录：arXiv 2、OpenAlex 2、Crossref 2，其他来源为 0。验证临时使用 25 秒读取超时、10 秒连接超时及一次尝试，未更改正式默认重试策略。
- IdeaSpark 有界 Phase-0 检查退出码 0，经检索、去重得到 4 条真实记录，来源为 arXiv 和 OpenAlex。最终检查复用了本次先前真实 arXiv 检索的缓存，OpenAlex 修复后重新查询。
- Phase-0 检查限定 arXiv/OpenAlex 各 2 条，关闭其他检索任务和语义扩展，只验证链路，不满足完整查新的覆盖要求。
- 已进入 `.pattern_summary_pending` 所指的模型分类步骤。这是待模型处理状态，不是依赖或编码崩溃；完整 IdeaSpark 后续阶段与新的 ScoopCheck 语义筛选、全文核验及结论尚未执行。

## 外部限制

- IdeaSpark OpenReview 未配置账号，依赖已经安装，但该连接器仍明确跳过；未读取、打印或编造凭据。
- Semantic Scholar 返回 HTTP 429。原多来源 IdeaSpark 验证在退避重试期间，已核对 PID 和任务路径，仅终止本次旧验证进程树，避免持续打限流接口；保留未完成尝试的日志。
- Paper Search 的 DBLP 响应无法解析为 JSON；OpenReview 本次为 0 条，不据此推断其账号故障或声称返回了论文。

准确状态：**本地依赖、编码、导出接口及已发现的 OpenAlex 参数故障已修复并验证；全部来源覆盖和完整科研评审并未完成。** 既有联合审查结论不因软件修复自动改变。Stage-215 仍待服务器真实结果。
