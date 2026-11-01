# 本地 embedding 与独立生成 API（任务 1.1–1.3）

任务 1.4 的共用双语词项与证据评分合同见 [retrieval-scoring.md](retrieval-scoring.md)。

将 `.env.example` 复制为 `.env`，分别填写两类配置。BGE-M3 是待评测候选，尚未代表真实双语质量通过。

| 配置 | 用途 |
| --- | --- |
| `LOCAL_EMBEDDING_MODEL` | 本机 HuggingFaceEmbeddings/SentenceTransformers 使用的模型 ID 或本地模型目录，默认 `BAAI/bge-m3` |
| `LOCAL_EMBEDDING_REVISION` | 模型 revision，建议填写已核对的固定 commit SHA；空值不允许加载 |
| `LOCAL_EMBEDDING_CACHE_DIR` | 模型缓存目录，默认 `./data/models` |
| `LOCAL_EMBEDDING_DEVICE` | `auto`（有 CUDA 优先 CUDA，否则 CPU）、`cpu`、`cuda` 或 `cuda:N` |
| `LOCAL_EMBEDDING_BATCH_SIZE` | 本地编码批大小，正整数，默认 32 |
| `LOCAL_EMBEDDING_MAX_BATCH_RETRIES` | CUDA 编码显存不足时减半批大小的次数上限，默认 3，范围 0–10 |
| `LOCAL_EMBEDDING_MAX_INPUT_TOKENS` | 编码输入上限，默认 512；实际限制还取本地模型和 tokenizer 上限的较小值 |
| `LOCAL_EMBEDDING_TEXT_NORMALIZATION` | `preserve`（默认，保留字符）或显式 `NFKC` 规范化；换行统一为空格 |
| `LOCAL_EMBEDDING_QUERY_PREFIX` / `LOCAL_EMBEDDING_DOCUMENT_PREFIX` | 问题/文档前缀，默认空；尾随空格保留，计入预算和指纹 |
| `LOCAL_EMBEDDING_NORMALIZE` | 向量归一化，默认 true，计入指纹 |
| `EMBEDDING_DISTANCE` | 显式距离配置，目前只支持 `cosine` |
| `GENERATION_API_BASE_URL` | OpenAI 兼容生成 API 地址，包括供应商要求的路径前缀（通常为 `/v1`） |
| `GENERATION_API_KEY` | 仅供生成使用的密钥 |
| `GENERATION_MODEL` | 生成模型 ID，统一覆盖旧核心/轻量模型配置 |

文档与问题在应用进程内向量化，入库入口、RAG 和向量服务共用同一套 `LOCAL_EMBEDDING_*` 配置。生成地址、模型或密钥变化不会覆盖本地模型参数。无需另起 embedding HTTP 服务，也没有远程 embedding 地址或凭据配置。`hf:` / `local:` 仅作为本地模型名前缀兼容；hash、远程 URL 或远程 provider 前缀会报配置错误。

模型资源首次下载可能需要访问下载站点，但只有模型标识与版本等资源请求，不携带保单正文或用户问题。编码使用本机模型；原 PDF、解析与检索资料保留本机。生成 API 接收问题、必要前文与必要原文证据。`OPENAI_EMBEDDING_MODEL` 和旧 `EMBEDDING_MODEL` 字段仅接纳旧 `.env`，不再决定实际模型，旧 demo 脚本设置这些字段也不能开启远程/hash 路径。

## 生成配置兼容顺序

对应 `GENERATION_*` 新字段未设置（None）时，按以下顺序取首个非空旧值；新字段显式设为空字符串会禁用该值，不回退：

- 地址：`OPENAI_BASE_URL` → `SILICONFLOW_BASE_URL`。
- 密钥：`OPENAI_API_KEY` → `SILICONFLOW_API_KEY`。
- 核心模型：`OPENAI_MODEL_CORE` → `OPENAI_MODEL` → `SILICONFLOW_MODEL`。
- 轻量模型：`OPENAI_MODEL_LIGHT` → `OPENAI_MODEL` → `SILICONFLOW_MODEL`。

不隐式填入默认生成供应商或模型。三项分别解析，因此迁移时建议完整填写新三项，避免混用不同供应商旧值。服务内部显式指定模型时使用该模型，地址和密钥仍使用生成配置。Pydantic Settings 保留“构造参数 > 环境变量 > `.env` > 默认值”的取值顺序；别名优先级在取到这些字段后应用。

生成配置不完整时不创建生成客户端，不能视为完整问答可用，本地向量化仍能独立运行。本地模型/revision/缓存配置缺失，或依赖、权重、设备加载失败时，`LocalEmbeddingUnavailable` 明确说明入库/语义检索不可用；不会调用生成服务代替向量化，也不会回退 hash 或其他模型。初始化不将文档/问题作为模型加载探针。编码异常只公开异常类别，避免输入或凭据从错误正文泄露。

日志在记录创建阶段隐藏已配置的生成密钥，以及生成地址中的用户名、密码、查询参数和片段，覆盖应用、HTTP 客户端/SDK、根日志和独立输出处理器；也处理 URL 编码或 JSON 转义后的已配置凭据。密钥和地址不出现在 Settings 的 repr 中。不要打印 `.env` 或完整 settings 字典。默认模型缓存 `data/models/` 不进入 Git。

## 本项验证范围

无网络测试用本地编码替身与生成 API 的内存 HTTP 替身检查配置隔离、同步/异步编码、生成请求、日志脱敏和出站请求边界；不下载真实权重，不把替身结果作为真实模型验收。

任务 1.2 已提供下述准备和诊断入口；任务 1.3 已提供 revision/tokenizer 指纹、兼容判断及输入预算。任务 4.1 负责将指纹接入 collection/索引快照的隔离与加载。任务 9.1–9.2 负责启动脚本、完整依赖与 Docker/健康诊断交付；现有健康接口尚不能用来证明真实模型或生成服务可用。

## 首次准备与缓存离线运行

在包含项目依赖的 Python 环境中配置模型 ID、固定 revision 和缓存路径。完整本地路径需要 `langchain-huggingface`、`sentence-transformers`、`torch` 及其依赖（现有 `requirements.txt` 包含本地模型依赖）。运行：

```powershell
python -m scripts.prepare_models --download
```

这是显式允许下载公开模型资源的操作。先尝试读取所选 revision 的本地缓存；缺失时才下载完整 snapshot，包含模型、tokenizer 及配置。已有 snapshot 但缺权重/tokenizer 时，会在显式允许下载的情况下补齐。下载调用只接收模型 ID、revision、缓存目录等参数，不接收文档、问题或生成凭据，也不携带隐式 HF 账号 token。模型较大，请按实际下载源和磁盘条件预留资源；未指定固定 revision 会直接失败，不猜测正式版本。

模型加载始终指向本地 snapshot，并设置 `local_files_only=True`、`trust_remote_code=False`。常规应用构造 `EmbeddingService()` 和下面的预检均不允许下载、不探测远程站点：

```powershell
python -m scripts.prepare_models
```

缓存完整时，即使下载站点断开也可加载；缓存缺失或损坏时明确未就绪，并提示带 `--download` 准备。配置现有本地模型目录时直接读取该目录，不调用下载服务，目录完整性由加载和预检确认；其版本标识由提供者填写，不宣称已核验其来源。

预检使用固定的公开中英测试短句，分别检查文档与问题编码；返回向量数量、维度一致性、非零及有限数值都正确后才输出 `local_embedding.status=ready`。报告包含所选 revision、实际 snapshot revision、实际设备、最终批大小、维度及警告。模型能构造但尚未完成预检时是 `not_checked`。本预检不读入用户文档、不验证跨语言检索质量。

## GPU 显存与 CPU 回退

`auto` 优先可用 CUDA，无 GPU 则使用 CPU。显式选择 CUDA 但 CUDA 不可用时也使用同模型 CPU 并给出提示。编码 OOM 时依次减半批大小，最多减小 `LOCAL_EMBEDDING_MAX_BATCH_RETRIES` 次且不低于 1；仍 OOM 时释放 GPU 模型引用和缓存，再从同一本地 snapshot 加载 CPU 模型。加载模型本身 OOM 时批大小无法减少模型权重占用，直接尝试同模型 CPU。

回退保留模型/revision、归一化和既有维度约束。非内存错误不会触发其他模型回退；CPU 加载或编码仍失败则明确不可用，不无限重试。`batch_reduced`、`cpu_execution`、`cpu_fallback` 警告记录实际变化，CPU 提示可能变慢。每个服务实例的编码串行执行，异步接口在线程中执行阻塞操作。

## 独立生成 API 预检与错误码

仅在希望发起真实生成请求时显式加上：

```powershell
python -m scripts.prepare_models --check-generation
```

也可与 `--download` 一起使用。该选项会向已配置生成服务发送固定短请求，可能产生少量费用。即使本地模型失败，也会独立检查生成服务。未指定此选项时生成状态保持 `not_checked`，不作付费调用；缺配置时为 `unavailable/not_configured`，也不作请求。退出码 0 表示本次明确要求的检查通过；退出码 1 表示至少一项失败。`model_services_ready` 只有本地预检和真实生成请求都成功才为 true，且仅表示模型服务就绪，不代表文档索引或整体验收通过。

| 错误码 | 处理 |
| --- | --- |
| `not_configured` | 生成地址/密钥/模型缺失，无请求 |
| `credentials` | 401/403，不重试，核对权限和凭据 |
| `quota` | 429 且明确为余额/配额耗尽，不重试 |
| `configuration` | 400/404/405/413/422，不重试，核对服务合同及模型配置 |
| `timeout`、`connection`、`rate_limit`、`server` | 超时、连接、临时限流或 5xx，按有界退避重试 |
| `invalid_response` | 无有效文本回答/流对象，不重试 |
| `queue_timeout`、`circuit_open` | 并发队列超时或熔断，诚实报告不可用 |
| `unexpected` | 其他未分类错误，不盲目重试 |

所有主要生成入口共用这些规则。`LLM_MAX_RETRIES` 实际表示总尝试次数（包含第一次），默认 3、最多 10；退避为 `min(LLM_BACKOFF_BASE * LLM_BACKOFF_FACTOR ** n, LLM_BACKOFF_MAX)`，默认 0.25、2、2.5 秒。`OPENAI_TIMEOUT_SECS` 限制每次 API 调用，SDK 内部重试关闭；服务不隐式换供应商。空流、仅空白或无有效文本的流按 `invalid_response` 失败，不能作为正常完成；流已经输出一部分后发生故障会关闭上游流并报告错误，不重放或将部分答案当作完成。取消不重试，等待队列许可会释放。

本项无网络故障注入覆盖完整/缺失/部分缓存、下载失败、GPU OOM、无 GPU、CPU 失败、向量校验、跨设备同 profile 数值容差及排名一致性，以及生成凭据/配额/超时/异常返回/取消。GPU 和跨设备测试使用替身，只证明恢复合同；真实 CPU/GPU 数值、资源与速度仍须在固定模型上实测。尚未提供本机真实模型权重及真实生成配置，因此本项不宣称实际联合验收完成。

## 指纹与兼容性（任务 1.3）

成功预检的报告新增 `local_embedding.embedding_profile`，其中 `profile` 是规范化的语义配置，`fingerprint` 是该配置规范 JSON 的 SHA-256。它包含：

- 模型 ID、配置 revision、实际 snapshot revision、实际输出维度；可修改的本地目录还包含模型权重/配置文件的内容摘要，避免同 revision 标签下覆盖权重后误用旧向量。
- 实际加载 tokenizer 的序列化规则摘要（包含词表、合并/规范化规则和特殊 token），而非仅凭 tokenizer 名称或词表大小。缓存路径不进入摘要；无法可靠识别 tokenizer 时明确失败。
- 问题/文档前缀摘要、文字预处理版本、归一化、显式距离、实际输入上限、隐式 prompt 禁用及超限拒绝策略。
- 编码/分词库版本，以及当前切块版本和参数。当前仍是 `legacy-character-v1`；任务 3.2 改为页/条款/token 预算切块时须同步更新版本。

生成地址、生成模型、密钥、设备、批大小和缓存位置不属于 embedding 语义指纹。只换生成模型无需重建向量，但仍要重新验证完整答案；换 embedding 模型，即便维度相同，或修改 revision、tokenizer、前缀、预处理、归一化、预算或切块合同，均要求重建。CPU 回退会核对新加载 tokenizer/预算与已有语义配置是否一致，不接受另一个语义空间。

`EmbeddingService.embedding_profile().to_record()` 在已知维度后返回完整记录；调用 `require_compatible(stored_record)` 校验记录是否完整、摘要是否正确、语义配置是否一致。缺失指纹的旧索引不能通过此校验。失败抛出 `EmbeddingCompatibilityError`，其 `changed_fields` 指出变化的顶层配置项，并要求从保留的 PDF 重建；它不自动采纳当前配置、覆写旧索引或迁移向量。

本任务提供兼容判断接口，尚未完成任务 4.1 的所有索引读写入口接入。现有索引不能仅因本任务完成而视为已校验，旧应用/脚本的索引加载安全性须随 4.1 一并验收。

## 精确输入预算

服务使用已加载的本地 tokenizer，对实际规范化并添加角色前缀后的文本调用 `encode(add_special_tokens=True, truncation=False)`。上限取配置、SentenceTransformer 的 `max_seq_length` 和 tokenizer 的有效 `model_max_length` 的最小值；无穷大占位上限不作为实际模型限制，模型限制完全未知时拒绝编码。计数不加载第二个模型、不访问网络，不用字符数冒充 token 数。

编码禁用模型隐式 prompt（`prompt=""`），只使用显式角色前缀。前缀、特殊 token 共同占用输入预算；内容恰好达到上限可编码，超限抛出 `EmbeddingInputTooLong`，仅报告角色/数量/限制，不回显正文。原文、产品名、数字、日期和条件不因预算而截断。`count_tokens(text, query=True/False)` 和 `max_input_tokens` 可供任务 3.2 的切块服务使用；当前 CLI 遇到超限片段会明确失败，自动按页/条款拆分由任务 3.2 完成。

这些检查仅保证 embedding 输入不被静默截断；生成 API 的整轮上下文分配与比较覆盖仍由后续问答任务接入。

## 重建步骤

1. 停止使用旧库的入库/查询，保存原 PDF、SQLite、索引目录、配置及删除日志。不要把修改模型配置当作已迁移向量，也不要覆盖或删除唯一源文件。
2. 固定新的本地模型 revision、预处理、前缀、预算与切块参数。运行 `python -m scripts.prepare_models --download` 准备资源，再运行不带下载选项的预检；记录返回的完整 embedding profile 和 fingerprint。
3. 使用新的空 Chroma 与 processed 目录，从仍有效的原 PDF 重新解析、编码并构建关键词资源。当前样本 CLI 可通过 `--persist-dir` 和 `PROCESSED_DATA_PATH` 指向新目录；不要对唯一旧目录使用 `--reset`。新建 collection 已显式使用 cosine；给已有 collection 传入新 metadata 不会把旧 L2 向量自动转换为 cosine。
4. 比较新索引保存的 profile 与当前 profile，验证来源、输入完整性和真实检索/回答，再切换应用配置到匹配的新目录。索引 profile 的自动持久化、按指纹隔离和发布接入由 4.1–4.3 完成；目前应保留预检记录并将该切换视为需人工核对的准备步骤，不能声称已通过完整发布验收。
5. 需要回退时恢复匹配的配置/模型/索引，保留新增来源并优先重放删除日志。不能从旧快照恢复已删除资料，也不能以丢弃新增文档实现回退。
