# Spec Delta

## Purpose

让用户以中文、英文或中英混合问题查询选定保险条款，并通过本地多语言 embedding 与独立配置的生成 API 获得原文可追溯回答。该能力统一语言、边界与真实语料验收，使跨语言检索和完整回答的可用性由真实本地模型、实际生成 API 与真实文档共同验证。

## ADDED Requirements

### Requirement: Configure local embeddings and generation API separately

系统 SHALL 在本机执行文档及问题的 embedding，单独配置本地模型、版本、缓存与运行参数；生成 SHALL 独立配置 API 地址、模型和凭据，不将生成服务用作远程 embedding 替代。

#### Scenario: Separate responsibilities
- **WHEN** 用户配置本地 embedding 模型与生成 API
- **THEN** 文档及问题在本机向量化，回答使用生成配置，改变生成模型不覆盖本地模型参数

#### Scenario: Missing local model
- **WHEN** 生成服务已配置但本地模型不可用
- **THEN** 明确说明入库或语义检索未就绪，不用生成服务、hash 或其他模型假装成功

### Requirement: Cache downloaded model assets for local operation

系统 SHALL 允许首次下载并缓存公开 embedding 模型权重与 tokenizer，记录实际版本；缓存完整后 SHALL 能在模型下载站点不可访问时本地加载与向量化，模型下载 SHALL 不携带文档正文或问题。

#### Scenario: First preparation
- **WHEN** 选定模型尚未缓存且下载源可访问
- **THEN** 下载并缓存该模型及必要资源，准备完成后在本地推理，不将保单发送到下载站点

#### Scenario: Cached model without download access
- **WHEN** 已完整缓存的模型后续运行时无法访问下载站点
- **THEN** 仍可本地加载和编码；生成 API 的网络依赖独立报告

### Requirement: Prefer GPU with same-model CPU fallback

系统 SHALL 优先使用可用 GPU；显存不足 SHALL 先有界降低批大小，必要时使用同模型 CPU 路径并提示速度变化。回退 SHALL 保留模型版本、预处理和归一化，不静默更换语义空间。

#### Scenario: GPU memory exhaustion
- **WHEN** GPU 编码显存不足且降低批大小后仍无法运行
- **THEN** 尝试同模型 CPU 编码，显示实际设备与可能变慢的提示，不切换远程、hash 或其他模型

#### Scenario: No GPU available
- **WHEN** 本机没有可用 GPU 而同模型可在 CPU 运行
- **THEN** 使用 CPU 并说明设备及速度变化，不将 CPU 路径误报为不支持

#### Scenario: All local execution paths fail
- **WHEN** 降低批次及同模型 CPU 路径仍不可用
- **THEN** 明确报告模型不可用，不无限重试或将失败伪装为证据不足

### Requirement: Keep the document library and retrieval local

系统 SHALL 在本机保存原 PDF、文档记录、向量和关键词索引并执行检索；文档解析、切块与向量化 SHALL 在本机执行，生成 API SHALL 仅接收本轮问题、理解追问必需的同范围前文和有效范围内必要证据，不整份上传 PDF 作为生成替代流程。

#### Scenario: Local ingestion and API generation
- **WHEN** 用户入库并查询一份文档
- **THEN** 本地模型处理需要向量化的文本，只有生成服务接收本轮所需资料，原文件和检索资料仍由本地文档库管理

### Requirement: Retrieve across Chinese and English

系统 SHALL 支持中问中、中问英、英问中、英问英及中英混排；问题语言 SHALL 不成为排除另一种文档语言的条件，真实跨语能力 SHALL 用真实本地 embedding 模型 验证。

#### Scenario: Four language combinations
- **WHEN** 等价中英文问题分别查询含对应条款的中英文文档
- **THEN** 四种组合均可检索原文证据并进入相应回答流程

#### Scenario: Mixed question
- **WHEN** 用户询问“这份 policy 的 waiting period 是多久？”
- **THEN** 保留产品与条件，不因混合语言直接拒答

### Requirement: Resolve answer language consistently

系统 SHALL 支持 `answer_language=auto|zh|en`，默认 auto；优先使用显式参数，其次问题中的明确语言要求，再跟随当前问题主要语言，无法判定时回退中文，并返回已解析语言。

#### Scenario: English question about Chinese source
- **WHEN** 以英文提问中文条款且未另行指定
- **THEN** 用英文回答并保留中文原文引用

#### Scenario: Parameter overrides text instruction
- **WHEN** 中文问题要求英文回答，但参数为 zh
- **THEN** 使用中文；参数为 auto 时遵循问题中的英文回答要求

#### Scenario: Ambiguous mixed language
- **WHEN** 没有语言指令且主要语言无法判定
- **THEN** 使用中文并明确返回 zh

### Requirement: Preserve source-language provenance

系统 SHALL 以原文、文档 ID、原文件名与原页码展示证据；解释或译文 SHALL 与原文区分，不能替换原文或改变金额、单位、期限、否定与适用条件。

#### Scenario: Chinese explanation of English exclusions
- **WHEN** 用户用中文询问英文除外条款
- **THEN** 中文解释对应英文原文证据，引用可打开正确页，译文不冒充原 PDF 文字

### Requirement: Apply bilingual insurance boundaries

系统 SHALL 一致处理中英文购买/配置建议、保证获赔与离题问题；合法条款问题 SHALL 不因缺中文关键词被拒绝。建议边界或拒答 SHALL 不携带公开引用，不用模型常识补保险事实。

#### Scenario: Legitimate short English question
- **WHEN** 问题是 `What is the waiting period?` 且所选文档有充分依据
- **THEN** 不因缺中文关键词拒答

#### Scenario: Purchase or guarantee
- **WHEN** 用户问 `Should I buy this policy?` 或 `Can you guarantee my claim will be paid?`
- **THEN** 以已解析语言返回 advice 边界，公开引用为空，不给购买建议或保证

#### Scenario: Unsupported fact or off-topic question
- **WHEN** 有效范围内缺少所问条款依据，或问题为中英文天气等离题内容
- **THEN** 诚实拒答并保持公开引用为空

### Requirement: Localize all terminal outcomes

正常回答、建议边界、证据不足、澄清、本地模型或生成 API 不可用、超时及来源删除导致中止 SHALL 使用已解析语言；流式/非流式 SHALL 保持等价终态，不把未完成输出当完整答案。

#### Scenario: Generation service unavailable
- **WHEN** 英文回答请求已有有效检索证据，但生成 API 无法使用
- **THEN** 英文说明生成不可用，可展示真实原文片段，不声称完成完整解释

#### Scenario: Local embedding fails
- **WHEN** 本地 embedding 模型加载或编码失败且同模型设备回退仍不可用
- **THEN** 明确显示入库/语义检索不可用，不伪装成“条款中没有依据”或生成正常答案

#### Scenario: Stream interrupted by source deletion
- **WHEN** 依赖来源的流式回答因删除被取消
- **THEN** 最终状态说明来源删除导致中止，不将部分比较作为完整答案

### Requirement: Preserve intent during retrieval preparation

分词、术语映射及追问改写 SHALL 保留产品、金额、日期、否定与限制；原问题 SHALL 被保留，改写和前文只辅助检索，不成为证据或改变文档范围。

#### Scenario: Negated accident condition
- **WHEN** 问题询问某产品非意外事故的等待期
- **THEN** 不能借“意外无等待期”条款回答所有事故均无等待期

### Requirement: Use consistent evidence scores

系统 SHALL 区分距离、证据相关性和检索排名，用与已验证本地 embedding 配置匹配的证据门槛决定是否回答；关键词或融合原始分数 SHALL 不冒充同尺度相关性或正确概率。

#### Scenario: Smaller distance
- **WHEN** 同一距离配置下候选距离更小
- **THEN** 转换后的相关性不更低

#### Scenario: Lexical score without supporting evidence
- **WHEN** 关键词排名高但证据相关性不足
- **THEN** 不能仅以原始关键词数值通过相似度门槛

### Requirement: Reject incompatible semantic indexes

系统 SHALL 校验查询与索引的实际本地 embedding 模型/版本、tokenizer、维度、文本处理与距离配置；不匹配 SHALL 要求重建或恢复兼容配置，不静默换空间，包括同维度模型更换。

#### Scenario: Equal dimensions different models
- **WHEN** 用户改为另一种同维度 embedding 模型
- **THEN** 仍识别为不兼容，不能复用旧索引假装正常

### Requirement: Use representative official real documents

首批真实验收 SHALL 使用 6 份保险公司官方条款，简体中文和英文各 3 份，覆盖医疗、重疾、寿险，不含繁体中文文档；SHALL 记录来源、发行者、地区、产品、可见版本与文件身份，避免跨地区事实混用。

#### Scenario: Acquire acceptance corpus
- **WHEN** 准备真实验收语料
- **THEN** 从官方公开来源取得满足语言/类别覆盖的 6 份文字 PDF，记录版本依据；不能将文件名中的日期擅自当发行日期

#### Scenario: Candidate fails extraction
- **WHEN** 官方候选文件未通过整份质量检查
- **THEN** 记录淘汰原因并替换候选，不降低扫描完整性门槛来凑足数量

### Requirement: Validate complete answers in four language combinations

真实本地 embedding 与实际生成 API 的端到端验收 SHALL 使用 40 个有明确依据的问题，四种组合各 10 个，每组至少 9 个获得正确、完整且有原文引用的回答。单纯检索命中或替身模型通过 SHALL 不代替该门槛。

#### Scenario: Per-group success gate
- **WHEN** 验收某语言组合的 10 个可回答问题
- **THEN** 至少 9 个完整正确且有引用才通过；无依据拒答、局部回答或只有检索结果均不算该正例成功

#### Scenario: Actual services unavailable
- **WHEN** 缺少可用本地 embedding 权重或真实生成 API 配置
- **THEN** 对应实际验收保持未完成，不将合成或替身测试结果算作通过

### Requirement: Zero critical factual and provenance errors in acceptance

固定验收集中已输出答案的关键金额、期限、否定、适用条件与引用归属 SHALL 不出现错误，范围外证据 SHALL 为零；达到正例通过数量 SHALL 不豁免任何关键事实错误。

#### Scenario: Nine successes and one wrong deadline
- **WHEN** 某组九个正例通过，但另一输出错误断言期限
- **THEN** 整体验收仍不通过，不能把关键事实错误视为允许的一次失败

### Requirement: Verify negative cases and interaction scenarios

真实验收 SHALL 至少包含 12 个证据不足、离题、购买建议及保证获赔负例且全部正确处理，并另行验证逐份比较、多轮追问和删除来源行为。

#### Scenario: Negative-case gate
- **WHEN** 执行至少 12 个标注负例
- **THEN** 全部符合边界与引用规则，任一编造事实或不当承诺导致验收失败

#### Scenario: Interaction coverage
- **WHEN** 执行跨文档比较、范围切换追问和删除场景
- **THEN** 分别验证逐份覆盖、上下文重置、取消及来源清理，不用 40 个独立问题代替这些检查

### Requirement: Separate regression calibration and real acceptance

合成语料 SHALL 用于稳定回归；校准与真实验收案例 SHALL 分开，报告记录本地模型/版本、设备、批大小、内存/显存表现、生成 API 模型、索引配置、各组完整回答成功数、错误、诊断性召回与调用时延。等待时间目标 SHALL 在实际接入测量后确定，不预先承诺未测时限。

#### Scenario: Report actual usability
- **WHEN** 准备宣称版本可供个人实际使用
- **THEN** 提供真实本地模型与生成 API 的完整联合验收及性能记录；合成样本通过不能替代真实文档验收，Recall@5 仅作为检索诊断指标
