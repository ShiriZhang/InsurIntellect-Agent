# Spec Delta

## Purpose

让单用户明确指定每次问答的文档范围，并区分普通相关性问答、逐文档比较和同范围多轮追问。该能力约束所有检索、生成、引用及历史，防止默认全库查询、比较遗漏、上下文串用或删除来源继续成为证据。

## ADDED Requirements

### Requirement: Require explicit document selection

界面 SHALL 提供独立于引用查看器的文档选择，仅允许选择可提问文档；首次无已保存有效选择或用户清空选择时 SHALL 阻止提交，不自动查询全库。

#### Scenario: Empty initial selection
- **WHEN** 文档库已有可提问文档但没有用户保存的选择
- **THEN** 界面要求明确选定文档后才能提问，不默认全选

#### Scenario: Explicitly clear all selections
- **WHEN** 用户取消全部选择
- **THEN** 页面保持空范围并提示先选文档，不恢复隐式全库范围

### Requirement: Keep selection unchanged after upload

新上传文档进入可提问状态时，系统 SHALL 刷新列表并提示用户可选，但 SHALL 不自动追加、替换或扩大当前选择。

#### Scenario: New C becomes ready
- **WHEN** 当前选择 A、B，后台 C 完成入库
- **THEN** 下一问仍只查询 A、B，只有用户手动选择才加入 C

### Requirement: Validate scope for ask and batch APIs

问答和批量查询 API SHALL 要求显式非空文档 ID 列表，去重后校验全部 ID；省略、null、空列表、未知、未就绪、删除中或已删除来源 SHALL 返回明确范围错误，不回退全库或仅使用有效子集。

#### Scenario: Missing null or empty scope
- **WHEN** ask 或 batch 请求省略文档范围、传 null 或空列表
- **THEN** 在生成前返回范围校验错误，不执行隐式全库问答

#### Scenario: Invalid ID among valid IDs
- **WHEN** 请求同时选择有效文档与未就绪或已删除文档
- **THEN** 整个范围被拒绝，指出问题来源，不静默缩成剩余子集

#### Scenario: Duplicate IDs
- **WHEN** 请求重复包含同一有效文档 ID
- **THEN** 按一份文档处理并返回去重后的实际范围

### Requirement: Enforce scope across all evidence paths

系统 SHALL 在检索前、最终证据筛选与生成前执行同一已发布文档范围；过滤失败 SHALL 禁用相关路径或诚实降级，不能无范围重试。

#### Scenario: Out-of-scope higher-ranked policy
- **WHEN** 只选择等待期 90 天的 A，未选 B 的 180 天条款匹配更高
- **THEN** 上下文、答案和引用只使用 A

#### Scenario: Many lexical distractors
- **WHEN** 范围外有超过 50 个高排名片段而范围内存在有效依据
- **THEN** 范围内候选仍参与检索，不因先截断全库结果而遗漏

#### Scenario: Scope filter failure
- **WHEN** 某检索路径无法限制候选范围
- **THEN** 不使用该路径未过滤内容生成答案或公开引用

### Requirement: Distinguish ordinary QA from comparison

普通问答 SHALL 在选定集合内寻找相关依据而不承诺逐份覆盖；明确比较 SHALL 对每份所选文档分别检索，分别呈现有依据事实或依据不足，不把不同保单合成单一合同结论。

#### Scenario: Ordinary clause question
- **WHEN** 用户以普通问答询问某术语含义
- **THEN** 系统引用选定范围内足够相关的证据，不声称已经逐份审阅全部文档

#### Scenario: Explicit waiting-period comparison
- **WHEN** 用户要求分别比较 A、B 的等待期，A 为 90 天、B 为 180 天
- **THEN** 分别呈现 A 与 B 的期限和引用，任一份排名较低也不被另一份挤出覆盖范围

#### Scenario: One document lacks evidence
- **WHEN** 比较 A、B、C，只有 A、B 存在足够依据
- **THEN** 可回答 A、B，并明确列出 C 未找到依据；不编造 C，也不把回答冒充全部文档都有证据

### Requirement: Resolve comparison intent visibly

系统 SHALL 自动识别明确比较意图、显示当前问答方式并允许手动选择普通问答或逐文档比较；意图无法明确时 SHALL 先澄清而不默默选择覆盖合同，手动方式优先于自动判断。

#### Scenario: Natural-language comparison
- **WHEN** 用户问“这几份的等待期分别多久”
- **THEN** 系统使用逐文档比较并显示该方式

#### Scenario: Ambiguous intent
- **WHEN** 在多份文档下的问题无法判断是一般解释还是逐份比较
- **THEN** 系统请求用户澄清，不生成暗含完整比较承诺的答案

#### Scenario: Manual override
- **WHEN** 用户明确选择普通问答或逐文档比较方式
- **THEN** 使用该方式，自动识别不能暗中覆盖

### Requirement: Limit comparison to five documents

系统 SHALL 将一次逐文档比较限制为最多 5 份文档；超过时 SHALL 要求缩小范围，不静默截取前 5 份。普通问答 SHALL 不因选择超过 5 份而被当作比较超限。

#### Scenario: Six-document comparison
- **WHEN** 用户明确或自动识别为比较且选择 6 份
- **THEN** 生成前提示最多比较 5 份，要求重新选择

#### Scenario: Broader ordinary search
- **WHEN** 用户以普通问答在 6 份有效文档中寻找相关条款
- **THEN** 系统可以检索该范围，不承诺逐份覆盖

### Requirement: Freeze each request scope

系统 SHALL 在接收问题时固定范围、索引快照和已解析方式，并记录至历史；之后改变选择或发布新文件 SHALL 不扩大正在进行的请求。来源删除时 SHALL 按取消规则中止相关请求。

#### Scenario: Change selection during a stream
- **WHEN** 提交基于 A 的问题后改选 B
- **THEN** 已有请求仍绑定 A，下一问使用 B，A 的迟到结果不进入 B 的当前上下文

#### Scenario: New document published mid-request
- **WHEN** 请求开始后 C 完成入库
- **THEN** C 不进入该请求的证据或选择，是否供后续使用由用户明确决定

#### Scenario: Source deleted mid-request
- **WHEN** 被请求范围包含的文档开始删除
- **THEN** 相关请求中止，不能依旧快照继续作为正常答案完成

### Requirement: Support follow-ups within the same scope

系统 SHALL 在同一会话且选定文档集合不变时利用前文理解追问；范围改变或相关来源删除 SHALL 重置或失效相关上下文，重新选择旧范围也不自动复活旧会话状态。

#### Scenario: Follow up on an exception
- **WHEN** 询问等待期后，在同一范围追问“那意外事故呢？”
- **THEN** 系统理解所指问题并在当前文档中重新检索相应条件

#### Scenario: Switch scope and switch back
- **WHEN** 从 A 改为 B 后又改回 A
- **THEN** 每次切换开始新的当前上下文，不把 B 的事实或先前 A 会话自动带入

### Requirement: Reground every follow-up

前文 SHALL 只用于消解指代与问题意图，每次回答 SHALL 重新检索有效原文并引用；旧助理答案、被删除来源或无充分证据的重写结果 SHALL 不成为新事实依据。

#### Scenario: Prior answer contains a number
- **WHEN** 前文提到某期限，本轮有效条款不足以支持该期限
- **THEN** 系统不从旧答案复制该期限作为有依据结论，而明确说明依据不足

#### Scenario: Follow-up intent is ambiguous
- **WHEN** “它呢”无法在当前会话与范围中唯一消解
- **THEN** 系统要求澄清，不猜测产品、文档或条件

### Requirement: Keep transport semantics consistent

流式与非流式 SHALL 共用范围、方式、比较上限和前文隔离规则；范围及已识别比较超限错误 SHALL 在打开答案流前返回。界面 SHALL 默认流式，现有主要答案字段 SHALL 保留并扩展范围、语言和方式信息。

#### Scenario: Invalid streaming request
- **WHEN** 流式请求没有有效范围或比较超过 5 份
- **THEN** 生成前返回相应校验错误，不先输出答案再报告错误

#### Scenario: Equivalent valid requests
- **WHEN** 同一有效问题分别通过两种传输提交
- **THEN** 两者具有相同的实际范围、方式和证据约束，最终引用遵循相同规则
