# Spec Delta

## Purpose

为本地单用户提供持久化保险 PDF 文档库，支持多份文字文档上传、状态查询、原文件查看与删除。该能力明确完整可提问、处理失败和来源已删除的区别，使用户不会把扫描件、不完整索引或已移除资料当作有效问答证据。

## ADDED Requirements

### Requirement: Upload multiple text-based PDFs

系统 SHALL 允许单用户连续上传多份可提取中英文文字的 PDF，为已接受文件返回稳定文档 ID 和处理状态，并在后台完成入库。

#### Scenario: Upload Chinese and English documents
- **WHEN** 用户提交一份中文 PDF 后又提交一份英文 PDF
- **THEN** 两份文件各有身份和处理状态，无需等待前一份入库完毕才能接受后一份

#### Scenario: Invalid upload
- **WHEN** 输入不是有效 PDF、超过公开文件大小限制，或属于无法读取的加密/损坏文件
- **THEN** 请求被拒绝或任务明确失败，不将文件标为可提问

### Requirement: Enforce page limits and disclose design capacity

系统 SHALL 拒绝超过 200 页的单份 PDF，并以 50 份作为累计文档设计与验收规模；超过 50 份 SHALL 给出提示但允许继续上传，不自动删除资料或将其当作总量硬上限。

#### Scenario: Page limit
- **WHEN** 用户上传 201 页 PDF
- **THEN** 系统明确说明 200 页限制，文档不能进入可提问状态

#### Scenario: Exceed verified library scale
- **WHEN** 上传使尚未删除的文档库超过 50 份
- **THEN** 页面提示超出已验证规模，仍接受符合单文件限制的上传

### Requirement: Reject incomplete scanned extraction as a whole

系统 SHALL 检查逐页文字提取质量；扫描页、明显乱码或无法读取的非空内容页被发现时，整份文档 SHALL 失败并显示页码和原因，不发布部分内容。纯空白页 SHALL 不被误判为扫描件。

#### Scenario: Scanned PDF
- **WHEN** 文档仅含无有效文本层的扫描内容
- **THEN** 系统提示暂不支持扫描件，整份文件不可提问

#### Scenario: Mixed scanned and text pages
- **WHEN** 100 页文档中 97 页可提取，另 3 页为无法读取的扫描内容
- **THEN** 系统列明受影响页并拒绝整份发布，不将 97 页宣称为完整文档

#### Scenario: Blank pages and decorations
- **WHEN** 文档含纯空白页或不影响有效正文提取的页眉标志
- **THEN** 系统不只因无文字空白页或存在图片而拒收，引用仍使用原 PDF 页码

### Requirement: Stable identity independent of filenames

系统 SHALL 区分同名不同内容文件并基于内容去重；正在处理或可提问文档的相同内容重复上传 SHALL 复用已有身份，已删除来源的重新上传 SHALL 使用新身份，不恢复旧引用。

#### Scenario: Same filename different contents
- **WHEN** 两份 PDF 名称相同但内容不同
- **THEN** 原文件、片段和引用互不覆盖，可按文档 ID 分别选择

#### Scenario: Duplicate active document
- **WHEN** 重复上传内容相同且正在处理或可提问的 PDF
- **THEN** 返回已有文档状态，不建立重复任务或索引

#### Scenario: Retry a failed document
- **WHEN** 重新上传同一失败文件
- **THEN** 对该文档启动新的幂等尝试，清除旧尝试残留而不改变其他文档

#### Scenario: Re-upload deleted source
- **WHEN** 完成删除后重新上传相同内容
- **THEN** 它作为新文档处理，旧答案仍显示原来源已删除

### Requirement: Persist observable document states

系统 SHALL 持久化处理阶段、失败原因与删除进度；列表和详情 SHALL 区分排队、解析、索引、可提问、失败及删除状态，不以磁盘文件存在推断可提问。

#### Scenario: Upload processing
- **WHEN** 文件已保存但必要索引尚未发布
- **THEN** 页面显示处理中且不可选择查询，完成发布后才显示可提问

#### Scenario: Orphan original file
- **WHEN** 磁盘存在没有已提交索引身份的 PDF
- **THEN** 系统不会把它直接列为可提问文档

### Requirement: Publish complete indexes without restart

系统 SHALL 仅在语义与关键词检索资源均完整后发布可提问状态，并让下一次请求无需重启即可查询；新增文档发布失败 SHALL 不暴露其残留片段或破坏先前可用文档。

#### Scenario: Newly ready document
- **WHEN** 文档显示可提问
- **THEN** 其语义和关键词检索均已可用，用户可手动选择后立即提问

#### Scenario: Partial indexing failure
- **WHEN** 新文档的任一必要索引构建失败
- **THEN** 文档保持不可提问，旧文档仍可用，失败片段不进入答案

### Requirement: Recover committed document state after restart

系统 SHALL 在重启后恢复已提交索引、重新排队中断的入库并继续未完成删除；恢复 SHALL 幂等，不能重复片段或重新发布已开始删除的来源。

#### Scenario: Ready and interrupted uploads
- **WHEN** 服务在一份文档可提问、另一份正在入库时重启
- **THEN** 已提交文档恢复可查询，中断任务继续处理而不直接升级为可提问

#### Scenario: Interrupted deletion
- **WHEN** 原文或旧索引尚未全部清除时服务重启
- **THEN** 该来源仍不可查询并继续删除，不从历史快照或磁盘扫描恢复为可用

### Requirement: View original PDFs by identity

系统 SHALL 按稳定文档身份展示原 PDF 与引用页，删除中的或已删除来源 SHALL 不再提供原文件或原文摘录。

#### Scenario: Same-name citation
- **WHEN** 点击两份同名文档之一的有效引用
- **THEN** 查看器打开该 ID 对应的原文件和页码

#### Scenario: Deleted file URL
- **WHEN** 打开来源已删除的历史引用或旧文件地址
- **THEN** 系统明确提示来源已删除，不返回缓存原文或其他同名文件

### Requirement: Reuse ingestion across entry points

系统 SHALL 对网页上传和支持的 CLI 导入使用相同身份、解析质量、索引兼容和可提问判定，使演示样本也成为可显式选择的文档。

#### Scenario: CLI sample import
- **WHEN** 用户通过 CLI 导入预置样本
- **THEN** 样本出现在同一文档库中，可按 ID 选择查询和查看引用

### Requirement: Remove source material when deleting a document

系统 SHALL 支持删除文档，清除原 PDF、提取正文、向量、关键词索引及历史引用原文摘录，也清理系统管理的旧快照与缓存中的可读来源副本；完成前 SHALL 不宣称删除成功。

#### Scenario: Completed deletion
- **WHEN** 删除任务显示成功
- **THEN** 该来源不能从文件接口、正文存储、检索索引、历史引用卡或系统管理的旧快照读取

#### Scenario: Cleanup failure
- **WHEN** 任一必要清理失败
- **THEN** 页面显示删除尚未完成及可诊断原因，文档保持不可查询并可继续恢复清理

### Requirement: Cancel source-dependent work during deletion

系统 SHALL 在开始删除时阻止该文档的新查询，取消所有依赖它的进行中回答与入库工作，并使相关前文不再可供追问；已取消任务 SHALL 不重新写回来源片段或发布正常终态。

#### Scenario: Delete during streamed comparison
- **WHEN** A、B 的比较正在输出，用户删除 A
- **THEN** 相关回答中止并说明来源删除，不能静默变成只比较 B 的正常答案

#### Scenario: Delete while upload is running
- **WHEN** 用户删除尚在入库的文档
- **THEN** 入库停止发布，迟到的处理结果不能让该文档重新变为可提问

### Requirement: Preserve answer history with deleted-source markers

系统 SHALL 保留已有答案正文和必要来源标识，将引用标为“来源已删除”，移除引用卡原文并禁止该来源及依赖它的旧上下文用于后续追问；保留的答案正文 SHALL 不被当作重新检索的事实证据。

#### Scenario: Read historical answer after deletion
- **WHEN** 用户打开曾引用已删除来源的答案
- **THEN** 已有答案正文仍可阅读，失效引用保留必要标识但无原文摘录、原文件或有效来源链接

#### Scenario: Follow up after deletion
- **WHEN** 用户针对包含已删除来源的旧答案继续追问
- **THEN** 系统不从旧原文或旧答案恢复来源事实，要求有效文档范围并按新上下文重新检索
