# 文档库数据与迁移（任务 2.1–2.3）

第 2 批交付持久化模型、schema 迁移和受控原文件登记服务；第 3 批接入解析、持久 worker 和共用 CLI；第 4 批接入 [一致索引发布与恢复](index-publication.md)。操作见 [共用入库说明](document-ingestion.md)，上传/删除 HTTP 接口仍由后续批次接入。登记成功仅表示 `queued`，两路索引完整且提交指针后才 ready。

## 初始化当前空库

在项目根目录、已安装项目依赖的 Python 环境运行：

```powershell
python -c "import asyncio; from app.core.database import init_db; asyncio.run(init_db())"
```

数据库位置遵守 `DATABASE_URL_ASYNC`，未配置则从 `DATABASE_URL` 推导。首次运行创建空表，重复运行不重置数据。准备好本地模型缓存后，可通过 `python -m scripts.simple_ingest --pdf-dir samples` 登记、准备并发布样本；同一入口使用 `DocumentStorage.accept` 和持久入库 worker。完整索引发布成功后，才能按返回的整数 ID 查询。

## 已有库升级与重建

1. 停止应用和所有入库/索引写入，记录数据库、PDF、processed、Chroma 和模型 profile 的实际配置路径。
2. 备份 SQLite（使用 SQLite backup API，或停写并关闭连接后连同 WAL/SHM 一起保存），并备份对应原文件、索引、身份映射和删除日志；不要只备份 `.db` 而遗漏尚未 checkpoint 的 WAL。
3. 用上述初始化命令执行版本化补列迁移。迁移不删除旧正文、答案或旧 `status`，失败会向调用方抛出，不忽略错误。`schema_migrations` 记录已完成版本；重复执行保持结果不变。
4. 旧 `status=active` 继续表示旧业务状态，新增 `processing_status=pending`，不会自动成为 `ready`。旧 chunk 的 `attempt=NULL` 表示身份尚未版本化。旧历史的 `scope_status=unknown`、范围/epoch 均为空，仍可作为存档读取，不能根据引用猜出范围，也不能用于新追问。
5. 从仍存在且未删除的原 PDF，经后续共用入库入口重建，保存旧/新身份映射；缺少原文件应明确记录不可重建。无 profile 的旧中文/hash 向量不可直接发布。不要通过恢复 manifest、磁盘扫描或把 `active` 改成 `ready` 跳过重建。
6. 恢复/回退时先保存升级后新增文件及最新删除记录，恢复匹配的应用/schema/索引后，优先重放删除 tombstone 和清理，再开放查询。不要仅恢复旧备份而复活已删除资料。应用不会遍历清除用户自行保存的外部备份。

## 状态与身份

- `status` 保留旧业务语义；`processing_status` 使用 pending/queued/parsing/indexing/ready/failed；`deletion_status` 使用 none/deleting/failed/deleted。删除失败仍不可查询。
- `attempt` 是入库尝试号，`published_attempt`、`published_fingerprint` 和 `index_commit_id` 标识已发布资源；只有完整发布且没有删除状态的来源可提问。`IndexPointer` 指向当前提交，`IndexCommit` 保存精确的文档/尝试配对。
- `SessionScope` 以 `(session_id, epoch)` 保存范围；返回旧集合也必须新建 epoch。历史单独保存请求范围、语言、方式、提交身份与来源状态；本批只建立持久化合同，实际范围切换由任务 6.3 接入。
- `DeletionTask` 保存删除 epoch、阶段、已完成步骤和失败原因；后续删除服务完成所有可读来源清理后才能标 deleted。tombstone 的 ID 不复用，不含来源正文。

## 受控原文件登记

`DocumentStorage.accept(binary_stream, original_name)` 是阻塞服务，异步上传入口必须在线程/受控执行器调用。它遵守 `PDF_STORAGE_PATH` 和 `UPLOAD_MAX_BYTES`（默认 20 MiB），在同目录暂存并计算 SHA-256，flush/fsync 后原子替换到 `<document_id>/original-<随机 UUID>.pdf`，最后提交 SQLite。每次新写入使用独立文件身份，失败请求的清理不会误删复用同一文档 ID 的后续上传。原名只作展示，路径分隔符被去除，文件读取从数据库 ID 和已登记的受控路径解析，兼容此前的 `original.pdf`，拒绝外部路径和删除来源；PDF 有效性/扫描/页数检查由后续解析服务完成。

内容拥有权用 `ActiveContentClaim` 的哈希主键和 SQLite `BEGIN IMMEDIATE` 跨连接串行化。queued/parsing/indexing/ready 同内容复用 ID 与尝试，不再建任务；同名不同内容建立独立 ID。failed 重传保留 ID、递增 attempt、清除旧 SQLite 片段及失败字段，旧任务标 superseded；已有受控原文件直接复用、不替换，提交失败仍保留原文件和原尝试状态；原文件缺失时才创建新的独立路径。旧向量 ID 合并保存在新尝试 `cleanup_json`，由后续索引 worker 在发布前幂等清理，重试失败不会丢掉待清理身份。所有清理只针对本 ID。

deleting 或删除清理失败时拒绝重传；deleted 后同内容分配新 ID，即使旧内容 claim 尚在也会在事务内释放它。旧历史和 tombstone 保留原 ID，不能指向新上传。此服务不执行删除流程；任务 7 负责清理后才标 deleted。

读取、写入、原子替换或提交失败会回滚登记并移除本次暂存/未提交原文件；整个进程被强杀可能留下孤立文件，但它没有有效数据库/索引提交身份，不会被列为 ready。不得通过扫描目录把这些文件自动注册或恢复来源；后续恢复/删除服务按内部身份处理临时残留。旧库可读路径也不会被当作新受控原文件路径。

## 本批验证

2026-11-01：新增 26 项临时 SQLite 测试覆盖模型、旧表补列/重复迁移、未知历史隔离、并发内容去重、重试、删除后新身份、读写/提交中断和路径边界；其中 3 项确定性交错测试验证失败清理不会误删后续上传，4 项验证新旧文件名的原文件复用和失败回滚。交错测试在修复前全部失败，修复后通过；全量回归 258 项通过。`ruff check . --select E9,F63,F7,F82` 与 OpenSpec 严格校验通过。验证不调用真实生成 API 或下载模型，不能替代后续真实模型与完整入库/删除场景验收。
