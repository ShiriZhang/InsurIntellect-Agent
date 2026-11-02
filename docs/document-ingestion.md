# 共用 PDF 入库（任务 3.1–3.4）

网页上传的服务入口与 CLI 都使用 `DocumentIngestionService.submit(stream, original_name)`，由第 2 批的受控存储分配整数 ID、内容去重和尝试号。上传 HTTP 端点由第 5 批接入；本批不新增该端点。

## 首次样本与实际文件

在项目根目录和已安装依赖的 Python 环境中运行：

```powershell
python -m scripts.prepare_models --download
python -m scripts.generate_sample_corpus
python -m scripts.simple_ingest --pdf-dir samples
python -m scripts.simple_ingest --pdf-dir D:/Policies --no-wait
```

先按 [模型配置](model-configuration.md) 配置固定的 `LOCAL_EMBEDDING_MODEL`、revision、cache、device 与 batch size。应用和 CLI 默认只读取已缓存权重，不自动下载、不使用 hash 或远程 embedding。`GENERATION_*` 独立配置；入库不构造生成客户端、不发送正文到生成服务，不需要生成 API 可用。

省略 `--pdf-dir` 时使用项目 `samples/`；支持直接运行 `python scripts/simple_ingest.py`。`--no-wait` 只保存任务，随后应用启动的 worker 会处理；默认等待发布终态，`--timeout`（默认 300 秒）只限制 CLI 等待，不是性能承诺。超时不丢掉持久任务，CLI 自己启动的 worker 会请求中断并在下次恢复。

CLI 返回 JSON：`documents` 给出 ID、attempt、处理/任务状态、页数、语言、失败原因/页码与实际设备诊断；`prepared=true` 代表本地候选已准备，`ready=true` 才代表已发布可提问。默认等待失败/取消/超时返回非零退出码；`--no-wait` 成功登记返回零。

准备阶段使用 `task_status=staged`、`processing_status=indexing`，将原文切片、完整模型指纹和候选向量存入 SQLite。第 4 批已接入 [Chroma/关键词一致发布](index-publication.md)：正常 worker 与 CLI 随后验证两路资源并提交 ready。默认等待模式仅在发布成功时返回成功退出码；入队模式仍只保证登记。CLI 与上传服务入口遵守相同判定，不写旧 manifest 或覆盖旧 BM25 文件冒充发布。索引 ready 不代替真实生成 API 问答验收。

`--reset` 已移除，传入会在任何登记/删除操作前明确拒绝。原 `--persist-dir`、`--collection` 不再使用；数据库和受控存储遵守 `DATABASE_URL[_ASYNC]`、`PDF_STORAGE_PATH`，后续索引发布遵守 `CHROMA_PERSIST_DIRECTORY`、`PROCESSED_DATA_PATH`。不要直接清空目录来重建；保留身份映射和删除记录，按 [迁移说明](document-library-storage.md) 处理旧库及无指纹旧索引。

## 逐页质量与切块

单文件默认最大 20 MiB，可用 `UPLOAD_MAX_BYTES` 配置；200 页是固定硬限制。检查 PDF 标识、打开失败、修复损坏和加密；逐页检查明显乱码、无有效文本层的大面积扫描图及显著非文字绘图。有任何受影响页就整份失败，记录原页码，既不提交部分证据也不标 ready。纯空白、边框和小型标志图可以跳过，原 PDF 页号不压缩。

这是保守的文字 PDF 质量门槛，不提供 OCR，也不宣称能理解任意图表或识别所有视觉乱码；大幅图片只有页眉/页脚文字时仍会拒绝，需换成完整可读的文字 PDF。

切块先按原页及中英条款/章节编号分段，保留源切片与字符偏移、标题、前段上下文和页内顺序。核心切片连接后与逐页提取原文一致；正文不翻译、不静默截断。每份输入（包括标题、上下文、前缀与特殊 token）都用实际本地 tokenizer 检查；同时按 UTF-8 字节数保守限制单块生成输入，并预留配置的提示/回答预算。请求总证据预算仍由后续问答准备负责。

`CHUNK_SIZE` 是字符上限，`CHUNK_OVERLAP` 是附带原文上下文的字符数；上下文过长时可缩短辅助重叠，核心正文不会跳过。配置预算不足以容纳标题及一个字符时明确失败。ID 由文档 ID、attempt、原页码、页内顺序和正文哈希派生；切块版本、预算和策略写入 embedding profile，策略变化要求新索引，不复用旧空间。

## 队列、取消与恢复

`IngestionAttempt` 是任务事实来源，`INGEST_QUEUE_SIZE` 只限制内存提示队列；提示队列满时 SQLite 任务仍会被轮询处理。单 worker 在独立线程串行执行解析、切块和本地编码，耗时操作期间不持有 SQLite 写锁，允许继续登记其他文件。

同一数据库的 `.ingestion.lock` 文件使用操作系统互斥锁，网页应用和 CLI 不会同时执行入库；进程退出自动释放，锁文件本身可保留。未取得锁的服务保留待机线程，原持有者退出后自动接管，仅在真正取得锁后恢复和执行任务。其他进程仍可入队和读取状态。应用空库启动不加载/下载模型，也不调用生成 API。

schema v2 幂等补齐候选向量、源偏移和尝试检查点，v3 增加已提交资源身份，均保留已有数据。每批编码后检查数量、维度、有限数值、指纹与取消/删除 epoch，再在短事务中写入。模型设备回退沿用第 1 批同模型策略，并在诊断中报告实际设备和提示。

`cancel(document_id)` 持久化取消并移除该尝试的候选片段；删除流程可复用此钩子。每个写入阶段都复核文档、attempt、删除状态与 epoch，迟到的解析/编码结果不再写回。原 PDF 的完整删除仍由第 7 批负责。

应用关闭等待 `INGEST_SHUTDOWN_TIMEOUT` 秒。Python 不能安全强杀正在执行的原生编码；超过等待后保留互斥锁，待计算返回，丢弃迟到结果并重新排队，随后释放资源。重启先恢复已提交索引、重放删除记录，再恢复当前且未删除的 parsing/indexing 尝试，重新准备同一 attempt，清理该尝试检查点后写入确定性片段，不产生重复身份。已 staged 候选无需重新运行模型即可继续发布，deleting/deleted 永不自动恢复为 ready。

本批无网络测试使用临时数据库、合成 PDF 和明确标注的本地模型替身；不替代真实模型质量、正式六份语料或完整问答验收。
