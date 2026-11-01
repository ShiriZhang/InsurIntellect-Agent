# 双语词项、检索排名与证据分数（任务 1.4）

CLI 入库与问答 BM25 检索共用 `app/services/retrieval_text.py`。原 PDF、片段正文和公开引用保留原文；词项只用于召回，不替换原文事实。

## 共用词项规则

检索副本使用 Unicode NFKC、大小写 casefold 和空白规范化。中文使用独立 jieba 分词器（HMM 关闭），英文保留词及产品标识。产品名、数字、金额、小数、负数、百分比、日期和否定词不被当作停用词删除。例如 `AlphaCare-X2`、`2026-10-25`、`10,000.50`、`-5.25`、`80%`、`非`、`不予`、`not`、`unless` 仍参与检索。

版本化术语表为等待期、免赔额、责任免除、重大疾病等增加 `term:...` 召回提示。中文“等待期”和英文 `waiting period` 可共享词项，但各自原词仍保留。英文术语须在词边界命中，不把 `SuperClaimsPlan-X2` 的内部子串当作 `claims`。这些提示不证明不同产品、地区、合同或否定条件等价，不参与答案事实推断。

BM25Plus 对未命中文档也可能给出正的基线分数。因此检索会额外检查实际词项重叠；没有任何命中词项的片段不贡献关键词排名。

## 分数合同

| 字段 | 含义及用途 |
| --- | --- |
| `metadata.raw_distance` | cosine 距离，原值保留，越小越相关；不是相似度 |
| `metadata.evidence_score` | `clamp(1 - raw_distance, 0, 1)`，用于证据门槛 |
| `metadata.vector_score` | 兼容字段，与 cosine 证据分数相同 |
| `similarity_score` | API 展示的 cosine 证据分数，与门槛使用同一尺度 |
| `metadata.bm25_score` | 关键词检索原始排名分数，不是证据分数 |
| `metadata.rrf_score` | 倒数排名融合分数，只排序，不用于回答门槛 |

向量检索正常路径和底层 collection 回退路径都按同一公式转换距离。cosine 小于零的相似性截为 0；缺失、NaN、无穷等无效输入不能产生有效证据分数。合法零分保留为零，不因另一个排名值较高而被覆盖。

BM25 独有候选需要使用本轮问题向量与对应片段的同空间存储向量计算 cosine 距离，不能直接使用 BM25/RRF 分数。向量缺失、维度不符或零范数时，片段标记为 `evidence_status=unavailable`，不能因此进入生成。实际 embedding 指纹与索引空间的一致加载由 4.1 接入；本项不把未验证的旧索引宣称为已兼容。

流式和非流式问答都在生成前逐片段检查 `SIMILARITY_THRESHOLD`。仅有高关键词/RRF 排名而缺乏足够 cosine 证据时，不会开始生成；存在其他有效证据时，低分候选也不会混入生成上下文。引用筛选中的词项提示只用于排名平局，不增加证据分数。

原始排名值、`ranking_details.original_similarity` 或 `final_score` 不再作为证据兜底。兼容旧 API 的纯 `similarity_score` 字段仍按其声明的相似度含义读取，但当记录带有排名信息而没有明确证据字段时，不猜测该数值的含义。显式的未知/零证据不会被其他字段替换。

保留旧字段名 `confidence_score`，RAG 主路径现在返回最佳 cosine 证据相关性，两种传输一致，不再返回固定的 0.9。它不表示答案正确概率。默认阈值仍为 0.32，实际发布值须结合固定本地模型、校准集和完整回答结果验证；本项没有用合成测试校准真实阈值。

## 旧关键词索引与重建

新 BM25 文件保存 `lexical_profile`，包含词项协议版本及摘要。摘要覆盖术语表、词项规则、Unicode/jieba 版本及基础词典内容。查询端只加载匹配该记录的关键词索引；旧文件缺标识或标识不匹配时明确禁用关键词路径并提示重建，不把新分词套在旧词项统计上。有效向量路径仍可独立检索。

先准备并验证当前模型（见 `model-configuration.md`），再从仍有效的原 PDF 在新目录重建。例如仅重建合成样本时：

```powershell
$env:PROCESSED_DATA_PATH="./data/processed/lexical-v1"
python -m scripts.simple_ingest --pdf-dir ./samples --persist-dir ./data/vector_db/lexical-v1 --collection insurance_documents
```

保留旧目录和源文件，核对新目录的模型/profile 与检索结果后再切换配置，不对唯一旧库使用 `--reset`。当前 CLI 会同时生成向量和关键词资源；自动恢复/重建、版本化发布由 4.1–4.3 完成。切换查询范围、逐文档覆盖、引用身份/顺序和双语边界的完整实现分别仍属于后续任务。

本项测试使用临时合成文档、本地向量替身和真实 BM25 算法，覆盖混排/大小写、关键文本保留、正常与回退向量路径、距离单调性、RRF 排名、零命中候选以及两种传输的门槛与生成上下文。测试不下载真实权重或调用生成服务，不代表真实跨语问答质量已验收。
