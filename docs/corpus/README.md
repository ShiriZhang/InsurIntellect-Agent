# 官方真实条款语料 v1（任务 1.5）

本清单固定六份保险公司公开条款：简体中文、英文各三份，每种语言各覆盖医疗、重疾、寿险。原 PDF 已采集到本机 `data/acceptance/official-v1/`，此目录由 Git 忽略；仓库保留可复现来源、文件身份与逐页解析报告，不把第三方条款作为项目 MIT 代码重新授权。合成样本仍保留在 `samples/`，与真实语料分开。

## 固定清单

完整 URL、采集时间、字节数、SHA-256、正文版本与定位依据见 [official-v1.json](official-v1.json)。下表中的版本是正文标识，不表示个人保单生效日期或产品仍在售。

| ID / 本地文件名 | 产品 | 语言 / 类别 | 发行者与地区 | 页数 | 正文可见版本（第 1 页） |
| --- | --- | --- | --- | ---: | --- |
| `zh-medical.pdf` | 平安住院安心短期医疗保险 | 简体中文 / 医疗 | 平安养老保险股份有限公司 / 中国内地 | 10 | 平安养老〔2018〕医疗保险041 号 |
| `zh-critical-illness.pdf` | 平安上海医保账户重大疾病保险 | 简体中文 / 重疾 | 平安养老保险股份有限公司 / 上海医保账户适用条件 | 12 | 平安养老[2018]疾病保险097 号 |
| `zh-life.pdf` | 平安团体定期寿险 | 简体中文 / 寿险 | 平安养老保险股份有限公司 / 中国内地团体 | 5 | 平安养老〔2018〕定期寿险010 号 |
| `en-medical.pdf` | Hospital Care | 英文 / 医疗 | Income Insurance Limited / 新加坡 | 14 | PL/Hospital Care/202204E |
| `en-critical-illness.pdf` | Complete Critical Protect | 英文 / 重疾 | Income Insurance Limited / 新加坡 | 75 | LHO/Complete Critical Protect/CCN1.202610.1 |
| `en-life.pdf` | DIRECT Star Term (non-renewable) | 英文 / 寿险 | Income Insurance Limited / 新加坡 | 4 | LHO/DIRECT Star Term (non-renewable)/TAN18.202510.1 |

中文来源是平安官方附件域 `resources.pingan.com`。英文附件来自 [Income 官方条款目录](https://www.income.com.sg/policy-documents-and-forms) 链接的 `assets-au-01.kc-usercontent.com`；CDN 附件的官网关联需保留，不能只凭 CDN 域名认定任意文件官方。

医疗类别包含住院津贴；Hospital Care 另有急诊意外医疗费用等责任，不将其描述为综合医疗费用报销险。中文寿险为团体合同，上海重疾有特定医保条件；跨地区、个人/团体及产品条件必须分别归属，不能因语言或类别相同推断条款等价。清单的 `region_basis` 和三组 `*_evidence` 记录正文页码及定位锚点。

没有单独确认的发布日期保留 `null`。例如英文重疾本次采集的正文为 `CCN1.202610.1`、75 页，不能用旧缓存检索摘要中的 `202510`、文件名或 URL 日期代替。版本变化须重新核对内容、哈希和后续人工题集，不能原地改日期凑齐旧清单。

## 采集与离线复核

在项目根目录、已安装现有项目依赖的 Python 环境中运行。这里只调用现有 PDF 提取入口，不下载 embedding 权重、不创建索引、不调用生成 API，也不需要 API Key。

```powershell
# 首次获取缺失文件，同时验证固定身份和逐页解析；现有文件不会覆盖。
python -m scripts.official_corpus --fetch --report data/acceptance/official-v1/acquisition-report.json

# 默认不联网，复核已经采集的文件。
python -m scripts.official_corpus --report data/acceptance/official-v1/offline-report.json

# 可选择独立本地目录；报告只包含身份和诊断，不含保单全文。
python -m scripts.official_corpus --fetch --destination data/acceptance/official-v1-copy --report data/acceptance/official-v1-copy/report.json
```

若当前环境没有数据库驱动，可沿用测试用 SQLite 环境变量 `DATABASE_URL=sqlite://`、`DATABASE_URL_ASYNC=sqlite+aiosqlite://`。现有提取脚本导入配置，但该采集命令不连接数据库、不载入模型、不发布可提问状态。

入口只接受清单中的六个受审核产品，HTTPS 官方来源和重定向受域名限制，单文件最多 20 MiB、请求超时 45 秒。下载字节必须符合固定长度、SHA-256 和 PDF 头；已存在文件只复核，绝不自动覆盖。默认执行离线检查，返回码 0 表示六份全部通过，任一缺失或失败返回非零，并在报告中列明原因。

官网文件可能被替换或撤下。若出现 `official_bytes_changed_review_required`，保留现有已核验文件与报告，在独立目录重新审核官方候选，确认正文版本、来源、地区、语言和完整解析后，再明确更新清单及相关题集。不要跳过哈希、将新文件改名成旧版本、改用非官方镜像或降低解析标准。未通过的候选不能补足六份覆盖；替换候选的原因要进入清单的 `rejected_candidates`。

## 本次检查结果及状态边界

[official-v1.parse-report.json](official-v1.parse-report.json) 记录此次实际六份、共 120 页的检查结果，绑定清单 SHA-256 与 PyMuPDF 版本。检查使用项目现有 `scripts.simple_ingest._extract_pages`，没有用独立解析器的成功结果冒充项目成功：

- PDF 未加密，页数和文件哈希匹配；每一个原始页号均出现在项目提取结果中。
- 每页有可读字母/数字，未发现替换字符、私用区或未分配字符；报告保留字符数和逐页文本哈希。
- 正文的产品、发行者、地区依据和可见版本锚点与清单一致。
- 中文原文检查为简体，英文原文检查为英文；未将繁体候选转换成简体来充数。脚本标记是诊断规则，配合固定文件的正文审核，不宣称通用语言分类能力。

本语料特意选择每页均有可提取内容的文件；漏页、图像扫描页、乱码或明显不足的文字会使本轮语料检查整份失败，不靠跳过失败页获得成功。此处的每页文字检查不能替代 task 3.1 将建立的完整上传质量服务，尤其不保证理解任意混合图表或识别所有局部扫描内容。

`collected_project_parse_checked` 只表示这份固定文件已采集且通过当前项目解析检查；报告中的 `application_ready` 始终为 `false`。它不表示已经建立向量/关键词索引或进入文档库，不表示真实模型或问答质量已验收。后续须经过完整入库质量服务与索引发布，才能成为可提问文档；不要用当前 `simple_ingest` 直接导入这些真实条款，其演示产品/日期元数据将在 task 3.4 中移除。

task 1.6 的核对题集、独立校准集及评分规则见 [evaluation/README.md](../evaluation/README.md)；task 10.2–10.4 才验证真实本地模型、生成 API、交互与性能。无网络单元测试使用临时合成 PDF，覆盖清单完整性、非繁体覆盖、来源/重定向限制、哈希变更、版本/来源锚点、漏页、传统字形拒绝和默认离线行为，不能计作真实问答验收。
