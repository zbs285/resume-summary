# Resume Summary

一个面向 macOS 的批量简历事实提取工具。它只整合简历中明确写出的：

- 姓名；
- 本科、硕士、博士学校及毕业 / 预计毕业时间；
- 工作或实习经历的公司与 Title。

工具不会评价候选人，也不会汇总项目经历、工作内容、论文、奖项或自我评价。

## 下载与首次配置

1. 在仓库右侧 **Releases** 下载 `Resume-Summary-macOS.zip` 并解压。
2. 首次使用双击 `Configure OpenRouter Key.command`，按照提示创建并保存 OpenRouter API Key。Key 只保存在 macOS 钥匙串。
3. 将 `Resume Summary.app` 移到“应用程序”文件夹。

## 每次怎么用

1. 双击 `Resume Summary.app`。
2. 浏览器会自动打开本地操作页面。
3. 任选一种方式添加简历：
   - 拖入多个 PDF；
   - 拖入一个文件夹；
   - 点击“选择 PDF”或“选择文件夹”；
   - 粘贴本机 PDF / 文件夹路径。文件夹会递归查找 PDF。
4. 点击“开始汇总”，页面会显示当前文件、进度和免费候选模型。
5. 完成后下载 PDF、CSV、HTML 或 JSON，也可以点击“在 Finder 中打开”。
6. 使用完毕点击页面右上角“退出工具”。

结果默认保存在：

```text
~/Downloads/Resume_Summary_时间
```

## 工作原理

```text
PDF
  → 本机提取文字
  → 删除邮箱、电话和链接
  → 扫描全文并定位姓名 / 教育 / 工作实习区块
  → 长内容按 12,000 字符左右分块
  → OpenRouter 免费模型输出结构化 JSON
  → 用 PDF 原文校验、去重和合并
  → 生成 PDF / CSV / HTML / JSON
```

几个关键设计：

- **不截断固定前缀**：程序扫描完整简历。常规简历通常只请求一次；很长的相关区块才会分块，最多 20 块。
- **动态免费模型**：每批开始时实时读取 OpenRouter 模型目录，只选择当前 prompt / completion 价格为 0、支持文本和结构化输出、上下文至少 16K 的模型。
- **零价格硬限制**：每次请求设置 `provider.max_price=0`。如果免费端点不可用，请求会失败，不会自动切换到收费端点。
- **稳定与回退**：同一批次优先固定使用一个模型；遇到限流或故障时自动尝试下一个合格的免费模型。
- **本机缓存**：相同 PDF 按 SHA-256 命中缓存，避免重复消耗免费请求次数。
- **原文核验**：学校、毕业时间、公司和岗位会与 PDF 提取文字交叉核对，并标记需要人工复核的结果。

## 隐私

- PDF 文件本身不会上传；工具先在本机提取文字。
- 邮箱、电话和 URL 会在请求模型前删除。
- 请求使用 `provider.data_collection=deny`，排除会收集或用于训练所传数据的路由。
- 请求使用零价格上限，避免误用收费端点。
- 姓名、学校和工作经历仍然属于个人信息，请确保你有合法的处理权限。

如需严格零数据留存，可在启动前设置：

```bash
export RESUME_SUMMARY_REQUIRE_ZDR=1
```

免费 ZDR 端点可能没有容量。

## 从源码运行

需要 Python 3.11 或更高版本：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m resume_summarizer.ui
```

也可以使用命令行批处理：

```bash
.venv/bin/python -m resume_summarizer ~/Downloads/resumes \
  --model auto-free \
  --provider openrouter \
  --recursive \
  --output ~/Downloads/Resume_Summary
```

## 测试与构建

```bash
.venv/bin/python -m unittest discover -s tests -v
./build_app.command
```

扫描型 PDF 需要额外的 OCR helper，可通过 `RESUME_SUMMARY_OCR_HELPER` 指定可执行文件。文本型 PDF 不需要 OCR。

## 当前限制

- 免费模型的可用性、速度和每日请求额度由 OpenRouter 决定，可能变化。
- 分块后的每一块都会占用一次免费请求。
- 模型抽取仍可能出错，分享前应检查“需复核”项目。

