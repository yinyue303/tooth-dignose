# 双语诊断报告评估实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为诊断测试集增加稳定的语言标签，按输入语言选择中英文提示词，并只在日志中输出英文三项、中文两项平均指标。

**Architecture:** 用一个不依赖模型或 NLTK 的纯 Python 辅助模块承载语言校验、中文字符级 tokenizer、分组和日志格式，以便本地快速测试。`dignose.py` 继续负责模型推理和 NLTK/Gemma 指标计算；Excel 通过 `openpyxl` 原地增加 `language` 列，提示词按样本语言路由。

**Tech Stack:** Python 3、pandas、openpyxl、PyTorch、Transformers、NLTK、NumPy、unittest

---

### Task 1: 建立可测试的双语评估辅助模块

**Files:**
- Create: `models/diagnose_evaluation.py`
- Create: `tests/test_diagnose_evaluation.py`

- [ ] **Step 1: 编写中文 tokenizer、语言校验、分组和日志格式的失败测试**

```python
import unittest

from models.diagnose_evaluation import (
    CHINESE,
    ENGLISH,
    chinese_metric_tokenize,
    detect_language,
    format_metric_summary,
    partition_by_language,
    validate_language,
)


class DiagnoseEvaluationTests(unittest.TestCase):
    def test_detect_language(self):
        self.assertEqual(detect_language("请生成口腔报告"), CHINESE)
        self.assertEqual(detect_language("Generate a dental report"), ENGLISH)

    def test_chinese_tokenizer_preserves_numbers_and_icd_codes(self):
        tokens = chinese_metric_tokenize("47号牙齿牙髓炎（K04.005）")
        self.assertEqual(
            tokens,
            ["47", "号", "牙", "齿", "牙", "髓", "炎", "(", "k04.005", ")"],
        )

    def test_validate_language_rejects_other_values(self):
        self.assertEqual(validate_language(0), CHINESE)
        self.assertEqual(validate_language(1), ENGLISH)
        with self.assertRaisesRegex(ValueError, "language"):
            validate_language(2)

    def test_partition_requires_both_languages(self):
        groups = partition_by_language(["中", "English"], ["中", "English"], [0, 1])
        self.assertEqual(groups[CHINESE], (["中"], ["中"]))
        self.assertEqual(groups[ENGLISH], (["English"], ["English"]))

    def test_summary_contains_only_requested_metrics(self):
        summary = format_metric_summary(1, 0.1, 0.2, 0.3, 1, 0.4, 0.5)
        self.assertIn("English METEOR: 0.200000", summary)
        self.assertIn("Chinese BLEU: 0.400000", summary)
        self.assertNotIn("Chinese METEOR", summary)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试并确认因模块不存在而失败**

Run: `python -m unittest tests.test_diagnose_evaluation -v`

Expected: FAIL，错误包含 `No module named 'models.diagnose_evaluation'`。

- [ ] **Step 3: 实现最小辅助模块**

```python
import re
import unicodedata

CHINESE = 0
ENGLISH = 1
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_ZH_TOKEN_RE = re.compile(
    r"[a-z]\d{2}(?:\.\d+)?(?:x\d+)?|[a-z]+|\d+(?:\.\d+)*|"
    r"[\u3400-\u4dbf\u4e00-\u9fff]|[^\w\s]",
    re.IGNORECASE,
)


def normalize_metric_text(text):
    return unicodedata.normalize("NFKC", str(text)).lower().strip()


def detect_language(text):
    return CHINESE if _CJK_RE.search(str(text)) else ENGLISH


def validate_language(value):
    if value in (CHINESE, ENGLISH):
        return int(value)
    raise ValueError(f"language 必须是 0 或 1，实际为：{value!r}")


def chinese_metric_tokenize(text):
    return _ZH_TOKEN_RE.findall(normalize_metric_text(text))


def partition_by_language(references, predictions, languages):
    if not (len(references) == len(predictions) == len(languages)):
        raise ValueError("参考答案、预测结果和语言标签数量必须一致")
    groups = {CHINESE: ([], []), ENGLISH: ([], [])}
    for reference, prediction, language in zip(references, predictions, languages):
        refs, preds = groups[validate_language(language)]
        refs.append(reference)
        preds.append(prediction)
    if not groups[CHINESE][0] or not groups[ENGLISH][0]:
        raise ValueError("中文和英文分组都必须至少包含一个样本")
    return groups


def format_metric_summary(
    english_count,
    english_bleu,
    english_meteor,
    english_cosine,
    chinese_count,
    chinese_bleu,
    chinese_cosine,
):
    return (
        "最终评估结果：\n\n"
        f"英文样本数量：{english_count}\n"
        f"English BLEU: {english_bleu:.6f}\n"
        f"English METEOR: {english_meteor:.6f}\n"
        f"English Cosine Similarity: {english_cosine:.6f}\n\n"
        f"中文样本数量：{chinese_count}\n"
        f"Chinese BLEU: {chinese_bleu:.6f}\n"
        f"Chinese Cosine Similarity: {chinese_cosine:.6f}"
    )
```

- [ ] **Step 4: 运行测试并确认通过**

Run: `python -m unittest tests.test_diagnose_evaluation -v`

Expected: 5 tests PASS。

### Task 2: 为 Excel 增加语言列

**Files:**
- Modify: `dataset/excel/test-diagnostic.xlsx`

- [ ] **Step 1: 用临时输出文件写入 `language` 列**

使用 `openpyxl.load_workbook()` 打开工作簿；在 E1 写入 `language`。对 E2:E1800 调用 `detect_language(q)`，中文写整数 0，英文写整数 1。复制 D 列相邻单元格的样式，并把工作表中的 Excel Table 范围扩展到 E 列。先保存为同目录临时 `.xlsx`，不要直接覆盖原文件。

- [ ] **Step 2: 验证临时工作簿**

重新加载临时文件并断言：数据行 1799、中文 917、英文 882、无空语言、集合严格等于 `{0, 1}`、A:D 单元格值与原工作簿完全一致、表格范围覆盖 E 列、公式错误数为 0。

- [ ] **Step 3: 原子替换原工作簿并复验**

验证成功后用临时文件替换 `dataset/excel/test-diagnostic.xlsx`，再次只读打开并检查 E1、E2:E1800 和计数。

### Task 3: 增加英文报告提示词

**Files:**
- Create: `prompt/diagnose--2-en.txt`
- Preserve: `prompt/diagnose--2.txt`

- [ ] **Step 1: 创建与中文规则等价的英文提示词**

英文提示词必须固定输出以下标题且各出现一次：

```text
Patient Record:
Examination:
X-Rays Examination:
Diagnosis:
Treatment Plan:
Treatment Recommendations:
Management:
Medical Instructions:
Remarks:
```

缺失内容统一写 `Not provided.`，禁止编造患者信息、牙位、疾病编码、影像结果、药物和已实施治疗。

- [ ] **Step 2: 静态检查两份提示词**

逐一检查中文和英文九个标题均存在、顺序一致、没有重复；确认英文提示词未要求中文输出。

### Task 4: 将双语分组接入诊断评估脚本

**Files:**
- Modify: `models/dignose.py`
- Test: `tests/test_diagnose_evaluation.py`

- [ ] **Step 1: 增加 JSON 图片解析的失败测试**

在测试文件中新增：合法 JSON 数组返回字符串列表；空字符串、字典、包含非字符串元素的数组均抛出 `ValueError`。

- [ ] **Step 2: 实现并测试安全图片解析**

在 `models/diagnose_evaluation.py` 中用 `json.loads()` 实现 `parse_image_list()`，禁止 `eval()`；运行单元测试至通过。

- [ ] **Step 3: 修改提示词加载和行级路由**

读取 `diagnose--2.txt` 为中文提示词，读取 `diagnose--2-en.txt` 为英文提示词。加载 Excel 后要求存在 `id、q、a、images、language` 五列，规范化 `language`，每条消息按 0/1 插入对应系统提示词。

- [ ] **Step 4: 拆分 BLEU 与 METEOR 计算**

英文使用 `wordpunct_tokenize(normalize_metric_text(text))` 计算 BLEU-4 和 METEOR；中文使用 `chinese_metric_tokenize()` 只计算 BLEU-4。两种 BLEU 均使用 `(0.25, 0.25, 0.25, 0.25)` 和 `SmoothingFunction().method1`。

- [ ] **Step 5: 一次编码后按语言汇总余弦相似度**

Gemma 仅加载一次并对全部 reference/prediction 编码；按 `language` 布尔索引分别计算中英文平均值，避免为两组重复加载模型。

- [ ] **Step 6: 删除指标文件和逐样本指标写入**

保留现有 `id、true、pred` 预测文件；删除 `result_df["bleu"]`、`result_df["meteor"]`、余弦列、`summary_df` 和 `_metrics.txt` 写入逻辑。

- [ ] **Step 7: 只打印最终分组平均值**

调用 `format_metric_summary()` 并打印，保证英文输出 BLEU/METEOR/Cosine，中文只输出 BLEU/Cosine，均保留六位小数。

### Task 5: 最终验证

**Files:**
- Verify: `models/diagnose_evaluation.py`
- Verify: `models/dignose.py`
- Verify: `prompt/diagnose--2-en.txt`
- Verify: `dataset/excel/test-diagnostic.xlsx`

- [ ] **Step 1: 运行所有轻量测试**

Run: `python -m unittest discover -s tests -v`

Expected: 全部 PASS。

- [ ] **Step 2: 运行语法检查**

Run: `python -m py_compile models/diagnose_evaluation.py models/dignose.py`

Expected: exit code 0。

- [ ] **Step 3: 验证工作簿数据与样式**

使用 openpyxl 重新打开最终文件，确认 1799 行、917 个 0、882 个 1、A:D 内容不变、E 列样式存在、所有公式无错误。

- [ ] **Step 4: 检查差异和范围**

Run: `git diff --check`

Expected: 无空白错误；未触碰 LoRA 路径、CUDA 映射及其他训练脚本。

- [ ] **Step 5: 记录完整推理限制**

本机缺少 PyTorch、Transformers 和 NLTK，且仓库中没有配置的 LoRA 权重，因此本地只执行纯逻辑测试、语法检查和 Excel 完整性检查；完整 GPU 推理由用户的 HPC 环境执行并在日志中验证最终五项输出。
