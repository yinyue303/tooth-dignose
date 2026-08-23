# 分类结果指标写入 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在零样本分类脚本生成的预测明细文件末尾追加 Accuracy、Precision、Recall 和 F1 指标表。

**Architecture:** 将指标计算和文件追加提取到一个不依赖模型加载的轻量模块，使推理脚本复用同一组指标进行终端输出和文件写入。结果文件仍由推理循环写入预测表，指标模块只在完整推理结束后追加一次两列表格。

**Tech Stack:** Python、scikit-learn、unittest

---

### Task 1: 指标计算与结果文件追加

**Files:**
- Create: `models/result_metrics.py`
- Create: `tests/test_result_metrics.py`
- Modify: `models/zero-shoot-classification.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_result_metrics.py`，通过文件路径加载模块，避免导入大模型脚本：

```python
import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "models" / "result_metrics.py"


def load_result_metrics_module():
    spec = importlib.util.spec_from_file_location("result_metrics", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ResultMetricsTest(unittest.TestCase):
    def test_appends_single_metrics_table_after_predictions(self):
        result_metrics = load_result_metrics_module()
        metrics = result_metrics.calculate_classification_metrics(
            ["A", "B"], ["A", "A"]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            result_file = Path(temp_dir) / "result.txt"
            result_file.write_text(
                "id\ttrue\tpred\n1\tA\tA\n2\tB\tA\n",
                encoding="utf-8",
            )
            result_metrics.append_metrics(result_file, metrics)

            self.assertEqual(
                result_file.read_text(encoding="utf-8"),
                "id\ttrue\tpred\n"
                "1\tA\tA\n"
                "2\tB\tA\n"
                "\n"
                "metric\tvalue\n"
                "Accuracy\t0.500000\n"
                "Precision\t0.250000\n"
                "Recall\t0.500000\n"
                "F1\t0.333333\n",
            )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试并确认正确失败**

Run: `python -m unittest tests/test_result_metrics.py -v`

Expected: `ERROR`，原因是 `models/result_metrics.py` 尚不存在。

- [ ] **Step 3: 实现最小指标模块**

创建 `models/result_metrics.py`：

```python
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def calculate_classification_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def append_metrics(result_file, metrics):
    with open(result_file, "a", encoding="utf-8", newline="") as file:
        file.write("\nmetric\tvalue\n")
        for name, value in metrics.items():
            file.write(f"{name}\t{value:.6f}\n")
```

- [ ] **Step 4: 接入零样本分类脚本**

在 `models/zero-shoot-classification.py` 中导入两个函数，用下面的逻辑替换四次独立计算与打印：

```python
from result_metrics import append_metrics, calculate_classification_metrics

metrics = calculate_classification_metrics(y_true, y_pred)
for metric_name, metric_value in metrics.items():
    print(f"{metric_name}:", metric_value)
append_metrics(result_file, metrics)
```

保留现有 `classification_report` 输出。

- [ ] **Step 5: 运行测试并确认通过**

Run: `python -m unittest tests/test_result_metrics.py -v`

Expected: `Ran 1 test` 和 `OK`。

- [ ] **Step 6: 做语法与差异检查**

Run: `python -m py_compile models/result_metrics.py models/zero-shoot-classification.py tests/test_result_metrics.py`

Expected: 无输出，退出码为 0。

Run: `git diff --check -- models/result_metrics.py models/zero-shoot-classification.py tests/test_result_metrics.py`

Expected: 无错误。

- [ ] **Step 7: 提交实现**

```bash
git add models/result_metrics.py models/zero-shoot-classification.py tests/test_result_metrics.py
git commit -m "feat: 将分类评估指标写入结果文件"
```
