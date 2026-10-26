import ast
import math
import numbers
import re
import unicodedata
from statistics import fmean


REQUIRED_COLUMNS = ("id", "q", "a", "images", "language")
_CJK_RANGES = (
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0x20000, 0x2EBEF),
    (0x30000, 0x323AF),
)
_CODE_RE = re.compile(r"[a-z]+\d+(?:[._/-][a-z0-9]+)*", re.IGNORECASE)
_WORD_RE = re.compile(r"[a-z]+(?:['’-][a-z]+)*", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\d+(?:[._/-]\d+)*")


def normalize_metric_text(text):
    if text is None:
        return ""
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    return " ".join(normalized.split())


def _is_cjk(character):
    codepoint = ord(character)
    return any(start <= codepoint <= end for start, end in _CJK_RANGES)


def classify_language(question):
    normalized = unicodedata.normalize("NFKC", str(question or ""))
    if not normalized.strip():
        raise ValueError("q 为空，无法判断 language")
    return 0 if any(_is_cjk(character) for character in normalized) else 1


def validate_required_columns(columns):
    missing = [column for column in REQUIRED_COLUMNS if column not in columns]
    if missing:
        raise ValueError(f"缺少必需列：{', '.join(missing)}")


def validate_language_values(values):
    languages = []
    for index, value in enumerate(values):
        if (
            isinstance(value, bool)
            or not isinstance(value, numbers.Integral)
            or int(value) not in (0, 1)
        ):
            raise ValueError(
                f"第 {index + 1} 条记录的 language 必须是整数 0 或 1，实际为 {value!r}"
            )
        languages.append(int(value))
    return languages


def prompt_for_language(language, chinese_prompt, english_prompt):
    validated = validate_language_values([language])[0]
    return chinese_prompt if validated == 0 else english_prompt


def parse_image_paths(value):
    if isinstance(value, list):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"images 不是有效的图片路径列表：{value!r}") from exc
    else:
        raise ValueError(f"images 必须是图片路径字符串列表，实际为 {value!r}")

    if not isinstance(parsed, list) or any(
        not isinstance(path, str) or not path.strip() for path in parsed
    ):
        raise ValueError(f"images 必须是图片路径字符串列表，实际为 {value!r}")
    return parsed


def tokenize_chinese(text):
    normalized = normalize_metric_text(text)
    tokens = []
    index = 0
    while index < len(normalized):
        if normalized[index].isspace():
            index += 1
            continue

        for pattern in (_CODE_RE, _WORD_RE, _NUMBER_RE):
            match = pattern.match(normalized, index)
            if match:
                tokens.append(match.group(0))
                index = match.end()
                break
        else:
            tokens.append(normalized[index])
            index += 1
    return tokens


def validate_aligned_lengths(references, predictions, languages, cosine_scores):
    lengths = {
        len(references),
        len(predictions),
        len(languages),
        len(cosine_scores),
    }
    if len(lengths) != 1:
        raise ValueError(
            "参考答案、预测结果、language 与余弦相似度的数量必须一致"
        )


def _score_pair(reference_tokens, prediction_tokens, scorer):
    if not reference_tokens or not prediction_tokens:
        return 0.0
    return float(scorer(reference_tokens, prediction_tokens))


def calculate_grouped_metrics(
    references,
    predictions,
    languages,
    cosine_scores,
    bleu_scorer,
    meteor_scorer,
    english_tokenizer,
):
    validate_aligned_lengths(references, predictions, languages, cosine_scores)
    languages = validate_language_values(languages)

    english_bleu = []
    english_meteor = []
    english_cosine = []
    chinese_bleu = []
    chinese_cosine = []

    for reference, prediction, language, cosine in zip(
        references,
        predictions,
        languages,
        cosine_scores,
    ):
        if not math.isfinite(float(cosine)):
            raise ValueError(f"余弦相似度必须是有限数值，实际为 {cosine!r}")

        if language == 0:
            reference_tokens = tokenize_chinese(reference)
            prediction_tokens = tokenize_chinese(prediction)
            chinese_bleu.append(
                _score_pair(reference_tokens, prediction_tokens, bleu_scorer)
            )
            chinese_cosine.append(float(cosine))
            continue

        reference_tokens = english_tokenizer(normalize_metric_text(reference))
        prediction_tokens = english_tokenizer(normalize_metric_text(prediction))
        english_bleu.append(
            _score_pair(reference_tokens, prediction_tokens, bleu_scorer)
        )
        english_meteor.append(
            _score_pair(reference_tokens, prediction_tokens, meteor_scorer)
        )
        english_cosine.append(float(cosine))

    if not chinese_bleu:
        raise ValueError("中文指标组不能为空")
    if not english_bleu:
        raise ValueError("英文指标组不能为空")

    return {
        "english_count": len(english_bleu),
        "english_bleu": fmean(english_bleu),
        "english_meteor": fmean(english_meteor),
        "english_cosine": fmean(english_cosine),
        "chinese_count": len(chinese_bleu),
        "chinese_bleu": fmean(chinese_bleu),
        "chinese_cosine": fmean(chinese_cosine),
    }


def format_metric_summary(metrics):
    return "\n".join(
        (
            f"English samples: {metrics['english_count']}",
            f"English BLEU: {metrics['english_bleu']:.6f}",
            f"English METEOR: {metrics['english_meteor']:.6f}",
            f"English Cosine Similarity: {metrics['english_cosine']:.6f}",
            f"Chinese samples: {metrics['chinese_count']}",
            f"Chinese BLEU: {metrics['chinese_bleu']:.6f}",
            f"Chinese Cosine Similarity: {metrics['chinese_cosine']:.6f}",
        )
    )
