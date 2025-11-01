import argparse
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from pypdf import PdfReader


API_ENDPOINT = "https://api.x.ai/v1/chat/completions"
DEFAULT_MODEL = "grok-4-fast-reasoning"
MAX_DEFAULT_BATCH_TOKENS = 12000
MAX_PARALLEL_BATCHES = 3
ESSAY_LIMIT = 30


@dataclass
class EssaySubmission:
    index: int
    filename: str
    student_name: str
    text: str


@dataclass
class BatchResult:
    batch_index: int
    essays: List[EssaySubmission]
    evaluations: List[Dict[str, Any]]
    usage: Dict[str, Any]


class EvaluationError(Exception):
    """Raised when the evaluation pipeline cannot complete."""


def load_env(env_path: Path) -> None:
    """
    Lightweight .env loader so the CLI can run without extra dependencies.
    Existing environment variables win if duplicates appear.
    """
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip()


def read_text_file(path: Path, required: bool = True) -> str:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"File not found: {path}")
        return ""
    if path.suffix.lower() == ".pdf":
        return read_pdf_file(path)
    return path.read_text(encoding="utf-8")


def collect_essays(essays_dir: Path) -> List[EssaySubmission]:
    if not essays_dir.exists():
        raise FileNotFoundError(f"Essays directory not found: {essays_dir}")

    essay_files = sorted(
        [
            p
            for p in essays_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".txt", ".pdf"}
        ]
    )

    if not essay_files:
        raise EvaluationError(f"No essay files found in directory: {essays_dir}")

    if len(essay_files) > ESSAY_LIMIT:
        logging.warning(
            "More than %d essays detected (%d total). Only the first %d will be evaluated.",
            ESSAY_LIMIT,
            len(essay_files),
            ESSAY_LIMIT,
        )
        essay_files = essay_files[:ESSAY_LIMIT]

    submissions: List[EssaySubmission] = []
    for idx, file_path in enumerate(essay_files):
        if file_path.suffix.lower() == ".pdf":
            text = read_pdf_file(file_path)
        else:
            text = read_text_file(file_path)

        if not text.strip():
            raise EvaluationError(f"No textual content extracted from: {file_path}")

        fallback_name = derive_student_name_from_filename(file_path)
        student_name = extract_student_name(text, fallback=fallback_name)
        submissions.append(
            EssaySubmission(
                index=idx,
                filename=file_path.name,
                student_name=student_name,
                text=text.strip(),
            )
        )
    return submissions


def extract_student_name(essay_text: str, fallback: str = "Unknown Student") -> str:
    for line in essay_text.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        lower_line = candidate.lower()
        for prefix in ("student:", "name:"):
            if lower_line.startswith(prefix):
                value = candidate.split(":", 1)[1].strip()
                return value or fallback
    return fallback


def derive_student_name_from_filename(path: Path) -> str:
    stem = path.stem.strip()
    if not stem:
        return "Unknown Student"
    normalized = stem.replace("_", " ").replace("-", " ")
    words = [word for word in normalized.split() if word]
    if not words:
        return "Unknown Student"
    return " ".join(word.capitalize() for word in words)


def read_pdf_file(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise EvaluationError(f"Failed to read PDF file {path}: {exc}") from exc

    pages: List[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:  # pragma: no cover - defensive
            raise EvaluationError(
                f"Failed to extract text from {path} page {page_number}: {exc}"
            ) from exc
        pages.append(page_text.strip())

    combined_text = "\n\n".join(filter(None, pages)).strip()
    if not combined_text:
        logging.warning("PDF %s did not yield extractable text.", path)
    return combined_text


def estimate_tokens(*texts: str) -> int:
    """
    Rough heuristic: 4 characters per token. Guards against zero division.
    """
    total_chars = sum(len(t) for t in texts if t)
    return max(1, total_chars // 4)


def build_prompt_header(reading_material: str, question: str, context: str) -> str:
    return (
        "You are an AI evaluator for 10th-grade high school essays on various literature topics. "
        "Evaluate the following essays based on the provided reading material, question, and context. "
        "Account for potential OCR artifacts in the essays (e.g., minor misspellings or garbled words)—"
        "interpret leniently but do not assume missing content.\n\n"
        f"Reading Material:\n{reading_material.strip()}\n\n"
        f"Test Question:\n{question.strip()}\n\n"
        f"Context:\n{context.strip()}\n\n"
        "Now evaluate the following essays:\n"
    )


def format_essay_block(seq_num: int, essay: EssaySubmission) -> str:
    essay_body = essay.text.strip()
    return f"Essay {seq_num} (Student: {essay.student_name}):\n{essay_body}\n"


def build_batches(
    essays: List[EssaySubmission],
    prompt_header: str,
    max_tokens: int,
) -> List[List[EssaySubmission]]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive.")

    batches: List[List[EssaySubmission]] = []
    current_batch: List[EssaySubmission] = []
    current_tokens = estimate_tokens(prompt_header)

    for seq_num, essay in enumerate(essays, start=1):
        block_text = format_essay_block(seq_num, essay)
        block_tokens = estimate_tokens(block_text)
        if current_batch and current_tokens + block_tokens > max_tokens:
            batches.append(current_batch)
            current_batch = []
            current_tokens = estimate_tokens(prompt_header)

        current_batch.append(essay)
        current_tokens += block_tokens

    if current_batch:
        batches.append(current_batch)

    return batches


def assemble_prompt(prompt_header: str, batch: List[EssaySubmission]) -> str:
    parts = [prompt_header]
    for seq_num, essay in enumerate(batch, start=1):
        parts.append(format_essay_block(seq_num, essay))
    parts.append(
        "For each essay, output in JSON format:\n"
        "{\n"
        '  "student_name": "...",\n'
        '  "summary": "1-2 sentence summary of response vs. question/context.",\n'
        '  "criterion_1": {\n'
        '    "explanation": "1-2 sentences on reasoning.",\n'
        '    "score": 1-5\n'
        "  },\n"
        '  "criterion_2": {\n'
        '    "explanation": "1-2 sentences on reasoning.",\n'
        '    "score": 1-5\n'
        "  },\n"
        '  "total_score": sum of scores,\n'
        '  "overall_comment": "1-2 sentences on strengths/areas for improvement."\n'
        "}\n\n"
        "Output an array of these JSON objects for all essays."
    )
    return "\n".join(parts)


def call_xai_api(
    api_key: str,
    prompt: str,
    model: str,
    timeout: int,
) -> Tuple[str, Dict[str, Any]]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": prompt,
            }
        ],
    }

    response = requests.post(
        API_ENDPOINT,
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    usage = data.get("usage", {})

    content: Optional[str] = None
    if isinstance(data.get("choices"), list) and data["choices"]:
        first_choice = data["choices"][0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message") or {}
            content = message.get("content")
            if content is None and "text" in first_choice:
                content = first_choice["text"]
    if not content:
        raise EvaluationError("No content returned from xAI Grok API response.")
    return content, usage


def parse_evaluations(raw_content: str) -> List[Dict[str, Any]]:
    try:
        return json.loads(raw_content)
    except json.JSONDecodeError:
        trimmed = extract_json_array(raw_content)
        if not trimmed:
            raise
        return json.loads(trimmed)


def extract_json_array(raw_text: str) -> Optional[str]:
    start = raw_text.find("[")
    end = raw_text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    return raw_text[start : end + 1]


def evaluate_batches(
    api_key: str,
    model: str,
    prompt_header: str,
    batches: List[List[EssaySubmission]],
    request_timeout: int,
) -> Tuple[List[BatchResult], Dict[str, int]]:
    results: List[BatchResult] = []
    usage_lock = threading.Lock()
    aggregated_usage: Dict[str, int] = {}

    def process_batch(batch_tuple: Tuple[int, List[EssaySubmission]]) -> BatchResult:
        batch_index, batch = batch_tuple
        prompt = assemble_prompt(prompt_header, batch)
        raw_content, usage = call_xai_api(api_key, prompt, model, request_timeout)
        evaluations = parse_evaluations(raw_content)
        if len(evaluations) != len(batch):
            logging.warning(
                "Batch %d: Expected %d evaluations, received %d. "
                "Matching will proceed by student name.",
                batch_index,
                len(batch),
                len(evaluations),
            )
        with usage_lock:
            for key, value in usage.items():
                if isinstance(value, int):
                    aggregated_usage[key] = aggregated_usage.get(key, 0) + value
        return BatchResult(batch_index, batch, evaluations, usage)

    with ThreadPoolExecutor(max_workers=min(len(batches), MAX_PARALLEL_BATCHES)) as executor:
        future_map = {
            executor.submit(process_batch, (idx, batch)): idx for idx, batch in enumerate(batches)
        }
        for future in as_completed(future_map):
            batch_idx = future_map[future]
            try:
                batch_result = future.result()
                logging.info("Completed batch %d.", batch_idx + 1)
                results.append(batch_result)
            except Exception as exc:  # pylint: disable=broad-except
                logging.error("Batch %d failed: %s", batch_idx + 1, exc)
                raise

    results.sort(key=lambda item: item.batch_index)
    return results, aggregated_usage


def align_results(batches: List[BatchResult]) -> List[Dict[str, Any]]:
    aligned: List[Dict[str, Any]] = []
    for batch in batches:
        used_indices: set[int] = set()
        for submission in batch.essays:
            evaluation = match_evaluation(submission, batch.evaluations, used_indices)
            if evaluation:
                evaluated_name = evaluation.get("student_name", "") if isinstance(evaluation, dict) else ""
                evaluated_name = (evaluated_name or "").strip()
                if not evaluated_name or evaluated_name.lower() == "unknown student":
                    evaluated_name = submission.student_name
                if (
                    not evaluated_name
                    or evaluated_name.lower() == "unknown student"
                ):
                    evaluated_name = derive_student_name_from_filename(Path(submission.filename))

                aligned.append(
                    {
                        "essay_index": submission.index,
                        "student_name": evaluated_name or "Unknown Student",
                        "summary": evaluation.get("summary", ""),
                        "criterion_1": evaluation.get("criterion_1", {}),
                        "criterion_2": evaluation.get("criterion_2", {}),
                        "total_score": evaluation.get("total_score"),
                        "overall_comment": evaluation.get("overall_comment", ""),
                    }
                )
            else:
                logging.warning(
                    "No evaluation found for student '%s' in batch %d.",
                    submission.student_name,
                    batch.batch_index + 1,
                )
    aligned.sort(key=lambda row: row["essay_index"])
    return aligned


def match_evaluation(
    submission: EssaySubmission,
    evaluations: List[Dict[str, Any]],
    used_indices: set[int],
) -> Optional[Dict[str, Any]]:
    target_name = submission.student_name.strip().lower()
    for idx, eval_item in enumerate(evaluations):
        if idx in used_indices:
            continue
        candidate = str(eval_item.get("student_name", "")).strip().lower()
        if candidate and candidate == target_name and target_name:
            used_indices.add(idx)
            return eval_item

    for idx, eval_item in enumerate(evaluations):
        if idx in used_indices:
            continue
        used_indices.add(idx)
        return eval_item

    return None


def write_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    records = []
    for row in rows:
        criterion_1_score = extract_score(row.get("criterion_1"))
        criterion_2_score = extract_score(row.get("criterion_2"))
        total_score = row.get("total_score")
        if total_score is None and criterion_1_score is not None and criterion_2_score is not None:
            total_score = criterion_1_score + criterion_2_score
        records.append(
            {
                "Student Name": row["student_name"],
                "Criterion 1 Score": criterion_1_score,
                "Criterion 2 Score": criterion_2_score,
                "Total Score": total_score,
            }
        )
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    logging.info("Wrote CSV report to %s", output_path)


def extract_score(criterion: Any) -> Optional[int]:
    if isinstance(criterion, dict):
        score = criterion.get("score")
        if isinstance(score, int):
            return score
        if isinstance(score, (float, str)):
            try:
                return int(float(score))
            except ValueError:
                return None
    elif isinstance(criterion, (int, float)):
        return int(criterion)
    return None


def write_pdf_reports(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        student_name = row["student_name"]
        slug = slugify(student_name or "student")
        pdf_path = output_dir / f"{slug}.pdf"
        create_pdf_report(pdf_path, row)
        logging.info("Wrote PDF report for %s to %s", student_name, pdf_path)


def slugify(value: str) -> str:
    sanitized = "".join(ch if ch.isalnum() else "-" for ch in value.strip())
    sanitized = "-".join(filter(None, sanitized.split("-")))
    return sanitized.lower() or "student"


def create_pdf_report(pdf_path: Path, evaluation: Dict[str, Any]) -> None:
    canvas_obj = canvas.Canvas(str(pdf_path), pagesize=letter)
    width, height = letter
    x_margin = inch
    y = height - inch
    line_height = 12

    def write_line(text: str, bold: bool = False) -> None:
        nonlocal y
        if y <= inch:
            canvas_obj.showPage()
            y = height - inch
        if bold:
            canvas_obj.setFont("Helvetica-Bold", 11)
        else:
            canvas_obj.setFont("Helvetica", 10)
        canvas_obj.drawString(x_margin, y, text)
        y -= line_height

    def write_block(title: str, body: str) -> None:
        write_line(title, bold=True)
        for wrapped_line in wrap_text(body, max_chars=90):
            write_line(wrapped_line)
        y_bottom_padding()

    def y_bottom_padding():
        nonlocal y
        y -= 4

    canvas_obj.setTitle(f"Evaluation - {evaluation['student_name']}")
    write_line(f"Student: {evaluation['student_name']}", bold=True)
    write_line("")

    summary = evaluation.get("summary", "")
    write_block("Summary:", summary or "No summary provided.")

    criterion_1 = evaluation.get("criterion_1", {}) or {}
    criterion_2 = evaluation.get("criterion_2", {}) or {}

    crit1_body = format_criterion_block("Criterion 1", criterion_1)
    crit2_body = format_criterion_block("Criterion 2", criterion_2)

    write_block("Criterion 1 (Question Response Quality):", crit1_body)
    write_block("Criterion 2 (Use of Evidence):", crit2_body)

    total_score = evaluation.get("total_score")
    if total_score is None:
        total_score = (
            extract_score(criterion_1) + extract_score(criterion_2)
            if extract_score(criterion_1) is not None and extract_score(criterion_2) is not None
            else "N/A"
        )
    write_line(f"Total Score: {total_score}", bold=True)
    write_line("")

    overall_comment = evaluation.get("overall_comment", "")
    write_block("Overall Comment:", overall_comment or "No overall comment provided.")

    canvas_obj.save()


def wrap_text(text: str, max_chars: int) -> Iterable[str]:
    if not text:
        return []
    words = text.split()
    lines: List[str] = []
    current_line: List[str] = []
    current_length = 0

    for word in words:
        word_length = len(word) + 1  # include space
        if current_length + word_length > max_chars:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word) + 1
        else:
            current_line.append(word)
            current_length += word_length
    if current_line:
        lines.append(" ".join(current_line))
    return lines


def format_criterion_block(title: str, criterion: Dict[str, Any]) -> str:
    explanation = criterion.get("explanation", "No explanation provided.")
    score = criterion.get("score", "N/A")
    return f"{title} Score: {score}\n{explanation}"


def summarise_material(
    api_key: str,
    material: str,
    model: str,
    timeout: int,
) -> str:
    prompt = (
        "Summarise the following reading material in 5-6 bullet points while retaining key plot "
        "and thematic elements relevant for essay evaluation:\n\n"
        f"{material.strip()}"
    )
    content, _ = call_xai_api(api_key, prompt, model, timeout)
    return content.strip()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate batches of student essays using the xAI Grok API. "
            "Requires essays (.txt files), reading material, question, and context."
        )
    )

    parser.add_argument("--essays_dir", required=True, help="Directory containing essay text files.")
    parser.add_argument("--material_file", required=True, help="Path to reading material text file.")
    parser.add_argument("--question_file", required=True, help="Path to test question text file.")

    context_group = parser.add_mutually_exclusive_group(required=True)
    context_group.add_argument("--context", help="Context string describing the class or unit.")
    context_group.add_argument(
        "--context_file", help="Path to a text file containing context for the evaluation."
    )

    parser.add_argument(
        "--env_file",
        default=".env",
        help="Path to a .env file containing XAI_API_KEY (default: .env in current directory).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"xAI Grok model to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--max_batch_tokens",
        type=int,
        default=MAX_DEFAULT_BATCH_TOKENS,
        help=(
            "Approximate maximum token budget per batch (default: "
            f"{MAX_DEFAULT_BATCH_TOKENS}). Adjust if requests are too large."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="evaluations_output",
        help="Directory to write PDF reports (default: evaluations_output).",
    )
    parser.add_argument(
        "--csv_path",
        default="evaluations.csv",
        help="Path for the CSV summary (default: evaluations.csv).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="HTTP timeout for API requests in seconds (default: 120).",
    )
    parser.add_argument(
        "--summarise-material",
        action="store_true",
        dest="summarise_material",
        help="Summarise the reading material via the API before evaluation.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for debugging.",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def ensure_api_key(env_file: Path) -> str:
    load_env(env_file)
    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        raise EvaluationError(
            "Missing XAI_API_KEY. Set it in your environment or provide it in the .env file."
        )
    return api_key


def main() -> None:
    args = parse_arguments()
    configure_logging(args.verbose)

    essays_dir = Path(args.essays_dir)
    material_file = Path(args.material_file)
    question_file = Path(args.question_file)
    context_file = Path(args.context_file) if args.context_file else None
    env_file = Path(args.env_file)
    output_dir = Path(args.output_dir)
    csv_path = Path(args.csv_path)

    api_key = ensure_api_key(env_file)

    logging.info("Loading source materials.")
    essays = collect_essays(essays_dir)
    reading_material = read_text_file(material_file)
    question_text = read_text_file(question_file)
    context_text = args.context if args.context else read_text_file(context_file) if context_file else ""

    if args.summarise_material:
        logging.info("Summarising reading material via xAI Grok API.")
        reading_material = summarise_material(api_key, reading_material, args.model, args.timeout)

    prompt_header = build_prompt_header(reading_material, question_text, context_text)
    heuristic_tokens = estimate_tokens(prompt_header)
    logging.info("Prompt header estimated token count: ~%d tokens.", heuristic_tokens)

    batches = build_batches(essays, prompt_header, args.max_batch_tokens)
    logging.info("Prepared %d batch(es) for evaluation.", len(batches))

    batch_results, usage = evaluate_batches(
        api_key=api_key,
        model=args.model,
        prompt_header=prompt_header,
        batches=batches,
        request_timeout=args.timeout,
    )

    aligned_results = align_results(batch_results)
    logging.info("Received evaluations for %d essays.", len(aligned_results))

    write_csv(aligned_results, csv_path)
    write_pdf_reports(aligned_results, output_dir)

    if usage:
        logging.info("Aggregated token usage from API: %s", usage)
    logging.info("Evaluation pipeline complete.")


if __name__ == "__main__":
    try:
        main()
    except EvaluationError as err:
        logging.error("Evaluation failed: %s", err)
        raise SystemExit(1) from err
    except FileNotFoundError as err:
        logging.error("File error: %s", err)
        raise SystemExit(1) from err
    except requests.HTTPError as err:
        logging.error("HTTP error from xAI API: %s", err)
        raise SystemExit(1) from err
    except json.JSONDecodeError as err:
        logging.error("Failed to parse JSON from API response: %s", err)
        raise SystemExit(1) from err
