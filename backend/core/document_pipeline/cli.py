import argparse
import json
import os
import sys
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.document_pipeline.pipeline import DocumentPipeline
from utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTS = {".txt", ".md", ".pdf", ".docx", ".pptx", ".xlsx"}


def main():
    parser = argparse.ArgumentParser(description="OmniFind Document Context-Extraction Pipeline CLI")
    parser.add_argument("path", help="File path or directory to process")
    parser.add_argument("--output", "-o", help="Optional output JSON file path", default=None)
    parser.add_argument("--pretty", action="store_true", help="Format JSON output with indentation")

    args = parser.parse_args()
    target_path = Path(args.path)

    if not target_path.exists():
        print(f"Error: Path does not exist: {target_path}", file=sys.stderr)
        sys.exit(1)

    pipeline = DocumentPipeline()
    results = []

    if target_path.is_file():
        try:
            res = pipeline.extract_context(str(target_path))
            results.append(res)
            print(f"Successfully processed: {target_path.name}")
        except Exception as e:
            print(f"Error processing {target_path.name}: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Batch directory run
        files = [p for p in target_path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
        print(f"Found {len(files)} supported document(s) in {target_path}")

        for f in files:
            try:
                res = pipeline.extract_context(str(f))
                results.append(res)
                print(f"  [OK] {f.name} ({res['word_count']} words, {len(res['chunks'])} chunks, {len(res['tables'])} tables)")
            except Exception as e:
                print(f"  [SKIPPED/ERROR] {f.name}: {e}", file=sys.stderr)

    output_payload = results[0] if (target_path.is_file() and len(results) == 1) else results
    json_str = json.dumps(output_payload, indent=2 if args.pretty else None)

    if args.output:
        Path(args.output).write_text(json_str, encoding="utf-8")
        print(f"Output saved to {args.output}")
    else:
        if target_path.is_file():
            print(json_str)


if __name__ == "__main__":
    main()
