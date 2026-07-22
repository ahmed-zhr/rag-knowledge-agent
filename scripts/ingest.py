"""CLI ingestion — index the sample documents into the knowledge base.

Usage:
    python scripts/ingest.py            # ingest every domain
    python scripts/ingest.py operations # ingest a single domain
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config  # noqa: E402
from src import ingestion  # noqa: E402


def main():
    targets = sys.argv[1:] or config.DOMAINS
    for domain in targets:
        report = ingestion.ingest_domain(domain)
        print(f"[{report['domain']:<12}] {report['files']} file(s) "
              f"-> {report['chunks_indexed']} chunk(s) indexed")
    print(f"\nVector store: {config.VECTOR_DB_DIR}")


if __name__ == "__main__":
    main()
