
"""Simple command‑line interface for querying the assistant."""
import argparse, asyncio, sys

from assistant.rag import get_chain
from assistant.ingest import create_vector_database  # optional import for convenience

def main() -> None:
    parser = argparse.ArgumentParser(description="Query the assistant from the CLI.")
    parser.add_argument("question", nargs="*", help="Your question to ask.")
    parser.add_argument("--ingest", action="store_true", help="Run ingestion before querying.")
    args = parser.parse_args()

    if args.ingest:
        print("Running ingestion …")
        create_vector_database()

    question = " ".join(args.question).strip()
    if not question:
        print("You must provide a question", file=sys.stderr)
        sys.exit(1)

    chain = get_chain()
    answer = asyncio.run(chain.ainvoke(question))
    print("\n=== ANSWER ===\n", answer["result"])

if __name__ == "__main__":
    main()
