"""Entry point for the AI detection prototype.

This script is intentionally tiny. It simply starts the application orchestration
layer and passes the CLI arguments through to the detection workflow.
"""

from src.application import main


if __name__ == "__main__":
    main()
