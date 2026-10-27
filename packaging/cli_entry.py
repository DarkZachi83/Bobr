"""Punkt startowy AI-Bobr-cli.exe: zawsze tryb tekstowy (program okienkowy nie ma konsoli)."""
import sys

import main

if __name__ == "__main__":
    sys.exit(main.main(["--cli", *sys.argv[1:]]))
