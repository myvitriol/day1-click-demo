"""day1-click-demo entry point.

usage:
  python run.py                      # mic capture + web UI (demo laptop)
  python run.py --source file --loop # rehearse with the golden cycle, no connector needed
  python run.py --selftest           # headless pipeline check against golden expectations
"""
from app.main import main

if __name__ == "__main__":
    main()
