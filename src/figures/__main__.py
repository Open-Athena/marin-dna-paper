"""Build all recipes supported by local data: python -m figures."""
import sys
from build_figure import main
if len(sys.argv) == 1:
    sys.argv.append('all')
raise SystemExit(main())
