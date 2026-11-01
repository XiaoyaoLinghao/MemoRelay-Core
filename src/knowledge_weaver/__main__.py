"""Allow `python -m knowledge_weaver` to dispatch to server.main()."""
import sys
from knowledge_weaver.server import main

if __name__ == "__main__":
    sys.exit(main())
