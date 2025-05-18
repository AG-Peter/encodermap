# Standard Library Imports
from pathlib import Path

# Third Party Imports
from livereload import Server, shell


if __name__ == "__main__":
    server = Server()
    files = list(Path(__file__).parent.rglob("*.rst"))
    files.extend(list(Path(__file__).parent.rglob("*.nblink")))
    files.extend(list(Path(__file__).parent.rglob("*.ipynb")))
    files.extend(list(Path(__file__).parent.parent.rglob("*.py")))
    files.extend(list(Path(__file__).parent.rglob("*.md")))
    files.extend(list((Path(__file__).parent / "_static").glob("*")))
    for file in files:
        server.watch(
            str(file),
            shell("make html"),
            delay=1,
        )
    server.serve(root="build/html")
