# Standard Library Imports
from pathlib import Path

# Third Party Imports
from livereload import Server, shell


if __name__ == "__main__":
    server = Server()
    server.watch("source/*.rst", shell("make html"), delay=1)
    server.watch("source/getting_started/*.rst", shell("make html"), delay=1)
    server.watch("source/reference/*.rst", shell("make html"), delay=1)
    server.watch("source/*.md", shell("make html"), delay=1)
    server.watch("source/*.py", shell("make html"), delay=1)
    server.watch("source/_static/*", shell("make html"), delay=1)
    server.watch("source/_templates/*", shell("make html"), delay=1)
    server.watch(
        str(Path(__file__).resolve().parent / "encodermap") + "/*",
        shell("make html"),
        delay=1,
    )
    server.watch(
        str(Path(__file__).resolve().parent / "tutorials") + "/*",
        shell("make html"),
        delay=1,
    )
    server.watch(
        str(Path(__file__).resolve().parent / "CONTRIBUTING.md") + "/*",
        shell("make html"),
        delay=1,
    )
    server.serve(root="build/html")
