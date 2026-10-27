import os
import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient


def main():
    root = Path(__file__).resolve().parent
    os.chdir(root)
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache" / "matplotlib"))
    path = root / "experiments.ipynb"
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    nbformat.write(notebook, path)

    def save(cell, cell_index, **kwargs):
        for output in cell.get("outputs", []):
            if output.output_type == "stream":
                print(output.text.rstrip(), flush=True)
        nbformat.write(notebook, path)
        print(f"Ячейка {cell_index} готова", flush=True)

    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv[0] = sys.executable
    client = NotebookClient(notebook, km=manager, timeout=7200,
                            resources={"metadata": {"path": str(root)}},
                            on_cell_executed=save)
    try:
        client.execute()
    finally:
        nbformat.write(notebook, path)
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    print("Saved executed experiments.ipynb", flush=True)


if __name__ == "__main__":
    main()
