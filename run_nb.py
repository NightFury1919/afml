"""
run_nb.py -- execute a notebook in place, headlessly, in the current env.

    python run_nb.py ch10\bet_sizing\chapter_10_bet_sizing.ipynb

WHY THIS EXISTS: `python -m nbconvert --execute` fails on this machine at
IMPORT time (nbconvert -> postprocessors/serve -> tornado.httpserver ->
netutil -> ssl.create_default_context -> ssl.SSLError [ASN1: NOT_ENOUGH_DATA],
a Windows cert-store parse failure under Python 3.10). nbclient is the layer
nbconvert delegates execution to and never imports tornado.httpserver, so it
avoids the whole chain. `jupyter nbconvert` is worse still -- it dispatches to
a stray Python 3.8 install on PATH.
"""

import os
import sys

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


def main():
    if len(sys.argv) < 2:
        print('usage: python run_nb.py <notebook.ipynb> [kernel_name]')
        return 2

    path = sys.argv[1]
    kernel = sys.argv[2] if len(sys.argv) > 2 else 'mlfinlab'

    if not os.path.exists(path):
        print(f'ERROR: no such notebook: {path}')
        return 1

    nb = nbformat.read(path, as_version=4)
    n_code = sum(1 for c in nb.cells if c.cell_type == 'code')
    workdir = os.path.dirname(os.path.abspath(path))

    print(f'executing {path}')
    print(f'  kernel   : {kernel}')
    print(f'  cwd      : {workdir}')
    print(f'  code cells: {n_code}')

    client = NotebookClient(
        nb,
        timeout=1800,
        kernel_name=kernel,
        allow_errors=False,
        resources={'metadata': {'path': workdir}},
    )

    try:
        client.execute()
    except CellExecutionError as exc:
        # Deliberately do NOT write on failure -- a half-executed notebook
        # committed to the repo is worse than an unexecuted one, because it
        # looks finished.
        print('\nFAILED -- notebook NOT written. A cell raised:\n')
        print(str(exc)[:3000])
        return 1

    nbformat.write(nb, path)
    ran = sum(1 for c in nb.cells
              if c.cell_type == 'code' and c.get('execution_count'))
    print(f'\nOK -- wrote {path}')
    print(f'  cells executed: {ran} / {n_code}')
    print(f'  kernel python : '
          f'{nb.metadata.get("language_info", {}).get("version", "?")}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
