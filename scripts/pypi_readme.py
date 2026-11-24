#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write the PyPI project description from README.md.

    python scripts/pypi_readme.py <README.md> <git ref> > python/README.md

PyPI renders the description without the repository around it, so this:

* drops ``<!-- github-only -->`` ... ``<!-- /github-only -->`` blocks (math, which PyPI does not
  render; images over PyPI's 10 MB image limit),
* uncomments ``<!-- pypi-only`` ... ``/pypi-only -->`` blocks (their replacements),
* makes repository paths absolute at ``<git ref>`` (the release tag): images (``src``, ``srcset``
  with one URL) on raw.githubusercontent.com, links (Markdown and ``href``) on github.com.

build_pycunls.sh runs it before building a wheel; a checkout's python/README.md is a symlink to
the README itself (for local builds).
"""

import re
import sys

REPO = "nvidia-isaac/cuNLS"
_GITHUB_ONLY = re.compile(r"<!-- github-only\b.*?-->.*?<!-- /github-only -->\n?", re.S)
_PYPI_ONLY = re.compile(r"<!-- pypi-only\n(.*?)/pypi-only -->\n?", re.S)
_IMAGE = re.compile(r'\b(src|srcset)="(?![a-z]+:|#)([^"]+)"')
_HREF = re.compile(r'\bhref="(?![a-z]+:|#)([^"]+)"')
_LINK = re.compile(r"\]\((?![a-z]+:|#)([^)\s]+)\)")


def pypi_readme(text: str, ref: str) -> str:
    """The PyPI version of README ``text``, with repository paths at git ``ref``."""
    for marker in ("github-only", "pypi-only"):
        opened = text.count(f"<!-- {marker}")
        closed = text.count(f"<!-- /{marker} -->" if marker == "github-only" else f"/{marker} -->")
        if opened != closed:
            raise SystemExit(f"pypi_readme.py: unbalanced {marker} markers in README.md (got {opened} "
                             f"openings and {closed} closings, expected equal). Fix: close each block.")
    text = _GITHUB_ONLY.sub("", text)
    text = _PYPI_ONLY.sub(r"\1", text)
    text = _IMAGE.sub(lambda m: f'{m[1]}="https://raw.githubusercontent.com/{REPO}/{ref}/{m[2]}"', text)
    text = _HREF.sub(lambda m: f'href="https://github.com/{REPO}/blob/{ref}/{m[1]}"', text)
    return _LINK.sub(lambda m: f"](https://github.com/{REPO}/blob/{ref}/{m[1]})", text)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python scripts/pypi_readme.py <README.md> <git ref>")
    with open(sys.argv[1], encoding="utf-8") as f:
        sys.stdout.write(pypi_readme(f.read(), sys.argv[2]))
