# Sphinx Documentation

Build the docs locally:

```bash
python -m pip install -r docs/sphinx/requirements.txt
python -m sphinx -b html docs/sphinx docs/sphinx/_build
```

Open:

`docs/sphinx/_build/index.html`

Build the docs in Docker (output to `docs/build` by default):

```bash
bash docs/build_in_docker.sh
```

Or choose a custom output folder:

```bash
bash docs/build_in_docker.sh /path/to/output
```
