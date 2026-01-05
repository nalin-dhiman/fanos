# Release checklist (PyPI)

1. Pick a final PyPI name (this project uses `fanos-optimizer`).
2. Replace `<YOUR_GITHUB>` URLs in `pyproject.toml`.
3. Run:
   - `python -m pip install -U build twine pytest`
   - `pytest`
   - `python -m build`
   - `python -m twine check dist/*`
4. Upload to TestPyPI:
   - `python -m twine upload --repository testpypi dist/*`
5. Install from TestPyPI in a clean venv and run the example.
6. Upload to PyPI:
   - `python -m twine upload dist/*`
