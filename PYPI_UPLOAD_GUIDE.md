# PyPI Upload Guide for SLM-Trainer

This guide will walk you through uploading the SLM-Trainer package to PyPI so users can install it with `pip install slm-trainer`.

## Prerequisites

Before you start, make sure you have:
1. A PyPI account (create one at https://pypi.org/account/register/)
2. A TestPyPI account for testing (create one at https://test.pypi.org/account/register/)
3. Your package code ready and tested

## Step 1: Update Package Metadata

Before publishing, update the following files with your information:

### 1.1 Update `pyproject.toml`

Edit lines 12-14 to add your name and email:
```toml
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
```

Update lines 34-37 with your GitHub repository URL:
```toml
[project.urls]
Homepage = "https://github.com/YOURUSERNAME/SLM-Trainer"
Documentation = "https://github.com/YOURUSERNAME/SLM-Trainer#readme"
Repository = "https://github.com/YOURUSERNAME/SLM-Trainer"
Issues = "https://github.com/YOURUSERNAME/SLM-Trainer/issues"
```

### 1.2 Update `setup.py`

Edit lines 32-33:
```python
author="Raja Kushal",
author_email="your.email@example.com",
```

Edit line 37:
```python
url="https://github.com/YOURUSERNAME/SLM-Trainer",
```

### 1.3 Update Version

When you're ready to publish a new version, update the version number in both:
- `pyproject.toml` line 7: `version = "0.1.0"`
- `setup.py` line 31: `version="0.1.0"`

Follow semantic versioning (MAJOR.MINOR.PATCH):
- MAJOR: Breaking changes
- MINOR: New features (backwards compatible)
- PATCH: Bug fixes

## Step 2: Install Build Tools

Install the necessary tools to build and upload your package:

```bash
pip install --upgrade build twine
```

- `build`: Creates distribution packages
- `twine`: Uploads packages to PyPI

## Step 3: Clean Previous Builds

Remove any old build artifacts:

```bash
# Windows
rmdir /s /q build dist slm_trainer.egg-info

# Linux/Mac
rm -rf build/ dist/ *.egg-info/
```

## Step 4: Build the Package

Build both source distribution and wheel:

```bash
python -m build
```

This creates two files in the `dist/` directory:
- `slm_trainer-0.1.0.tar.gz` (source distribution)
- `slm_trainer-0.1.0-py3-none-any.whl` (wheel distribution)

## Step 5: Test Upload to TestPyPI (Recommended)

Before uploading to the real PyPI, test with TestPyPI:

### 5.1 Create TestPyPI Account
Go to https://test.pypi.org/account/register/ and create an account.

### 5.2 Create API Token
1. Go to https://test.pypi.org/manage/account/
2. Scroll to "API tokens"
3. Click "Add API token"
4. Set the scope to "Entire account"
5. Copy the token (starts with `pypi-`)

### 5.3 Upload to TestPyPI

```bash
python -m twine upload --repository testpypi dist/*
```

When prompted:
- Username: `__token__`
- Password: Paste your TestPyPI API token

### 5.4 Test Installation from TestPyPI

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ slm-trainer
```

Test that it works:
```python
from slm_trainer import SLMTrainer
print("Success!")
```

## Step 6: Upload to Real PyPI

Once you've tested everything works:

### 6.1 Create PyPI Account
Go to https://pypi.org/account/register/ and create an account.

### 6.2 Create API Token
1. Go to https://pypi.org/manage/account/
2. Scroll to "API tokens"
3. Click "Add API token"
4. Set the scope to "Entire account" (or specific to slm-trainer after first upload)
5. Copy the token (starts with `pypi-`)

### 6.3 Upload to PyPI

```bash
python -m twine upload dist/*
```

When prompted:
- Username: `__token__`
- Password: Paste your PyPI API token

## Step 7: Verify Upload

1. Go to https://pypi.org/project/slm-trainer/
2. Check that all information is displayed correctly
3. Test installation:

```bash
pip install slm-trainer
```

## Step 8: Save Your API Token Securely

To avoid entering the token every time, create a `.pypirc` file:

**Windows:** `C:\Users\YOURUSERNAME\.pypirc`
**Linux/Mac:** `~/.pypirc`

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR-ACTUAL-TOKEN-HERE

[testpypi]
username = __token__
password = pypi-YOUR-TEST-TOKEN-HERE
```

**Important:** Never commit this file to git!

## Publishing Updates

When you need to release a new version:

1. **Update version number** in `pyproject.toml` and `setup.py`
2. **Update CHANGELOG** or release notes
3. **Clean old builds:** `rm -rf build/ dist/ *.egg-info/`
4. **Build:** `python -m build`
5. **Upload:** `python -m twine upload dist/*`

## Common Issues

### Issue: "Package already exists"
You can't upload the same version twice. Increment the version number.

### Issue: "Invalid credentials"
Make sure you're using `__token__` as username (not your PyPI username) and your API token as password.

### Issue: "File not found"
Make sure you've run `python -m build` and the `dist/` folder exists.

### Issue: Long description rendering error
Your README.md has formatting issues. Test with:
```bash
python -m twine check dist/*
```

## Best Practices

1. **Always test with TestPyPI first**
2. **Use semantic versioning**
3. **Keep a CHANGELOG**
4. **Tag releases in git:** `git tag v0.1.0 && git push --tags`
5. **Write clear release notes**
6. **Test installation in a fresh virtual environment**
7. **Never include sensitive data in your package**

## Automated Publishing with GitHub Actions

For automated releases, create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.x'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    - name: Build package
      run: python -m build
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*
```

Add your PyPI token to GitHub secrets as `PYPI_API_TOKEN`.

## Resources

- PyPI Documentation: https://packaging.python.org/
- Twine Documentation: https://twine.readthedocs.io/
- Semantic Versioning: https://semver.org/
- Python Packaging Guide: https://packaging.python.org/tutorials/packaging-projects/

## Quick Reference Commands

```bash
# Install tools
pip install --upgrade build twine

# Clean builds
rm -rf build/ dist/ *.egg-info/

# Build package
python -m build

# Check package
python -m twine check dist/*

# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Upload to PyPI
python -m twine upload dist/*
```

Good luck with your package!
