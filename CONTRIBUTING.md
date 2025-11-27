# Contributing to Offshore

Thank you for your interest in contributing to Offshore! This document provides guidelines and instructions for contributing.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

## 📜 Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before contributing.

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:

   ```bash
   git clone https://github.com/YOUR_USERNAME/offshore.git
   cd offshore
   ```

3. **Add upstream remote**:

   ```bash
   git remote add upstream https://github.com/Udhay-Adithya/offshore.git
   ```

4. **Create a branch** for your changes:

   ```bash
   git checkout -b feature/your-feature-name
   ```

## 🤝 How to Contribute

### Types of Contributions We Welcome

- 🐛 **Bug fixes**: Found a bug? Feel free to fix it!
- ✨ **New features**: Have an idea? Propose it first via an issue
- 📚 **Documentation**: Improvements to README, docstrings, or tutorials
- 🧪 **Tests**: Additional test coverage is always appreciated
- 🔧 **Refactoring**: Code quality improvements
- 🌐 **Translations**: Help translate documentation

### Contribution Ideas

- Add new model architectures (TCN, N-BEATS, Informer, etc.)
- Implement additional technical indicators
- Add support for more data sources
- Improve backtesting with transaction costs and slippage
- Create Jupyter notebook tutorials
- Add hyperparameter optimization (Optuna integration)
- Implement real-time prediction API

## 💻 Development Setup

### Prerequisites

- Python 3.11+
- Git
- Virtual environment tool (venv, conda, etc.)

### Installation

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Or install from requirements
pip install -r requirements.txt
pip install pytest pytest-cov black isort mypy ruff
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_models.py -v

# Run specific test
pytest tests/test_models.py::TestLSTMClassifier::test_forward_pass -v
```

### Code Quality Tools

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/
```

## 📝 Coding Standards

### Python Style Guide

- Follow [PEP 8](https://pep8.org/) style guide
- Use [Black](https://black.readthedocs.io/) for formatting (line length: 100)
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Use type hints for all function signatures

### Code Structure

```python
"""
Module docstring explaining the purpose.
"""
from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


class MyClass:
    """
    Class docstring with description.
    
    Args:
        param1: Description of param1.
        param2: Description of param2.
    
    Example:
        >>> obj = MyClass(param1=1, param2="test")
        >>> obj.method()
    """
    
    def __init__(self, param1: int, param2: str) -> None:
        self.param1 = param1
        self.param2 = param2
    
    def method(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method docstring.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features).
        
        Returns:
            Output tensor of shape (batch, num_classes).
        """
        pass
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

Examples:

```
feat(models): add TCN classifier architecture
fix(data): handle missing values in OHLCV data
docs(readme): add Indian stock examples
test(metrics): add ROC-AUC edge case tests
```

## 🔄 Pull Request Process

### Before Submitting

1. **Update your branch** with the latest upstream changes:

   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run all checks**:

   ```bash
   # Format
   black src/ tests/
   isort src/ tests/
   
   # Lint
   ruff check src/ tests/
   
   # Test
   pytest tests/ -v
   ```

3. **Update documentation** if needed

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Tests added/updated for changes
- [ ] Documentation updated if needed
- [ ] All tests pass locally
- [ ] Commit messages follow convention

### PR Template

When creating a PR, please include:

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Refactoring

## Testing
Describe how you tested the changes.

## Checklist
- [ ] Tests pass
- [ ] Code formatted
- [ ] Documentation updated
```

### Review Process

1. Submit PR with clear description
2. Automated checks will run
3. Maintainers will review
4. Address any feedback
5. Once approved, PR will be merged

## 🐛 Reporting Issues

### Bug Reports

When reporting bugs, please include:

1. **Environment info**:
   - Python version
   - OS
   - Package versions (`pip list`)

2. **Steps to reproduce**:

   ```python
   # Minimal code example
   from src.models import LSTMClassifier
   model = LSTMClassifier(...)
   # Error occurs here
   ```

3. **Expected behavior**: What should happen

4. **Actual behavior**: What actually happens

5. **Error messages**: Full traceback if applicable

### Feature Requests

When requesting features:

1. **Use case**: Why is this needed?
2. **Proposed solution**: How should it work?
3. **Alternatives**: Other approaches considered
4. **Additional context**: Examples, references, etc.

## 📞 Getting Help

- 💬 **Discussions**: Use GitHub Discussions for questions
- 🐛 **Issues**: Use GitHub Issues for bugs/features
- 📧 **Email**: For sensitive matters, contact maintainers directly

## 🙏 Acknowledgments

Thank you to all contributors who help make Offshore better!

---

**Happy Contributing!** 🚀
