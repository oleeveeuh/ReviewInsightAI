# Contributing to ReviewInsight AI

Thank you for your interest in contributing to ReviewInsight AI! This document provides guidelines and instructions for contributing to the project.

## Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/yourusername/reviewinsight-ai.git
   cd reviewinsight-ai
   ```

2. **Set up a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-langchain.txt  # For LangChain features
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## Code Style

- Follow PEP 8 guidelines for Python code
- Use type hints where appropriate
- Write docstrings for all functions and classes
- Keep functions focused and modular

## Testing

Before submitting a pull request, please:
- Test your changes with sample data
- Ensure the API server starts without errors
- Verify that existing functionality is not broken

## Submitting Changes

1. Create a new branch for your feature/bugfix
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

3. Push to your fork
   ```bash
   git push origin feature/your-feature-name
   ```

4. Create a pull request with a clear description of your changes

## Project Structure

- `src/agent/` - Agent orchestration and tools
- `src/analysis/` - Batch processing and evaluation
- `src/processing/` - Data merging and validation
- `src/prompts/` - Prompt templates
- `api/` - FastAPI REST service
- `dashboard/` - Streamlit dashboard
- `notebooks/` - Exploratory analysis

## Questions?

Feel free to open an issue for questions or discussion about potential contributions.
