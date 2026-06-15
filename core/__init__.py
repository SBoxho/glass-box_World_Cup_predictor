"""Glass-Box World Cup Predictor — core package.

UI-agnostic. Contains all data, feature, model, explanation, and simulation logic.
``app`` (Streamlit) and ``api`` (FastAPI) import from here; this package imports from
neither. That one-directional dependency keeps the model layer reusable and testable.
"""

__version__ = "0.1.0"
