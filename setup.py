from setuptools import setup, find_packages

setup(
    name="dlp-ai-monitor",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "pydantic",
        "catboost",
        "pandas",
        "numpy",
        "scikit-learn",
        "mlflow",
    ],
)