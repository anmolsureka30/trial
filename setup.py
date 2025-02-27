from setuptools import setup, find_packages

setup(
    name="insurance_verification",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.31.1",
        "pandas>=2.1.4",
        "numpy>=1.26.3",
        "plotly>=5.18.0",
        "langchain>=0.1.0",
        "langchain-community>=0.0.13",
        "sentence-transformers>=2.2.2",
        "chromadb>=0.4.22",
        "google-generativeai>=0.3.2",
        "python-dotenv>=1.0.0",
    ]
) 