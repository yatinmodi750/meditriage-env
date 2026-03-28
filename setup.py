from setuptools import setup, find_packages

setup(
    name="meditriage-env",
    version="1.0.0",
    description="OpenEnv medical triage environment for RL/LLM agents",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "pydantic>=2.0",
        "openai>=1.0",
    ],
    extras_require={
        "dev":  ["pytest", "pyyaml"],
        "demo": ["gradio"],
    },
)