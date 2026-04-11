from setuptools import setup, find_packages

setup(
    name="llm_workout",
    version="0.1.0",
    description="A from-scratch educational LLM library and training set.",
    author="LLM Workout",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "numpy"
    ],
    python_requires=">=3.8",
)
