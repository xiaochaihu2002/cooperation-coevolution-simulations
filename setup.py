from setuptools import setup, find_packages

setup(
    name="cooperation-coevolution",
    version="1.0.0",
    description="Critical selection intensity in adaptive network evolution",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "networkx>=3.0",
        "joblib>=1.2.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
    ],
)
