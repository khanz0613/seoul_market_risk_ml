from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="seoul-market-risk-ml",
    version="0.1.0",
    author="NH Digital Hanaro",
    description="Seoul Market Risk ML System for Small Business Revenue Prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nh-digital/seoul-market-risk-ml",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "mypy>=1.7.1",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.26.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "seoul-risk-predict=src.cli:main",
            "seoul-risk-train=src.models.train:main",
            "seoul-risk-preprocess=src.preprocessing.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
)