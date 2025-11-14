"""
Setup script for EV Energy Optimization package.
"""

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ev-energy-optimizer",
    version="1.0.0",
    author="B.Tech CSE Student",
    author_email="your.email@example.com",
    description="ML-based energy consumption prediction for hybrid EVs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kapil7818/ML-Energy-Optimization-EVs",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "web": [
            "streamlit>=1.28.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ev-energy-train=main:main",
            "ev-energy-app=app:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
