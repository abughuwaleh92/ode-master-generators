from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ode-master-generators",
    version="2.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Production-ready ODE Master Generators System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ode-master-generators",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "sympy>=1.12",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pyarrow>=11.0.0",
        "pyyaml>=6.0",
        "dill>=0.3.6",
        "tqdm>=4.65.0",
    ],
    entry_points={
        "console_scripts": [
            "ode-generate=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.md"],
    },
)