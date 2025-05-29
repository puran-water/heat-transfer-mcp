#!/usr/bin/env python3
"""Setup script for Heat Transfer MCP Server."""

from setuptools import setup, find_packages
import os

# Read requirements from requirements.txt
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    try:
        with open(requirements_path, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        return []

# Read long description from README
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    try:
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "MCP server for thermal engineering calculations with automatic unit conversion"

setup(
    name="heat-transfer-mcp",
    version="1.0.0",
    description="MCP server for thermal engineering calculations with automatic unit conversion",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="hvkshetry",
    author_email="hvkshetry@gmail.com",
    url="https://github.com/puran-water/heat-transfer-mcp",
    packages=find_packages(),
    include_package_data=True,
    install_requires=read_requirements(),
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    keywords="mcp heat-transfer thermal-engineering unit-conversion wastewater",
    project_urls={
        "Bug Reports": "https://github.com/puran-water/heat-transfer-mcp/issues",
        "Source": "https://github.com/puran-water/heat-transfer-mcp",
        "Documentation": "https://github.com/puran-water/heat-transfer-mcp/blob/main/README.md",
    },
)