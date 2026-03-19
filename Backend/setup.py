from setuptools import setup, find_packages

classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Operating System :: Microsoft :: Windows",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
]

setup(
    name="rpfnet",
    version="0.1.4",
    description="RPFNet: Attack-Agnostic Tabular Data Poisoning Detection via Meta-Learned Relational Fingerprints",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/lakshancosta/RPFNet",
    author="Lakshan Costa",
    author_email="lakshancosta2@gmail.com",
    license="MIT",
    classifiers=classifiers,
    keywords="poisoning detection machine-learning data-science",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "fastapi",
        "uvicorn",
        "pydantic",
        "scikit-learn",
        "pandas",
        "numpy"
    ],
    package_data={
        "RPFNet": ["*.pt"],
        "rpfnet": ["*.pt"]
    }
)