import pathlib

from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="coinstac-sparse-dinunet",
    version="2.0.1",
    description="Distributed Sparse Neural Network implementation on COINSTAC.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/bishalth01/coinstac-sparse-dinunet",
    author="Bishal Thapaliya",
    author_email="bishalthapaliya16@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    packages=[
        'coinstac_sparse_dinunet', 'coinstac_sparse_dinunet.config', 'coinstac_sparse_dinunet.data',
        'coinstac_sparse_dinunet.metrics', 'coinstac_sparse_dinunet.distrib',
        'coinstac_sparse_dinunet.distrib.nodes',
        'coinstac_sparse_dinunet.nn', 'coinstac_sparse_dinunet.utils', 'coinstac_sparse_dinunet.vision'
    ],
    include_package_data=True,
    install_requires=['numpy', 'pillow', 'matplotlib', 'opencv-python-headless', 'pandas', 'sklearn']
)
