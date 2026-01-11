from setuptools import setup, find_packages

setup(
    name="scheffe-bo-benchmark",
    version="0.1.0",
    package_dir={"": "src"},
    # packages=find_packages(where="src"), # Found nothing because it is a file, not a dir
    py_modules=["scheffe_generator"],
    install_requires=[
        "numpy",
        "pandas", 
        "matplotlib",
        "scipy",
        "botorch",
        "gpytorch",
        "imageio",
        "torch"
    ],
)
