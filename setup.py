from setuptools import find_packages, setup
import glob

setup(
    name="tune-experiment",
    packages=find_packages(),
    version="0.0.1",
    author="Ray Team",
    description="Tune experiment",
    url="https://github.com/Yard1/tune_experiment",
    install_requires=["ray[tune]"],
    scripts=list(glob.glob("bin/*"))
)
