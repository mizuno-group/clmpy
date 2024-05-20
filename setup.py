from setuptools import setup, find_packages

setup(
    name="clmpy",
    version="0.0.1",
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "clmpy=clmpy.clmpy"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3.12"
    ]
)