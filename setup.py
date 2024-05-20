from setuptools import setup, find_packages

setup(
    name="clmpy",
    version="0.0.1",
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "clmpy.gruvae.train=clmpy.gruvae.train:main",
            "clmpy.gruvae.evaluate=clmpy.gruvae.evaluate:main",
            "clmpy.gruvae.generate=clmpy.gruvae.generate:main",
            "clmpy.gruvae.encode=clmpy.gruvae.encode:main"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3.12"
    ]
)