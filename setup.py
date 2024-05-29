from setuptools import setup, find_packages

setup(
    name="clmpy",
    version="0.0.1",
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "clmpy.gruvae.train=clmpy.GRU_VAE.train:main",
            "clmpy.gruvae.evaluate=clmpy.GRU_VAE.evaluate:main",
            "clmpy.gruvae.generate=clmpy.GRU_VAE.generate:main",
            "clmpy.gruvae.encode=clmpy.GRU_VAE.encode:main"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3.12"
    ]
)