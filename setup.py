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
            "clmpy.gruvae.encode=clmpy.GRU_VAE.encode:main",
            "clmpy.gru.train=clmpy.GRU.train:main",
            "clmpy.gru.evaluate=clmpy.GRU.evaluate:main",
            "clmpy.gru.generate=clmpy.GRU.generate:main",
            "clmpy.gru.encode=clmpy.GRU.encode:main",
            "clmpy.transformerlatent.train=clmpy.Transformer_latent.train:main",
            "clmpy.transformerlatent.train_mlp=clmpy.Transformer_latent.train_MLP:main",
            "clmpy.transformerlatent.evaluate=clmpy.Transformer_latent.evaluate:main",
            "clmpy.transformerlatent.generate=clmpy.Transformer_latent.generate:main",
            "clmpy.transformerlatent.encode=clmpy.Transformer_latent.encode:main",
            "clmpy.transformervae.train=clmpy.Transformer_VAE.train:main",
            "clmpy.transformervae.evaluate=clmpy.Transformer_VAE.evaluate:main",
            "clmpy.transformervae.generate=clmpy.Transformer_VAE.generate:main",
            "clmpy.transformervae.encode=clmpy.Transformer_VAE.encode:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3.12"
    ]
)