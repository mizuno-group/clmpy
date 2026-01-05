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
            "clmpy.transformerlatent.evaluate=clmpy.Transformer_latent.evaluate:main",
            "clmpy.transformerlatent.generate=clmpy.Transformer_latent.generate:main",
            "clmpy.transformerlatent.encode=clmpy.Transformer_latent.encode:main",
            "clmpy.transformerlatent_rpe.train=clmpy.Transformer_latent_RPE.train:main",
            "clmpy.transformerlatent_rpe.evaluate=clmpy.Transformer_latent_RPE.evaluate:main",
            "clmpy.transformerlatent_rpe.generate=clmpy.Transformer_latent_RPE.generate:main",
            "clmpy.transformerlatent_rpe.encode=clmpy.Transformer_latent_RPE.encode:main",
            "clmpy.transformerlatent_rope.train=clmpy.Transformer_latent_RoPE.train:main",
            "clmpy.transformerlatent_rope.evaluate=clmpy.Transformer_latent_RoPE.evaluate:main",
            "clmpy.transformerlatent_rope.generate=clmpy.Transformer_latent_RoPE.generate:main",
            "clmpy.transformerlatent_rope.encode=clmpy.Transformer_latent_RoPE.encode:main",
            "clmpy.transformerlatent_local.train=clmpy.Transformer_latent_local.train:main",  
            "clmpy.transformerlatent_local.evaluate=clmpy.Transformer_latent_local.evaluate:main",
            "clmpy.transformerlatent_local.generate=clmpy.Transformer_latent_local.generate:main",
            "clmpy.transformerlatent_local.encode=clmpy.Transformer_latent_local.encode:main",
            "clmpy.transformerlatent_mix.train=clmpy.Transformer_latent_mix.train:main",
            "clmpy.transformerlatent_mix.evaluate=clmpy.Transformer_latent_mix.evaluate:main",
            "clmpy.transformerlatent_mix.generate=clmpy.Transformer_latent_mix.generate:main",
            "clmpy.transformerlatent_mix.encode=clmpy.Transformer_latent_mix.encode:main",
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