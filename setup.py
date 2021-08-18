from setuptools import setup, find_packages

setup(
  name = 'rgn2-replica',
  packages = find_packages(),
  version = '0.0.1',
  license='CreativeCommons4.0',
  description = 'RGN2-REPLICA: Replicating a SoTA model for Protein Folding with no homologs (wip)',
  author = 'Eric Alcaide',
  author_email = 'ericalcaide1@gmail.com',
  url = 'https://github.com/hypnopump/rgn2-replica',
  keywords = [
    'artificial intelligence',
    'bioinformatics',
    'protein folding', 
    'protein structure prediction'
  ],
  install_requires=[
    'einops>=0.3',
    'numpy',
    'torch>=1.6',
    'sidechainnet',
    'proDy',
    'tqdm',
    'mp-nerf',
    'datasets>=1.10',
    'transformers>=4.2',
    'x-transformers>=0.16.1',
    'pytorch-lightning>=1.4',
    'wandb'
  ],
  setup_requires=[
    'pytest-runner',
  ],
  tests_require=[
    'pytest'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Programming Language :: Python :: 3.7',
  ],
)
