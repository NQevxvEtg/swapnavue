# setup.py
from setuptools import setup

# This setup configuration is designed to work with your specific file structure.
setup(
    name='swapnavue',
    version='0.0.1',
    # This tells setuptools that the root for the modules is the 'src' directory.
    package_dir={'': 'src'},
    # This explicitly lists all the .py files that should be made importable.
    py_modules=[
        "consolidation",
        "encoders",
        "functions",  # <-- Add this line
        "network",
        "spatial_pooler",
        "temporal_memory",
    ],
)