from pathlib import Path
from setuptools import setup, find_packages

with (Path(__file__).resolve().parent / "README.md").open("r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='flair-hub',
    version='2.0.0',
    author='IGNF | Anatol Garioud',
    author_email='flair@ign.fr',
    description='Baseline and demo code for the FLAIR-HUB dataset',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/IGNF/FLAIR-HUB',
    license='Apache-2.0',
    project_urls={
        'Bug Tracker': 'https://github.com/IGNF/FLAIR-HUB/issues'
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition'
    ],
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    python_requires=">=3.11,<3.14",
    install_requires=[
        'torch>=2.6,<3',
        'torchvision>=0.21,<1',
        'timm==1.0.15',
        'numpy>=1.26,<3',
        'PyYAML>=6,<7',
        'scipy>=1.11,<2',
        'shapely>=2,<3',
        'tqdm>=4.66,<5',
        'opencv-python-headless>=4.8,<5',
        'geopandas==1.1.2',
        'rasterio==1.4.3',
        'pandas==2.2.3',
        'scikit-image==0.25.2',
        'scikit-learn==1.6.1',
        'pillow',
        'torchmetrics==1.7.0',
        'pytorch-lightning==2.5.1',
        'segmentation-models-pytorch==0.4.0',
        'tensorboard==2.19.0',
        'pyarrow==23.0.1',
        'pyogrio==0.10.0',
        'safetensors==0.5.3'
    ],
    extras_require={'download': ['huggingface-hub>=0.27,<2']},
    include_package_data=True,
    package_data={'': ['*.yml', '*.yaml']},
    entry_points={
        'console_scripts': [
            'flairhub=flair_hub.main:main',
            'flairhub_zonal=flair_zonal_detection.main:main',
        ]
    }
)
