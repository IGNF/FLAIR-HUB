from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='flair-hub',
    version='1.0.0',
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
    python_requires='>=3.11',
    install_requires=[
        'geopandas==1.1.2',
        'rasterio==1.4.3',
        'omegaconf',
        'jsonargparse',
        'matplotlib',
        'pandas==2.2.3',
        'scikit-image==0.25.2',
        'scikit-learn==1.6.1',
        'pillow',
        'torchmetrics==1.7.0',
        'pytorch-lightning==2.5.1',
        'segmentation-models-pytorch==0.4.0',
        'tensorboard==2.19.0',
        'pyarrow==19.0.0',
        'pyogrio==0.10.0',
        'safetensors==0.5.3'
    ],
    include_package_data=True,
    package_data={'': ['*.yml']},
    entry_points={
        'console_scripts': [
            'flairhub=flair_hub.main:main',
            'flairhub_zonal=flair_zonal_detection.main:main',
        ]
    }
)
