from setuptools import setup, find_packages

setup(
    name='flair-hub',
    version='0.1.0',  # Change this as needed or implement dynamic version reading from VERSION file
    author='IGNF | Anatol Garioud',
    author_email='flair@ign.fr',
    description='baseline and demo code for FLAIR-HUB dataset',
    long_description='French Land-cover from Arospace ImageRy',
    long_description_content_type='French Land-cover from Arospace ImageRy',
    url='https://github.com/IGNF/FLAIR-HUB',
    project_urls={
        'Bug Tracker': 'https://github.com/IGNF/FLAIR-HUB'
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',        
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition'
    ],
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    python_requires='==3.11.0',
    install_requires=[
        'geopandas==1.0.1',
        'rasterio==1.4.3',
        'omegaconf',
        'jsonargparse',
        "matplotlib>=3.8.2",
        "pandas==2.2.3",
        "scikit-image==0.25.2",
        "scikit-learn==1.6.1",
        "pillow>=9.3.0",
        "torchmetrics==1.2.0",
        "pytorch-lightning==2.1.1",
        "segmentation-models-pytorch==0.4.0",
        "tensorboard==2.15.1",
        "codecarbon>=2.8.1"
    ],
    include_package_data=True,
    package_data={
        '': ['*.yml']
    },
    entry_points={
        'console_scripts': [
            'flairhub=flair_hub.main:main',
            'flairhub_zonal=flair_zonal_detection.main:main',
        ]
    }
)

