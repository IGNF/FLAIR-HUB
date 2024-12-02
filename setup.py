from setuptools import setup, find_packages

setup(
    name='flair-inc',
    version='0.0.1',  # Change this as needed or implement dynamic version reading from VERSION file
    author='Anatol Garioud',
    author_email='flair@ign.fr',
    description='baseline and demo code for FLAIR-INC dataset',
    long_description='French Land-cover from Arospace ImageRy',
    long_description_content_type='French Land-cover from Arospace ImageRy',
    url='https://github.com/IGNF/FLAIR-INC',
    project_urls={
        'Bug Tracker': 'https://github.com/IGNF/FLAIR-INC'
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
    python_requires='>=3.10',
    install_requires=[
        'geopandas>=0.10',
        'rasterio>=1.1.5',
        'omegaconf',
        'jsonargparse',
        "matplotlib>=3.8.2",
        "pandas>=2.1.4",
        "scikit-image>=0.22.0",
        "pillow>=9.3.0",
        "torchmetrics==1.2.0",
        "pytorch-lightning==2.1.1",
        "segmentation-models-pytorch==0.3.3",
        "albumentations==1.3.1",
        "tensorboard==2.15.1",
        "transformers>=4.41.2"
    ],
    include_package_data=True,
    package_data={
        '': ['*.yml']
    },
    entry_points={
        'console_scripts': [
            'flair_inc_detect=zone_detect.main:main',
            'flair_inc=flair_inc.main:main'
        ]
    }
)

