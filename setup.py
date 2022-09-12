from setuptools import find_packages, setup

setup(
    name='dcgpen',
    version='0.1dev',
    packages=find_packages(),
    license='',
    python_requires=">=3.8.*",
    install_requires=[
        'torch>=1.7',
        'opencv-python',
        'torchvision',
        'scipy',
        'tqdm',
        'lmdb',
        'ninja',
        'numpy',
        'scikit-image',
        'pillow',
    ],
    include_package_data=True,
    package_data={'': ['*.yaml', '*.cu', '*.cpp', '*.h']},
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown'
)
