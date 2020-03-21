from setuptools import setup, find_packages

setup(
    name='torch_audioset',
    version='0.0.1',
    author='Haochen Wang',
    author_email='whcw@cmu.edu',
    packages=find_packages(exclude=("tests", )),
    python_requires='>=3',
    install_requires=[],
    package_data={
        '': ['*.yml', '*.json'],
    },
    zip_safe=False  # accessing config files without using pkg_resources. lazy for now
)
