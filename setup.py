import os
from setuptools import setup


README_PATH = 'README.rst'
LONG_DESC = ''
if os.path.exists(README_PATH):
    with open(README_PATH) as readme:
        LONG_DESC = readme.read()

INSTALL_REQUIRES = ['Pillow']
PACKAGE_NAME = 'image2cad'
PACKAGE_DIR = 'src'

setup(
    name=PACKAGE_NAME,
    version='0.1.1',
    author='x',
    author_email='x',
    maintainer='x',
    maintainer_email='x',
    description=(
        "Python-image2cad"
    ),
    long_description=LONG_DESC,
    license='',
    keywords='image2cad OCR Python',
    url='',
    packages=[PACKAGE_NAME],
    package_dir={PACKAGE_NAME: PACKAGE_DIR},
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    entry_points={
        'console_scripts': ['{0} = {0}.{0}:main'.format(PACKAGE_NAME)]
    },
    classifiers=[
        'Programming Language :: Python',        
        'Programming Language :: Python :: 3',
    ]
)