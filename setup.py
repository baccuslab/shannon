try:
  from setuptools import setup
except:
  from distutils.core import setup

config = {
    'description': "Compute entropy, Shannon's information and several related quantities",
    'author': 'Pablo Jadzinsky and Lane McIntosh',
    'url': 'no url yet',
    'download_url': 'git@github.com:baccuslab/shannon.git',
    'author_email': 'pjadzinsky@gmail.com;lmcintosh@stanford.edu',
    'version': '0.1',
    'install_requires': ['nose', 'numpy', 'scipy'],
    'packages': ['shannon'],
    'scripts': [],
    'name', 'shannon'
    }

setup(**config)
