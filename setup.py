from setuptools import find_packages
from setuptools import setup

with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content if 'git+' not in x]

setup(name='Wine_recommender_api',
      version="1.0",
      description="Project Description",
      packages=find_packages(),
      test_suite = 'tests',
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      scripts=['scripts/Wine_recommender_api-run'],
      zip_safe=False)
