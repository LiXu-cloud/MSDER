from setuptools import setup, find_packages

setup(name='udo_db',
      version='0.20',
      description='a universal optimizer for database system',
      author='Junxiong Wang',
      author_email='chuangzhetianxia@gmail.coma',
      url='https://ovss.github.io/UDO/',
      download_url='https://github.com/OVSS/UDO/archive/refs/tags/0.01.tar.gz',
      keywords=['Database Optimization', 'OLAP', 'Index selection', 'System parameters'],
      packages=find_packages(),
      license='MIT',
      install_requires=[  # I get to this in a second
          'numpy',
          'h5py',
          'testresources1',
          'gym',
          'udo_optimization',
          'tensorflow',
          'keras',
          'keras-rl',
          'mysqlclient',
          'psycopg2'
      ],
      classifiers=[
          'Development Status :: 3 - Alpha',
          # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
          'Intended Audience :: Developers',  # Define that your audience are developers
          'Topic :: Database',
          'License :: OSI Approved :: MIT License',  # Again, pick a license
          'Programming Language :: Python :: 3',  # Specify which python versions that you want to support
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
      ])
