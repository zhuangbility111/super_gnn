import os
from setuptools import setup, find_packages

ops_setup_path = os.path.join(os.path.dirname(__file__), 'supergnn', 'ops', 'setup.py')

setup(
    name='super_gnn',
    version='0.1',
    packages=find_packages(),  # 包含所有的 Python 包
    install_requires=[
        # 列出项目的依赖项
        'requests',
        'numpy',
    ],
    setup_requires=[
        f'file://{ops_setup_path}',
    ],
    description='a distributed gnns training system for cpu supercomputers',
)
