from setuptools import setup, find_packages

setup(
    name='super_gnn',
    version='0.1',
    packages=find_packages(),  # 包含所有的 Python 包
    install_requires=[
        # 列出项目的依赖项
        'requests',
        'numpy',
    ],
    author='zhuangbility111',
    author_email='zhuang.c.aa@m.titech.ac.jp',
    description='a distributed gnns training system for cpu supercomputers',
    url='https://github.com/zhuangbility111/super_gnn',
)
