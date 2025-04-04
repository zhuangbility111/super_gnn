import os
import subprocess
from setuptools import setup, find_packages
from setuptools.command.install import install

ops_setup_path = os.path.join(os.path.dirname(__file__), 'supergnn', 'ops', 'setup.py')

class CustomInstallCommand(install):
    def run(self):
        # Run the original install command
        super().run()
        # Run the setup.py in the subdirectory
        if os.path.exists(ops_setup_path):
            subprocess.check_call(['python', ops_setup_path, 'install'])

setup(
    name='super_gnn',
    version='0.1',
    packages=find_packages(),  # 包含所有的 Python 包
    install_requires=[
        # 列出项目的依赖项
        'requests',
        'numpy',
    ],
    description='a distributed gnns training system for cpu supercomputers',
    cmdclass={
        'install': CustomInstallCommand,
    },
)
