import codecs
import os
from setuptools import setup, find_packages

# these things are needed for the README.md show on pypi (if you dont need delete it)
here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

# you need to change all these
VERSION = '0.0.2'
DESCRIPTION = 'A package for tagging photo, all the tags will be combine to a sentence.'

setup(
    name="light_tagger",
    version=VERSION,
    author="bubbliiiing",
    author_email="bubbliiiing@qq.com",
    description=DESCRIPTION,
    # 长描述内容的类型设置为markdown
    long_description_content_type="text/markdown",
    # 长描述设置为README.md的内容
    long_description=long_description,
    # 使用find_packages()自动发现项目中的所有包
    packages=find_packages(),
    # 许可协议
    license='Apache-2.0',
    # 要安装的依赖包
    install_requires=[
        "pillow>=9.0.0",
        "requests>=2.30.0",
        "opencv-python",
        "onnxruntime-gpu",
        "numpy",
        "tqdm",
    ],
    # keywords=['python', 'menu', 'dumb_menu','windows','mac','linux'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
