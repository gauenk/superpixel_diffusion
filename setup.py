#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
from setuptools import setup, find_packages
# from distutils.core import setup
import os
import stat
import shutil
import platform
import sys
import site
import glob

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# -- file paths --
long_description="""Code for publication"""
setup(
    name='superpixel_diffusion',
    version='100.100.100',
    description='Some shared basics',
    long_description=long_description,
    url='https://github.com/gauenk/superpixel_diffusion',
    author='Kent Gauen',
    author_email='gauenk@purdue.edu',
    license='MIT',
    keywords='neural network',
    install_requires=[],
    package_dir={"": "lib"},
    packages=find_packages("lib"),
    package_data={'': ['*.so']},
    include_package_data=True,
    # ext_modules=[
    #     CUDAExtension('superpixel_cuda', [
    #         # -- search --
    #         "lib/superpixel_diffusion/pwd/pair_wise_distance_cuda_source.cu",
    #         "lib/superpixel_diffusion/pwd/grad_pair_wise_distance_cuda_source.cu",
    #         "lib/superpixel_diffusion/pybind.cpp",
    #         # 'lib/superpixel_diffusion/nsp/nsa_agg_cuda_source.cu',
    #     ],
    #                   extra_compile_args={'cxx': ['-g','-w'],
    #                                       'nvcc': ['-O2','-w']})
    # ],
    cmdclass={'build_ext': BuildExtension},
)
