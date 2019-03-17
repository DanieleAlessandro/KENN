#!/usr/bin/env python

from distutils.core import setup


setup(name='kenn',
      version='1.0',
      license='bsd-3-clause',
      description='Knowledge Enhanced Neural Networks',
      author='Alessandro Daniele, Luciano Serafini',
      author_email='daniele@fbk.eu, serafini@fbk.eu',
      url='https://github.com/DanieleAlessandro/KENN',
      packages=['kenn', 'kenn.delta_functions'],
      keywords=['kenn', 'Statistical Relational Learning', 'Neural Symbolic Learning', 'Learning with logic'],
      install_requires=['numpy', 'tensorflow'],
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Programming Language :: Python :: 2.7'
      ],
      download_url='https://github.com/DanieleAlessandro/KENN/archive/v1.0.tar.gz'
      )
