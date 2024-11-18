from setuptools import setup

setup(
    name='rag_kit',
    version='0.1.0',
    description='A short description of your project',
    author='Uladzimir Tsimochaŭ',
    author_email='ucmochau@outlook.com',
    packages=['rag_kit'],
    install_requires=[
        'chromadb',
        'cohere',
    ],
)