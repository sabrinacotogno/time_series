from setuptools  import setup, find_packages

setup(
    name='time_series_utils',
    version='0.1.0',
    description='Utilities for time series data processing',
    author='Sabrina Cotogno',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.8',
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'ipykernel',
        'jupyter',
        'notebook',
        'streamlit',],
    extras_require={
        'dev': [
            'pytest',
            'twine',],
    }
)