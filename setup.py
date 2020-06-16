import setuptools

setuptools.setup(
    name = "haloprops",
    version = "0.0.1",
    author = "Yangyao Chen",
    author_email = "yangyaochen.astro@foxmail.com",
    description = "relation of halo properties", 
    python_requires = ">=3.4",
    install_requires = ["h5py", "scikit-learn", "numpy", "scipy"],
    packages = ["haloprops"],
    package_data = {'haloprops': ['data/*']}
)