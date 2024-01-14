import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="arhdfa",
    version="0.0.1",
    author="Dongah Kim",
    author_email="donga0223@gmail.com",
    description="A package of Autoregressive heteroskedasticiy dynamic factor analysis (ARH DFA) model.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    url="https://github.com/donga0223/comparing-forecasting-performance/archdfa",
    py_modules=['archdfa']
)
