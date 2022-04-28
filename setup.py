import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deq-vdp",
    version="0.0.1",
    author="Bruno M. Pacheco",
    author_email="mpacheco.bruno@gmail.com",
    description="DEQ to model a Van der Pol oscillator.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/brunompacheco/deq-vdp",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
