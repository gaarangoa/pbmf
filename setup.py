import setuptools 

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="BiomarkerDiscovery",
    version="0.0.1",
    author="Gustavo Arango, Elly Kipkogei, Damian Bikiel",
    author_email="gustavo.arango@astrazeneca.com",
    description="A custom loss function for identifying predictive biomarker(s)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    #url="https://bitbucket.astrazeneca.com/scm/ecoeds/biomarkerdiscovery",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires = [],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Restricted",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    entry_points=''' '''
)