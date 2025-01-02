from setuptools import setup, find_packages

setup(
    name='bedrock_agent',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "boto3==1.35.81",
        "pydantic==2.10.3",
        "python-dotenv==1.0.1",
        "loguru==0.7.3"
    ],
    package_data={
        '': ['pricing_info.json'],
    },
    include_package_data=True,
)