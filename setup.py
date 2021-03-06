from setuptools import setup, find_packages


def load_requirements(path):
    with open(path) as fin:
        return [
            line
            for line in map(lambda l: l.strip(), fin.readlines())
            if line and not line.startswith('#')
        ]


requirements = load_requirements('requirements.txt')

setup(
    name='sru_lm',
    version='0.1.77',
    packages=find_packages(),
    install_requires=requirements,
)
