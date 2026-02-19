from setuptools import setup, find_packages

def get_requirements(file_path: str) -> list[str]:
    """
    This function will return a list of requirements
    """
    requirements = []
    with open(file_path) as file:
        requirements = file.read().splitlines()
        if "-e ." in requirements:
            requirements.remove("-e .")
    return requirements


setup(
    name="loan_approval_project",
    version="0.1.0",
    author="Rishi",
    author_email="rishikanthsuggula@email.com",
    description="Loan approval prediction and data preprocessing pipeline",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
    python_requires=">=3.8",
)
