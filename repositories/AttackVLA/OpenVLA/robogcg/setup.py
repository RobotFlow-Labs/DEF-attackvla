from setuptools import setup, find_packages

setup(
    name="roboGCG",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch>=2.0",
        "transformers>=4.4",
        "pillow>=9.0.0",
        "numpy>=1.20.0",
        "tqdm>=4.62.0",
        "matplotlib>=3.4.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0", 
        "tensorboard>=2.8.0",
        "nltk>=3.6.0",
        "pytest>=6.2.5",
    ],
    extras_require={
        "dev": [
            "black",
            "isort",
            "flake8",
            "pytest-cov",
        ],
    },
    python_requires=">=3.9",
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for adversarial attacks on vision-language robot control models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/roboGCG",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)