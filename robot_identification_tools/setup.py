from setuptools import find_packages, setup

package_name = "robot_identification_tools"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md", "LICENSE"]),
    ],
    python_requires=">=3.10",
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="RobotIdentification contributors",
    maintainer_email="maintainers@example.com",
    description="Offline robot identification datasets and artifact workflows",
    license="MIT",
    entry_points={
        "console_scripts": [
            "dataset_inspect = robot_identification_tools.dataset_inspect:main",
            "identification_artifact = robot_identification_tools.identification_artifact:main",
        ],
    },
)
