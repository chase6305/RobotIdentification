from setuptools import find_packages, setup

package_name = "robot_identification_ros2"

setup(
    name=package_name,
    version="0.1.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md", "../LICENSE"]),
    ],
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="RobotServer maintainers",
    maintainer_email="maintainers@example.com",
    description=
    "ROS 2 recording and session tools for robot parameter identification",
    license="MIT",
    entry_points={
        "console_scripts": [
            "joint_state_recorder = robot_identification_ros2.joint_state_recorder:main",
        ],
    },
)
