from pathlib import Path

from setuptools import setup

package_name = "voice_control"
configs = [str(p) for p in (Path(__file__).parent / "configs").glob("*.yaml")] + [
    str(Path(__file__).parent / "configs" / "keywords.txt")
]

setup(
    name=package_name,
    version="0.1.0",
    packages=[
        package_name,
        f"{package_name}.nodes",
        f"{package_name}.prompts",
        f"{package_name}.demos",
    ],
    package_data={f"{package_name}.prompts": ["*.txt"]},
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/configs", configs),
    ],
    install_requires=["numpy", "pyyaml"],
    zip_safe=True,
    maintainer="xjtu-edsa",
    maintainer_email="team@example.com",
    description="M2 voice control: KWS wake word + streaming ASR + rule/LLM hybrid parsing",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            f"asr_node = {package_name}.nodes.asr_node:main",
            f"llm_service = {package_name}.nodes.llm_service:main",
            f"parser_node = {package_name}.nodes.parser_node:main",
        ],
    },
)
