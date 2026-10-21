# The Thymio escape

![EPFL MICRO-452](https://img.shields.io/badge/EPFL-MICRO--452%20%C2%B7%20Fall%202025-FF0000)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)
![OpenCV 4.12](https://img.shields.io/badge/OpenCV-4.12-5C3EE8?logo=opencv&logoColor=white)
![NumPy 2.3](https://img.shields.io/badge/NumPy-2.3-013243?logo=numpy&logoColor=white)
![Robot: Thymio](https://img.shields.io/badge/robot-Thymio-0A9FD8)
[![Report: Jupyter notebook](https://img.shields.io/badge/report-Jupyter%20notebook-F37626?logo=jupyter&logoColor=white)](Group_T24_report.ipynb)

For our 2025 fall semester project for the _Basics of mobile robotics_ course at EPFL, we chose to save Thymio from its world. To escape, the robot needs to join the portal as soon as possible and avoid any obstacles in his way.

> 📓 **How does Thymio find its way out?** The [project report](Group_T24_report.ipynb) takes it apart piece by piece: ArUco vision, A\* path planning, local obstacle avoidance and Kalman filtering, with the figures and tuning behind each choice.

| EPFL                                     | Full name                          |
|:-----------------------------------------|:-----------------------------------|
| [341237](https://people.epfl.ch/341237)  | Louis Grange                       |
| [362819](https://people.epfl.ch/362819)  | Alessandro De Zen                  |
| [328691](https://people.epfl.ch/328691)  | Verter Stoilov                     |

| Project GUI                                     | External top view                     |
|:-----------------------------------------|:-----------------------------------|
| <img src="assets/project.webp" height="200" alt="Picture of Thymio's Adventure" />  | <img src="assets/project-cam.webp" height="200" alt="Picture of Thymio's Adventure" />                       |

## ⚙️ Quickstart

1. Clone the repository
2. Open a terminal and execute `python -m venv /PATH/TO/PROJECT/.venv` to create a virtual environment
3. Activate the environment using `source /PATH/TO/PROJECT/.venv/bin/activate`
4. Execute `pip install -r requirements.txt` to install the project dependencies
5. Launch the main file using `python3 /PATH/TO/PROJECT/main.py`
