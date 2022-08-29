<div id="top"></div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This Master project explores the application of reinforcement learning in a simulation problem involving car racing.  First, the problem is simplified using a 2-dimensional environment in order to verify that the learning is feasible. Secondly, the implementation of a custom PID that allows to verify that the communication interface between Assetto Corsa and a script is feasible and fast enough to collect data from the environment while sending actions to the car (in other words that the feedback loop is fast enough to ensure the control of the car). Third, the implementation of a reinforcement learning algorithm to control the car using an AI.

<p align="right">(<a href="#top">back to top</a>)</p>



### Built With

This section list all major frameworks/libraries used to bootstrap this project.

* [Ray](https://docs.ray.io/en/latest/#)
* [RLLIB](https://docs.ray.io/en/latest/rllib/index.html)
* [GYM](https://gym.openai.com/)
* [Assetto Corsa](https://www.instant-gaming.com/fr/1263-acheter-jeu-steam-assetto-corsa/)
* [Rocket Master](https://github.com/danuo/rocket-meister)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

please install requirements using :
* pip
  ```sh
  pip install -r requirements.txt
  ```

### Installation

1. Go to https://drive.google.com/drive/u/1/folders/178Kwb2h7dLC7Th5rQbz7sLuDLrCDoxjT and download the folder Data (to the base of the project)
2. For evaluating the current policy, download the folder SAC_evaluate_new_lidar
3. Work done with VS so better to download it https://visualstudio.microsoft.com/fr/

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage
In order to evaluate the policy, run the file ray_rollout.py

In order to train a new policy, run the file ray_train.py

In order to rune the controller PID, run the file unit_test_env

The are several options to select in env_config section inside ray_train.py and ray_rollout.py. Don't hesitate to look at env.py to see what they are doing.

lidar.py contains the code about the lidars

env.py contains the code about the environment, the reward function and all things needed to train a policy

controller.py contains everything needed for the controller

<p align="right">(<a href="#top">back to top</a>)</p>




<!-- CONTACT -->
## Contact

Florian Genilloud - florian.genilloud@gmail.com


<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
