![Splash Figure](docs/splash_fig.png)

# RoVi-Aug
RoVi-Aug uses state-of-the-art diffusion models to augment robotics demonstration datasets with different robots and viewpoints.

This repository provides the official implementation of [RoVi-Aug: Robot and Viewpoint Augmentation for Cross-Embodiment Robot Learning
](https://rovi-aug.github.io/).

# Installation
The installation process will install companion repos into a folder called `deps` inside of this repository. Since there are several companents of the pipeline (see below), there will be four conda environments created.

Run the following script from the root directory of the repository and follow the prompts provided to decide what to install or not install:
```
./install.sh
```

# Code Structure
![Pipeline](docs/pipeline.png)

# Citation
If you found this paper / code useful, please consider citing: 

```
@inproceedings{
    chen2024roviaug,
    title={RoVi-Aug: Robot and Viewpoint Augmentation for Cross-Embodiment Robot Learning},
    author={Lawrence Yunliang Chen and Chenfeng Xu and Karthik Dharmarajan and Muhammad Zubair Irshad and Richard Cheng and Kurt Keutzer and Masayoshi Tomizuka and Quan Vuong and Ken Goldberg},
    booktitle = {Conference on Robot Learning (CoRL)},
    address  = {Munich, Germany},
    year = {2024},
}
```

# License
This project is licensed under the MIT License. See the LICENSE file for more details.

# Contact
For questions or issues, please reach out to Lawrence Chen or open an issue.