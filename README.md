# Awesome mmWave Radar Perception

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

<p align="center">
  <img src="https://media.tenor.com/ZQIRn2UeT8YAAAAd/dragon-radar.gif" alt="Dragon Radar scanning for Dragon Balls" width="70%" />
</p>

<p align="center">
  <sub>Dragon Radar scanning for breakthroughs in mmWave perception. Animation from Dragon Ball by Akira Toriyama (Bird Studio/Shueisha).</sub>
</p>

A curated list of awesome papers, datasets, and resources for millimeter-wave (mmWave) radar perception, with a focus on deep learning methods.

For a broader overview of radar perception beyond mmWave, including research involving modalities such as UWB or LiDAR, please refer to the more comprehensive [awesome-radar-perception](https://github.com/ZHOUYI1023/awesome-radar-perception) repository, which covers a wider spectrum of radar technologies and applications.

Author: Armor

Contact: htkstudy@163.com

Last Updated: **October 15, 2025**

## Inclusion Criteria

This list is curated based on the following standards, in order of priority:

- **Reproducibility:** Papers with publicly available code or datasets are the highest priority.
- **Research Focus:** Articles that explore the intersection of deep learning with mmWave radar perception, presenting novel or interesting ideas, are included even without open-source code.
- **Timeliness:** The latest relevant publications from top-tier academic conferences and journals are regularly added to keep the list current.

## Contents

- [🌐 Radar Foundational Technologies](#-radar-foundational-technologies)
  - [Signal Processing & Parameter Estimation](#signal-processing--parameter-estimation)
  - [High-Resolution Imaging & SAR Imaging](#high-resolution-imaging--sar-imaging)
  - [Data Synthesis, Enhancement & Simulation](#data-synthesis-enhancement--simulation)
  - [Foundational Models & Representation Learning About Radar Signals](#foundational-models--representation-learning-about-radar-signals)
- [🤖 Embodied AI & Robotics](#-embodied-ai--robotics)
- [🚗 Autonomous Driving & Drone](#-autonomous-driving--drone)
  - [3D Object Detection & Classification](#3d-object-detection--classification)
  - [Semantic & Instance Segmentation](#semantic--instance-segmentation)
  - [Scene Flow & Motion Prediction](#scene-flow--motion-prediction)
  - [Radar Odometry & Ego-Motion Estimation](#radar-odometry--ego-motion-estimation)
  - [Multi-Object Tracking](#multi-object-tracking)
  - [Simultaneous Localization and Mapping (SLAM)](#simultaneous-localization-and-mapping-slam)
  - [Sensor Fusion Techniques](#sensor-fusion-techniques)
- [🩺 Human Sensing & Healthcare](#-human-sensing--healthcare)
  - [Human Activity Recognition (HAR)](#human-activity-recognition-har)
  - [Gesture Recognition & Hand Tracking](#gesture-recognition--hand-tracking)
  - [Occupancy, Presence & Fall Detection](#occupancy-presence--fall-detection)
  - [Pose Estimation & Skeletal Tracking & Human Motion](#pose-estimation--skeletal-tracking--human-motion)
  - [Vital Signs & Biometric Identification](#vital-signs--biometric-identification)
  - [Sleep Monitoring](#sleep-monitoring)
  - [Fatigue driving detection](#fatigue-driving-detection)
  - [Identity Recognition & Person Re-identification](#identity-recognition--person-re-identification)
- [🌱 Agriculture Areas](#-agriculture-areas)
- [🏭 Industrial Areas](#-industrial-areas)
- [🔒 Forensics & Privacy Security](#-forensics--privacy-security)
- [📦 Other Areas](#-other-areas)

## 🌐 Radar Foundational Technologies


### Signal Processing & Parameter Estimation

| Title | Code | Publish / Year | Keywords |
| :--- | :---: | :---: | :--- |
| [A Data-centric Supervised Transfer Learning Framework for DOA Estimation with Array Imperfections](https://arxiv.org/abs/2504.13394) | [Code](https://github.com/zzb-nice/DOA_est_Master) | arxiv / 2025 | DOA Estimation, Transfer Learning, Data-centric |
| [NEAR: Neural Electromagnetic Array Response](https://proceedings.mlr.press/v267/bu25c.html) | [Code](https://github.com/J1mmyYu1/NEAR) | PMLR / 2025 | Array Response, Neural Fields, Electromagnetics, DOA Estimation |
| [Advancing Single-Snapshot DOA Estimation with Siamese Neural Networks for Sparse Linear Arrays](https://ieeexplore.ieee.org/abstract/document/10890598/) | [Code](https://github.com/ruxinzh/SNNS_SLA) | ICASSP / 2025 | DOA Estimation, Siamese Networks, Sparse Arrays |
| [Advancing High-Resolution and Efficient Automotive Radar Imaging through Domain-Informed 1D Deep Learning](https://ieeexplore.ieee.org/document/10890731) | - | ICASSP / 2025 | Radar Imaging, High-Resolution, DOA Estimation |
| [Model-Based Knowledge-Driven Learning Approach for Enhanced High-Resolution Automotive Radar Imaging](https://ieeexplore.ieee.org/abstract/document/10974998) | [Code](https://github.com/ruxinzh/SR-SPECNet) | IEEE Transactions on Radar Systems / 2025 | Radar Imaging, Super-Resolution, Deep Unfolding, DOA Estimation |
| [BFAR: improving radar odometry estimation using a bounded false alarm rate detector](https://link.springer.com/article/10.1007/s10514-024-10176-2) | - | Autonomous Robots / 2024 | Radar Odometry, SLAM, False Alarm Rate |
| [Single-Frame MIMO Radar Velocity Vector Estimation via Multi-Bounce Scattering](https://ieeexplore.ieee.org/document/11103510) | - | IEEE Transactions on Computational Imaging / 2025 | Velocity Estimation, MIMO, Multi-Bounce |

### High-Resolution Imaging & SAR Imaging

| Title | Code | Publish / Year | Keywords |
| :--- | :---: | :---: | :--- |
| [Unsupervised 3D SAR Imaging Network Based on Generative Adversary Learning](https://ieeexplore.ieee.org/document/10919030) | [Code](https://github.com/WMMWWM/Unsupervised-3D-SAR-Imaging-Network-Based-on-Generative-Adversary-Learning) |  IEEE Transactions on Antennas and Propagation / 2025 | 3D SAR, Imaging, Unsupervised, GAN, Synthetic aperture radar |
| [RF4D:Neural Radar Fields for Novel View Synthesis in Outdoor Dynamic Scenes](https://arxiv.org/abs/2505.20967) | [Code](https://github.com/zhan0618/RF4D_code) | arxiv / 2025 | Neural Fields, Novel View Synthesis, 4D Radar |
| [Millimeter-Wave SAR imaging of Sparse Trajectory via Untrained Complex-valued Neural Network](https://arxiv.org/abs/2505.00536) | [Code](https://github.com/Armorhtk/mmUSAR) | IEEE Transactions on Aerospace and Electronic Systems / 2025 | SAR Imaging, Sparse Trajectory, Untrained Neural Network |


### Data Synthesis, Enhancement & Simulation

| Title | Code | Publish / Year | Keywords |
| :--- | :---: | :---: | :--- |
| [Wideband RF Radiance Field Modeling Using Frequency-embedded 3D Gaussian Splatting](https://arxiv.org/abs/2505.20714) | [Code](https://github.com/sim-2-real/Wideband3DGS) | arxiv / 2025 | Radiance Field, 3D Gaussian Splatting, Wideband RF |
| [Talk is Not Always Cheap: Promoting Wireless Sensing Models with Text Prompts](https://arxiv.org/abs/2504.14621) | [Code](https://github.com/zk-b612/WiTalk) | arxiv / 2025 | Wireless Sensing, Text Prompts, LLM |
| [One Snapshot is All You Need: A Generalized Method for mmWave Signal Generation](https://ieeexplore.ieee.org/abstract/document/10416806) | - |  IEEE INFOCOM / 2025 | Signal Generation, Data Synthesis, Simulation |
| [Synthetic Radar Signal Generator for Human Motion Analysis](https://ieeexplore.ieee.org/abstract/document/10804837) | - | IEEE Transactions on Radar Systems / 2025 | Signal Generation, Simulation, motion-capture |
| [Diffusion^2: Turning 3D Environments into Radio Frequency Heatmaps](https://arxiv.org/abs/2510.02274) | [Project](https://rfvision-project.github.io/) | arxiv / 2025 | RF Heatmap, Diffusion Model, Channel Modeling |
| [Inverse Rendering of Near-Field mmWave MIMO Radar for Material Reconstruction](https://ieeexplore.ieee.org/document/10892639/) | [Code](https://github.com/nihofm/inverse-radar-rendering) | IEEE Journal of Microwaves / 2025 | Inverse Rendering, Material Reconstruction, Near-Field |
| [Simulate Any Radar: Attribute-Controllable Radar Simulation via Waveform Parameter Embedding](https://arxiv.org/abs/2506.03134) | [Code](https://github.com/zhuxing0/SA-Radar) | arxiv / 2025 | Controllable Simulation, Data Synthesis, 2D/3D object detection, radar semantic segmentation |
| [L2RDaS: Synthesizing 4D Radar Tensors for Model Generalization via Dataset Expansion](https://arxiv.org/abs/2503.03637) | [Project](https://github.com/kaist-avelab/K-Radar) | arxiv / 2025 | Data Synthesis, 4D Radar, Generalization |
| [RF-Diffusion: Radio Signal Generation via Time-Frequency Diffusion](https://dl.acm.org/doi/10.1145/3636534.3649348) | [Code](https://github.com/yourusername/RF-Diffusion) | MobiCom  / 2024 | Radio Signal Generation, Time-Frequency Diffusion |

### Foundational Models & Representation Learning About Radar Signals

| Title | Code | Publish / Year | Keywords |
| :--- | :---: | :---: | :--- |
| [Unlocking Interpretability for RF Sensing: A Complex-Valued White-Box Transformer](https://arxiv.org/abs/2507.21799) | [Code](https://github.com/rfcrate/RF_CRATE) | arxiv / 2025 | Interpretability, RF Sensing, Transformer, Complex-Valued |
| [Multi-View Radar Detection Transformer with Differentiable Positional Encoding](https://ieeexplore.ieee.org/document/10889849/) | - | ICASSP / 2025 | Multi-View Radar, Detection, Transformer |
| [Towards Foundational Models for Single-Chip Radar](https://arxiv.org/abs/2509.12482) | [Project](https://wiselabcmu.github.io/grt/) | arxiv / 2025 | Foundation Model, Self-Supervised, Representation |
| [SpikingRTNH: Spiking Neural Network for 4D Radar Object Detection](https://arxiv.org/abs/2502.00074) | [Code](https://github.com/kaist-avelab/K-Radar/tree/main/models/skeletons) | arxiv / 2025 | Spiking Neural Network, 4D Radar, Object Detection |

## 🤖 Embodied AI & Robotics

| Title | Code | Publish / Year | Keywords |
| :--- | :---: | :---: | :--- |
| [Towards Robust and Secure Embodied AI: A Survey on Vulnerabilities and Attacks](https://arxiv.org/abs/2502.13175) | - | arxiv / 2025 | Survey, Embodied AI, Security, Robustness |
| [FuseGrasp: Radar-Camera Fusion for Robotic Grasping of Transparent Objects](https://ieeexplore.ieee.org/document/10909339) | - | IEEE Transactions on Mobile Computing / 2025 | Robotic Grasping, Sensor Fusion, Material recognition, Synthetic aperture radar |
| [MITO: A Millimeter-Wave Dataset and Simulator for Non-Line-of-Sight Perception](https://arxiv.org/abs/2502.10259) | [Code](https://github.com/signalkinetics/MITO_Codebase/tree/main) | arxiv / 2025 | Dataset, Simulator, NLOS, Synthetic aperture radar |
| [Non-Line-of-Sight 3D Object Reconstruction via mmWave Surface Normal Estimation](https://dl.acm.org/doi/10.1145/3711875.3729138) | [Code](https://github.com/signalkinetics/mmNorm) | MobiSys / 2025 | 3D Reconstruction, NLOS, Surface Normal, Synthetic aperture radar |
| [Loosely coupled 4D-Radar-Inertial Odometry for Ground Robots](https://arxiv.org/abs/2411.17289) | [Code](https://github.com/robotics-upo/4D-Radar-Odom) | arxiv / 2025 | Robot Odometry, 4D Radar, SLAM |


## 🚗 Autonomous Driving & Drone


### 3D Object Detection & Classification

| Title | Code | Publish / Year | Keywords |
| :--- | :---: | :---: | :--- |
| [RICCARDO: Radar Hit Prediction and Convolution for Camera-Radar 3D Object Detection](https://openaccess.thecvf.com/content/CVPR2025/html/Long_RICCARDO_Radar_Hit_Prediction_and_Convolution_for_Camera-Radar_3D_Object_CVPR_2025_paper.html) | [Code](https://github.com/longyunf/riccardo) | CVPR / 2025 | 3D Object Detection, Sensor Fusion, Hit Prediction |
| [DoppDrive: Doppler-Driven Temporal Aggregation for Improved Radar Object Detection](https://arxiv.org/abs/2508.12330) | [Project](https://yuvalhg.github.io/DoppDrive/) | arXiv / 2025 | Object Detection, Temporal Aggregation, Doppler |
| [RadarNeXt: Real-Time and Reliable 3D Object Detector Based On 4D mmWave Imaging Radar](https://arxiv.org/abs/2501.02314) | [Code](https://github.com/Pay246-git468/RadarNeXt) | arXiv / 2025 | 3D Object Detection, 4D Radar, Real-Time |
| [RADLER: Radar Object Detection Leveraging Semantic 3D City Models and Self-Supervised Learning](https://openaccess.thecvf.com/content/CVPR2025W/PBVS/html/Luo_RADLER_Radar_Object_Detection_Leveraging_Semantic_3D_City_Models_and_CVPRW_2025_paper.html) | [Project](https://gpp-communication.github.io/RADLER/) | CVRP 2025 | Object Detection, 3D City Models, Self-Supervised |
| [Beyond Pillars: Advancing 3D Object Detection with Salient Voxel Enhancement of LiDAR-4D Radar Fusion](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5518546) | [Code](https://github.com/icdm-adteam/SVEFusion) | SSRN / 2025 | 3D Object Detection, LiDAR, Radar Fusion |
| [RCDFNet: A 4-D Radar and Camera Dual-Level Fusion Network for 3-D Object Detection](https://ieeexplore.ieee.org/abstract/document/11006930) | [Code](https://github.com/D-Hourse/RCDFNet/tree/master) | IEEE Sensors Journal / 2025 | 3D Object Detection, Sensor Fusion, Dual-Level Fusion |
| [V2X-R: Cooperative LiDAR-4D Radar Fusion for 3D Object Detection with Denoising Diffusion](https://arxiv.org/abs/2411.08402) | [Code](https://github.com/ylwhxht/V2X-R) | CVPR / 2025 | 3D Object Detection, V2X, Sensor Fusion, Diffusion Model |
| [RCDINO: Enhancing Radar-Camera 3D Object Detection with DINOv2 Semantic Features](https://arxiv.org/abs/2508.15353) | [Code](https://github.com/OlgaMatykina/RCDINO) | arXiv / 2025 | 3D Object Detection, Sensor Fusion, DINOv2 |

### Semantic & Instance Segmentation

| Title | Code | Publish / Year | Keywords |
| :--- | :---: | :---: | :--- |
| [RETR: Multi-View Radar Detection Transformer for Indoor Perception](https://neurips.cc/virtual/2024/poster/95530) | [Code](https://github.com/merlresearch/radar-detection-transformer) | NeurIPS / 2024 | Multi-View Radar, Detection, Transformer |
| [RC-ROSNet: Fusing 3D Radar Range-Angle Heat Maps and Camera Images for Radar Object Segmentation](https://ieeexplore.ieee.org/document/11112643) | [Code](https://github.com/Zhuanglong2/RC-ROSNet) | IEEE Transactions on Circuits and Systems for Video Technology  / 2025 | Object Segmentation, Sensor Fusion, Range-Angle Map |
| [M2CNet: LiDAR 3D Semantic Segmentation Based on Multi-level Multi-view Cross-attention Fusion for Autonomous Vehicles](https://ieeexplore.ieee.org/document/11125962/) | [Code](https://github.com/Terminal-lidar/M2CNet) | IEEE Transactions on Vehicular Technology / 2025 | Semantic Segmentation, Sensor Fusion, Cross-Attention |
| [RadarMask: A Novel End-to-End Sparse Millimeter-Wave Radar Sequence Panoptic Segmentation and Tracking Method](https://ieeexplore.ieee.org/abstract/document/11128555) | [Code](https://github.com/yb-guo/RadarMask) | ICRA / 2025 | Panoptic Segmentation, Tracking, Sparse Radar |
| [4D Radar And Vision Fusion Detection Model Based On Segmentation-assisted](https://www.researchsquare.com/article/rs-5358941/v1) | [Code](https://github.com/Huniki/RVASANET) | arXiv / 2024 | 4D Radar, Vision Fusion, Segmentation |
| [AdaPKC: PeakConv with Adaptive Peak Receptive Field for Radar Semantic Segmentation](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f6b22ac37beb5da61efd4882082c9ecd-Abstract-Conference.html) | [Code](https://github.com/lihua199710/AdaPKC) | NeurIPS / 2024 | Semantic Segmentation, Adaptive Receptive Field |
| [TARSS-Net: Temporal-Aware Radar Semantic Segmentation Network](https://neurips.cc/virtual/2024/poster/96608) | [Code](https://github.com/zlw9161/TARSS-NeT) | NeurIPS / 2024 | Semantic Segmentation, Temporal-Aware |


### Scene Flow & Motion Prediction

| Title | Code | Publish / Year | Keywords |
| :--- | :---: | :---: | :--- |
| [Self-Supervised Diffusion-Based Scene Flow Estimation and Motion Segmentation with 4D Radar](https://ieeexplore.ieee.org/document/10974572) | [Code](https://github.com/nubot-nudt/RadarSFEMOS) | IRAL / 2025 | Scene Flow, Motion Segmentation, Self-Supervised |

### Radar Odometry & Ego-Motion Estimation

| Title | Code | Publish / Year | Keywords |
| :--- | :---: | :---: | :--- |
| [Digital Beamforming Enhanced Radar Odometry](https://ieeexplore.ieee.org/document/11127292) | [Code](https://github.com/SenseRoboticsLab/DBE-Radar) | ICRA / 2025 | Radar Odometry, Digital Beamforming|
| [DRO: Doppler-Aware Direct Radar Odometry](https://arxiv.org/abs/2504.20339) | [Code](https://github.com/utiasASRL/dro) | RSS / 2025 | Radar Odometry, SLAM, Doppler |
| [GaRLIO: Gravity enhanced Radar-LiDAR-Inertial Odometry](https://arxiv.org/abs/2502.07703) | [Code](https://github.com/ChiyunNoh/GaRLIO) | arXiv / 2025 | Odometry, SLAM, Sensor Fusion |
| [Ground-Optimized 4D Radar-Inertial Odometry via Continuous Velocity Integration using Gaussian Process](https://arxiv.org/abs/2502.08093) | [Code](https://github.com/wooseongY/Go-RIO) | arXiv / 2025 | Radar Odometry, SLAM, Gaussian Process |
| [Equi-RO: A 4D mmWave Radar Odometry via Equivariant Networks](https://arxiv.org/abs/2509.20674)| - | arxiv / 2025 | Radar Odometry, Equivariant Network |
| [EFEAR-4D：Ego-velocity Filtering for Efficient and Accurate 4D radar Odometry](https://ieeexplore.ieee.org/document/10685149) | [Code](https://github.com/CLASS-Lab/EFEAR-4D) | IEEE Robotics and Automation Letters / 2024 | Radar Odometry, Ego-velocity, SLAM |

### Multi-Object Tracking

| Title | Code | Publish / Year | Keywords |
| :--- | :---: | :---: | :--- |
| [Real-Time Multi-object Tracking and Identification Using Sparse Point-Cloud Data from Low-Cost mmWave Radar](https://link.springer.com/chapter/10.1007/978-3-031-92011-0_12) | - | Robot Intelligence Technology and Applications / 2024 | MOT, Tracking, Object Identification |
| [USVTrack: USV-Based 4D Radar-Camera Tracking Dataset for Autonomous Driving in Inland Waterways](https://arxiv.org/abs/2506.18737) | [Dataset](https://github.com/USVTrack/USVTrack) | arXiv / 2025 | Dataset, Tracking, Radar-Camera Fusion |

### Simultaneous Localization and Mapping (SLAM)

| Title | Code | Publish / Year | Keywords |
| :--- | :---: | :---: | :--- |
| [Doppler-SLAM: Doppler-Aided Radar-Inertial and LiDAR-Inertial SLAM](https://arxiv.org/abs/2504.11634) | [Code](https://github.com/Wayne-DWA/Doppler-SLAM) | IEEE Robotics and Automation Letters / 2025 | SLAM, Sensor Fusion, Doppler, Odometry |
| [S^3E: Self-Supervised State Estimation for Radar-Inertial System](https://arxiv.org/abs/2509.25984) |  - | arxiv / 2025 | SLAM, Self-Supervised |
| [MapKD: Unlocking Prior Knowledge with Cross-Modal Distillation for Efficient Online HD Map Construction](https://arxiv.org/abs/2508.15653) | [Code](https://github.com/2004yan/MapKD2026) | arxiv / 2025 | HD Map, Cross-Modal, Distillation |
| [Towards Dense and Accurate Radar Perception via Efficient Cross-Modal Diffusion Model](https://ieeexplore.ieee.org/document/10592769) | [Code](https://github.com/ZJU-FAST-Lab/Radar-Diffusion) | IEEE Robotics and Automation Letters / 2024 | Radar Perception, Cross-Modal, Diffusion Model |


### Sensor Fusion Techniques

| Title | Code | Publish / Year | Keywords |
| :--- | :---: | :---: | :--- |
| [RadarRGBD: A Multi-Sensor Fusion Dataset for Perception with RGB-D and mmWave Radar](https://arxiv.org/abs/2505.15860) | [Dataset](https://github.com/song4399/RadarRGBD) | arxiv / 2025 | Dataset, Sensor Fusion, RGB-D |
| [Artemis: Contour-Guided 3-D Sensing and Localization With mmWave Radar for Infrastructure-Assisted AVs](https://ieeexplore.ieee.org/document/10891135) | - | IEEE Internet of Things Journal / 2025 | Infrastructure-based, Localization, 3D Sensing |
| [CoVeRaP: Cooperative Vehicular Perception through mmWave FMCW Radars](https://www.arxiv.org/abs/2508.16030) | [Code](https://github.com/John1001Song/FMCW_Vehicle_Fusion) | arxiv / 2025 | Cooperative Perception, V2X, Sensor Fusion |
| [Ultra-High-Frequency Harmony: mmWave Radar and Event Camera Orchestrate Accurate Drone Landing](https://dl.acm.org/doi/10.1145/3715014.3722048) | [Project](https://mme-loc.github.io/) | SenSys / 2025 | Drone Landing, UAV, Sensor Fusion, Event Camera |
| [Rehearse-3d: A Multi-Modal Emulated Rain Dataset for 3d Point Cloud De-Raining](https://arxiv.org/abs/2504.21699) | [Dataset](https://sporsho.github.io/REHEARSE3D) | arxiv / 2025 | Dataset, De-Raining, Sensor Fusion |
| [4D-ROLLS: 4D Radar Occupancy Learning via LiDAR Supervision](https://arxiv.org/abs/2505.13905) | [Code](https://github.com/CLASS-Lab/4D-ROLLS) | arxiv / 2025 | Occupancy Learning, Sensor Fusion, LiDAR Supervision |
| [MIPD: A Multi-Sensory Interactive Perception Dataset for Embodied Intelligent Driving](https://ieeexplore.ieee.org/abstract/document/11112801) | [Dataset](https://github.com/BUCT-IUSRC/Dataset__MIPD) | IEEE Transactions on Intelligent Transportation Systems / 2025 | Dataset, Driver Monitoring, Sensor Fusion |
| [MetaOcc: Spatio-Temporal Fusion of Surround-View 4D Radar and Camera for 3D Occupancy Prediction with Dual Training Strategies](https://arxiv.org/abs/2501.15384) | [Code](https://github.com/LucasYang567/MetaOcc) | arxiv / 2025 | Occupancy Prediction, Sensor Fusion, Spatio-Temporal |


## 🩺 Human Sensing & Healthcare


### Human Activity Recognition (HAR)

| Title | Code | Publish / Year | Keywords |
| :--- | :---: | :---: | :--- |
| [Human Activity Recognition Based on Multipath Fusion in Non-line-of-sight Corner](https://ieeexplore.ieee.org/document/11177013) | [Code](https://github.com/tlz1111/Multipath-Fusion-Network) | IEEE Internet of Things Journal / 2025 | HAR, NLOS, Multipath Fusion |
| [Enhancing Activity Recognition: Motion Waveform Preprocessing from Millimeter-Wave Radar Data for Transformer-Based Classification](https://arxiv.org/abs/2403.02324) | [Code](https://github.com/Alan-cs1/MmWave-Motion-Waveform-HAR) | IEEE International Conference on Multimedia and Expo Workshops (ICMEW) / 2025 | HAR, Preprocessing, Transformer |
| [Resolution-Adaptive Micro-Doppler Spectrogram for Human Activity Recognition](https://arxiv.org/abs/2411.15057) | - | arxiv / 2025 | HAR, Micro-Doppler, Adaptive Resolution |
| [A Novel Multimodal LLM-Driven RF Sensing Method for Human Activity Recognition](https://ieeexplore.ieee.org/document/11003262) | [Code](https://github.com/ci4r/CI4R-MULTI3) | International Conference on Microwave, Antennas & Circuits (ICMAC) / 2025 | HAR, LLM, Multimodal |
| [RadMamba: Efficient Human Activity Recognition through Radar-based Micro-Doppler-Oriented Mamba State-Space Model](https://arxiv.org/abs/2504.12039) | [Code](https://github.com/lab-emi/AIRHAR) | arxiv / 2025 | HAR, Mamba, State-Space Model, Micro-Doppler |
| [DGAR: A Unified Domain Generalization Framework for RF-Based Human Activity Recognition](https://arxiv.org/abs/2503.17667) | [Code](https://github.com/Junshuo-Lau/HUST_HAR_LFM) | arxiv / 2025 | HAR, Domain Generalization |
| [RadProPoser: Uncertainty-Aware Human Pose Estimation and Activity Classification from Raw Radar Data](https://arxiv.org/abs/2508.03578) | [Code](https://github.com/jonasmueler/RadProPoser) | arxiv / 2025 | Pose Estimation, HAR, Uncertainty |


### Gesture Recognition & Hand Tracking
| Title | Code | Publish / Year | Keywords |
| :--- | :---: | :---: | :--- |
| [mmWave Radar-based Unsupervised Gesture Recognition via Image-Aligned Heterogeneous Domain Transfer](https://ieeexplore.ieee.org/document/11180134) | [Code](https://github.com/onlinehuazai/mmGesture) | IEEE Transactions on Mobile Computing / 2025 | Gesture Recognition, Unsupervised, Domain Transfer |
| [mmPencil: Toward Writing-Style-Independent In-Air Handwriting Recognition via mmWave Radar and Large Vision-Language Model](https://dl.acm.org/doi/10.1145/3749504) | [Dataset](https://www.kaggle.com/datasets/mmpencil/mmpencil-dataset/data) | Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies / 2025 | Handwriting Recognition, In-Air Writing, VLM |
| [Human-Centered Fully Adaptive Radar for Gesture Recognition in Smart Environments](https://ieeexplore.ieee.org/abstract/document/11126867) | [Dataset](https://github.com/ci4r) |  IEEE Transactions on Human-Machine Systems / 2025 | Gesture Recognition, Adaptive Radar, Human-Centered |
| [mmEgoHand: Egocentric Hand Pose Estimation and Gesture Recognition with Head-mounted Millimeter-wave Radar and IMU](https://arxiv.org/abs/2501.13805) | [Code](https://github.com/WhisperYi/mmVR) | arxiv / 2025 | Hand Pose, Gesture Recognition, Egocentric |
| [mmDigit: A Real-Time Digit Recognition Framework in Air-Writing Using FMCW Radar](https://ieeexplore.ieee.org/document/10771807/) | [Dataset](https://github.com/Tjkjjc/gesture) | IEEE INTERNET OF THINGS JOURNAL / 2025 | In-Air Writing, Digit Recognition |
| [mmHand: Toward Pixel-Level-Accuracy Hand Localization Using a Single Commodity mmWave Device](https://ieeexplore.ieee.org/document/10906525) | - | IEEE Internet of Things Journal / 2025 | Hand Localization, Gesture Recognition |
| [Rodar: Robust Gesture Recognition Based on mmWave Radar Under Human Activity Interference](https://ieeexplore.ieee.org/document/10533689) | [Code](https://github.com/Xlab2024/MvDeFormer) | IEEE Transactions on Mobile Computing / 2024 | Gesture Recognition, HAR, Interference |


### Occupancy, Presence & Fall Detection
| Title | Code | Publish / Year | Keywords |
| :--- | :---: | :---: | :--- |
| [BSENSE: In-vehicle Child Detection and Vital Sign Monitoring with a Single mmWave Radar and Synthetic Reflectors](https://dl.acm.org/doi/abs/10.1145/3666025.3699352) | [Code](https://github.com/mtang724/BSENSE-in-cabin) | SenSys / 2024 | Child Detection, In-cabin Sensing, Vital Signs |
| [Exploration of Low-Cost but Accurate Radar-Based Human Motion Direction Determination](https://arxiv.org/abs/2507.22567) | [Code](https://github.com/JoeyBGOfficial/Low-Cost-Accurate-Radar-Based-Human-Motion-Direction-Determination) | arxiv / 2025 | Motion Direction, Low-Cost Radar |
| [End-to-End Radar Human Segmentation with Differentiable Positional Encoding](https://eusipco2025.org/wp-content/uploads/pdfs/0000631.pdf) | - | EUSIPCO / 2025 | Human Segmentation, Transformer |
| [MVDoppler-Pose: Multi-Modal Multi-View mmWave Sensing for Long-Distance Self-Occluded Human Walking Pose Estimation](https://ieeexplore.ieee.org/abstract/document/11093407) | [Code](https://github.com/gogoho88/MVDoppler-Pose) | CVPR / 2025 | Pose Estimation, Multi-View, Self-Occlusion |
| [SelaFD:Seamless Adaptation of Vision Transformer Fine-tuning for Radar-based Human Activity Recognition](https://ieeexplore.ieee.org/document/10888271/) | [Code](https://github.com/wangyijunlyy/SelaFD) | ICASSP / 2025 | HAR, Vision Transformer, Activity Learning |
| [Advanced Millimeter-Wave Radar System for Real-Time Multiple-Human Tracking and Fall Detection](https://www.mdpi.com/1424-8220/24/11/3660) | [Code](https://github.com/DarkSZChao/MMWave_Radar_Human_Tracking_and_Fall_detection) | Sensors / 2024 | Fall Detection, Multi-Human Tracking |


### Pose Estimation & Skeletal Tracking & Human Motion

| Title | Code | Publish / Year | Keywords |
| :--- | :---: | :---: | :--- |
| [Learning to Analyze Human Skeletal by Radar–Camera Supervision](https://ieeexplore.ieee.org/document/10930633) | [Code](https://github.com/zylofor/STC-HSANet) | WACV / 2024 | Skeleton Estimation, Sensor Fusion, Supervision |
| [RadarLLM: Empowering Large Language Models to Understand Human Motion from Millimeter-wave Point Cloud Sequence](https://arxiv.org/abs/2504.09862) | [Project](https://inowlzy.github.io/RadarLLM/) | CVPR / 2024 | Human Motion, LLM, Point Cloud |
| [Few-shot Human Motion Recognition through Multi-Aspect mmWave FMCW Radar Data](https://arxiv.org/abs/2501.11028) | [Code](https://github.com/MountainChenCad/channel-DN4) | arxiv / 2025 | HAR, Few-shot Learning, Multi-Aspect |

### Vital Signs & Biometric Identification
| Title | Code | Publish / Year | Keywords |
| :--- | :---: | :---: | :--- |
| [ReBP: Short-Term Blood Pressure Estimation by Reconstructing PPG Signals Based on mmWave Radar](https://ieeexplore.ieee.org/document/11028948) | [Code](https://github.com/JialMa/ReBP) |IEEE Sensors Journal / 2025 | Blood Pressure, Vital Signs, rPPG |
| [Atrial Fibrillation Detection via Contactless Radio Monitoring and Knowledge Transfer](https://www.nature.com/articles/s41467-025-59482-y) | [Code](https://github.com/yyuqin/Atrial-Fibrillation-Detection) | nature communications / 2025 | Healthcare, Atrial Fibrillation, Vital Signs |
| [Hierarchical and Multimodal Data for Daily Activity Understanding](https://arxiv.org/abs/2504.17696) | [Dataset](https://alregib.ece.gatech.edu/software-and-datasets/darai-daily-activity-recordings-for-artificial-intelligence-and-machine-learning/) | arxiv / 2025 | Dataset, HAR, Multimodal |
| [CardiacMamba: A Multimodal RGB-RF Fusion Framework with State Space Models for Remote Physiological Measurement](https://arxiv.org/abs/2502.13624) | [Code](https://github.com/WuZheng42/CardiacMamba) | arxiv / 2025 | Vital Signs, rPPG, Mamba, Sensor Fusion |
| [RadEye: Tracking Eye Motion Using FMCW Radar](https://dl.acm.org/doi/10.1145/3706598.3713775) | - | Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems / 2025 | Eye Tracking, Gaze Estimation |
| [Realistic Facial Expression Reconstruction Using Millimeter Wave](https://ieeexplore.ieee.org/abstract/document/10904120) | - | IEEE Transactions on Mobile Computing / 2025 | Facial Reconstruction, Generative Models |

### Sleep Monitoring

| Title | Code | Publish / Year | Keywords |
| :--- | :---: | :---: | :--- |

### Fatigue driving detection

| Title | Code | Publish / Year | Keywords |
| :--- | :---: | :---: | :--- |
| [CarVision: Vehicle Ranging and Tracking Using mmWave Radar for Enhanced Driver Safety](https://www.computer.org/csdl/proceedings-article/percom/2025/355100a215/27fizQ2avXG) | [Code](https://github.com/srajib826/CarVision) | IEEE International Conference on Pervasive Computing and Communications (PerCom) / 2025 | Vehicle Tracking, Driver Safety, Ranging |
| [PhysDrive: A Multimodal Remote Physiological Measurement Dataset for In-vehicle Driver Monitoring](https://arxiv.org/abs/2507.19172) | [Dataset](https://github.com/WJULYW/PhysDrive-Dataset) | arxiv / 2025 | Dataset, Driver Monitoring, Vital Signs, rPPG |

### Identity Recognition & Person Re-identification

| Title | Code | Publish / Year | Keywords |
| :--- | :---: | :---: | :--- |
| [I Sense You Fast: Simultaneous Action and Identity Inference by Slimming Multi-Branch RadarNet](https://ieeexplore.ieee.org/document/11005642) | [Code](https://github.com/MagicalLiHua/PolyLite-RadarNet) | IEEE Transactions on Mobile Computing / 2025 | HAR, Person Identification, Model Slimming |
| [Open-Set Gait Recognition from Sparse mmWave Radar Point Clouds](https://ieeexplore.ieee.org/document/11080220) | [Code](https://github.com/rmazzier/OpenSetGaitRecognition_PCAA) | IEEE Sensors Journal / 2025 | Gait Recognition, Open-Set, Biometrics |
| [mmReID: Person Reidentification Based on Commodity Millimeter-Wave Radar](https://ieeexplore.ieee.org/document/10937945) | [Code](https://github.com/ci4r/mmReID) | IEEE Internet of Things Journal  / 2025 | Person Re-identification, Biometrics |
| [Through-Wall Cross-Domain User Identification via Lip Movement Micro-Doppler and MIMO Radar: an Unsupervised Domain Adaptation Approach](https://ieeexplore.ieee.org/abstract/document/11153694) | [Code](https://github.com/KaiYCode/Lip-TWCDID) | IEEE Transactions on Mobile Computing / 2025 | User Identification, Cross-Domain, Lip Movement | 


## 🌱 Agriculture Areas

| Title | Code | Publish / Year | Keywords |
| :--- | :---: | :---: | :--- |
| [Hydra: Accurate Multi-Modal Leaf Wetness Sensing with mm-Wave and Camera Fusion](https://dl.acm.org/doi/10.1145/3636534.3690662) | [Code](https://github.com/liuyime2/MobiCom24-Hydra) | ACM MobiCom / 2024 | Agriculture, Leaf Wetness, Sensor Fusion |


## 🏭 Industrial Areas

| Title | Code | Publish / Year | Keywords |
| :--- | :---: | :---: | :--- |
| [CRFusion: Fine-Grained Object Identification Using RF-Image Modality Fusion](https://ieeexplore.ieee.org/document/10835118) | - | IEEE Transactions on Mobile Computing / 2025 | Sensor Fusion, Object Identification, RF-Image |

## 🔒 Forensics & Privacy Security

| Title | Code | Publish / Year | Keywords |
| :--- | :--- | :--- | :--- |
| - | - | - | - |

## 📦 Other Areas

| Title | Code | Publish / Year | Keywords |
| :--- | :--- | :--- | :--- |
| - | - | - | - |

## Contribution

Contributions are always welcome! This list is actively maintained.

Please read the [**contribution guidelines**](CONTRIBUTING.md) before submitting your pull request.

## License

[![CC0](https://mirrors.creativecommons.org/presskit/buttons/88x31/svg/cc-zero.svg)](https://creativecommons.org/publicdomain/zero/1.0/)