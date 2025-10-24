# 🚀 Awesome mmWave Radar Perception [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

<p align="center">
  <img src="https://media.tenor.com/ZQIRn2UeT8YAAAAd/dragon-radar.gif" alt="Dragon Radar scanning for Dragon Balls" width="70%" />
</p>

<p align="center">
  <sub>Dragon Radar scanning for breakthroughs in mmWave perception. Animation from Dragon Ball by Akira Toriyama (Bird Studio/Shueisha).</sub>
</p>

**Welcome to the Awesome mmWave Radar Perception repository!**

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
  - [Point Cloud Processing](#point-cloud-processing)
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
- [🤝 How to Contribute](#contribution)

## 🌐 Radar Foundational Technologies


### Signal Processing & Parameter Estimation

| Title | Code | Publication | Date | Summary |
| :--- | :---: | :---: | :---: | :--- |
| [A Data-centric Supervised Transfer Learning Framework for DOA Estimation with Array Imperfections](https://arxiv.org/abs/2504.13394) | [Code](https://github.com/zzb-nice/DOA_est_Master) | arXiv | 2025 | Addresses DOA estimation accuracy degradation caused by array imperfections through a supervised transfer learning framework that adapts models trained on ideal arrays to real-world conditions. |
| [NEAR: Neural Electromagnetic Array Response](https://proceedings.mlr.press/v267/bu25c.html) | [Code](https://github.com/J1mmyYu1/NEAR) | PMLR | 2025 | Models electromagnetic array responses as continuous neural fields to improve DOA estimation by capturing complex antenna patterns and mutual coupling effects. |
| [Advancing Single-Snapshot DOA Estimation with Siamese Neural Networks for Sparse Linear Arrays](https://ieeexplore.ieee.org/abstract/document/10890598/) | [Code](https://github.com/ruxinzh/SNNS_SLA) | ICASSP | 2025 | Achieves accurate single-snapshot DOA estimation for sparse linear arrays using Siamese neural networks that learn similarity metrics between spatial patterns. |
| [Advancing High-Resolution and Efficient Automotive Radar Imaging through Domain-Informed 1D Deep Learning](https://ieeexplore.ieee.org/document/10890731) | N/A | ICASSP | 2025 | Enhances automotive radar imaging resolution and computational efficiency by incorporating radar signal processing domain knowledge into 1D convolutional neural networks. |
| [Model-Based Knowledge-Driven Learning Approach for Enhanced High-Resolution Automotive Radar Imaging](https://ieeexplore.ieee.org/abstract/document/10974998) | [Code](https://github.com/ruxinzh/SR-SPECNet) | IEEE Transactions on Radar Systems | 2025 | Combines model-based signal processing with deep unfolding networks to achieve super-resolution radar imaging while maintaining physical interpretability. |
| [BFAR: improving radar odometry estimation using a bounded false alarm rate detector](https://link.springer.com/article/10.1007/s10514-024-10176-2) | N/A | Autonomous Robots | 2024 | Improves radar odometry accuracy by implementing a bounded false alarm rate detector that filters spurious detections while preserving valid targets. |
| [Single-Frame MIMO Radar Velocity Vector Estimation via Multi-Bounce Scattering](https://ieeexplore.ieee.org/document/11103510) | N/A | IEEE Transactions on Computational Imaging | 2025 | Estimates full velocity vectors from single-frame MIMO radar data by exploiting multi-bounce scattering paths that provide additional geometric constraints. |
| [Doppler Former: Velocity Supervision of Raw Radar Data](https://ieeexplore.ieee.org/abstract/document/11127345) | [Code](https://github.com/coconut-zs/Fvidar-DopplerFormer) | ICRA | 2025 | Introduces a novel Doppler supervision mechanism for raw radar data to enhance velocity estimation accuracy in challenging scenarios. |
| [Direction of Arrival Estimation with Virtual Antenna Array Using FMCW Radar Simulated Data](https://arxiv.org/abs/2508.07513) | [Code](https://github.com/ekurtgl/FMCW-MIMO-Radar-Simulation) | arxiv | 2025 | Addresses the underutilization of velocity information in raw radar data by introducing a plug-and-play 'Doppler Former' module designed to explicitly extract Doppler features and boost perception model performance. |
| [Radar-STDA: A High-Performance Spatial-Temporal Denoising Autoencoder for Interference Mitigation of FMCW Radars](https://arxiv.org/abs/2307.09063) | [Code](https://github.com/GuanRunwei/rd_map_temporal_spatial_denoising_autoencoder) | arxiv | 2023 | Tackles the inefficiency of existing deep learning-based radar interference mitigation by proposing a nano-sized, high-speed spatial-temporal denoising autoencoder that achieves high performance with minimal computational cost. |
| [Experiments with mmWave Automotive Radar Test-bed](https://ieeexplore.ieee.org/document/9048939) | [Code](https://github.com/Xiangyu-Gao/mmWave-radar-signal-processing-and-microDoppler-classification) | 53rd Asilomar Conference on Signals, Systems, and Computers | 2019 | Details the creation of a lab-scale FMCW radar test-bed using a TI chipset and presents a new, large raw dataset collected with it, validating the platform's utility for ADAS research through preliminary object recognition results. |


### High-Resolution Imaging & SAR Imaging

| Title | Code | Publication | Date | Summary |
| :--- | :---: | :---: | :---: | :--- |
| [Millimeter Wave Inverse Pinhole Imaging](https://www.arxiv.org/abs/2510.13904) | N/A | arxiv | 2025 | Enhances the angular resolution of static, compact radars by introducing a rotating physical 'inverse pinhole,' and demonstrates that drone propellers can naturally serve this function for high-resolution imaging while hovering. |
| [Unsupervised 3D SAR Imaging Network Based on Generative Adversary Learning](https://ieeexplore.ieee.org/document/10919030) | [Code](https://github.com/WMMWWM/Unsupervised-3D-SAR-Imaging-Network-Based-on-Generative-Adversary-Learning) | IEEE Transactions on Antennas and Propagation | 2025 | Achieves 3D SAR image reconstruction without paired training data by leveraging generative adversarial networks to learn imaging transformations in an unsupervised manner. |
| [RF4D:Neural Radar Fields for Novel View Synthesis in Outdoor Dynamic Scenes](https://arxiv.org/abs/2505.20967) | [Code](https://github.com/zhan0618/RF4D_code) | arxiv | 2025 | Synthesizes novel radar views of dynamic outdoor scenes by representing 4D radar signals as continuous neural fields that encode spatial and temporal information. |
| [Millimeter-Wave SAR imaging of Sparse Trajectory via Untrained Complex-valued Neural Network](https://arxiv.org/abs/2505.00536) | [Code](https://github.com/Armorhtk/mmUSAR) | IEEE Transactions on Aerospace and Electronic Systems | 2025 | Reconstructs high-quality SAR images from sparse trajectory measurements using untrained complex-valued neural networks that exploit inherent signal structure without requiring pre-training. |
| [Novel Hybrid-Learning Algorithms for Improved Millimeter-Wave Imaging Systems](https://arxiv.org/abs/2306.15341) | [Code](https://github.com/josiahwsmith10/THz-and-Sub-THz-Imaging-Toolbox) | IEEE Journal of Microwaves | 2025 | Proposes a suite of novel hybrid-learning algorithms that interleave signal processing and deep learning to achieve a superior balance of performance and generalizability across diverse mmWave imaging applications. |
| [High-resolution mmWave Imaging using Metasurface and Diffusion](https://dl.acm.org/doi/abs/10.1145/3711875.3729162) | N/A | MobiSys | 2025 | Achieves high-resolution, static mmWave imaging by using an optimized metasurface to encode the scene and a diffusion model to reconstruct the image, replacing mechanical scanning and compressive sensing. |

### Data Synthesis, Enhancement & Simulation

| Title | Code | Publication | Date | Summary |
| :--- | :---: | :---: | :---: | :--- |
| [Wideband RF Radiance Field Modeling Using Frequency-embedded 3D Gaussian Splatting](https://arxiv.org/abs/2505.20714) | [Code](https://github.com/sim-2-real/Wideband3DGS) | arxiv | 2025 | Models wideband RF propagation in 3D scenes by embedding frequency information into Gaussian splatting representations for accurate multi-frequency signal prediction. |
| [Talk is Not Always Cheap: Promoting Wireless Sensing Models with Text Prompts](https://arxiv.org/abs/2504.14621) | [Code](https://github.com/zk-b612/WiTalk) | arxiv | 2025 | Enhances wireless sensing model performance by leveraging text prompts and large language models to provide semantic guidance for radar data interpretation. |
| [One Snapshot is All You Need: A Generalized Method for mmWave Signal Generation](https://ieeexplore.ieee.org/abstract/document/10416806) | N/A | IEEE INFOCOM | 2025 | Generates diverse mmWave signals from a single snapshot by learning generalizable representations that capture signal patterns across different scenarios. |
| [Synthetic Radar Signal Generator for Human Motion Analysis](https://ieeexplore.ieee.org/abstract/document/10804837) | N/A | IEEE Transactions on Radar Systems | 2025 | Synthesizes realistic radar signals for human motion scenarios by integrating motion-capture data with radar signal models to overcome data scarcity. |
| [Diffusion^2: Turning 3D Environments into Radio Frequency Heatmaps](https://arxiv.org/abs/2510.02274) | [Project](https://rfvision-project.github.io/) | arxiv | 2025 | Predicts RF signal propagation patterns in 3D environments using cascaded diffusion models that generate spatially-aware channel heatmaps. |
| [Inverse Rendering of Near-Field mmWave MIMO Radar for Material Reconstruction](https://ieeexplore.ieee.org/document/10892639/) | [Code](https://github.com/nihofm/inverse-radar-rendering) | IEEE Journal of Microwaves | 2025 | Reconstructs material properties of objects from near-field radar measurements by formulating an inverse rendering problem that recovers dielectric characteristics. |
| [Simulate Any Radar: Attribute-Controllable Radar Simulation via Waveform Parameter Embedding](https://arxiv.org/abs/2506.03134) | [Code](https://github.com/zhuxing0/SA-Radar) | arxiv | 2025 | Enables controllable radar simulation across different configurations by embedding waveform parameters into generative models for flexible data augmentation. |
| [L2RDaS: Synthesizing 4D Radar Tensors for Model Generalization via Dataset Expansion](https://arxiv.org/abs/2503.03637) | [Project](https://github.com/kaist-avelab/K-Radar) | arxiv | 2025 | Improves model generalization by synthesizing realistic 4D radar tensors through learning-to-learn approaches that expand training dataset diversity. |
| [MITO: A Millimeter-Wave Dataset and Simulator for Non-Line-of-Sight Perception](https://arxiv.org/abs/2502.10259) | [Code](https://github.com/signalkinetics/MITO_Codebase/tree/main) | arxiv | 2025 | Provides a comprehensive dataset and physics-based simulator for NLOS radar perception enabling research on seeing around corners with synthetic aperture techniques. |
| [RadaRays: Real-time Simulation of Rotating FMCW Radar for Mobile Robotics via Hardware-accelerated Ray Tracing](https://ieeexplore.ieee.org/abstract/document/10845807) | [Code](https://github.com/uos/radarays) | IEEE Robotics and Automation Letters | 2025 | Develops a real-time radar simulator that uses hardware-accelerated ray tracing to more accurately model complex wave phenomena like reflection and scattering, overcoming the limitations of simplistic, lidar-like simulations in existing platforms.|
| [RF-Diffusion: Radio Signal Generation via Time-Frequency Diffusion](https://dl.acm.org/doi/10.1145/3636534.3649348) | [Code](https://github.com/yourusername/RF-Diffusion) | MobiCom | 2024 | Generates realistic radio signals by applying diffusion models in the time-frequency domain to capture complex signal characteristics and variations. |
| [Generative Artificial Intelligence Meets Synthetic Aperture Radar: A survey](https://ieeexplore.ieee.org/abstract/document/10752552) | [Code](https://github.com/XAI4SAR/GenAIxSAR) | IEEE Geoscience and Remote Sensing Magazine | 2024 | Surveys the intersection of generative AI techniques with SAR imaging, highlighting advancements in data synthesis, super-resolution, and anomaly detection. |
| [Generation of Realistic Synthetic Raw Radar Data for Automated Driving Applications using Generative Adversarial Networks](https://arxiv.org/abs/2308.02632) | [Code](https://github.com/eduardo-candioto-fidelis/raw-radar-data-generation) | arxiv | 2023 | Addresses the speed and noise-modeling limitations of ray tracing by using a Generative Adversarial Network (GAN) to generate realistic, raw radar sensor data for data augmentation in critical scenarios.|
| [An Open Framework to Model Diffraction by Dynamic Blockers in Millimeter Wave Simulations](https://ieeexplore.ieee.org/document/9810361) | [Code](https://github.com/signetlabdei/rt-blockage-manager) | Addresses the challenge of simulating multiple, dynamic blockers in mmWave channels by releasing an open-source framework that models diffraction and interfaces with existing ray-tracing software. |


### Foundational Models & Representation Learning About Radar Signals

| Title | Code | Publication | Date | Summary |
| :--- | :---: | :---: | :---: | :--- |
| [Unlocking Interpretability for RF Sensing: A Complex-Valued White-Box Transformer](https://arxiv.org/abs/2507.21799) | [Code](https://github.com/rfcrate/RF_CRATE) | arxiv | 2025 | Develops an interpretable transformer architecture for RF sensing that processes complex-valued signals while maintaining transparency in feature learning and decision-making. |
| [Multi-View Radar Detection Transformer with Differentiable Positional Encoding](https://ieeexplore.ieee.org/document/10889849/) | N/A | ICASSP | 2025 | Enhances multi-view radar detection by introducing differentiable positional encodings that adapt to radar geometry and improve cross-view feature aggregation. |
| [Towards Foundational Models for Single-Chip Radar](https://arxiv.org/abs/2509.12482) | [Project](https://wiselabcmu.github.io/grt/) | arxiv | 2025 | Builds foundation models for single-chip radar through self-supervised pre-training on diverse radar data to enable transfer learning across multiple sensing tasks. |
| [SpikingRTNH: Spiking Neural Network for 4D Radar Object Detection](https://arxiv.org/abs/2502.00074) | [Code](https://github.com/kaist-avelab/K-Radar/tree/main/models/skeletons) | arxiv | 2025 | Achieves energy-efficient 4D radar object detection using spiking neural networks that exploit temporal sparsity in radar signals for low-power processing. |
| [Talk2Radar: Bridging Natural Language with 4D mmWave Radar for 3D Referring Expression Comprehension](https://arxiv.org/abs/2405.12821) | [Code](https://github.com/GuanRunwei/Talk2Radar) | arxiv | 2024 | Establishes a new research direction by creating the first-ever dataset and a baseline model that connects natural language expressions to objects within 4D mmWave radar point clouds for 3D referring expression comprehension.|
| [FMCW Radar Sensing for Indoor Drones Using Learned Representations](https://arxiv.org/abs/2301.02451) | [Dataset](https://thesmartrobot.github.io/datasets) | arxiv | 2023 | Addresses the inefficiency of task-specific deep learning for radar by using unsupervised learning to generate reusable, low-dimensional representations from raw FMCW data for various downstream tasks. |

## 🤖 Embodied AI & Robotics

| Title | Code | Publication | Date | Summary |
| :--- | :---: | :---: | :---: | :--- |
| [Towards Robust and Secure Embodied AI: A Survey on Vulnerabilities and Attacks](https://arxiv.org/abs/2502.13175) | N/A | arxiv | 2025 | Comprehensively surveys security vulnerabilities in embodied AI systems including sensor spoofing and adversarial attacks on perception and decision-making modules. |
| [FuseGrasp: Radar-Camera Fusion for Robotic Grasping of Transparent Objects](https://ieeexplore.ieee.org/document/10909339) | N/A | IEEE Transactions on Mobile Computing | 2025 | Enables robust grasping of transparent objects by fusing radar material sensing with camera visual information to overcome camera limitations on reflective surfaces. |
| [Non-Line-of-Sight 3D Object Reconstruction via mmWave Surface Normal Estimation](https://dl.acm.org/doi/10.1145/3711875.3729138) | [Code](https://github.com/signalkinetics/mmNorm) | MobiSys | 2025 | Reconstructs hidden 3D objects by estimating surface normals from mmWave multipath reflections using synthetic aperture radar imaging principles. |
| [Loosely coupled 4D-Radar-Inertial Odometry for Ground Robots](https://arxiv.org/abs/2411.17289) | [Code](https://github.com/robotics-upo/4D-Radar-Odom) | arxiv | 2025 | Achieves robust ground robot odometry in challenging conditions through loosely-coupled fusion of 4D radar measurements with inertial sensor data. |


## 🚗 Autonomous Driving & Drone


### 3D Object Detection & Classification

| Title | Code | Publication | Date | Summary |
| :--- | :---: | :---: | :---: | :--- |
| [RICCARDO: Radar Hit Prediction and Convolution for Camera-Radar 3D Object Detection](https://openaccess.thecvf.com/content/CVPR2025/html/Long_RICCARDO_Radar_Hit_Prediction_and_Convolution_for_Camera-Radar_3D_Object_CVPR_2025_paper.html) | [Code](https://github.com/longyunf/riccardo) | CVPR | 2025 | Enhances camera-radar fusion by predicting where radar hits should occur on objects and performing convolutions in predicted hit space for improved 3D detection. |
| [DoppDrive: Doppler-Driven Temporal Aggregation for Improved Radar Object Detection](https://arxiv.org/abs/2508.12330) | [Project](https://yuvalhg.github.io/DoppDrive/) | arXiv | 2025 | Improves radar object detection by using Doppler velocity measurements to guide temporal aggregation of radar frames for motion-aware feature learning. |
| [RadarNeXt: Real-Time and Reliable 3D Object Detector Based On 4D mmWave Imaging Radar](https://arxiv.org/abs/2501.02314) | [Code](https://github.com/Pay246-git468/RadarNeXt) | arXiv | 2025 | Achieves real-time 3D object detection from 4D radar by designing efficient architectures that balance accuracy and computational efficiency for automotive deployment. |
| [RADLER: Radar Object Detection Leveraging Semantic 3D City Models and Self-Supervised Learning](https://openaccess.thecvf.com/content/CVPR2025W/PBVS/html/Luo_RADLER_Radar_Object_Detection_Leveraging_Semantic_3D_City_Models_and_CVPRW_2025_paper.html) | [Project](https://gpp-communication.github.io/RADLER/) | CVRP | 2025 | Improves radar object detection by leveraging semantic 3D city models as prior knowledge and using self-supervised learning to reduce annotation requirements. |
| [Beyond Pillars: Advancing 3D Object Detection with Salient Voxel Enhancement of LiDAR-4D Radar Fusion](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5518546) | [Code](https://github.com/icdm-adteam/SVEFusion) | SSRN | 2025 | Advances beyond pillar-based representations by identifying and enhancing salient voxels in LiDAR-radar fusion for more discriminative 3D object features. |
| [RCDFNet: A 4-D Radar and Camera Dual-Level Fusion Network for 3-D Object Detection](https://ieeexplore.ieee.org/abstract/document/11006930) | [Code](https://github.com/D-Hourse/RCDFNet/tree/master) | IEEE Sensors Journal | 2025 | Performs radar-camera fusion at both feature and decision levels to complement strengths of each modality for robust 3D object detection across conditions. |
| [V2X-R: Cooperative LiDAR-4D Radar Fusion for 3D Object Detection with Denoising Diffusion](https://arxiv.org/abs/2411.08402) | [Code](https://github.com/ylwhxht/V2X-R) | CVPR | 2025 | Enables cooperative perception in V2X scenarios by fusing LiDAR and 4D radar across vehicles using denoising diffusion to handle communication noise. |
| [RCDINO: Enhancing Radar-Camera 3D Object Detection with DINOv2 Semantic Features](https://arxiv.org/abs/2508.15353) | [Code](https://github.com/OlgaMatykina/RCDINO) | arXiv | 2025 | Enhances radar-camera 3D detection by incorporating DINOv2's rich semantic features to provide better object understanding and context awareness. |

### Semantic & Instance Segmentation

| Title | Code | Publication | Date | Summary |
| :--- | :---: | :---: | :---: | :--- |
| [RC-ROSNet: Fusing 3D Radar Range-Angle Heat Maps and Camera Images for Radar Object Segmentation](https://ieeexplore.ieee.org/document/11112643) | [Code](https://github.com/Zhuanglong2/RC-ROSNet) | IEEE Transactions on Circuits and Systems for Video Technology | 2025 | Performs radar object segmentation by fusing 3D range-angle heatmaps with camera images to leverage complementary spatial and semantic information. |
| [M2CNet: LiDAR 3D Semantic Segmentation Based on Multi-level Multi-view Cross-attention Fusion for Autonomous Vehicles](https://ieeexplore.ieee.org/document/11125962/) | [Code](https://github.com/Terminal-lidar/M2CNet) | IEEE Transactions on Vehicular Technology | 2025 | Enhances LiDAR semantic segmentation through multi-level multi-view cross-attention that captures both local details and global context for autonomous driving. |
| [RadarMask: A Novel End-to-End Sparse Millimeter-Wave Radar Sequence Panoptic Segmentation and Tracking Method](https://ieeexplore.ieee.org/abstract/document/11128555) | [Code](https://github.com/yb-guo/RadarMask) | ICRA | 2025 | Performs end-to-end panoptic segmentation and tracking on sparse mmWave radar sequences by jointly learning instance masks and temporal associations. |
| [RETR: Multi-View Radar Detection Transformer for Indoor Perception](https://neurips.cc/virtual/2024/poster/95530) | [Code](https://github.com/merlresearch/radar-detection-transformer) | NeurIPS | 2024 | Achieves robust indoor perception by aggregating multi-view radar data through transformer architecture that handles varying spatial perspectives and occlusions. |
| [4D Radar And Vision Fusion Detection Model Based On Segmentation-assisted](https://www.researchsquare.com/article/rs-5358941/v1) | [Code](https://github.com/Huniki/RVASANET) | arXiv | 2024 | Improves 4D radar-vision fusion detection by using segmentation results as intermediate supervision to guide feature learning and alignment between modalities. |
| [AdaPKC: PeakConv with Adaptive Peak Receptive Field for Radar Semantic Segmentation](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f6b22ac37beb5da61efd4882082c9ecd-Abstract-Conference.html) | [Code](https://github.com/lihua199710/AdaPKC) | NeurIPS | 2024 | Addresses radar sparsity in semantic segmentation through adaptive peak convolutions that dynamically adjust receptive fields based on local point density. |
| [TARSS-Net: Temporal-Aware Radar Semantic Segmentation Network](https://neurips.cc/virtual/2024/poster/96608) | [Code](https://github.com/zlw9161/TARSS-NeT) | NeurIPS | 2024 | Enhances radar semantic segmentation by explicitly modeling temporal dependencies across consecutive frames to leverage motion information for better predictions. |
| [Achelous++: Power-Oriented Water-Surface Panoptic Perception Framework on Edge Devices based on Vision-Radar Fusion and Pruning of Heterogeneous Modalities](https://arxiv.org/abs/2312.08851) | [Code](https://github.com/GuanRunwei/Achelous) | arxiv | 2024 | Addresses the high power consumption of water-surface perception models by proposing a lightweight, vision-radar fusion framework with a novel multi-modal pruning strategy to enable efficient multi-task learning on edge devices. |
| [ASY-VRNet: Waterway Panoptic Driving Perception Model based on Asymmetric Fair Fusion of Vision and 4D mmWave Radar](https://ieeexplore.ieee.org/document/10802447) | [Code](https://github.com/GuanRunwei/ASY-VRNet) | IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) | 2024 | Addresses the differing feature requirements of object detection and semantic segmentation by proposing an asymmetric fusion module that treats both vision and radar as point sets to create a task-specific shared feature space.|

### Scene Flow & Motion Prediction

| Title | Code | Publication | Date | Summary |
| :--- | :---: | :---: | :---: | :--- |
| [Self-Supervised Diffusion-Based Scene Flow Estimation and Motion Segmentation with 4D Radar](https://ieeexplore.ieee.org/document/10974572) | [Code](https://github.com/nubot-nudt/RadarSFEMOS) | IRAL | 2025 | Estimates scene flow and segments moving objects from 4D radar using self-supervised diffusion models that learn motion patterns without manual annotations. |
| [DoGFlow: Self-Supervised LiDAR Scene Flow via Cross-Modal Doppler Guidance](https://arxiv.org/abs/2505.06056) | [Project](https://ajinkyakhoche.github.io/DogFlow/) | arxiv | 2025 | Overcomes the bottleneck of manual annotation for LiDAR scene flow by using a self-supervised method that generates pseudo-labels from radar Doppler measurements, significantly improving performance and data efficiency. |

### Radar Odometry & Ego-Motion Estimation

| Title | Code | Publication | Date | Summary |
| :--- | :---: | :---: | :---: | :--- |
| [Digital Beamforming Enhanced Radar Odometry](https://ieeexplore.ieee.org/document/11127292) | [Code](https://github.com/SenseRoboticsLab/DBE-Radar) | ICRA | 2025 | Improves radar odometry accuracy by applying digital beamforming to enhance angular resolution and reduce multipath interference for better feature matching. |
| [DRO: Doppler-Aware Direct Radar Odometry](https://arxiv.org/abs/2504.20339) | [Code](https://github.com/utiasASRL/dro) | RSS | 2025 | Achieves accurate direct radar odometry by explicitly incorporating Doppler velocity measurements into optimization framework to constrain ego-motion estimation. |
| [GaRLIO: Gravity enhanced Radar-LiDAR-Inertial Odometry](https://arxiv.org/abs/2502.07703) | [Code](https://github.com/ChiyunNoh/GaRLIO) | arXiv | 2025 | Enhances multi-sensor odometry by incorporating gravity direction as additional constraint to improve radar-LiDAR-inertial fusion accuracy and robustness. |
| [Ground-Optimized 4D Radar-Inertial Odometry via Continuous Velocity Integration using Gaussian Process](https://arxiv.org/abs/2502.08093) | [Code](https://github.com/wooseongY/Go-RIO) | arXiv | 2025 | Optimizes radar-inertial odometry for ground vehicles by using Gaussian processes to model continuous velocity integration and handle ground constraint. |
| [Equi-RO: A 4D mmWave Radar Odometry via Equivariant Networks](https://arxiv.org/abs/2509.20674)| N/A | arxiv | 2025 | Achieves robust radar odometry by leveraging equivariant neural networks that respect geometric transformations and improve generalization across diverse motion patterns. |
| [Learning Point Correspondences In Radar 3D Point Clouds For Radar-Inertial Odometry](https://arxiv.org/abs/2506.18580) | [Code](https://github.com/aau-cns/radar_transformer) | arxiv | 2025 | Improves radar-inertial odometry by using a self-supervised Transformer network to find robust point correspondences, uniquely generating training data by solving a linear sum assignment problem to avoid manual labeling. |
| [Radar4VoxMap: Accurate Odometry from Blurred Radar Observations](https://ieeexplore.ieee.org/abstract/document/11128118) | [Code](https://github.com/ailab-hanyang/Radar4VoxMap) | IEEE International Conference on Robotics and Automation (ICRA) | 2025 | Tackles the issue of submap blurring from cumulative errors in radar odometry by using an RCS-weighted voxel map and a fixed-lag graph optimization that jointly refines both the map and the vehicle's pose. |
| [EFEAR-4D：Ego-velocity Filtering for Efficient and Accurate 4D radar Odometry](https://ieeexplore.ieee.org/document/10685149) | [Code](https://github.com/CLASS-Lab/EFEAR-4D) | IEEE Robotics and Automation Letters | 2024 | Improves 4D radar odometry efficiency by filtering radar points based on ego-velocity consistency to remove dynamic objects and outliers. |
| [RadarMOSEVE: A Spatial-Temporal Transformer Network for Radar-Only Moving Object Segmentation and Ego-Velocity Estimation](https://ojs.aaai.org/index.php/AAAI/article/view/28240) | [Code](https://github.com/ORCA-Uboat/RadarMOSEVE) | AAAI | 2024 | Proposes a Transformer network with custom self- and cross-attention mechanisms designed to leverage radar's radial velocity information to overcome data sparsity and noise for simultaneous moving object segmentation and ego-velocity estimation. |
| [DeRO: Dead Reckoning Based on Radar Odometry With Accelerometers Aided for Robot Localization](https://ieeexplore.ieee.org/document/10801645) | [Code](https://github.com/hoangvietdo/dero) |IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) | 2024 | Mitigates localization drift by using radar Doppler velocity and gyroscope data for direct dead reckoning, while using accelerometer-derived tilt angles and scan matching for Kalman filter updates, thus avoiding error-prone accelerometer double integration.|
| [A Robust Baro-Radar-Inertial Odometry M-Estimator for Multicopter Navigation in Cities and Forests](https://ieeexplore.ieee.org/document/10705761) | [Dataset](https://www.research-collection.ethz.ch/entities/researchdata/99624de7-3f89-4411-9894-c8ff8ba0f009) | IEEE International Conference on Multisensor Fusion and Integration for Intelligent Systems (MFI) | 2024 | Enhances multicopter navigation in challenging environments by fusing barometer, radar, and inertial data using an M-estimator framework that robustly handles outliers and sensor noise. |
| [Successive Pose Estimation and Beam Tracking for mmWave Vehicular Communication Systems](https://ieeexplore.ieee.org/document/10464790) | [Code](https://github.com/Cen-Liu/Fast-CFEAR-Radar-Odometry) | IEEE Transactions on Vehicular Technology | 2024 | Alleviates beam training overhead in vehicular mmWave communications by first estimating the vehicle's pose via radar odometry and then using this information within an Extended Kalman Filter to proactively track the communication beam. |


### Multi-Object Tracking

| Title | Code | Publication | Date | Summary |
| :--- | :---: | :---: | :---: | :--- |
| [USVTrack: USV-Based 4D Radar-Camera Tracking Dataset for Autonomous Driving in Inland Waterways](https://arxiv.org/abs/2506.18737) | [Dataset](https://github.com/USVTrack/USVTrack) | arXiv | 2025 | Provides the first 4D radar-camera tracking dataset for inland waterway autonomous navigation featuring unique challenges of water surface reflections. |
| [Real-Time Multi-object Tracking and Identification Using Sparse Point-Cloud Data from Low-Cost mmWave Radar](https://link.springer.com/chapter/10.1007/978-3-031-92011-0_12) | N/A | Robot Intelligence Technology and Applications | 2024 | Enables real-time multi-object tracking from sparse mmWave point clouds through efficient data association algorithms optimized for low-cost radar sensors. |


### Simultaneous Localization and Mapping (SLAM)

| Title | Code | Publication | Date | Summary |
| :--- | :---: | :---: | :---: | :--- |
| [Doppler-SLAM: Doppler-Aided Radar-Inertial and LiDAR-Inertial SLAM](https://arxiv.org/abs/2504.11634) | [Code](https://github.com/Wayne-DWA/Doppler-SLAM) | IEEE Robotics and Automation Letters | 2025 | Enhances SLAM accuracy by integrating Doppler velocity measurements into radar-inertial and LiDAR-inertial frameworks to provide additional motion constraints. |
| [S^3E: Self-Supervised State Estimation for Radar-Inertial System](https://arxiv.org/abs/2509.25984) | N/A | arxiv | 2025 | Achieves self-supervised radar-inertial state estimation by learning motion patterns from unlabeled data to reduce dependency on ground truth annotations. |
| [MapKD: Unlocking Prior Knowledge with Cross-Modal Distillation for Efficient Online HD Map Construction](https://arxiv.org/abs/2508.15653) | [Code](https://github.com/2004yan/MapKD2026) | arxiv | 2025 | Enables efficient online HD map construction by distilling knowledge from offline maps using cross-modal learning to reduce computational overhead. |
| [Towards Dense and Accurate Radar Perception via Efficient Cross-Modal Diffusion Model](https://ieeexplore.ieee.org/document/10592769) | [Code](https://github.com/ZJU-FAST-Lab/Radar-Diffusion) | IEEE Robotics and Automation Letters | 2024 | Generates dense radar representations from sparse measurements using cross-modal diffusion models that leverage complementary sensor information for accurate perception. |
| [Radarize: Enhancing Radar SLAM with Generalizable Doppler-Based Odometry](https://dl.acm.org/doi/10.1145/3643832.3661871) | [Project](https://radarize.github.io/) | MOBISYS | 2024 | Introduces a self-contained, radar-only SLAM system that significantly outperforms existing radar-inertial methods by leveraging Doppler-based odometry and multipath suppression, eliminating the need for any auxiliary sensors. |
| [Enhancing mmWave Radar Point Cloud via Visual-inertial Supervision](https://ieeexplore.ieee.org/document/10610091) | [Code](https://github.com/ClarenceZSK/mmEMP) |ICRA | 2024| Replaces expensive LiDAR supervision with a low-cost visual-inertial system to densify radar point clouds and remove artifacts, making high-quality radar enhancement accessible for commercial vehicles.|
| [BatMobility: Towards Flying Without Seeing for Autonomous Drones](https://dl.acm.org/doi/10.1145/3570361.3592532) | [Project](https://batmobility.github.io/) | ACM MobiCom | 2023 | Enables drones to 'fly without seeing' by introducing a radar-only perception system that uses a novel 'radio flow' technique as a robust, non-optical alternative to optical flow for navigation. |


### Sensor Fusion Techniques

| Title | Code | Publication | Date | Summary |
| :--- | :---: | :---: | :---: | :--- |
| [4D mmWave Radar for Sensing Enhancement in Adverse Environments: Advances and Challenges](https://arxiv.org/abs/2503.24091) | [Code](https://github.com/XiangyPeng/4D-mmWave-Radar-in-Adverse-Environments) | arxiv | 2025 | Fills a research gap by providing the first comprehensive review of 4D mmWave radar datasets and learning-based methods specifically for perception in adverse environments. |
| [Availability-aware Sensor Fusion via Unified Canonical Space for 4D Radar, LiDAR, and Camera](https://arxiv.org/abs/2503.07029) | [Code](https://github.com/kaist-avelab/K-Radar) | arxiv | 2025 | Addresses the vulnerability of multi-sensor fusion systems to sensor failure by projecting all sensor data into a unified representation, enabling robust cross-modal attention that is aware of sensor availability. |
| [RadarRGBD: A Multi-Sensor Fusion Dataset for Perception with RGB-D and mmWave Radar](https://arxiv.org/abs/2505.15860) | [Dataset](https://github.com/song4399/RadarRGBD) | arxiv | 2025 | Provides a comprehensive RGB-D and mmWave radar fusion dataset enabling research on complementary depth sensing and material property estimation. |
| [MIPD: A Multi-Sensory Interactive Perception Dataset for Embodied Intelligent Driving](https://ieeexplore.ieee.org/abstract/document/11112801) | [Code](https://github.com/BUCT-IUSRC/Dataset__MIPD) | IEEE Transactions on Intelligent Transportation Systems | 2025 | Addresses the lack of embodied sensory information in autonomous driving datasets by introducing a new dataset (MIPD) that integrates sound, light, and vibration with traditional sensors to foster more interactive and comprehensive perception. |
| [Artemis: Contour-Guided 3-D Sensing and Localization With mmWave Radar for Infrastructure-Assisted AVs](https://ieeexplore.ieee.org/document/10891135) | N/A | IEEE Internet of Things Journal | 2025 | Achieves precise infrastructure-based localization by using mmWave radar to extract object contours that guide 3D sensing and vehicle positioning. |
| [CoVeRaP: Cooperative Vehicular Perception through mmWave FMCW Radars](https://www.arxiv.org/abs/2508.16030) | [Code](https://github.com/John1001Song/FMCW_Vehicle_Fusion) | arxiv | 2025 | Enables cooperative perception among vehicles by sharing and fusing mmWave FMCW radar data to expand sensing coverage and reduce occlusions. |
| [Ultra-High-Frequency Harmony: mmWave Radar and Event Camera Orchestrate Accurate Drone Landing](https://dl.acm.org/doi/10.1145/3715014.3722048) | [Project](https://mme-loc.github.io/) | SenSys | 2025 | Achieves precise drone landing by fusing mmWave radar range measurements with event camera's high-temporal-resolution motion detection capabilities. |
| [Rehearse-3d: A Multi-Modal Emulated Rain Dataset for 3d Point Cloud De-Raining](https://arxiv.org/abs/2504.21699) | [Dataset](https://sporsho.github.io/REHEARSE3D) | arxiv | 2025 | Provides multi-modal rainy weather dataset enabling research on point cloud de-raining algorithms for robust perception under adverse conditions. |
| [4D-ROLLS: 4D Radar Occupancy Learning via LiDAR Supervision](https://arxiv.org/abs/2505.13905) | [Code](https://github.com/CLASS-Lab/4D-ROLLS) | arxiv | 2025 | Learns 4D radar occupancy representations using LiDAR as supervisory signal to overcome radar annotation challenges and enable dense scene understanding. |
| [MIPD: A Multi-Sensory Interactive Perception Dataset for Embodied Intelligent Driving](https://ieeexplore.ieee.org/abstract/document/11112801) | [Dataset](https://github.com/BUCT-IUSRC/Dataset__MIPD) | IEEE Transactions on Intelligent Transportation Systems | 2025 | Provides multi-sensory dataset capturing driver-vehicle interactions for research on embodied intelligence and driver monitoring systems. |
| [MetaOcc: Spatio-Temporal Fusion of Surround-View 4D Radar and Camera for 3D Occupancy Prediction with Dual Training Strategies](https://arxiv.org/abs/2501.15384) | [Code](https://github.com/LucasYang567/MetaOcc) | arxiv | 2025 | Predicts 3D occupancy by fusing surround-view 4D radar and camera through spatio-temporal networks with dual training strategies for improved generalization. |
| [Multi-Modal Fusion Sensing: A Comprehensive Review of Millimeter-Wave Radar and Its Integration With Other Modalities](https://ieeexplore.ieee.org/document/10525189) | N/A | IEEE Communications Surveys & Tutorials | 2024 | Comprehensively surveys mmWave radar fusion with various modalities covering fusion architectures, challenges, and applications across different domains. |
| [The Oxford Radar RobotCar Dataset: A Radar Extension to the Oxford RobotCar Dataset](https://arxiv.org/abs/1909.01300) | [Dataset](https://github.com/oxford-robotics-institute/radar-robotcar-dataset-sdk) | arxiv | 2019 | Extends the popular Oxford RobotCar dataset with a high-resolution scanning radar to provide the research community with a large-scale, multi-sensor dataset for developing autonomous driving perception algorithms that are robust to adverse weather. |

### Point Cloud Processing
| Title | Code | Publication | Date | Summary |
| :--- | :---: | :---: | :---: | :--- |
| [RLCNet: A Novel Deep Feature-Matching-Based Method for Online Target-Free Radar-LiDAR Calibration](https://ieeexplore.ieee.org/abstract/document/11127342) | [Code](https://github.com/nubot-nudt/RLCNet) | ICRA | 2025 | Tackles radar-LiDAR extrinsic calibration by using a deep network that first establishes sparse keypoint matches and then refines them into dense correspondences to overcome radar point cloud sparsity. |
| [RadarSplat: Radar Gaussian Splatting for High-Fidelity Data Synthesis and 3D Reconstruction of Autonomous Driving Scenes](https://arxiv.org/abs/2506.01379) | [Code](https://github.com/umautobots/radarsplat) | ICCV| 2025 | Overcomes the poor performance of neural representations in noisy radar scenarios by integrating Gaussian Splatting with a novel noise model, enabling high-fidelity 3D reconstruction and realistic data synthesis. |
| [NeuRadar: Neural Radiance Fields for Automotive Radar Point Clouds](https://openaccess.thecvf.com/content/CVPR2025W/WAD/html/Rafidashti_NeuRadar_Neural_Radiance_Fields_for_Automotive_Radar_Point_Clouds_CVPRW_2025_paper.html) | [Code](https://github.com/mrafidashti/neuradar) | CVPR | 2025 | Pioneers the use of Neural Radiance Fields for radar by proposing a model that jointly generates multi-sensor data and uses a probabilistic representation to accurately capture the stochastic nature of radar point clouds. |

## 🩺 Human Sensing & Healthcare


### Human Activity Recognition (HAR)

| Title | Code | Publication | Date | Summary |
| :--- | :---: | :---: | :---: | :--- |
| [Human Activity Recognition Based on Multipath Fusion in Non-line-of-sight Corner](https://ieeexplore.ieee.org/document/11177013) | [Code](https://github.com/tlz1111/Multipath-Fusion-Network) | IEEE Internet of Things Journal | 2025 | Achieves NLOS activity recognition by fusing multipath radar signals that bounce around corners to sense hidden human activities. |
| [Enhancing Activity Recognition: Motion Waveform Preprocessing from Millimeter-Wave Radar Data for Transformer-Based Classification](https://ieeexplore.ieee.org/document/11152068) | [Code](https://github.com/Alan-cs1/MmWave-Motion-Waveform-HAR) | IEEE International Conference on Multimedia and Expo Workshops (ICMEW) | 2025 | Improves transformer-based activity recognition by designing specialized preprocessing that extracts motion waveforms from mmWave radar signals. |
| [Resolution-Adaptive Micro-Doppler Spectrogram for Human Activity Recognition](https://arxiv.org/abs/2411.15057) | N/A | arxiv | 2025 | Enhances HAR performance by adaptively adjusting micro-Doppler spectrogram resolution based on activity characteristics and computational constraints. |
| [A Novel Multimodal LLM-Driven RF Sensing Method for Human Activity Recognition](https://ieeexplore.ieee.org/document/11003262) | [Code](https://github.com/ci4r/CI4R-MULTI3) | International Conference on Microwave, Antennas & Circuits (ICMAC) | 2025 | Leverages large language models to interpret multimodal RF sensing data by bridging radar signals with semantic understanding for improved activity recognition. |
| [RadMamba: Efficient Human Activity Recognition through Radar-based Micro-Doppler-Oriented Mamba State-Space Model](https://arxiv.org/abs/2504.12039) | [Code](https://github.com/lab-emi/AIRHAR) | arxiv | 2025 | Achieves efficient HAR using Mamba state-space models that capture micro-Doppler temporal dependencies with linear complexity for real-time processing. |
| [DGAR: A Unified Domain Generalization Framework for RF-Based Human Activity Recognition](https://arxiv.org/abs/2503.17667) | [Code](https://github.com/Junshuo-Lau/HUST_HAR_LFM) | arxiv | 2025 | Addresses cross-domain HAR challenges through unified domain generalization framework that learns domain-invariant radar representations. |
| [RadProPoser: Uncertainty-Aware Human Pose Estimation and Activity Classification from Raw Radar Data](https://arxiv.org/abs/2508.03578) | [Code](https://github.com/jonasmueler/RadProPoser) | arxiv | 2025 | Jointly performs pose estimation and activity classification from raw radar with uncertainty quantification to improve reliability in ambiguous scenarios. |
| [RadHARSimulator V1: Model-Based FMCW Radar Human Activity Recognition Simulator](https://arxiv.org/abs/2509.06751) | [Code](https://github.com/JoeyBGOfficial/RadHARSimulatorV1-Model-Based-FMCW-Radar-Human-Activity-Recognition-Simulator) | arxiv | 2025 | Overcomes the challenge of acquiring diverse, high-fidelity radar data for Human Activity Recognition (HAR) by developing a model-based simulator that generates realistic micro-Doppler signatures for various activities.|
| [Few-Shot Adaptation to Unseen Conditions for Wireless-Based Human Activity Recognition Without Fine-Tuning](https://ieeexplore.ieee.org/abstract/document/10681641) | N/A | IEEE Transactions on Mobile Computing | 2024 | Eliminates the need for fine-tuning in wireless activity recognition by using meta-learning to learn a condition-invariant latent space, where recognition is performed based on activity similarity rather than a condition-dependent mapping. |
| [CubeLearn: End-to-end Learning for Human Motion Recognition from Raw mmWave Radar Signals](https://ieeexplore.ieee.org/document/10018429) | [Code](https://github.com/zhaoymn/cubelearn) | IEEE Internet of Things Journal | 2023 | Replaces the conventional, fixed DFT preprocessing in radar-based motion recognition with a learnable, end-to-end module to extract task-optimized features directly from raw signals, boosting performance especially for lightweight models. |
| [MiliPoint: a point cloud dataset for mmWave radar](https://dl.acm.org/doi/10.5555/3666122.3668861) | [Dataset](https://github.com/yizzfz/MiliPoint) | NIPS | 2023 | Fosters research into point-based deep learning for radar by releasing a large-scale, diverse dataset for human activity recognition, complete with established performance baselines. |
| [MM-Fi: multi-modal non-intrusive 4D human dataset for versatile wireless sensing](https://dl.acm.org/doi/abs/10.5555/3666122.3666944) | [Dataset](https://github.com/ybhbingo/MMFi_dataset) | NeurIPS | 2023 | Provides a large-scale multi-modal dataset combining 4D radar, RGB-D, and motion capture data to facilitate research in wireless human sensing. |

### Gesture Recognition & Hand Tracking
| Title | Code | Publication | Date | Summary |
| :--- | :---: | :---: | :---: | :--- |
| [mmWave Radar-based Unsupervised Gesture Recognition via Image-Aligned Heterogeneous Domain Transfer](https://ieeexplore.ieee.org/document/11180134) | [Code](https://github.com/onlinehuazai/mmGesture) | IEEE Transactions on Mobile Computing | 2025 | Achieves unsupervised gesture recognition by aligning radar spectrograms with image features through heterogeneous domain transfer learning. |
| [mmPencil: Toward Writing-Style-Independent In-Air Handwriting Recognition via mmWave Radar and Large Vision-Language Model](https://dl.acm.org/doi/10.1145/3749504) | [Dataset](https://www.kaggle.com/datasets/mmpencil/mmpencil-dataset/data) | Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies | 2025 | Enables writing-style-independent in-air handwriting recognition by combining mmWave radar with vision-language models for semantic understanding. |
| [Human-Centered Fully Adaptive Radar for Gesture Recognition in Smart Environments](https://ieeexplore.ieee.org/abstract/document/11126867) | [Dataset](https://github.com/ci4r) | IEEE Transactions on Human-Machine Systems | 2025 | Develops fully adaptive radar that automatically adjusts parameters based on user behavior and environment for robust gesture recognition. |
| [mmEgoHand: Egocentric Hand Pose Estimation and Gesture Recognition with Head-mounted Millimeter-wave Radar and IMU](https://arxiv.org/abs/2501.13805) | [Code](https://github.com/WhisperYi/mmVR) | arxiv | 2025 | Achieves egocentric hand tracking by fusing head-mounted mmWave radar with IMU for VR/AR applications with minimal occlusion. |
| [mmDigit: A Real-Time Digit Recognition Framework in Air-Writing Using FMCW Radar](https://ieeexplore.ieee.org/document/10771807/) | [Dataset](https://github.com/Tjkjjc/gesture) | IEEE Internet of Things Journal | 2025 | Enables real-time in-air digit writing recognition using FMCW radar by tracking hand motion trajectories and extracting stroke patterns. |
| [mmHand: Toward Pixel-Level-Accuracy Hand Localization Using a Single Commodity mmWave Device](https://ieeexplore.ieee.org/document/10906525) | N/A | IEEE Internet of Things Journal | 2025 | Achieves pixel-level hand localization accuracy from single mmWave device by combining advanced signal processing with learning-based refinement. |
| [Hand Gesture Recognition With Uncertainty Awareness via FMCW Radar Sensing and Deep Learning](https://ieeexplore.ieee.org/abstract/document/11023089) | [Code](https://github.com/thetuantrinh/Hand-Gesture-Recognition) | IEEE Sensors Journal | 2025 | Addresses the lack of reliability in deep learning-based gesture recognition by introducing a model that quantifies its own prediction uncertainty using a combination of techniques like Monte Carlo dropout and deep ensembles. |
| [Rodar: Robust Gesture Recognition Based on mmWave Radar Under Human Activity Interference](https://ieeexplore.ieee.org/document/10533689) | [Code](https://github.com/Xlab2024/MvDeFormer) | IEEE Transactions on Mobile Computing | 2024 | Achieves robust gesture recognition under activity interference by disentangling hand gestures from body movements using multi-view deformable attention. |
| [Eat-Radar: Continuous Fine-Grained Intake Gesture Detection Using FMCW Radar and 3D Temporal Convolutional Network with Attention](https://ieeexplore.ieee.org/abstract/document/10342867) | [Dataset](https://github.com/Pituohai/Eat-Radar) |  IEEE Journal of Biomedical and Health Informatics  | 2024 | Achieves fine-grained, continuous detection of eating and drinking gestures by applying a 3D temporal convolutional network with attention to radar Range-Doppler data, validated on a new public dataset featuring diverse eating styles in realistic meal sessions. |
| [GaitSADA: Self-Aligned Domain Adaptation for mmWave Gait Recognition](https://www.computer.org/csdl/proceedings-article/mass/2023/243300a218/1RIEetqCvpC) | [Code](https://github.com/exitudio/GaitSADA) | International Conference on Mobile Ad Hoc and Smart Systems (MASS) | 2023 | Tackles the poor generalization of mmWave gait recognition by using a two-stage semi-supervised domain adaptation method that combines contrastive learning with consistency training to align spatial and temporal domain shifts. |
| [Towards Domain-Independent and Real-Time Gesture Recognition Using mmWave Signal](https://dl.acm.org/doi/10.1109/TMC.2022.3207570) | [Code](https://github.com/DI-HGR/cross_domain_gesture_dataset) | IEEE Transactions on Mobile Computing | 2023 | Achieves domain-independent, real-time gesture recognition by using a novel data augmentation framework to improve generalization and a spatial-temporal segmentation algorithm to enable continuous recognition.|

### Occupancy, Presence & Fall Detection
| Title | Code | Publication | Date | Summary |
| :--- | :---: | :---: | :---: | :--- |
| [Exploration of Low-Cost but Accurate Radar-Based Human Motion Direction Determination](https://arxiv.org/abs/2507.22567) | [Code](https://github.com/JoeyBGOfficial/Low-Cost-Accurate-Radar-Based-Human-Motion-Direction-Determination) | arxiv | 2025 | Achieves accurate motion direction estimation with low-cost radar by developing efficient algorithms that maximize information extraction from limited hardware. |
| [End-to-End Radar Human Segmentation with Differentiable Positional Encoding](https://eusipco2025.org/wp-content/uploads/pdfs/0000631.pdf) | N/A | EUSIPCO | 2025 | Performs end-to-end human segmentation from radar by introducing differentiable positional encodings that adapt to irregular radar point distributions. |
| [MVDoppler-Pose: Multi-Modal Multi-View mmWave Sensing for Long-Distance Self-Occluded Human Walking Pose Estimation](https://ieeexplore.ieee.org/abstract/document/11093407) | [Code](https://github.com/gogoho88/MVDoppler-Pose) | CVPR | 2025 | Estimates human poses at long distances under self-occlusion by fusing multi-view mmWave Doppler signatures across multiple radar perspectives. |
| [SelaFD:Seamless Adaptation of Vision Transformer Fine-tuning for Radar-based Human Activity Recognition](https://ieeexplore.ieee.org/document/10888271/) | [Code](https://github.com/wangyijunlyy/SelaFD) | ICASSP | 2025 | Seamlessly adapts pre-trained vision transformers to radar-based HAR through selective fine-tuning strategies that preserve learned representations. |
| [Advanced Millimeter-Wave Radar System for Real-Time Multiple-Human Tracking and Fall Detection](https://www.mdpi.com/1424-8220/24/11/3660) | [Code](https://github.com/DarkSZChao/MMWave_Radar_Human_Tracking_and_Fall_detection) | Sensors | 2024 | Enables real-time multi-human tracking and fall detection by developing advanced algorithms for handling multiple targets and detecting sudden motion changes. |
| [BSENSE: In-vehicle Child Detection and Vital Sign Monitoring with a Single mmWave Radar and Synthetic Reflectors](https://dl.acm.org/doi/abs/10.1145/3666025.3699352) | [Code](https://github.com/mtang724/BSENSE-in-cabin) | SenSys | 2024 | Detects in-vehicle child presence and monitors vital signs using synthetic reflectors to enhance radar signal coverage in confined cabin spaces. |
| [mmFall: Fall Detection using 4D MmWave Radar and a Hybrid Variational RNN AutoEncoder](https://arxiv.org/abs/2003.02386) | [Code](https://github.com/radar-lab/mmfall) | arxiv | 2022 | DCircumvents the need for fall data collection by training a variational RNN autoencoder in a semi-supervised manner on only normal activities, detecting falls as anomalies in the radar signal. |

### Pose Estimation & Skeletal Tracking & Human Motion

| Title | Code | Publication | Date | Summary |
| :--- | :---: | :---: | :---: | :--- |
| [Few-shot Human Motion Recognition through Multi-Aspect mmWave FMCW Radar Data](https://arxiv.org/abs/2501.11028) | [Code](https://github.com/MountainChenCad/channel-DN4) | arxiv | 2025 | Improves radar-inertial odometry for drone navigation in GNSS-denied environments by incorporating a barometer and using a robust m-estimator to more adaptively handle sensor outliers than binary filters. |
| [Learning to Analyze Human Skeletal by Radar–Camera Supervision](https://ieeexplore.ieee.org/document/10930633) | [Code](https://github.com/zylofor/STC-HSANet) | WACV | 2024 | Learns radar-based skeleton estimation through camera supervision that provides ground truth skeletal annotations during training. |
| [RadarLLM: Empowering Large Language Models to Understand Human Motion from Millimeter-wave Point Cloud Sequence](https://arxiv.org/abs/2504.09862) | [Project](https://inowlzy.github.io/RadarLLM/) | CVPR | 2024 | Empowers LLMs to interpret human motion from radar point clouds by developing specialized encoders and language-aligned representations. |
| [Robust Human Detection under Visual Degradation via Thermal and mmWave Radar Fusion](https://dl.acm.org/doi/10.5555/3639940.3639965) | [Code](https://github.com/ramdrop/utm)| EWSN | 2023 | Improves human detection in visually degraded conditions by fusing thermal and radar data using a novel uncertainty-guided method, which leverages Bayesian feature extraction to mitigate sensor-specific noise. |
| [mmPose-NLP: A Natural Language Processing Approach to Precise Skeletal Pose Estimation Using mmWave Radars](https://ieeexplore.ieee.org/document/9723439) | [Code](https://github.com/radar-lab/mmPose-NLP) | IEEE Transactions on Neural Networks and Learning Systems | 2022 | Pioneers the estimation of a large number of skeletal key points from radar by uniquely framing the problem as a natural language processing task, where a sequence-to-sequence model interprets voxelized point clouds like text. |
| [mmBody Benchmark: 3D Body Reconstruction Dataset and Analysis for Millimeter Wave Radar](https://dl.acm.org/doi/10.1145/3503161.3548262) | [Project](https://chen3110.github.io/mmbody/index) | ACM MM | 2022 | Establishes a comprehensive benchmark for 3D body reconstruction by creating a large-scale dataset to systematically compare radar and cameras, revealing radar's superior accuracy to RGB but inferiority to depth under normal conditions, and its greater robustness in adverse weather. |
| [mRI: Multi-modal 3D Human Pose Estimation Dataset using mmWave, RGB-D, and Inertial Sensors](https://dl.acm.org/doi/10.5555/3600270.3602258) | [Project](https://sizhean.github.io/mri) | NeurIPS | 2022 | Fills the gap in home-based health monitoring research by releasing a large-scale, multi-modal dataset (mRI) of rehabilitation exercises captured with synchronized mmWave, RGB-D, and inertial sensors.|
| [Fast and scalable human pose estimation using mmWave point cloud](https://dl.acm.org/doi/10.1145/3489517.3530522) | [Code](https://github.com/anonymousIoTresearcher/FUSE) | DAC | 2022 | Tackles the sparsity and data scarcity issues in mmWave pose estimation by combining a multi-frame representation with a meta-learning framework to enable rapid adaptation to unseen scenarios. |

### Vital Signs & Biometric Identification
| Title | Code | Publication | Date | Summary |
| :--- | :---: | :---: | :---: | :--- |
| [From High-SNR Radar Signal to ECG: A Transfer Learning Model with Cardio-Focusing Algorithm for Scenarios with Limited Data](https://arxiv.org/abs/2506.19358) | N/A | arxiv | 2025 | Tackles data scarcity in radar-based ECG recovery by using a transfer learning model that pre-trains on unlabeled radar data and only requires a few paired samples for fine-tuning, complemented by an algorithm that actively tracks the heart to ensure high-quality signal acquisition. |
| [Simulation for Noncontact Radar-Based Physiological Sensing Using Depth-Camera-Derived Human 3D Model with Electromagnetic Scattering Analysis](https://arxiv.org/abs/2507.13826) | N/A | arxiv | 2025 | Enhances the realism of radar-based physiological simulations by replacing simplified geometric models with actual human body shape and motion data captured by a depth camera.|
| [Enhanced sparse optimization approach for vital signal extraction from millimeter-wave radar](https://www.sciencedirect.com/science/article/abs/pii/S1746809425009577) | [Code](https://github.com/Z-H-XU/DCT-Vital-Signs) | Biomedical Signal Processing and Control | 2025 | Improves vital sign extraction from radar by using a sparse optimization framework based on the Discrete Cosine Transform (DCT), whose energy compaction properties enable better separation of weak and spectrally overlapping signals. |
| [ReBP: Short-Term Blood Pressure Estimation by Reconstructing PPG Signals Based on mmWave Radar](https://ieeexplore.ieee.org/document/11028948) | [Code](https://github.com/JialMa/ReBP) | IEEE Sensors Journal | 2025 | Estimates blood pressure by reconstructing PPG-like signals from mmWave radar that capture cardiovascular pulse wave characteristics. |
| [Atrial Fibrillation Detection via Contactless Radio Monitoring and Knowledge Transfer](https://www.nature.com/articles/s41467-025-59482-y) | [Code](https://github.com/yyuqin/Atrial-Fibrillation-Detection) | nature communications | 2025 | Detects atrial fibrillation from contactless radar monitoring by transferring knowledge from clinical ECG data to radar cardiac patterns. |
| [Hierarchical and Multimodal Data for Daily Activity Understanding](https://arxiv.org/abs/2504.17696) | [Dataset](https://alregib.ece.gatech.edu/software-and-datasets/darai-daily-activity-recordings-for-artificial-intelligence-and-machine-learning/) | arxiv | 2025 | Provides hierarchical multimodal dataset capturing daily activities with synchronized radar, video, and sensor data for comprehensive activity understanding. |
| [CardiacMamba: A Multimodal RGB-RF Fusion Framework with State Space Models for Remote Physiological Measurement](https://arxiv.org/abs/2502.13624) | [Code](https://github.com/WuZheng42/CardiacMamba) | arxiv | 2025 | Measures remote physiological signals by fusing RGB and RF modalities using Mamba models that efficiently capture temporal cardiac dynamics. |
| [RadEye: Tracking Eye Motion Using FMCW Radar](https://dl.acm.org/doi/10.1145/3706598.3713775) | N/A | Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems | 2025 | Tracks eye motion using FMCW radar by detecting minute movements through high-resolution Doppler analysis for non-contact gaze estimation. |
| [Realistic Facial Expression Reconstruction Using Millimeter Wave](https://ieeexplore.ieee.org/abstract/document/10904120) | N/A | IEEE Transactions on Mobile Computing | 2025 | Reconstructs realistic facial expressions from mmWave radar by learning mappings between radar signatures and detailed facial muscle movements. |
| [A Heart Rate Measurement Model Based on Koopman Predictors With FMCW Radar](https://ieeexplore.ieee.org/abstract/document/11122469) | [Code](https://github.com/licongsheng/HRKNet) | IEEE Transactions on Instrumentation and Measurement | 2025 | Introduces a Koopman theory-based deep model that improves radar heart rate measurement by explicitly separating and processing the signal's time-invariant and time-variant dynamics. |
| [AirECG: Contactless Electrocardiogram for Cardiac Disease Monitoring via mmWave Sensing and Cross-domain Diffusion Model](https://dl.acm.org/doi/abs/10.1145/3678550) | [Code](https://github.com/LangchengZhao/AirECG) | IMWUT/Ubicomp | 2024 | Enables accurate, contactless ECG reconstruction for cardiac patients by using a cross-domain diffusion model, whose iterative denoising process can synthesize abnormal waveforms, and a guidance mechanism to ensure high fidelity. |
| [radarODE: An ODE-Embedded Deep Learning Model for Contactless ECG Reconstruction from Millimeter-Wave Radar](https://arxiv.org/abs/2408.01672) | [Code](https://github.com/ZYY0844/radarODE-MTL) | arxiv | 2024 | Improves radar-to-ECG reconstruction by embedding an Ordinary Differential Equation (ODE) into the deep learning decoder, which provides a morphological prior of the ECG waveform to guide the model and enhance robustness against body movements. |
| [radarODE-MTL: A Multi-Task Learning Framework with Eccentric Gradient Alignment for Robust Radar-Based ECG Reconstruction](https://arxiv.org/abs/2410.08656) | [Code](https://github.com/ZYY0844/radarODE-MTL) | arxiv | 2024 | Improves the robustness of radar-based ECG reconstruction against noise by deconstructing the problem into a multi-task learning framework and using a novel gradient alignment strategy to manage conflicting optimization goals. |
| [Machine Learning for Healthcare Radars: Recent Progresses in Human Vital Sign Measurement and Activity Recognition](https://ieeexplore.ieee.org/document/10322785) | N/A |  IEEE Communications Surveys & Tutorials  | 2023 | Surveys the use of machine learning in radar-based healthcare, highlighting its role in enhancing traditional vital sign algorithms versus its foundational necessity for activity recognition. |


### Sleep Monitoring

| Title | Code | Publication | Date | Keywords |
| :--- | :---: | :---: | :---: | :--- |
| [RestAware: Non-Invasive Sleep Monitoring Using FMCW Radar and AI-Generated Summaries](https://arxiv.org/abs/2508.00848v1) | N/A | arxiv | 2025 | Develops a non-invasive sleep monitoring system that uses FMCW radar for posture classification and uniquely integrates a large language model to automatically generate human-readable summaries from the sleep data. |
| [A Robust and Accurate FMCW MIMO Radar Vital Sign Monitoring Framework With 4-D Cardiac Beamformer and Heart-Rate Trace Carving Technique](https://ieeexplore.ieee.org/document/10516282) | N/A | IEEE Transactions on Microwave Theory and Techniques | 2024 | Improves radar-based heart rate monitoring in low-SNR environments by using a 4-D beamformer to focus on the heart in both space and time-frequency, and a trace carving technique to reconstruct faint signals from noise. |
| [MiSleep: Human Sleep Posture Identification from Deep Learning Augmented Millimeter-wave Wireless Systems](https://dl.acm.org/doi/full/10.1145/3643866) | N/A | ACM Transactions on Internet of Things | 2023 | Achieves detailed sleep posture monitoring by predicting 3D body joint locations from radar, uniquely using a state machine to first detect toss-and-turn events and only performing the prediction during stable rest periods. |

### Fatigue driving detection

| Title | Code | Publication | Date | Summary |
| :--- | :---: | :---: | :---: | :--- |
| [CarVision: Vehicle Ranging and Tracking Using mmWave Radar for Enhanced Driver Safety](https://www.computer.org/csdl/proceedings-article/percom/2025/355100a215/27fizQ2avXG) | [Code](https://github.com/srajib826/CarVision) | IEEE International Conference on Pervasive Computing and Communications (PerCom) | 2025 | Enhances driver safety by using mmWave radar for precise vehicle ranging and tracking to provide collision warnings and maintain safe distances. |
| [PhysDrive: A Multimodal Remote Physiological Measurement Dataset for In-vehicle Driver Monitoring](https://arxiv.org/abs/2507.19172) | [Dataset](https://github.com/WJULYW/PhysDrive-Dataset) | arxiv | 2025 | Provides multimodal dataset for driver monitoring with synchronized physiological measurements enabling research on fatigue and distraction detection. |

### Identity Recognition & Person Re-identification

| Title | Code | Publication | Date | Summary |
| :--- | :---: | :---: | :---: | :--- |
| [mmPPT: Hierarchical-Serialization-Enhanced Point Transformer for mmWave Pedestrian Reconstruction](https://www.sciencedirect.com/science/article/abs/pii/S1566253525008978) | N/A | Information Fusion | 2025 | Improves 3D pedestrian reconstruction from sparse radar by using a Point Transformer with a novel hierarchical serialization strategy that captures both fine-grained local details and global structural relationships. |
| [I Sense You Fast: Simultaneous Action and Identity Inference by Slimming Multi-Branch RadarNet](https://ieeexplore.ieee.org/document/11005642) | [Code](https://github.com/MagicalLiHua/PolyLite-RadarNet) | IEEE Transactions on Mobile Computing | 2025 | Achieves simultaneous action and identity recognition using slimmed multi-branch networks that efficiently share features across tasks for real-time inference. |
| [Open-Set Gait Recognition from Sparse mmWave Radar Point Clouds](https://ieeexplore.ieee.org/document/11080220) | [Code](https://github.com/rmazzier/OpenSetGaitRecognition_PCAA) | IEEE Sensors Journal | 2025 | Performs open-set gait recognition from sparse radar point clouds by learning discriminative features that generalize to unseen identities. |
| [mmReID: Person Reidentification Based on Commodity Millimeter-Wave Radar](https://ieeexplore.ieee.org/document/10937945) | [Code](https://github.com/ci4r/mmReID) | IEEE Internet of Things Journal | 2025 | Enables person re-identification across camera views using commodity mmWave radar by extracting gait and body shape signatures. |
| [Through-Wall Cross-Domain User Identification via Lip Movement Micro-Doppler and MIMO Radar: an Unsupervised Domain Adaptation Approach](https://ieeexplore.ieee.org/abstract/document/11153694) | [Code](https://github.com/KaiYCode/Lip-TWCDID) | IEEE Transactions on Mobile Computing | 2025 | Achieves through-wall user identification by detecting lip movement micro-Doppler with MIMO radar and using domain adaptation across environments. | 


## 🌱 Agriculture Areas

| Title | Code | Publication | Date | Summary |
| :--- | :---: | :---: | :---: | :--- |
| [Sem-RaDiff: Diffusion-Based 3D Radar Semantic Perception in Cluttered Agricultural Environments](https://arxiv.org/abs/2509.02283) | N/A | arxiv | 2025 | Tackles radar perception in cluttered agricultural scenes by using a hierarchical diffusion model that first filters sidelobe artifacts and then generates a fine-grained 3D semantic point cloud, enabling the accurate reconstruction of thin structures. |
| [Hydra: Accurate Multi-Modal Leaf Wetness Sensing with mm-Wave and Camera Fusion](https://dl.acm.org/doi/10.1145/3636534.3690662) | [Code](https://github.com/liuyime2/MobiCom24-Hydra) | ACM MobiCom | 2024 | Achieves accurate leaf wetness sensing for precision agriculture by fusing mm-wave radar's dielectric sensing with camera's visual information. |


## 🏭 Industrial Areas

| Title | Code | Publication | Date | Summary |
| :--- | :---: | :---: | :---: | :--- |
| [CRFusion: Fine-Grained Object Identification Using RF-Image Modality Fusion](https://ieeexplore.ieee.org/document/10835118) | N/A | IEEE Transactions on Mobile Computing | 2025 | Enables fine-grained object identification in industrial settings by fusing RF material signatures with visual appearance for enhanced discrimination. |

## 🔒 Forensics & Privacy Security

| Title | Code | Publication | Date | Summary |
| :--- | :---: | :---: | :---: | :--- |
| - | - | - | - | - |
| [Attacking mmWave Imaging With Neural Meta-Material Rendering](https://ieeexplore.ieee.org/document/11007126) | N/A | IEEE Transactions on Information Forensics and Security | 2025 | Demonstrates vulnerabilities in mmWave imaging systems by using neural meta-materials to create adversarial patterns that deceive radar-based detection. |
| [mmHSE: Enhanced Eavesdropping Attack on Headsets Leveraging COTS mmWave Radar](https://dl.acm.org/doi/abs/10.1145/3729475) | N/A | Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies | 2025 | Enhances a radar-based eavesdropping attack on headsets by combining a high-resolution beamformer (MVDR+CZT) for precise vibration capture, a spatiotemporal fitting method for signal isolation, and a GAN for audio refinement. |
| [SPACE: Speaker Adaptation for Acoustic Eavesdropping using mmWave Radio Signals](https://ieeexplore.ieee.org/abstract/document/11123722) | N/A | IEEE Transactions on Mobile Computing | 2025 | Tackles the challenge of speaker variability in mmWave eavesdropping by using a speaker encoder to extract and adapt to individual speaker characteristics, enabling more accurate speech reconstruction. |
| [Acoustic Eavesdropping From Sound-Induced Vibrations With Multi-Antenna mmWave Radar](https://ieeexplore.ieee.org/abstract/document/10925836) | N/A | IEEE Transactions on Mobile Computing | 2025 | Overcomes the need for prior training data in acoustic eavesdropping by using multi-antenna signal processing to achieve micrometer-level vibration extraction for speech recovery. |


## 📦 Other Areas

| Title | Code | Publication | Date | Summary |
| :--- | :---: | :---: | :---: | :--- |
| [A Comprehensive Radar-Based Berthing-Aid Dataset (R-BAD) and Onboard System for Safe Vessel Docking](https://www.mdpi.com/2079-9292/14/20/4065#B12-electronics-14-04065) | [Dataset](https://zenodo.org/records/16936465) [Code](https://github.com/Radar-Uniwa/FMCW-Radar-Based-Berthing-System) | Electronics | 2025 | Addresses the lack of public data for autonomous ship berthing by introducing a large-scale radar and video dataset (R-BAD) and a corresponding onboard system to enable the development of all-weather docking solutions. |
| [How to Talk to Your Classifier: Conditional Text Generation with Radar–Visual Latent Space](https://www.mdpi.com/1424-8220/25/14/4467) | N/A | Sensors | 2025 | Addresses the interpretability of radar classifiers by training a model to simultaneously classify an image and generate descriptive text from its latent space, using the text to reveal how the classifier interprets the visual data. |

## Contribution

Contributions are always welcome! This list is actively maintained.

Please read the [**contribution guidelines**](CONTRIBUTING.md) before submitting your pull request.

## License

[![CC0](https://mirrors.creativecommons.org/presskit/buttons/88x31/svg/cc-zero.svg)](https://creativecommons.org/publicdomain/zero/1.0/)