\# Reference Paper



\## Citation



\*\*Yiqun Shang, Minrui Zheng, Jiayang Li, and Xinqi Zheng.\*\*



"An effective feature selection approach based on hybrid Grey Wolf Optimizer and Genetic Algorithm for hyperspectral image."



Scientific Reports, Volume 15, Article 1968, 2025.



DOI: 10.1038/s41598-024-84934-8



\## Relevance to This Project



This paper introduces GWOGA, a hybrid feature-selection approach that combines Grey Wolf Optimizer (GWO) and Genetic Algorithm (GA) for hyperspectral image feature selection.



The paper motivates the hybrid approach by combining the complementary characteristics of GWO and GA:



\- GWO provides fast convergence during the early optimization stage.

\- GA provides broader global exploration and helps avoid local optima.

\- Chaotic initialization and Opposition-Based Learning are used to improve population diversity.

\- Elite learning is used to improve the search process.



The proposed approach was evaluated on the Indian Pines, KSC, and Botswana hyperspectral datasets.



\## Relation to This Implementation



This project takes inspiration from the hybrid GWO-GA optimization strategy described in the reference paper.



However, the current implementation differs from the original paper.



Instead of directly selecting physical hyperspectral spectral bands, this implementation first extracts a 512-dimensional deep feature representation using ConvNeXt.



The feature-selection pipeline is:



Image Dataset

→ ConvNeXt Feature Extraction

→ 512-D Deep Features

→ GWO Feature Selection

→ GA Refinement

→ SVM Classification



Therefore, the selected features in this implementation represent dimensions of the extracted deep feature vector rather than the original physical spectral bands.



\## Reference



Shang, Y., Zheng, M., Li, J., \& Zheng, X. (2025).



"An effective feature selection approach based on hybrid Grey Wolf Optimizer and Genetic Algorithm for hyperspectral image."



Scientific Reports, 15, 1968.



DOI: 10.1038/s41598-024-84934-8

