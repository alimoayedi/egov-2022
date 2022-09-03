# egov-2022

About

This directory contains the code and dataset used for the publication "Automated Topic Categorisation of Citizens’ Contributions: Reducing Manual Labelling Efforts Through Active Learning.", published in the Proceedings of the International Conference on Electronic Government EGOV 2022.

This work is based on research in the project CIMT/Partizipationsnutzen, which is funded by the Federal Ministry of Education and Research as part of its Social-Ecological Research funding priority, funding no. 01UU1904. (for more information, visit https://www.cimt-hhu.de/en/)

----------

Content

Code

-> active-learning

-> full-supervision-learning

The folders contain the experimental code. The deep learning experiments were realised using Jupyter Notebooks on Google Colab.

Data

-> Cycling-Dialogues contains the original dataset and the terms of use.

-> "dataset-preprocessed.pkl" is a preprocessed file which enriches the original dataset with preprocessed text (title and text were concatenated, tokenized and lowercased). This file is used for running the traditional machine learning experiments.

Results

-> An empty folder in which models are stored (after running the code for full supervision baselines).

----------

Citation

If you use the dataset, please cite the following paper:

@InProceedings{10.1007/978-3-031-15086-9_24, author="Romberg, Julia and Escher, Tobias", editor="Janssen, Marijn and Cs{\'a}ki, Csaba and Lindgren, Ida and Loukis, Euripidis and Melin, Ulf and Viale Pereira, Gabriela and Rodr{\'i}guez Bol{\'i}var, Manuel Pedro and Tambouris, Efthimios", title="Automated Topic Categorisation of Citizens' Contributions: Reducing Manual Labelling Efforts Through Active Learning", booktitle="Electronic Government", year="2022", publisher="Springer International Publishing", address="Cham", pages="369--385", isbn="978-3-031-15086-9"}

----------

License

The annotated data corpus is available under the Creative Commons CC BY-SA License (https://creativecommons.org/licenses/by-sa/4.0/).

----------

Contact Person

Julia Romberg, julia.romberg@hhu.de, https://www.cimt-hhu.de/en/team/romberg/
