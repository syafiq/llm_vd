## Description
This repository contains the code, data, and models used in our study on the effectiveness of Large Language Models (LLMs) for vulnerability detection across multiple programming languages. We focus on JavaScript, Java, Python, PHP, Go, and C/C++, leveraging the CVEfixes dataset to create language-specific subsets. We fine-tune and evaluate state-of-the-art LLMs on these subsets to assess their performance in detecting vulnerabilities across different languages.

Our research addresses the following questions:

- RQ1: How does the detection performance vary for different programming languages when using state-of-the-art LLMs on code from the same curated dataset?
- RQ2: Is there any dependency between code complexity and the detection performances using LLM?

## Downloading the Dataset and Models

Due to size constraints, the dataset and models are provided separately.

Datasets: Download the dataset from [Google Drive](https://drive.google.com/drive/folders/1gef7O2592a5BEdjX_yPz78HTN7-p_Zmo?usp=sharing). 

Models: Download the finetuned models from [Google Drive](https://drive.google.com/drive/folders/1RhyW2CkIvLzsDmIfdmBmYNFhxNQansOv?usp=sharing). 


## RQ1
### Data Processing
To replicate the data processing for each programming language, run `getdata.py` from `RQ1/data_processing/` directory.

Example
```bash
python RQ1/data_processing/getdata.py {lang}
```

where `{lang}` can be `[JavaScript, PHP, C, Java, Python, Go, C++]` (case sensitive). This produces 3 sets: `{lang}_date_[train,valid,test].json`, where `train` is used for training, `valid` is used for development, and `test` is used for the measurement that is shown in the paper. 

### Fine-tuning the Models
To replicate the finetuning process of our created models, use the following script in `RQ1/finetuning/`:

Example:
```bash
python RQ1/finetuning/finetune.py
```

Or, if the GPUs are accessible through [slurm](https://slurm.schedmd.com/slurm.html), one can use the script:

```bash
sbatch RQ1/finetuning/finetune.sh
```

### Measurement on MegaVul (Java)
To replicate the measurement on MegaVul for Java, use the following script:

```bash
python RQ1/test_on_megavul/megavul.py
```

The dataset can be downloaded from [this GitHub repo](https://github.com/Icyrockton/MegaVul). Note that the script only consumes the code snippets (named as column `text`) and the respective label (named as column `label`).

## RQ2
### Code Complexity Analysis
To replicate the analysis of the dependency between code complexity and detection performance, use RQ2/measure.py.

Example:
```bash
python RQ2/measure.py {lang}
```

where lang is `[js, python, java, php, go, c_cpp]`. For each measurement, the output will be the mean value of the following metrics: `[h_volume, h_difficulty, h_effort, cyclomatic_complexity, nloc]` that can be used to perform the analysis in RQ2. Note that each measurement requires the dataset (`train` and `test`) for each language, which can be taken from the previous step or directly from our Google Drive mentioned above.  

## Results
Our findings indicate significant variations in detection performance across different programming languages when using fine-tuned LLMs:

JavaScript demonstrated higher vulnerability detection performance, achieving better F1 scores and lower Vulnerability Detection Scores (VD-S).
C/C++ showed lower performance, with significantly lower F1 scores and higher VD-S values.
We did not find a strong correlation between code complexity and the detection capabilities of LLMs. The correlation coefficients were weak and not statistically significant, suggesting that code complexity, as measured by metrics like Cyclomatic Complexity and Halstead Effort, may not be a determining factor in the effectiveness of LLM-based vulnerability detection.

For detailed results and analysis, please refer to our paper.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
This project is licensed under the Apache-2.0 license.

## Acknowledgments
We would like to thank the contributors to the CVEfixes dataset and the developers of the DeepSeek-Coder models.

## Contact
For any inquiries, please open an issue in this repository.
