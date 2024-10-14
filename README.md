## Description
This repository contains the code, data, and models used in our study on the effectiveness of Large Language Models (LLMs) for vulnerability detection across multiple programming languages. We focus on JavaScript, Java, Python, PHP, Go, and C/C++, leveraging the CVEfixes dataset to create language-specific subsets. We fine-tune and evaluate state-of-the-art LLMs on these subsets to assess their performance in detecting vulnerabilities across different languages.

Our research addresses the following questions:

- RQ1: How does the detection performance vary for different programming languages when using state-of-the-art LLMs on code from the same curated dataset?
- RQ2: Is there any dependency between code complexity and the detection performances using LLM?

## Downloading the Dataset and Models

Due to size constraints, the dataset and models are provided separately.

Dataset: Download the dataset from [Google Drive](https://drive.google.com/drive/folders/1gef7O2592a5BEdjX_yPz78HTN7-p_Zmo?usp=sharing). 

Models: Download the fine-tuned models from [Google Drive](https://drive.google.com/drive/folders/1RhyW2CkIvLzsDmIfdmBmYNFhxNQansOv?usp=sharing). 

## Data Processing
Process the data for each programming language using the provided Jupyter notebooks in RQ1/data_processing/.

Example
```bash
jupyter notebook RQ1/data_processing/js-dataproc.ipynb
```

## Fine-tuning the Models
Fine-tune the models using the script in RQ1/finetuning/.

Example:
```bash
python RQ1/finetuning/finetune.py --config RQ1/finetuning/default_config.yaml
```

## Code Complexity Analysis
Analyze the dependency between code complexity and detection performance using RQ2/measure.py.

Example:
```bash
python RQ2/measure.py
```

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
