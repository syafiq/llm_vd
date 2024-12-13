## Description
This repository contains the code, data, and models used in our study on the effectiveness of Language Models (LMs) for vulnerability detection across multiple programming languages. We focus on JavaScript, Java, Python, PHP, Go, and C/C++, leveraging the CVEfixes dataset to create language-specific subsets. We fine-tune and evaluate state-of-the-art LMs on these subsets to assess their performance in detecting vulnerabilities across different languages.

- We clean and adapt the large CVEFixes dataset to perform an empirical study on the vulnerability detection performance differences between JavaScript, Java, Python, PHP, Go, and C/C++.
- We analyze and present correlation figures for the relationship between code complexity and vulnerability detection performance in the investigated dataset.

## Downloading the Dataset and Models

Due to size constraints, the dataset and models are provided separately.

Datasets: Download the dataset from [Google Drive](https://drive.google.com/drive/folders/1gef7O2592a5BEdjX_yPz78HTN7-p_Zmo?usp=sharing). 

Models: Download the finetuned models from [Google Drive](https://drive.google.com/drive/folders/1RhyW2CkIvLzsDmIfdmBmYNFhxNQansOv?usp=sharing). 


## Performance 
### Data Processing
To replicate the data processing of [CVEFixes](https://github.com/secureIT-project/CVEfixes) for each programming language, run `getdata.py` from `RQ1/data_processing/` directory.

Example
```bash
python RQ1/data_processing/getdata.py {lang}
```

where `{lang}` can be `[JavaScript, PHP, C, Java, Python, Go, C++]` (case sensitive). This produces 3 sets: `{lang}_date_[train,valid,test].json`, where `train` is used for training, `valid` is used for development, and `test` is used for the measurement that is shown in the paper. To get both C and C++ at the same time, one can modify the query a bit to something like:

```python
query = """
       SELECT m.code, m.before_change, c.committer_date
       FROM file_change f, method_change m, commits c
       WHERE m.file_change_id = f.file_change_id
       AND c.hash = f.hash
       AND f.programming_language IN ('C', 'C++')
       """
```

Another possible approach is to process them separately and then join the dataset by merging two dataframes, for instance. 

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

### Measurement on Other Datasets
To replicate the measurement on MegaVul, synth-vuln-fixes, and SARD use the following script:

```bash
python RQ1/test_on_other/_other.py
```

The dataset can be downloaded from [MegaVul](https://github.com/Icyrockton/MegaVul), [synth-vuln-fixes](https://huggingface.co/datasets/patched-codes/synth-vuln-fixes), [SARD](https://samate.nist.gov/SARD/test-suites/103). Note that the script only consumes the code snippets (named as column `text`) and the respective label (named as column `label`).

## Correlation between performance (F1) and complexity
### Code Complexity Analysis
To replicate the analysis of the dependency between code complexity and detection performance, use RQ2/measure.py.

Example:
```bash
python RQ2/measure.py {lang}
```

where lang is `[js, python, java, php, go, c_cpp]`. For each measurement, the output will be the mean value of the following metrics: `[h_volume, h_difficulty, h_effort, cyclomatic_complexity, nloc]` that can be used to perform the analysis. Note that each measurement requires the dataset (`train` and `test`) for each language, which can be taken from the previous step or directly from our Google Drive mentioned above.  

## Results
Our findings indicate significant variations in detection performance across different programming languages when using fine-tuned LMs:

JavaScript demonstrated higher vulnerability detection performance, achieving better F1 scores.
C/C++ showed lower performance, with significantly lower F1 scores.
We did not find a strong correlation between code complexity and the detection capabilities of LMs. The correlation coefficients were weak and not statistically significant, suggesting that code complexity, as measured by metrics like Cyclomatic Complexity and Halstead Effort, may not be a determining factor in the effectiveness of LM-based vulnerability detection.

For detailed results and analysis, please refer to our paper.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
This project is licensed under the Apache-2.0 license.

## Acknowledgments
We would like to thank the contributors to the CVEfixes dataset and the developers of the DeepSeek-Coder models.

## Contact
For any inquiries, please open an issue in this repository.
