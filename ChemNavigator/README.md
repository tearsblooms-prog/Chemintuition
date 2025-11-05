# Learned Interpretable Chemical Intuition Enables De Novo Catalytic Reaction Discovery

## Project Overview

ChemNavigator is a deep learning-based framework for autonomous chemical reaction discovery, focusing on uncovering potential new reactions from vast chemical spaces. The project integrates a reaction condition prediction model, **ReactNet**, with a reaction yield prediction model, **YieldMPNN**, capturing interpretable chemical intuition and simulating the reasoning process of human chemists. Through high-throughput computational screening, ChemNavigator successfully identified multiple high-yield candidate reactions, and experimental validation led to the discovery of four entirely new chemical reactions, providing a scalable solution for AI-driven chemical innovation.

## Key Features

- **Reaction Space Construction**: Builds virtual chemical reaction spaces.
- **Reaction Yield Prediction**: Predicts yields for Suzuki-Miyaura and Buchwald-Hartwig coupling reactions.
- **Reaction Condition Prediction**: Predicts optimal catalysts, reagents, and solvents based on reactants.
- **Multi-Dataset Support**: Supports multiple reaction datasets, including USPTO-Condition and Reaxys-TotalSyn-Condition.

## Project Structure

```
ChemNavigator/
|── reaction_space                  # Reaction space construction
├── yield_task/                     # Yield prediction tasks
│   ├── suzuki_miyaura/            # Suzuki-Miyaura dataset yield prediction
│   └── buchwald_hartwig/          # Buchwald-Hartwig dataset yield prediction
├── condition_task/                 # Condition prediction tasks
│   ├── USPTO-Condition/           # USPTO condition prediction
│   └── Reaxys-TotalSyn-Condition/ # Reaxys condition prediction
└── README.md
```

## Datasets
Before running the code, download the datasets into the `data` directories of each task.

## ChemNavigator model parameters
If you want to make predictions using our parameters, please first [download](https://drive.google.com/drive/folders/1Glpf9KERL7Dv-YPjly0I6p-FiW_vqqcW?usp=sharing) the model 
parameters and place them in the corresponding results folder.
### Yield Prediction Datasets
- **Suzuki-Miyaura**: Contains reactants, catalysts, reagents, solvents, and yields. The Suzuki-Miyaura yield dataset is from Perera et al., [A platform for automated nanomole-scale reaction screening and micromole-scale synthesis in flow](https://www.science.org/doi/10.1126/science.aap9112).
- **Buchwald-Hartwig**: Yield prediction data for C–N coupling reactions, published by Ahneman et al., [Predicting reaction performance in C–N cross-coupling using machine learning](https://www.science.org/doi/10.1126/science.aar5169).

### Condition Prediction Datasets
- **USPTO-Condition**: Open-source dataset of reaction conditions from the US Patent Office, published by Ahneman et al., [Generic Interpretable Reaction Condition Predictions with Open Reaction Condition Datasets and Unsupervised Learning of Reaction Center](https://spj.science.org/doi/10.34133/research.0231). Freely available at [GitHub](https://github.com/wangxr0526/Parrot).
- **Reaxys-TotalSyn-Condition**: Full synthesis reaction condition data from the commercial Reaxys database.

## Installation and Usage

### Quick Start

1. Clone the repository:
```bash
git clone https://github.com/your-username/ChemNavigator.git
cd ChemNavigator
```

2. Create a conda environment and install dependencies:
```bash
conda env create -f environment.yml
conda activate ChemNavigator
```

3. Construct the reaction space:
```bash
# Build the chemical reaction space from reactants
cd reaction_space
python generate_from_pools.py
```

4. Run yield prediction:
```bash
# Suzuki-Miyaura yield prediction
cd yield_task/suzuki_miyaura
cd data
#data preprocessing
python 1_suzuki_miyaura_preprocess.py
python 2_generate_fingerprint.py
python 3_preprocess_data.py

cd ../
python Train_YieldMPNN.py   # Train
python predict.py            # Predict

# Buchwald-Hartwig yield prediction
cd yield_task/buchwald_hartwig
cd data
#data preprocessing
python 1_buchwald_hartwig_preprocess.py
python 2_generate_fingerprint.py
python 3_preprocess_data.py

cd ../
python Run_YieldMPNN_Experiments.py
```

5. Run condition prediction:
```bash
# USPTO condition prediction
cd condition_task/USPTO-Condition
cd data
#data preprocessing
python USTPO_condition_preprocess.py

cd ../
python train_ReactNet.py         # Train
python predict_conditions.py     # Predict

# Reaxys condition prediction
cd condition_task/Reaxys-TotalSyn-Condition
python train_Reaxys.py           # Train
```

## Citation

If you use this project in your research, please cite:

```bibtex
@article{paper,
  title={Learned Interpretable Chemical Intuition Enables De Novo Catalytic Reaction Discovery},
  author={Jun Zhou},
  journal={},
  year={2025}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or suggestions, please contact:
- Email: tearsblooms@gmail.com
- GitHub Issues: [Project Issues Page](https://github.com/your-username/ChemNavigator/issues)
