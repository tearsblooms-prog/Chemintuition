# Chemist's intuition-driven exploration enables non-obvious reaction discovery beyond reaction space

## Project Overview

Chemintuition is a deep learning-based framework for autonomous chemical reaction discovery, focusing on uncovering potential new reactions from vast chemical spaces. The project integrates a reaction condition prediction model, **ConditionGen**, with a reaction yield prediction model, **YieldEvaluator**, capturing interpretable chemical intuition and simulating the reasoning process of human chemists. Through high-throughput computational screening, ChemNavigator successfully identified multiple high-yield candidate reactions, and experimental validation led to the discovery of four entirely new chemical reactions, providing a scalable solution for AI-driven chemical innovation.

## Key Features

- **Reaction Space Construction**: Builds virtual chemical reaction spaces.
- **Reaction Yield Prediction**: Predicts yields for Suzuki-Miyaura and Buchwald-Hartwig coupling reactions.
- **Reaction Condition Prediction**: Predicts optimal catalysts, reagents, and solvents based on reactants.
- **Reaction Feasibility Agent**: Uses RDKit-based local evidence and an LLM to rank whether predicted major products are chemically plausible.
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
├── analysis_reactions_agent/       # LLM-assisted reaction feasibility ranking
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
git clone https://github.com/tearsblooms-prog/Chemintuition.git
cd Chemintuition
```

2. Create a conda environment and install dependencies:
```bash
conda env create -f environment.yml
conda activate Chemintuition
```

3. Construct the reaction space:
```bash
# Build the chemical reaction space from reactants
cd reaction_space
python 1_build_fragment_pools.py
python 2_generate_from_pools.py
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
python train_10SEED.py   # Train
python predict.py            # Predict

# Buchwald-Hartwig yield prediction
cd yield_task/buchwald_hartwig
cd data
#data preprocessing
python 1_buchwald_hartwig_preprocess.py
python 2_generate_fingerprint.py
python 3_preprocess_data.py

cd ../
python run_experiments_10seed.py
```

5. Run condition prediction:
```bash
# USPTO condition prediction
cd condition_task/USPTO-Condition
cd data
#data preprocessing
python USTPO_condition_preprocess.py

cd ../
python train_ConditionGen_ReactNet.py         # Train
python predict_conditions.py     # Predict

# Reaxys condition prediction
cd condition_task/Reaxys-TotalSyn-Condition
python train_ConditionGen_Reaxys.py           # Train
```

6. Run the reaction feasibility agent:
```powershell
cd analysis_reactions_agent
conda env create -f environment.yml
Test-Path .\local_reaction_evidence.py
conda run -n yieldnet-reaction-agent python run_reaction_feasibility_agent.py --dry-run --no-resume --limit 20
```

## Reaction Feasibility Agent

The `analysis_reactions_agent/` module provides an LLM-assisted post-screening step for candidate reactions generated by the yield and condition workflows. It evaluates whether the written major product is chemically plausible from the supplied reactants and optional condition context.

Main behavior:

- Reads candidate reactions from a CSV file.
- Builds RDKit-based local structural evidence.
- Sends sanitized batches to an LLM provider such as Gemini or DeepSeek.
- Writes ranked feasibility results as CSV and JSONL.
- Keeps `predicted_yield` for output sorting/bookkeeping but does not send it to the LLM.

Default files:

- Input: `analysis_reactions_agent/data/demo_reactions.csv`
- Ranked output: `analysis_reactions_agent/data/demo_reactions_ranked.csv`
- Raw output: `analysis_reactions_agent/data/demo_reactions_raw.jsonl`
- Progress log: `analysis_reactions_agent/data/demo_reactions_feasibility_progress.txt`

The agent has its own Conda environment because RDKit is a hard dependency:

```powershell
cd analysis_reactions_agent
conda env create -f environment.yml
Test-Path .\local_reaction_evidence.py
$env:GEMINI_API_KEY="your_key"
conda run -n yieldnet-reaction-agent python run_reaction_feasibility_agent.py --batch-size 8 --max-workers 4
```

`local_reaction_evidence.py` must exist before the agent can run. See [`analysis_reactions_agent/README.md`](analysis_reactions_agent/README.md) for the full input schema, CLI options, provider configuration, resume behavior, and current source-completeness checks.

## Citation

If you use this project in your research, please cite:

```bibtex
@article{paper,
  title={Capturing interpretable chemist’s intuition enables high-yield catalytic reaction
discovery},
  author={Jun Zhou},
  journal={},
  year={2026}
}
```

## Contact

For questions or suggestions, please contact:
- Email: tearsblooms@gmail.com
- GitHub Issues: [Project Issues Page](https://github.com/your-username/ChemNavigator/issues)

