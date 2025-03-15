# LLM FACT CHECKING

This repository contains the experimentation code for the llm-fact-checking project, performed on the BioASQ dataset.

## Preparation

To set up your environment, follow these steps for the installation of requirements
```bash
  python -m vritualenv venv
  source venv/bin/activate
  pip install -r requirements.txt
```

You also need the OpenAI API Key specified. You can do this by:

```export OPENAI_API_KEY=your_key```

## Running Experiment

To execute the experiment, use the following commands:

1. Run the default experiment:

```bash
python run_experiment.py -e experiment_name
```
You should input the argument experiment_name at every experiment run

Optionally, you can run with a limited number of test samples, instead of the whole test set.

```bash
python run_experiment.py -e experiment_name --num_test_samples {num_test_samples}
```
Replace `{num_test_samples}` with the desired number of test samples you wish to run.

Other options include setting the log leve to debug for example:

```bash
python run_experiment.py -e experiment_name -l DEBUG
```

To modify experimentation settings, please update the `config.json`.

## Direct text comparison test

For the sake of testing and experimenting, one can use the direct text comparison and perform the fact checking by manually
providing the LLM answer (called as `inline_answer` and the reference document, called as `inline_reference`)

```bash
python compare_text_pair.py -e "Direct text experimentation" --inline_answer "answer text" --inline_reference "reference text"
```

