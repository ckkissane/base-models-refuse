Code to reproduce key results accompanying "Base LLMs refuse too".

* [Blog post](https://www.alignmentforum.org/posts/YWo2cKJgL7Lg8xWjj/base-llms-refuse-too)

## Contents

* `apply_refusal_dir.py` contains code to reproduce refusal evaluations for base models with and without interventions (steering or ablations) applied. Example usage:

`python apply_refusal_dir.py --model_name qwen1.5-0.5b --intervention_type addition --instruction_type harmful`

* `extract_refusal_dir.py` contains code to extract the "refusal direction" from each model with harmful / harmless contrast pairs. There's no need to run this to reproduce the results, since we've already provided each refusal direction that we use in the `refusal_directions` directory. Example usage:

`python extract_refusal_dir.py --model_name qwen1.5-0.5b-chat`

We build on code from [Arditi et al.](https://github.com/andyrdt/refusal_direction)
