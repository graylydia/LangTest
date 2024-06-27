import os
from langtest import Harness
from huggingface_hub import login

# Log in to Hugging Face Hub
login(token="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")

# Set environment for HF hub
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Configure Harness
harness = Harness(
    task="question-answering",
    model={"model": "meta-llama/Meta-Llama-3-8B", "hub": "huggingface"},
    data={"data_source": "Open-Orca/OpenOrca", "source": "huggingface", "split": "train"}
)

harness.configure({
    'model_parameters': {
        'temperature': 0,
        'max_tokens': 64
    },
    'tests': {
        'defaults': {
            'min_pass_rate': 1.00
        },
        'robustness': {
            'lowercase': {'min_pass_rate': 0.70}
            # 'uppercase': {'min_pass_rate': 0.70}
        }
    }
})

# Generate test cases
harness.generate()

# Display test cases
harness.testcases()

# Run the tests outlined in harness configuration
harness.run()

# Generate results
harness.generated_results

# Summary of results
harness.report()
