#Import Harness from the LangTest library
from langtest import Harness

# Setup and configure harness
harness = Harness(
    task="summarization",
    model={"model": "facebook/opt-1.3b", "hub":"huggingface"},
    data={"data_source" :"XSum",
          "split":"test-tiny"},
    config={
      'model_parameters': {
        'max_tokens': 32
      },

      'tests': {
        'defaults':{
          'min_pass_rate': 1.00
        },

        'robustness':{
          'lowercase': {'min_pass_rate': 0.70},
          'add_typo': {'min_pass_rate': 0.70}
        }
      }
    })

#harness.data = harness.data[:10]

harness.generate()

harness.testcases()

harness.run()

harness.generated_results()

harness.report()
