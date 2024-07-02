#Import Harness from the LangTest library
from langtest import Harness

import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "<API_TOKEN>"

harness = Harness(
                  task="question-answering", 
                  model={"model": "google/flan-t5-small","hub": "huggingface-inference-api"},
                  data={"data_source" :"BoolQ",
                        "split":"test-tiny"}
                  )

harness.configure({
    'model_parameters': {
        'temperature': 0,
        'max_tokens': 64
    },
    
    'tests': {
      'defaults':{
        'min_pass_rate': 1.00
      },

      'robustness':{
        'lowercase': {'min_pass_rate': 0.70},
        'uppercase': {'min_pass_rate': 0.70}
      }
    }
})

harness.generate()

harness.testcases()

harness.run()

harness.generated_results()

harness.report()