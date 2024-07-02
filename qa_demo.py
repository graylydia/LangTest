# #Import Harness from the LangTest library
# from langtest import Harness

# import os
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# harness = Harness(
#                   task="question-answering", 
#                   model={"model": "google/flan-t5-small","hub": "huggingface-inference-api"},
#                   data={"data_source" :"BoolQ",
#                         "split":"test-tiny"}
#                   )

# harness.configure({
#     'model_parameters': {
#         'temperature': 0,
#         'max_tokens': 64
#     },
    
#     'tests': {
#       'defaults':{
#         'min_pass_rate': 1.00
#       },

#       'robustness':{
#         'lowercase': {'min_pass_rate': 0.70},
#         'uppercase': {'min_pass_rate': 0.70}
#       }
#     }
# })

# harness.generate()

# harness.testcases()

# harness.run()

# harness.generated_results()

# harness.report()

# Import necessary libraries
import os

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = ""

#Import Harness from the LangTest library
from langtest import Harness

# Define the model and data source
model={"model": "text-davinci-003","hub":"openai"}
data={"data_source" :"CommonsenseQA-test-tiny"}

# Create a Harness object
harness = Harness(task="question-answering", model=model, data=data)

harness.configure(
{
 "evaluation": {"metric":"embedding_distance","distance":"cosine","threshold":0.9},
 "embeddings":{"model":"text-embedding-ada-002","hub":"openai"},
 # Note: To switch to the Hugging Face model, change the "hub" to "huggingface" and set the "model" to the desired Hugging Face embedding model.
 
'tests': {'defaults': {'min_pass_rate': 0.65},

           'robustness': {'add_ocr_typo': {'min_pass_rate': 0.66},
                          'dyslexia_word_swap':{'min_pass_rate': 0.60}
                         }
          }
 }

)

harness.generate().run().generated_results()
