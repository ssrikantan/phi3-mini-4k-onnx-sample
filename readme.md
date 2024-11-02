# Contoso Gaming Customer AI Assistant powered by Phi3-mini-4k onnxruntime

This demo uses the microsoft/Phi-3-mini-4k-instruct-onnx model that uses directml.
This model is available in Hugging face - [here](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx)

## Reference
This demo is based off the sample code provided [here](https://onnxruntime.ai/docs/genai/tutorials/phi3-python.html#run-with-directml)

The steps below are based on the documentation above.

**This will work only with python 9, since the onnxruntime-genai only works with it**

```sh

# Download the Model
huggingface-cli download microsoft/Phi-3-mini-4k-instruct-onnx --include directml/* --local-dir .

# Install the following in a virtual environment that has Python 9
pip install numpy
pip install --pre onnxruntime-genai-directml

```

# Running the Demo

1. The phi3-qa.py is from the documentation above
2. The Streamlit app can be run to show this in the browser
