import json
import numpy as np
import torch
from transformers import pipeline

class InferlessPythonModel:
        
    def initialize(self):
        self.generator = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v2",
            device_map="auto",
        )

    def infer(self, inputs):
        audio_url = inputs["audio_url"]
        pipeline_output = self.generator(audio_url)
        return {"transcribed_output": pipeline_output}

    def finalize(self):
        self.processor = None
        self.model = None

