import os

import onnxruntime as ort

from . import download


class CrepeInferenceSession(ort.InferenceSession):
    def __init__(self, model='full', sess_options=None, providers=None, provider_options=None, **kwargs):
        # Ensure model is available, downloading if necessary
        model_path = download.ensure_model_available(model, verbose=True)
        super().__init__(str(model_path), sess_options, providers, provider_options, **kwargs)
