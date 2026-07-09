import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("models/mpg_regr_model_es.onnx")

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


def predict(input_list: list) -> np.ndarray:
    """Function that takes a list of inputs and returns the model's predictions"""
    x = np.asarray(input_list, dtype=np.float32)

    if x.ndim == 1:
        x = x[np.newaxis, :]

    prediction = session.run(
        [output_name],
        {input_name: x},
    )[0]

    return prediction.flatten()
