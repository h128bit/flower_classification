from typing import Tuple, Callable
import torch
from sklearn.preprocessing import LabelEncoder
import toml
import numpy as np
import os


class ModelProxy:

    def __init__(self,
                 model: torch.nn.Module,
                 data_preprocess: Callable[[np.array], torch.tensor]) -> None:
        self._model = model

        # load weights and classes labels
        ROOT_FOLDER = os.path.dirname(os.getcwd())
        path_to_config = os.path.join(ROOT_FOLDER + '/app', 'modelproxy_config.toml')
        with open(path_to_config, 'r') as f:
            toml_string = f.read()
            read_toml = toml.loads(toml_string)

        self._path_to_weights = os.path.join(ROOT_FOLDER + '/app', read_toml['path_to_weights'])
        self._path_to_classes = os.path.join(ROOT_FOLDER + '/app', read_toml['path_to_encoded_classes'])

        self._model.load_state_dict(torch.load(self._path_to_weights, weights_only=True, map_location=torch.device('cpu')))

        self._model.eval()

        self._encoder = LabelEncoder()
        self._encoder.classes_ = np.load(self._path_to_classes, allow_pickle=True)

        self._data_preprocessing = data_preprocess

    def _sort_by_confidence(self, probability_list: np.array):
        top_indexes = np.array(sorted(range(len(probability_list)), key=lambda k: probability_list[k])[::-1])
        labels = self._encoder.inverse_transform(top_indexes)
        probability = np.array([probability_list[idx] for idx in top_indexes])
        return probability, labels

    def __call__(self, img: np.array) -> Tuple[np.array, np.array]:
        img = self._data_preprocessing(img)
        res = self._model(img.unsqueeze(0))
        res = torch.softmax(res[0], dim=0)

        res = res.detach().numpy()

        return self._sort_by_confidence(res)
