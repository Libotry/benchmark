import json
import base64

from datasets import Dataset

from ais_bench.benchmark.openicl import BaseEvaluator
from ais_bench.benchmark.registry import LOAD_DATASET
from ais_bench.benchmark.datasets.utils.datasets import get_data_path
from ais_bench.benchmark.utils.file import check_mm_custom
from ais_bench.benchmark.datasets.utils.video import VideoAsset, image_to_base64
from ais_bench.benchmark.utils.logging.logger import AISLogger
from ais_bench.benchmark.utils.logging.exceptions import ConfigError
from ais_bench.benchmark.utils.logging.error_codes import UTILS_CODES

from .base import BaseDataset

logger = AISLogger()


@LOAD_DATASET.register_module()
class MMCustomDataset(BaseDataset):
    
    @staticmethod
    def load(path, mm_type, num_frames=5):
        """
        Load a MM-style dataset that pairs images with question-answer
        annotations.

        Parameters
        ----------
        path : str
            Full path to the **question file** (usually `*_questions.json` or
            similar).  The corresponding ground-truth answer file is expected to
            live in the same directory and to have the suffix
            `_annotations.json`, e.g.
                /foo/bar/train_questions.json   # <-- `path`
                /foo/bar/train_annotations.json # <-- auto-detected
        mm_type : str
            How the image should be returned:
            - "path"   : keep the original file path (str)
            - "base64" : read the file and return it as a base64-encoded
            string (str)
        """
        path = get_data_path(path, local_mode=True)
        if not check_mm_custom(path):
            raise ConfigError(UTILS_CODES.MM_CUSTOM_DATASET_WRONG_FORMAT,"Invalid dataset, please check it!")

        dataset = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line.strip())
                data_type = line["type"]
                if mm_type == "path":
                    line['mm_url'] = line["path"]
                elif mm_type == "base64":
                    if data_type in ["image", "audio"]:
                        with open(line['path'], 'rb') as f:
                            binary_data = f.read()
                        line['mm_url'] = base64.b64encode(binary_data).decode("utf-8")
                    elif data_type == "video":
                        base64_frames = []
                        frames = VideoAsset(video_path=line['path'], num_frames=num_frames).pil_images
                        for frame in frames:
                            base64_frame = image_to_base64(frame)
                            base64_frames.append(base64_frame)
                        line['mm_url'] = ','.join(base64_frames)
                else:
                    raise ConfigError(UTILS_CODES.MM_CUSTOM_DATASET_WRONG_FORMAT,"Invalid type in dataset, please check it!")
                dataset.append(line)
        return Dataset.from_list(dataset)
        

class MMCustomEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        result = {'accuracy': 1}
        return result