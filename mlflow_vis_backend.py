import os
import os.path as osp

from mmengine.registry import VISBACKENDS
from mmengine.utils import scandir
from mmengine.visualization import MLflowVisBackend as BaseMLflowVisBackend


@VISBACKENDS.register_module()
class MLflowVisBackend(BaseMLflowVisBackend):
    def close(self) -> None:
        """Close the mlflow."""
        if not hasattr(self, '_mlflow'):
            return

        file_paths = dict()
        for filename in scandir(self.cfg.work_dir, self._artifact_suffix,
                                True):
            file_path = osp.join(self.cfg.work_dir, filename)
            relative_path = os.path.relpath(file_path, self.cfg.work_dir)
            dir_path = os.path.dirname(relative_path)
            file_paths[file_path] = dir_path

        for file_path, dir_path in file_paths.items():
            self._mlflow.log_artifact(file_path, dir_path)

        self._mlflow.end_run()


# import numpy as np
# import mlflow

# array = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# dataset = mlflow.data.from_numpy(array, source="data.csv")

# # Log an input dataset used for training
# with mlflow.start_run():
#     mlflow.log_input(dataset, context="training")
#          mlflow.log_input(dataset, context="training")
