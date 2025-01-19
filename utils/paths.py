import os

PROJECT_ROOT_DIR = '/home/kiriyamagk/桌面/AlignAnything'
TRAINED_MODELS_DIR = os.path.join(PROJECT_ROOT_DIR, "trained_models")
LOG_DIR = os.path.join(PROJECT_ROOT_DIR, "logs")



def path_completion(asset_path: str, root: str = None) -> str:
    """
        Takes in a local asset path and returns a full path.
            if @asset_path is absolute, do nothing
            if @asset_path is not absolute, load xml that is shipped by the package

        Args:
            asset_path (str): local asset path
            root (str): root folder for asset path. If not specified defaults to MODELS_DIR/assets

        Returns:
            str: Full (absolute) xml path
        """
    if asset_path.startswith("/"):
        full_path = asset_path
    else:
        full_path = os.path.join(root, asset_path)
    return full_path

