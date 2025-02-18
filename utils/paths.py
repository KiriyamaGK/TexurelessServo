import os

current_file_path=os.path.abspath(__file__)
ROOT=os.path.dirname(current_file_path)
PROJECT_ROOT_DIR = os.path.dirname(ROOT)
print(PROJECT_ROOT_DIR)
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

def return_disc_route(pth:str):
    if pth[0]=='/':
        return pth
    elif pth.startswith("One Touch"):
        user_home = os.path.expanduser('~')  # 获取用户主目录
        return os.path.join('/media', os.path.basename(user_home), pth)
    else:
        raise ValueError("路径必须以 '/' 或 'One Touch' 开头")