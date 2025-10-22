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

def determine_index_from_ckpt_name(ckpt_name):
    idx = ""
    for i in range(6,10):  #ckpt name starts with "epoch_"
        if ckpt_name[i].isdigit():
            idx+=ckpt_name[i]
        else:
            break
    if idx != "":
        return int(idx)
    return -1

def determine_ckpt_dirs(cfg,ckpt_base):
    '''
    :param itm: string or list eg. "all","epoch_441_validation_loss_0.032405721955001354.pth",[50,100,150]
    :return: list: list of ckpt dirs
    '''
    assert isinstance(cfg, str) or isinstance(cfg, list)
    if isinstance(cfg, str):
        if cfg =='all':
            ckpts_dirs = [os.path.join(ckpt_base, itm) for itm in os.listdir(ckpt_base) if "epoch" in itm]
        else:
            assert 'epoch_' in cfg
            ckpts_dirs = [os.path.join(ckpt_base, cfg)]
    else:
        all_dirs=[itm for itm in os.listdir(ckpt_base) if "epoch" in itm]
        all_epochs={}
        ckpts_dirs=[]
        for epoch in all_dirs:
            k=determine_index_from_ckpt_name(epoch)
            v=epoch
            all_epochs[str(k)]=v
        for test_epoch in cfg:
            if str(test_epoch) in all_epochs.keys():
                ckpts_dirs.append(os.path.join(ckpt_base, all_epochs[str(test_epoch)]))
            elif test_epoch in all_dirs:
                ckpts_dirs.append(os.path.join(ckpt_base, test_epoch))
    return ckpts_dirs

if __name__ == '__main__':
    a=determine_ckpt_dirs([1,29,300],"/home/kiriyamagk/桌面/AlignAnything/trained_models/trial/2025-03-16_01-41-23")
    b=determine_ckpt_dirs("all","/home/kiriyamagk/桌面/AlignAnything/trained_models/trial/2025-03-16_01-41-23")
    c = determine_ckpt_dirs("epoch_130_best_validation_loss_0.03146909167990088.pth", "/home/kiriyamagk/桌面/AlignAnything/trained_models/trial/2025-03-16_01-41-23")
    print(a)
    print(b)
    print(c)