def is_in_dagger_episode(episode, dagger_config):
    """
    判断当前episode是否应该使用DAgger策略
    
    Args:
        episode: 当前episode编号
        dagger_config: DAgger配置
        
    Returns:
        bool: 是否应该使用DAgger策略
    """
    if not dagger_config.get("utilized", False):
        return False
        
    dagger_episodes_config = dagger_config.get("dagger_episodes", {})
    use_type_config = dagger_episodes_config.get("use_type", [])
    
    # 遍历配置的区间，检查当前episode是否在DAgger区间内
    for start_ep, end_ep in use_type_config:
        if start_ep <= episode <= end_ep:
            return True
    
    return False

def _normalize_proportion(value):
    """将比例字段标准化为 None 或 [0,1] 的浮点值。"""
    if value is None:
        return None
    if isinstance(value, str) and value.lower() == "none":
        return None
    try:
        p = float(value)
    except Exception:
        return None
    p = max(0.0, min(1.0, p))
    return p


def should_train_policy(episode, dagger_config):
    """
    判断当前episode是否应该训练策略
    
    Args:
        episode: 当前episode编号
        dagger_config: DAgger配置
        
    Returns:
        tuple: (should_train, epochs, dagger_proportion)
            - should_train (bool): 是否应该训练策略
            - epochs (int): 训练轮数
            - dagger_proportion (float|None): 训练数据中来自DAgger数据集的比例，None表示不指定
    """
    if not dagger_config.get("utilized", False):
        return False, 0, None
        
    train_config = dagger_config.get("train", {})
    use_type_config = train_config.get("use_type", [])
    
    # 遍历配置，检查当前episode是否在训练列表中
    for entry in use_type_config:
        if not isinstance(entry, (list, tuple)):
            continue
        if len(entry) >= 2:
            train_episode, epochs = entry[0], entry[1]
            proportion = _normalize_proportion(entry[2]) if len(entry) >= 3 else None
            if episode == train_episode:
                return True, epochs, proportion
    
    return False, 0, None

def print_dagger_status(episode, is_dagger, should_train, train_epochs=0):
    """打印DAgger状态信息"""
    train_info = f"Train={should_train}" if not should_train else f"Train={should_train}({train_epochs} epochs)"
    print(f"[DAgger] Episode {episode}: DAgger={is_dagger}, {train_info}") 