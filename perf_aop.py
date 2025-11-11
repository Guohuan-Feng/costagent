import time, inspect, types

def aop_inject_timing(target_module, only_prefix=None):
    """统一为模块内所有函数/类方法加上计时装饰器"""
    def timed(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.time() - start
                print(f"[AOP] {func.__module__}.{func.__qualname__} took {duration:.4f}s")
        return wrapper

    for name, obj in vars(target_module).items():
        if inspect.isfunction(obj):
            if only_prefix and not obj.__module__.startswith(only_prefix):
                continue
            setattr(target_module, name, timed(obj))
        elif inspect.isclass(obj):
            for m_name, m in vars(obj).items():
                if isinstance(m, (types.FunctionType, types.MethodType)):
                    setattr(obj, m_name, timed(m))
