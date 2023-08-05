from .global_vars import load_prof_result


def initialize_profiler(file_path: str, tp: int) -> None:
    load_prof_result(file_path, tp)
