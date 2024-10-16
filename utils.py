import os

from llama_index.core import Settings
from llms.SetLLM import SetLLM

def set_llm(base_llm='gpt-3.5-turbo-0301'):
        Settings.llm = SetLLM(model=base_llm)
        Settings.context_window = Settings.llm.max_tokens + Settings.llm.max_input_tokens
        Settings.num_output = Settings.llm.max_tokens


# Importing module and class dynamically
def import_class(module_path, class_name):
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)

def truncate_filename(output_file, max_length=255, prefix="short_"):
    # 分割目录和文件名
    directory, filename = os.path.split(output_file)
    
    # 保证包括路径的完整文件名不超过限制
    required_length = len(directory) + 1 + len(prefix)
    max_filename_length = max_length - required_length
    
    if len(filename) > max_filename_length:
        # 如果文件名过长，截断并添加前缀
        new_filename = prefix + filename[-max_filename_length:]
        # 重新构建输出文件路径
        output_file = os.path.join(directory, new_filename)
        print(f"rename {filename} to {new_filename} to fit legal file length")
    return output_file