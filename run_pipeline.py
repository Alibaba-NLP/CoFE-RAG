import json
import sys
import os
import logging
from utils import import_class

# set logging level
logging.basicConfig(level=logging.INFO)
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

if len(sys.argv) != 2:
    print("Usage: python run_pipeline.py <config_path>")
    sys.exit(1)

config_path = sys.argv[1]
config = json.load(open(config_path))
prev_operators = ''

for dataset in config["datasets"]:
    inp_folder = os.path.join('datasets', dataset, config["start_point"])
    save_folder_prefix_list = config["save_folder_prefix_list"]
    current_prefix = []
    if config["start_prefix"]:
        current_prefix.append(config["start_prefix"])
    for operation in config["pipeline"]:
        operator_name = operation["operator"]

        config_name = operation["config_name"]
        operator_config_path = os.path.join(operator_name, "config", f"{config_name}.json")
        operator_config = json.load(open(operator_config_path))
        class_name = operator_config["class_name"]
        class_file = operator_config["class_file"]
        module_path = f"{operator_name}.{class_file}"
        current_prefix += [operation[key] for key in save_folder_prefix_list if key in operation]
        ouput_folder = os.path.join('datasets', dataset, '_'.join(current_prefix))
        # Dynamically importing the operator class
        OpClass = import_class(module_path, class_name)
        
        # Instantiate the operator with its configuration
        operator = OpClass(operator_config, inp_folder)

        if hasattr(operator, "process"):
            if not os.path.exists(ouput_folder):
                os.makedirs(ouput_folder)
            ouput_folder = operator.process(inp_folder, ouput_folder)
            operation['input_folder'] = inp_folder
            operation['operator_config'] = operator_config
            inp_folder = ouput_folder
            with open(os.path.join(ouput_folder, f"config.json"), "w") as f:
                json.dump(operation, f, indent=2, ensure_ascii=False)
        else:
            logging.info(f"The operator {class_name} does not support processing")
            exit(1)
        logging.info(f"Finished processing {operator_name}")
