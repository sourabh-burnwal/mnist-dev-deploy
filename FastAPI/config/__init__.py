import yaml

with open('config/config.yml', 'r') as file:
    project_config = yaml.safe_load(file)
