import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
config = load_config('VGG/configs/vgg_config.yaml')
print(config['model'])