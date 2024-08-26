import json

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def load_json_from_file(filename):
    try:
        with open(filename, 'r') as json_file:
            data = json.load(json_file)
        return data
    except FileNotFoundError:
        print(f"{Colors.RED}Error: File '{filename}' not found.{Colors.ENDC}")
        return None
    except json.JSONDecodeError:
        print(f"{Colors.RED}Error: Invalid JSON format in '{filename}'.{Colors.ENDC}")
        return None