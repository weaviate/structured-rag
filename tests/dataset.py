import json

def count_objects_in_json_file(filename):
  """Loads JSON data from a file and returns the number of objects in the list."""
  with open(filename, "r") as f:
      data = json.load(f)
  
  if isinstance(data, list):  # Check if data is a list of objects
      return len(data)
  else:
      raise ValueError("The JSON file does not contain a list of objects.")

count = count_objects_in_json_file("wiki-abstract-titles.json")
print(count)