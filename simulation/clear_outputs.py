import os
import re

def clear_timestamped_outputs(directory):
    # Regular expression to match timestamped files
    timestamp_pattern = re.compile(r'.*\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}.*')

    for filename in os.listdir(directory):
        if timestamp_pattern.match(filename):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    print(f"Deleted file: {file_path}")
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)
                    print(f"Deleted directory: {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

if __name__ == "__main__":
    output_directory = "/path/to/your/output/directory"
    clear_timestamped_outputs(output_directory)
    output_directory = os.path.join(os.path.dirname(__file__), 'output')
    clear_timestamped_outputs(output_directory)