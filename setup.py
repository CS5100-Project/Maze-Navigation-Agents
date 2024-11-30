import os


def create_project_structure():
    # Define the project structure
    structure = {
        "experiments": {
            "gym_version": ["checkpoints", "plots", "configs", "logs"],
            "custom_version": ["checkpoints", "plots", "configs", "logs"],
        },
        "gym_version": [],
        "custom_version": [],
        "utils": [],
    }

    # Get the project root directory (where this script is located)
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Create directories
    for dir_name, subdirs in structure.items():
        dir_path = os.path.join(project_root, dir_name)
        os.makedirs(dir_path, exist_ok=True)

        # Create __init__.py in each Python package directory
        if dir_name in ["gym_version", "custom_version", "utils"]:
            init_file = os.path.join(dir_path, "__init__.py")
            if not os.path.exists(init_file):
                open(init_file, "a").close()

        # Create subdirectories if any
        if isinstance(subdirs, list):
            for subdir in subdirs:
                os.makedirs(os.path.join(dir_path, subdir), exist_ok=True)


if __name__ == "__main__":
    create_project_structure()
    print("Project structure created successfully!")
