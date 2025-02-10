# Neuro-symbolic Interaction

## Setup Instructions

### 1. Load Required Keys
Before executing the notebook, ensure that you have set the following environment variables:

- `WATSONX_API_KEY`
- `PROJECT_ID`
- `WATSONX_URL`
- `GITHUB_TOKEN_PERSONAL`

These keys are essential for accessing IBM Watson services and cloning the repository.

### 2. Execute the Notebook

Click the button below to open the notebook in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ruslanmv/Neuro-symbolic-interaction/blob/main/Neuro_symbolic_interaction.ipynb)

### 3. Clone the Repository
The notebook includes the following script to clone the repository using your GitHub token:

```python
import os
import shutil
from IPython.display import clear_output
from google.colab import userdata

# Retrieve the GitHub token from Colab secrets
GITHUB_TOKEN = userdata.get('GITHUB_TOKEN_PERSONAL')
repo_url = "https://github.com/ruslanmv/Neuro-symbolic-interaction.git"
repo_name = "Neuro-symbolic-interaction"

# Clone the repository
clone_command = f"git clone https://{GITHUB_TOKEN}@{repo_url.replace('https://', '')}"
if not os.path.exists(repo_name):
    print(f"Cloning {repo_name} repository...")
    !{clone_command}
    clear_output()
    print(f"{repo_name} repository cloned successfully!")
    
    # Move contents to the current directory
    for item in os.listdir(repo_name):
        s = os.path.join(repo_name, item)
        d = os.path.join(".", item)
        try:
            shutil.move(s, d)
        except Exception as e:
            print(f"Error moving {item}: {e}")
    
    shutil.rmtree(repo_name)
    print(f"Contents of {repo_name} moved successfully!")
else:
    print(f"{repo_name} repository already exists. Skipping cloning.")
print("Finished.")
```

### 4. Install Dependencies
After cloning the repository, install the necessary dependencies using:

```python
import os
from IPython.display import clear_output

def install_requirements(requirements_file="requirements.txt"):
    if os.path.exists(requirements_file):
        print(f"Installing requirements from {requirements_file}...")
        try:
             !pip install -r {requirements_file}
             clear_output()
             print(f"Requirements from {requirements_file} installed successfully!")
        except Exception as e:
            print(f"Error installing requirements: {e}")
    else:
        print(f"Requirements file {requirements_file} not found.")

install_requirements()
print("Finished installing requirements (or skipped if not found).")
```

### 5. Launch the Gradio Application
To launch the Gradio interface, the following script is included:

```python
import gradio as gr
import threading
from app import demo

def launch_gradio(demo):
    demo.launch(inline=True, share=True)

thread = threading.Thread(target=launch_gradio, args=(demo,))
thread.start()
```

### Summary
This repository provides a neuro-symbolic interaction model using IBM Watson and Gradio for user interaction. Follow the steps above to set up and run the application successfully.

