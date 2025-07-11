# Installation and use

This guide will walk you through setting up and running the project on your local machine.

## 1. Clone the Repository

First, clone the repository to your local machine using Git:

```sh
git clone https://github.com/Pixelguy14/UI_AG_python
cd UI_AG_python
```

Alternatively, you can download the source code as a ZIP file from the [repository page](https://github.com/Pixelguy14/UI_AG_python).

## 2. Create and Activate a Virtual Environment

It's best practice to use a virtual environment to keep project dependencies isolated.

Create the virtual environment:
```sh
python3 -m venv env_AG
```

Now, activate it. The command varies based on your operating system:

### For macOS and Linux:
```sh
source env_AG/bin/activate
```

### For Windows:
In Command Prompt:
```cmd
env_AG\Scripts\activate.bat
```
In PowerShell:
```powershell
env_AG\Scripts\Activate.ps1
```

## 3. Install Dependencies

With the virtual environment active, install the necessary packages:
```sh
pip install -r src/utils/requirements.txt
```

## 4. Run the Project

You can now launch the application:
```sh
python main.py
```