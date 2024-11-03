# setup.ps1

# Create virtual environment
Write-Host "Creating virtual environment..."
python -m venv .venv

# Activate virtual environment
Write-Host "Activating virtual environment..."
.\.venv\Scripts\Activate.ps1

# Install requirements
Write-Host "Installing requirements..."
pip install -r requirements.txt

# Run setup script
Write-Host "Running setup script..."
python setup.py

Write-Host "Setup completed successfully! You can now run the main script with 'python main.py'" -ForegroundColor Green
Read-Host -Prompt "Press Enter to exit"