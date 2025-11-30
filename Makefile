install_deps:
	pip install --upgrade pip
	pip install -r requirements.txt
install_conch:
	pip install conch @ git+https://github.com/Mahmoodlab/CONCH.git@141cc09c7d4ff33d8eda562bd75169b457f71a62
