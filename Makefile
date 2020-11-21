.PHONY: env env-update env-remove install format lint test requirements

help:
	@echo "Available commands:"
	@echo "env              create the conda environment 'gnn-env'."
	@echo "env-update       update 'gnn-env'."
	@echo "env-remove       remove 'gnn-env'."
	@echo "format           format code."
	@echo "lint             run linters."

env:
	conda env create --file environment.yml

env-update:
	conda env update --name gnn-env --file environment.yml

env-remove:
	conda remove --name gnn-env --all

format:
	yapf -i --recursive gnn
	isort --atomic gnn
	docformatter -i -r gnn

lint:
	yapf --diff --recursive gnn
	pylint -v gnn
	mypy gnn
