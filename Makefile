preprocess: dlmi/preprocess/pipeline.py
	python dlmi/preprocess/pipeline.py

train: dlmi/train/pipeline.py
	python dlmi/train/pipeline.py

.PHONY: clean
clean:
	rm -rf data/train/processed/
	rm -rf data/train/patch/
	rm -rf data/test/processed/
	rm -rf data/test/patch/

update_dependencies:
	poetry export --output=requirements/requirements.txt --without-hashes --with=dev

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  help                 Show this help message"
	@echo "  clean                Clean the processed data directory"
	@echo "  update_dependencies  Export poetry dependencies to requirements files"
	@echo "  preprocess           Run the preprocessing script"
	@echo "  train                Run the train script"
