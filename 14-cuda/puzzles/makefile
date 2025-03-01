.PHONY: run clean

run:
	@if [ -z "$(word 2,$(MAKECMDGOALS))" ]; then \
		echo "Error: Please specify a CUDA file (e.g., make run guard.cu)"; \
		exit 1; \
	fi; \
	CU_FILE="$(word 2,$(MAKECMDGOALS))"; \
	if [ ! -f "$$CU_FILE" ]; then \
		echo "Error: File $$CU_FILE not found"; \
		exit 1; \
	fi; \
	BASENAME=$$(basename "$$CU_FILE" .cu); \
	nvcc "$$CU_FILE" -o "$$BASENAME" && \
	./$$BASENAME && \
	rm -f "$$BASENAME" "$$BASENAME.exe" *.exp *.lib

%:
	@:

clean:
	rm -f *.o *.exp *.lib *.exe