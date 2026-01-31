setup: apptainer-build-client docker-build-server 

apptainer-build-client: gpu.sif

docker-build-server:
	docker build -t diffmath-server -f docker/Dockerfile.SERVER .

apptainer/%.sif: apptainer/%.def
	apptainer build $@ $<

.PHONY: docker-build-server
