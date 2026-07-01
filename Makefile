setup: apptainer-build-client docker-build-server 

apptainer-build-client: apptainer/gpu.sif

apptainer-build-chimera-client: apptainer/chimera-generator.sif apptainer/chimera-blis.sif apptainer/chimera-openblas.sif

docker-build-server:
	docker build -t diffmath-server -f docker/Dockerfile.SERVER .

apptainer/chimera-blis.sif: apptainer/chimera-blis.def apptainer/install_chimera_backend.sh
	apptainer build --force $@ $<

apptainer/chimera-openblas.sif: apptainer/chimera-openblas.def apptainer/install_chimera_backend.sh
	apptainer build --force $@ $<

apptainer/%.sif: apptainer/%.def
	apptainer build --force $@ $<

.PHONY: docker-build-server apptainer-build-client apptainer-build-chimera-client
