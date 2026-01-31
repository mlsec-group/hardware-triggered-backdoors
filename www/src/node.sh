#!/bin/bash

docker run --rm -it -v$(pwd)/..:/app node-build /bin/bash -c "$*"
