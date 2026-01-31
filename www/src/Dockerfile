# Use an official Node.js image as the base, selecting a stable version
FROM node:23-alpine

# Set the working directory inside the container
WORKDIR /app/src

# Install essential development tools
RUN apk add --no-cache bash curl

# Install TypeScript globally
RUN npm install -g typescript
