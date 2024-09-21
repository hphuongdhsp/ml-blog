# FROM fastai/jekyll:2020-12-30

FROM fastai/jekyll:latest
# Set the working directory
WORKDIR /fastpages

# Update RubyGems to the required version
# RUN gem update --system 3.3.22
# Copy the project files to the container
COPY . .

# Ensure the shell scripts are executable
RUN chmod u+x action_entrypoint.sh
RUN chmod u+x word2post.sh
RUN dos2unix /fastpages/*.sh

# Update RubyGems to the required version
# RUN gem update --system

# Set the entrypoint command
CMD [ "/fastpages/action_entrypoint.sh" ]
