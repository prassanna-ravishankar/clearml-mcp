# :rocket: clearml-mcp

lightweight MCP server that interacts with the ClearML API

## Setup Dev Environment

Installation is using [UV](https://docs.astral.sh/uv/) to manage everything.

**Step 1**: Create a virtual environment

```
uv venv
```

**Step 2**: Activate your new environment

```
# on windows
.venv\Scripts\activate

# on mac / linux
source .venv/bin/activate
```

**Step 3**: Install all the cool dependencies

```
uv sync
```

## Github Repo Setup

To add your new project to its Github repository, firstly make sure you have created a project named **clearml-mcp** on Github.
Follow these steps to push your new project.

```
git remote add origin git@github.com:prassanna-ravishankar/clearml-mcp.git
git branch -M main
git push -u origin main
```

## Built-in CLI Commands

We've included a bunch of useful CLI commands for common project tasks using [taskipy](https://github.com/taskipy/taskipy).

```
# run src/clearml_mcp/clearml_mcp.py
task run

# run all tests
task tests



# run test coverage and generate report
task coverage

# typechecking with Ty or Mypy
task type

# ruff linting
task lint

# format with ruff
task format
```

## Docker Setup

A Dockerfile optimized to reduce the image size has been included. To get it up and running follow these steps.

**Step 1**: Build your Docker image.

```
docker build --progress=plain -t "clearml_mcp:Dockerfile" .
```

**Step 2**: Run your new image.

```
docker run --rm clearml_mcp:Dockerfile
```

## PyPI Deployment

1. Register your project and create an API Token on [PyPI](https://pypi.org/).
2. Add the API Token to your projects secrets with the name `PYPI_TOKEN`
3. Create a new release on Github.
4. Create a new tag in the form `*.*.*`.

## Docs Generation + Publishing

Doc generation is setup to scan everything inside `/src`, files with a prefix `_` will be ignored. Basic doc functions for generating, serving, and publishing can be done through these CLI commands:

```
# generate docs & serve
task docs

# serve docs
task serve

# generate static HTML docs (outputs to ./site/)
task html

# publish docs to Github Pages
task publish
```

Note: Your repo must be public or have an upgraded account to deploy docs to Github Pages.

## Dependabot Setup

1. Go to the "Settings -> Advanced Security" tab in your repository.
2. Under the "Dependabot" section enable the options you want to monitor, we recommend the "Dependabot security updates" at the minimum.

Dependabot is configured to do _weekly_ scans of your dependencies, and pull requests will be prefixed with "DBOT". These settings can be adjusted in the `./.github/dependabot.yml` file.

## References

- [Cookiecutter Python Project](https://github.com/wyattferguson/pattern) - A modern cookiecutter template for your next Python project.

## License

MIT license

## Contact

Created by [Prass, The Nomadic Coder](https://github.com/prassanna-ravishankar)
