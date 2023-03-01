# Contributing to JAX-GalSim

Everyone is welcome to contribute to this project, and contributions can take many different forms, from helping to answer questions on the [discussions page](https://github.com/GalSim-developers/JAX-GalSim/discussions), to contributing to the code-base by making a Pull Request.

In order to foster an open, inclusive, and welcoming community, all contributors agree to adhere to [JAX-GalSim code of conduct](CODE_OF_CONDUCT.md).

## Contributing code using Pull Requests

Code contributions are most welcome. You can in particular look for GitHub issues marked as `contributions welcome` or `good first issue`. But you can also propose adding a new functionality, in which case you may find it beneficial to first open a GitHub issue to discuss the feature you want to implement ahead of opening a Pull Request.

Once you have some code you wish to contribute, you can follow this procedure to prepare a Pull Request:

- Fork the JAX-GalSim repository under your own account, using the **Fork** button on the top right of the [GitHub page](https://github.com/GalSim-developers/JAX-GalSim).

- Clone and pip install your fork of the repository like so:

  ```bash
  git clone --recurse-submodules https://github.com/YOUR_USERNAME/JAX-GalSim
  cd JAX-GalSim
  pip install --user -e .
  ```

  This will install your local fork in editable mode, meaning you can directly modify source files in this folder without having to reinstall the package for them to be taken into account. Note the option `--recurse-submodules` needed to donwload the submodules required to run the tests.

- Open a branch for your developments:

  ```bash
  git checkout -b name-that-describes-my-feature
  ```

- Add your changes to the code using your favorite editor. You may at any moment test that everything is still working by running the test suite. From the root folder of the repository, run:

  ```bash
  pytest
  ```

- Once you are happy with your modifications, commit them, and push your changes to GitHub:

  ```python
  git add file_I_changed.py
  git commit -m "a message that describes your modifications"
  git push -set-upstream origin name-that-describes-my-feature
  ```

- From your GitHub interface, you should now be able to open a Pull Request to the JAX-GalSim repository.

Before submitting your PR, have a look at the procedure documented below.

### Checklist before opening a Pull Request

- Pull Requests should be self-contained and limited in scope, otherwise they become too difficult to review. If your modifications are broad, consider opening several smaller Pull Requests.

- Make sure your fork and branch are up-to-date with the `main` branch of JAX-GalSim. To update your local branch, you may do so from the GitHub interface, or you may use this CLI command:

  ```bash
  # Only needs to be done once:
  git remote add upstream http://www.github.com/GalSim-developers/JAX-GalSim
  # This will update your local branch
  git fetch upstream
  git rebase upstream/main
  ```

- Make sure the unit tests still work:

  ```bash
  pytest
  ```

  Ideally there should be some new unit tests for the new functionality, unless the work is completely covered by existing unit tests.

- Make sure your code conforms to the [Black](https://github.com/psf/black) style:

  ```bash
  black .
  ```

  If you haven't installed it already, you can install Black with `pip install black`.

- Update CHANGELOG.md to mention your change.

- Make sure any new files have BSD license at the top.

- If your changes contain multiple commits, we encourage you to squash them into a single (or very few) commit, before opening the PR. To do so, you can using this command:

```bash
git rebase -i
```

### Opening the Pull Request

- On the GitHub site, go to "Code". Then click the green "Compare and Review" button. Your branch is probably in the "Example Comparisons" list, so click on it. If not, select it for the "compare" branch.

- Make sure you are comparing your new branch to the upstream `main`. Press Create Pull Request button.

- Give a brief title. (We usually leave the branch number as the start of the title.)

- Explain the major changes you are asking to be code reviewed. Often it is useful to open a second tab in your browser where you can look through the diff yourself to remind yourself of all the changes you have made.

### After submitting the pull request

- Check to make sure that the PR can be merged cleanly. If it can, GitHub will report that "This branch has no conflicts with the base branch." If it doesn't, then you need to merge from master into your branch and resolve any conflicts.

- Wait a few minutes for the continuous integration tests to be run. Then make sure that the tests reports no errors. If not, click through to the details and try to figure out what is causing the error and fix it.

### After code review

- Once at least 1 and preferably 2 people have reviewed the code, and you have responded to all of their comments, we generally solicit for "any other comments" and give people a few more days before merging.

- Click the "Merge pull request" button at the bottom of the PR page.

- Click the "Delete branch" button.

## Guidelines for Code Style and Documentation

### Code Style

In this project we follow the [Black](https://github.com/psf/black) code formatting guidelines` (Any color you like...) This means that all code should be automatically formatted using Black and CI will fail if that's not the case.

You can install Black locally like so:

```bash
pip install black
```

And run it manually from the root directory of your local clone with `black .`

We highly recommend installing `pre-commit`, which will take care of running Black for you before any commit you make:

```bash
pip install pre-commit
pre-commit install
```

And that's all you need to do from now on.

### Documentation style

JAX-GalSim follows the NumPy/SciPy format: <https://numpydoc.readthedocs.io/en/latest/format.html>

However, most JAX-GalSim function will directly inherit the documentation from the reference GalSim project. We recommend avoid copy/pasting documentation, and instead using the `_wraps` utility to automatically reuse GalSim documentation:

```python
import galsim as _galsim
from jax._src.numpy.util import _wraps
from jax.tree_util import register_pytree_node_class

@_wraps(_galsim.Add,
        lax_description="Does not support `ChromaticObject` at this point.")
def Add(*args, **kwargs):
    return Sum(*args, **kwargs)
```

Note that this tool has the option of providing a `lax_description` which will be added to the reference documentation, and which can be use to document any differences with GalSim, typically a restricted set of features being available.
