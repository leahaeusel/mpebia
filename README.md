# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/leahaeusel/mpebia/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                  |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------------------------ | -------: | -------: | ------: | --------: |
| src/mpebia/\_\_init\_\_.py                            |        0 |        0 |    100% |           |
| src/mpebia/electromechanical\_model/\_\_init\_\_.py   |        0 |        0 |    100% |           |
| src/mpebia/electromechanical\_model/cube\_model.py    |       70 |        5 |     93% |92, 163, 209-212 |
| src/mpebia/entropies.py                               |       33 |        3 |     91% |39, 123, 136 |
| src/mpebia/logging.py                                 |       17 |        0 |    100% |           |
| src/mpebia/observations.py                            |       35 |        3 |     91% |     75-78 |
| src/mpebia/output.py                                  |        7 |        2 |     71% |     28-33 |
| src/mpebia/porous\_medium\_model/recycle\_metadata.py |       17 |        2 |     88% |     67-68 |
| src/mpebia/porous\_medium\_model/smart\_driver.py     |       50 |        6 |     88% |100, 113, 147, 155, 172-173 |
| src/mpebia/spacing.py                                 |        4 |        0 |    100% |           |
| src/mpebia/truncated\_gaussian\_prior.py              |       51 |        0 |    100% |           |
| **TOTAL**                                             |  **284** |   **21** | **93%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/leahaeusel/mpebia/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/leahaeusel/mpebia/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/leahaeusel/mpebia/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/leahaeusel/mpebia/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fleahaeusel%2Fmpebia%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/leahaeusel/mpebia/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.