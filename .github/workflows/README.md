# GitHub Action Testing

GitHub actions run on virtual Runners within the GitHub ecosystem.  However you can replicate this setup using the [act](https://github.com/nektos/act) utility.
`act` will use Docker images to replicate the GitHub Runner setup and execute locally.  This will allow you to test GitHub Actions locally for development or QA purposes before pushing to main branch.

## Installing act

You can [install act](https://nektosact.com/installation/index.html) into ./bin folder via:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
```

## Running act

From within the `dehullicinator` folder you can run GitHub Actions locally via act using:

```bash
../bin/act -s GITHUB_TOKEN=[Personal Access Token] workflow_dispatch
```

**Note:** GITHUB_TOKEN is generated via your Profile / Settings / Developer Settings / Personal Access Tokens.

![personal access token screen](/docs/images/pat.png)

The above command will execute the `workflow_dispatch` trigger of the workflow (i.e. manaully trigger the workflow) which should run all of the jobs defined in the workflow.
