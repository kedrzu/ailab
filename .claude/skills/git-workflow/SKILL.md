---
name: git-workflow
description: How changes move in ailab — commit and push straight to main, no branches, no PRs. The dev pipeline (/dev:fix … /dev:deliver) relies on it; reach for it on any commit or push in this repo.
---

# git-workflow — ailab

Fulfils the `dev` pipeline's `git-workflow` contract.

One-person learning repo (`kedrzu/ailab` on GitHub, private). **Every change goes straight to
`main`**: no feature branches, no pull requests, no tracker. `.claude/git.json` is absent on
purpose, so the `git` plugin's hook lets pushes to `main` through.

## Start

Work directly on `main` — do not create a branch. Bring it up to date first:

```bash
git pull --rebase origin main
```

In a Paseo worktree (checked out on its own branch) work there as is; the Checkpoint push below
sends the commits to `main` anyway.

## Checkpoint

Commit message: short imperative summary in English (`Add tokenizer experiment`), optional body
explaining why. End it with the attribution lines the session asks for.

Commit only a given set of files (nothing else staged):

```bash
git commit -m "<message>" -- <file> [<file> ...]
```

New, untracked files must be added first: `git add -- <file> ...` then the same `git commit -- <files>`.

Push what is already committed (works on `main` and in a worktree branch):

```bash
git pull --rebase origin main
git push origin HEAD:main
```

Confirm the push landed — both lines must print the same hash:

```bash
git fetch origin && git rev-parse HEAD origin/main
```

No submodules or other repositories to publish first.

## Publish

Unsupported — there are no pull requests or tracker issues; a pushed commit on `main` is
already published.

## Stack

Unsupported — changes are not split into chains of dependent PRs; each change is one or more
commits pushed to `main` in order.

## Land

Nothing separate to do: the push to `main` in Checkpoint is the landing. Never force-push `main`.
