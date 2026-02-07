# 09. Advanced Git Techniques

## Learning Objectives
- Automate with Git Hooks
- Manage external dependencies with Submodules
- Work on multiple branches simultaneously with Worktrees
- Understand Git internals and low-level commands

## Table of Contents
1. [Git Hooks](#1-git-hooks)
2. [Git Submodules](#2-git-submodules)
3. [Git Worktrees](#3-git-worktrees)
4. [Advanced Commands](#4-advanced-commands)
5. [Git Internals](#5-git-internals)
6. [Troubleshooting](#6-troubleshooting)
7. [Practice Exercises](#7-practice-exercises)

---

## 1. Git Hooks

### 1.1 Git Hooks Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Git Hooks Types                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Client Hooks (Local):                                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Commit Workflow:                                    │   │
│  │  • pre-commit    : Before commit (lint, test)       │   │
│  │  • prepare-commit-msg : Prepare commit message      │   │
│  │  • commit-msg    : Validate commit message          │   │
│  │  • post-commit   : After commit                     │   │
│  │                                                      │   │
│  │  Email Workflow:                                     │   │
│  │  • applypatch-msg                                   │   │
│  │  • pre-applypatch                                   │   │
│  │  • post-applypatch                                  │   │
│  │                                                      │   │
│  │  Other:                                              │   │
│  │  • pre-rebase    : Before rebase                    │   │
│  │  • post-checkout : After checkout                   │   │
│  │  • post-merge    : After merge                      │   │
│  │  • pre-push      : Before push                      │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Server Hooks (Remote):                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  • pre-receive   : Before receiving push            │   │
│  │  • update        : Before each branch update        │   │
│  │  • post-receive  : After receiving push             │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Basic Hook Setup

```bash
# Hook location
ls .git/hooks/
# pre-commit.sample, commit-msg.sample, ...

# Activate hook (remove .sample from sample)
cp .git/hooks/pre-commit.sample .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit

# Or create directly
touch .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

### 1.3 pre-commit Hook Example

```bash
#!/bin/bash
# .git/hooks/pre-commit

echo "Running pre-commit checks..."

# 1. Lint check
echo "Running ESLint..."
npm run lint
if [ $? -ne 0 ]; then
    echo "❌ ESLint failed. Please fix the errors."
    exit 1
fi

# 2. Type check
echo "Running TypeScript check..."
npm run type-check
if [ $? -ne 0 ]; then
    echo "❌ TypeScript check failed."
    exit 1
fi

# 3. Unit tests
echo "Running tests..."
npm test -- --watchAll=false
if [ $? -ne 0 ]; then
    echo "❌ Tests failed."
    exit 1
fi

# 4. Check for secrets
echo "Checking for secrets..."
if git diff --cached --name-only | xargs grep -l -E "(password|secret|api_key)\s*=" 2>/dev/null; then
    echo "❌ Potential secrets detected!"
    exit 1
fi

# 5. Check file sizes
echo "Checking file sizes..."
MAX_SIZE=5242880  # 5MB
for file in $(git diff --cached --name-only); do
    if [ -f "$file" ]; then
        size=$(wc -c < "$file")
        if [ $size -gt $MAX_SIZE ]; then
            echo "❌ File $file is too large ($size bytes)"
            exit 1
        fi
    fi
done

echo "✅ All pre-commit checks passed!"
exit 0
```

### 1.4 commit-msg Hook Example

```bash
#!/bin/bash
# .git/hooks/commit-msg

COMMIT_MSG_FILE=$1
COMMIT_MSG=$(cat "$COMMIT_MSG_FILE")

# Check Conventional Commits format
# type(scope): description
PATTERN="^(feat|fix|docs|style|refactor|test|chore|perf|ci|build|revert)(\(.+\))?: .{1,100}$"

if ! echo "$COMMIT_MSG" | head -1 | grep -qE "$PATTERN"; then
    echo "❌ Invalid commit message format!"
    echo ""
    echo "Commit message must follow Conventional Commits:"
    echo "  <type>(<scope>): <description>"
    echo ""
    echo "Types: feat, fix, docs, style, refactor, test, chore, perf, ci, build, revert"
    echo ""
    echo "Examples:"
    echo "  feat(auth): add login functionality"
    echo "  fix(api): resolve null pointer exception"
    echo "  docs: update README"
    echo ""
    exit 1
fi

# Check message length
FIRST_LINE=$(echo "$COMMIT_MSG" | head -1)
if [ ${#FIRST_LINE} -gt 72 ]; then
    echo "❌ First line must be 72 characters or less"
    exit 1
fi

echo "✅ Commit message is valid!"
exit 0
```

### 1.5 pre-push Hook Example

```bash
#!/bin/bash
# .git/hooks/pre-push

REMOTE=$1
URL=$2

# Prevent direct push to main/master
PROTECTED_BRANCHES="main master"
CURRENT_BRANCH=$(git symbolic-ref HEAD | sed 's!refs/heads/!!')

for branch in $PROTECTED_BRANCHES; do
    if [ "$CURRENT_BRANCH" = "$branch" ]; then
        echo "❌ Direct push to $branch is not allowed!"
        echo "Please create a pull request instead."
        exit 1
    fi
done

# Run full test suite
echo "Running full test suite before push..."
npm run test:ci
if [ $? -ne 0 ]; then
    echo "❌ Tests failed. Push aborted."
    exit 1
fi

# Verify build
echo "Verifying build..."
npm run build
if [ $? -ne 0 ]; then
    echo "❌ Build failed. Push aborted."
    exit 1
fi

echo "✅ All pre-push checks passed!"
exit 0
```

### 1.6 Managing Hooks with Husky

```bash
# Install Husky
npm install husky -D
npx husky init

# Add prepare script to package.json
# "prepare": "husky"

# Add pre-commit hook
echo "npm run lint && npm test" > .husky/pre-commit

# Add commit-msg hook
npm install @commitlint/cli @commitlint/config-conventional -D
echo "npx --no -- commitlint --edit \$1" > .husky/commit-msg

# commitlint.config.js
# module.exports = { extends: ['@commitlint/config-conventional'] };
```

```javascript
// lint-staged.config.js
module.exports = {
  '*.{js,jsx,ts,tsx}': [
    'eslint --fix',
    'prettier --write',
    'jest --findRelatedTests --passWithNoTests'
  ],
  '*.{json,md,yml,yaml}': [
    'prettier --write'
  ],
  '*.css': [
    'stylelint --fix',
    'prettier --write'
  ]
};
```

---

## 2. Git Submodules

### 2.1 Submodules Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Git Submodules                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Main Repository                                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  my-project/                                         │   │
│  │  ├── src/                                           │   │
│  │  ├── tests/                                         │   │
│  │  ├── .gitmodules      ← Submodule config           │   │
│  │  └── libs/                                          │   │
│  │      ├── shared-ui/   ← Submodule (external repo)  │   │
│  │      └── common-utils/← Submodule (external repo)  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Features:                                                  │
│  • Include external repos as subdirectories                 │
│  • Fixed to specific commits                                │
│  • Independent version control                              │
│  • Useful for shared libraries and dependencies            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Basic Submodule Commands

```bash
# Add submodule
git submodule add https://github.com/example/shared-ui.git libs/shared-ui

# .gitmodules file created
# [submodule "libs/shared-ui"]
#     path = libs/shared-ui
#     url = https://github.com/example/shared-ui.git

# Track specific branch
git submodule add -b develop https://github.com/example/lib.git libs/lib

# Clone repository with submodules
git clone --recursive https://github.com/example/main-project.git

# Or initialize after cloning
git clone https://github.com/example/main-project.git
git submodule init
git submodule update

# Or all at once
git submodule update --init --recursive
```

### 2.3 Updating Submodules

```bash
# Update submodule (to configured commit)
git submodule update

# Update submodule to latest
git submodule update --remote

# Update specific submodule only
git submodule update --remote libs/shared-ui

# Execute command in all submodules
git submodule foreach 'git checkout main && git pull'

# Check submodule status
git submodule status
# -abc1234 libs/shared-ui (v1.0.0)    ← - means not initialized
# +def5678 libs/common-utils (heads/main)  ← + means different commit

# Commit changes
cd libs/shared-ui
git checkout main
git pull
cd ../..
git add libs/shared-ui
git commit -m "Update shared-ui submodule"
```

### 2.4 Removing Submodules

```bash
# 1. Remove from .gitmodules
git config -f .gitmodules --remove-section submodule.libs/shared-ui

# 2. Remove from .git/config
git config --remove-section submodule.libs/shared-ui

# 3. Remove from staging
git rm --cached libs/shared-ui

# 4. Remove from .git/modules
rm -rf .git/modules/libs/shared-ui

# 5. Remove from working directory
rm -rf libs/shared-ui

# 6. Commit
git commit -m "Remove shared-ui submodule"
```

### 2.5 Submodule Warnings

```bash
# ⚠️ Check branch in submodule
cd libs/shared-ui
git branch
# * (HEAD detached at abc1234)  ← Detached HEAD!

# Checkout to branch to work in submodule
git checkout main
# Now can make changes

# ⚠️ Auto-update submodules on pull
git pull --recurse-submodules

# Or configure
git config --global submodule.recurse true

# ⚠️ Need to commit in main repo after submodule changes
git status
# modified:   libs/shared-ui (new commits)
git add libs/shared-ui
git commit -m "Update shared-ui to latest"
```

---

## 3. Git Worktrees

### 3.1 Worktrees Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Git Worktrees                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  One repository, multiple working directories               │
│                                                             │
│  ~/.git/my-project/     ← Main repository                  │
│  ├── .git/                                                  │
│  ├── src/                                                   │
│  └── (current branch: main)                                 │
│                                                             │
│  ~/worktrees/feature-a/ ← Worktree 1                       │
│  ├── .git (file, references main .git)                     │
│  ├── src/                                                   │
│  └── (current branch: feature/a)                            │
│                                                             │
│  ~/worktrees/hotfix/    ← Worktree 2                       │
│  ├── .git (file, references main .git)                     │
│  ├── src/                                                   │
│  └── (current branch: hotfix/urgent)                        │
│                                                             │
│  Advantages:                                                │
│  • Switch branches without stash                            │
│  • Work on multiple branches simultaneously                 │
│  • Work on other tasks during long builds                   │
│  • Parallel builds of multiple branches in CI               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Worktree Commands

```bash
# List worktrees
git worktree list
# /home/user/my-project        abc1234 [main]

# Add new worktree (existing branch)
git worktree add ../feature-a feature/a
# Preparing worktree (checking out 'feature/a')

# Add new worktree (create new branch)
git worktree add -b feature/b ../feature-b main

# Add to specific path
git worktree add ~/worktrees/hotfix hotfix/urgent

# List worktrees
git worktree list
# /home/user/my-project        abc1234 [main]
# /home/user/feature-a         def5678 [feature/a]
# /home/user/worktrees/hotfix  ghi9012 [hotfix/urgent]

# Work in worktree
cd ../feature-a
# Perform normal Git operations
git add .
git commit -m "Work on feature A"
git push

# Remove worktree
git worktree remove ../feature-a

# Or delete directory then clean up
rm -rf ../feature-a
git worktree prune  # Clean up invalid worktrees

# Lock/unlock (prevent accidental deletion)
git worktree lock ../feature-a
git worktree unlock ../feature-a
```

### 3.3 Worktree Use Cases

```bash
# Case 1: Urgent bug fix
# Currently working on feature, urgent bug occurs
git worktree add ../hotfix main
cd ../hotfix
git checkout -b hotfix/critical-bug
# Fix bug
git add . && git commit -m "Fix critical bug"
git push -u origin hotfix/critical-bug
# Create PR, merge
cd ../my-project
git worktree remove ../hotfix

# Case 2: Code review
# Check PR code locally
git fetch origin
git worktree add ../pr-123 origin/feature/new-feature
cd ../pr-123
npm install && npm test
# After review, remove
git worktree remove ../pr-123

# Case 3: Parallel builds (CI)
git worktree add ../build-debug main
git worktree add ../build-release main
cd ../build-debug && npm run build:debug &
cd ../build-release && npm run build:release &
wait

# Case 4: Version comparison
git worktree add ../v1.0 v1.0.0
git worktree add ../v2.0 v2.0.0
diff -r ../v1.0/src ../v2.0/src
```

---

## 4. Advanced Commands

### 4.1 Git Bisect (Binary Search)

```bash
# Find commit that introduced bug
git bisect start

# Current state (has bug)
git bisect bad

# Commit that was good
git bisect good abc1234

# Git checks out middle commit
# Test, then mark result
git bisect good  # or git bisect bad

# Repeat...
# Result:
# abc1234 is the first bad commit

# Exit
git bisect reset

# Automated bisect
git bisect start HEAD abc1234
git bisect run npm test
# Automatically determines good/bad and finds
```

### 4.2 Git Reflog

```bash
# All HEAD movement history
git reflog
# abc1234 HEAD@{0}: commit: Add feature
# def5678 HEAD@{1}: checkout: moving from main to feature
# ghi9012 HEAD@{2}: reset: moving to HEAD~1
# ...

# Reflog for specific branch
git reflog show main

# Recover deleted commit
git reflog
# abc1234 HEAD@{5}: commit: Important work  ← Recover this
git checkout abc1234
git checkout -b recovered-branch

# Undo incorrect reset
git reset --hard HEAD@{2}

# Reflog expiration period (default 90 days)
git config gc.reflogExpire 180.days
```

### 4.3 Advanced Git Stash

```bash
# Basic stash
git stash
git stash push -m "Work in progress on feature X"

# Stash specific files only
git stash push -m "Partial work" -- src/file1.js src/file2.js

# Include untracked files
git stash push -u -m "Include untracked"

# Include all files (including ignored)
git stash push -a -m "Include all"

# List stashes
git stash list
# stash@{0}: On feature: Work in progress
# stash@{1}: On main: Bug fix attempt

# Apply specific stash (don't delete)
git stash apply stash@{1}

# Apply and delete specific stash
git stash pop stash@{1}

# View stash contents
git stash show -p stash@{0}

# Convert stash to branch
git stash branch new-feature stash@{0}

# Delete stash
git stash drop stash@{0}
git stash clear  # Delete all
```

### 4.4 Advanced Git Cherry-pick

```bash
# Basic cherry-pick
git cherry-pick abc1234

# Multiple commits
git cherry-pick abc1234 def5678 ghi9012

# Range cherry-pick
git cherry-pick abc1234..ghi9012  # Exclude abc1234
git cherry-pick abc1234^..ghi9012  # Include abc1234

# Apply changes without committing
git cherry-pick -n abc1234

# Continue after resolving conflict
git cherry-pick --continue

# Abort
git cherry-pick --abort

# Cherry-pick merge commit (need -m option)
git cherry-pick -m 1 abc1234
# -m 1: Based on first parent (usually main)
# -m 2: Based on second parent (merged branch)
```

### 4.5 Advanced Git Rebase

```bash
# Interactive rebase
git rebase -i HEAD~5
# pick, reword, edit, squash, fixup, drop

# Rebase from specific commit
git rebase -i abc1234

# Autosquash (automatically handle fixup! prefix)
git commit --fixup abc1234
git rebase -i --autosquash abc1234^

# During rebase conflict
git rebase --continue
git rebase --skip
git rebase --abort

# onto option (move branch)
git rebase --onto main feature-base feature
# Move commits between feature-base and feature onto main

# preserve-merges (keep merge commits) - deprecated
git rebase --rebase-merges main
```

---

## 5. Git Internals

### 5.1 Git Objects

```
┌─────────────────────────────────────────────────────────────┐
│                    Git Object Types                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Blob (file content)                                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  SHA-1: abc123...                                   │   │
│  │  Content: (binary data of file)                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Tree (directory)                                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  SHA-1: def456...                                   │   │
│  │  100644 blob abc123... README.md                    │   │
│  │  100644 blob bcd234... main.js                      │   │
│  │  040000 tree cde345... src                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Commit                                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  SHA-1: ghi789...                                   │   │
│  │  tree def456...                                     │   │
│  │  parent efg567...                                   │   │
│  │  author John <john@example.com> 1234567890 +0900   │   │
│  │  committer John <john@example.com> 1234567890 +0900│   │
│  │                                                      │   │
│  │  Commit message                                      │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Tag (annotated tag)                                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  SHA-1: jkl012...                                   │   │
│  │  object ghi789... (commit)                          │   │
│  │  type commit                                        │   │
│  │  tag v1.0.0                                         │   │
│  │  tagger John <john@example.com> 1234567890 +0900   │   │
│  │                                                      │   │
│  │  Release version 1.0.0                               │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Low-level Commands (Plumbing)

```bash
# Check object type
git cat-file -t abc1234
# commit

# View object content
git cat-file -p abc1234
# tree def456789...
# parent ...
# author ...

# View current commit's tree
git cat-file -p HEAD^{tree}

# View blob content
git cat-file -p abc1234:README.md

# Calculate object hash
echo "Hello" | git hash-object --stdin
# Or from file
git hash-object README.md

# Store object
echo "Hello" | git hash-object -w --stdin

# Create tree
git write-tree

# Create commit
echo "Commit message" | git commit-tree <tree-sha> -p <parent-sha>

# Update reference
git update-ref refs/heads/new-branch abc1234
```

### 5.3 Git Directory Structure

```
.git/
├── HEAD              # Current branch reference
├── config            # Repository config
├── description       # GitWeb description
├── hooks/            # Git hooks
├── info/
│   └── exclude       # Local .gitignore
├── objects/          # All objects stored
│   ├── pack/         # Packed objects
│   ├── info/
│   └── ab/
│       └── c123...   # Object file (first 2 chars are directory)
├── refs/
│   ├── heads/        # Local branches
│   │   └── main
│   ├── remotes/      # Remote branches
│   │   └── origin/
│   │       └── main
│   └── tags/         # Tags
│       └── v1.0.0
├── logs/             # reflog storage
│   ├── HEAD
│   └── refs/
├── index             # Staging area
└── COMMIT_EDITMSG    # Last commit message
```

---

## 6. Troubleshooting

### 6.1 Common Problem Solutions

```bash
# Amend last commit (before push)
git commit --amend -m "New message"
git commit --amend --no-edit  # Keep message

# Modify pushed commit (dangerous!)
git commit --amend
git push --force-with-lease  # Safer force push

# Committed to wrong branch (before push)
git branch correct-branch    # New branch at current commit
git reset --hard HEAD~1      # Rewind current branch
git checkout correct-branch  # Switch to correct branch

# Remove file from commit
git reset HEAD~ -- file.txt
git commit --amend

# Remove sensitive info (from all history)
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch secrets.txt" \
  --prune-empty --tag-name-filter cat -- --all

# Or use BFG Repo-Cleaner (faster)
bfg --delete-files secrets.txt
bfg --replace-text passwords.txt
```

### 6.2 Conflict Resolution

```bash
# Check merge conflicts
git status
git diff --name-only --diff-filter=U

# Conflict markers
# <<<<<<< HEAD
# Current branch content
# =======
# Merging branch content
# >>>>>>> feature

# Choose by file
git checkout --ours file.txt    # Choose current branch
git checkout --theirs file.txt  # Choose merging branch

# Use merge tool
git mergetool

# After resolving conflicts
git add file.txt
git commit

# Abort merge
git merge --abort
```

### 6.3 Large Repository Management

```bash
# Find large files
git rev-list --objects --all | \
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
  sed -n 's/^blob //p' | \
  sort -nk2 | \
  tail -20

# Setup Git LFS
git lfs install
git lfs track "*.psd"
git lfs track "*.zip"
git add .gitattributes
git add large-file.psd
git commit -m "Add large file with LFS"

# Reduce repository size
git gc --aggressive --prune=now
git repack -a -d --depth=250 --window=250

# Shallow clone
git clone --depth 1 https://github.com/repo.git

# Sparse checkout
git sparse-checkout init
git sparse-checkout set src/ tests/
```

---

## 7. Practice Exercises

### Exercise 1: Setup Git Hooks
```bash
# Requirements:
# 1. pre-commit: Check code formatting
# 2. commit-msg: Validate Conventional Commits
# 3. pre-push: Run tests
# 4. Setup with Husky for team sharing

# Write hook scripts:
```

### Exercise 2: Submodule Project
```bash
# Requirements:
# 1. Create main project
# 2. Add shared library as submodule
# 3. Write submodule update script
# 4. Build with submodules in CI

# Write commands and scripts:
```

### Exercise 3: Using Worktrees
```bash
# Requirements:
# 1. Scenario: urgent bug fix during main work
# 2. Parallel work with worktrees
# 3. Clean up after work complete

# Write commands:
```

### Exercise 4: Find Bug with Bisect
```bash
# Requirements:
# 1. Write test script
# 2. Automate with git bisect run
# 3. Find bug commit

# Write commands:
```

---

## Next Steps

- [10_Monorepo_Management](10_Monorepo_Management.md) - Large repository management
- [08_Git_Workflow_Strategies](08_Git_Workflow_Strategies.md) - Review workflows
- [Pro Git Book](https://git-scm.com/book) - Advanced learning

## References

- [Git Hooks Documentation](https://git-scm.com/docs/githooks)
- [Git Submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules)
- [Git Worktree](https://git-scm.com/docs/git-worktree)
- [Git Internals](https://git-scm.com/book/en/v2/Git-Internals-Plumbing-and-Porcelain)

---

[← Previous: Git Workflow Strategies](08_Git_Workflow_Strategies.md) | [Next: Monorepo Management →](10_Monorepo_Management.md) | [Contents](00_Overview.md)
