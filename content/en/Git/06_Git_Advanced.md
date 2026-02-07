# Advanced Git Commands

## 1. git stash - Temporarily Save Work

Temporarily save work in progress and restore it later.

### Use Case

```
Working on branch A...
↓
Urgently need to switch to branch B
↓
Current work is incomplete for a commit
↓
Save temporarily with git stash!
```

### Basic Usage

```bash
# Temporarily save current changes
git stash

# Save with message
git stash save "Working on login feature"

# Or (newer method)
git stash push -m "Working on login feature"
```

### List Stashes

```bash
git stash list

# Output example:
# stash@{0}: WIP on main: abc1234 Recent commit message
# stash@{1}: On feature: def5678 Other work
```

### Restore Stash

```bash
# Restore most recent stash (keep stash)
git stash apply

# Restore most recent stash + delete
git stash pop

# Restore specific stash
git stash apply stash@{1}
git stash pop stash@{1}
```

### Delete Stash

```bash
# Delete specific stash
git stash drop stash@{0}

# Delete all stashes
git stash clear
```

### View Stash Contents

```bash
# View stash changes
git stash show

# Detailed diff
git stash show -p

# Specific stash details
git stash show -p stash@{1}
```

### Practice Example

```bash
# 1. Modify file
echo "Work in progress..." >> README.md

# 2. Save with stash
git stash push -m "Working on README"

# 3. Switch to other branch
git switch other-branch

# 4. Return after urgent work
git switch main

# 5. Restore stash
git stash pop
```

---

## 2. git rebase - Clean Up Commit History

Reorganize commit history cleanly.

### Merge vs Rebase

```
# Merge (creates merge commit)
      A---B---C  feature
     /         \
D---E---F---G---M  main  (M = merge commit)

# Rebase (linear history)
              A'--B'--C'  feature
             /
D---E---F---G  main
```

### Basic Rebase

```bash
# Rebase feature branch onto main
git switch feature
git rebase main

# Or in one line
git rebase main feature
```

### Rebase Flow

```bash
# 1. Work on feature branch
git switch -c feature
echo "feature" > feature.txt
git add . && git commit -m "feat: add feature"

# 2. New commit appears on main (someone else pushes)
git switch main
echo "main update" > main.txt
git add . && git commit -m "update main"

# 3. Rebase feature onto main
git switch feature
git rebase main

# 4. Now feature is on top of main's latest commit
git log --oneline --graph --all
```

### Interactive Rebase

You can modify, combine, delete, or reorder commits.

```bash
# Modify last 3 commits
git rebase -i HEAD~3
```

In editor:
```
pick abc1234 First commit
pick def5678 Second commit
pick ghi9012 Third commit

# Commands:
# p, pick = use commit
# r, reword = modify commit message
# e, edit = edit commit
# s, squash = combine with previous commit
# f, fixup = combine (discard message)
# d, drop = delete commit
```

### Squashing Commits

```bash
git rebase -i HEAD~3

# In editor:
pick abc1234 Implement feature
squash def5678 Fix bug
squash ghi9012 Refactor

# Saves and combines 3 commits into 1
```

### Resolving Rebase Conflicts

```bash
# When conflict occurs
git status  # Check conflicting files

# After resolving conflict
git add .
git rebase --continue

# Cancel rebase
git rebase --abort
```

### Warning

```bash
# ⚠️ Don't rebase commits that have been pushed!
# Changing shared history causes conflicts

# Only rebase commits that are local
# Use when cleaning up history before pushing
```

---

## 3. git cherry-pick - Pick Specific Commits

Bring specific commits from another branch to current branch.

### Use Case

```
Urgent bug fix needed on main
↓
Fix commit already exists on feature branch
↓
Get just that commit without merging everything
↓
git cherry-pick!
```

### Basic Usage

```bash
# Pick specific commit
git cherry-pick <commit-hash>

# Example
git cherry-pick abc1234

# Pick multiple commits
git cherry-pick abc1234 def5678

# Pick range (A not included, B included)
git cherry-pick A..B

# Include A too
git cherry-pick A^..B
```

### Options

```bash
# Get changes without committing
git cherry-pick --no-commit abc1234
git cherry-pick -n abc1234

# Continue after resolving conflict
git cherry-pick --continue

# Cancel cherry-pick
git cherry-pick --abort
```

### Practice Example

```bash
# 1. Fix bug on feature branch
git switch feature
echo "bug fix" > bugfix.txt
git add . && git commit -m "fix: critical bug fix"

# 2. Check commit hash
git log --oneline -1
# Output: abc1234 fix: critical bug fix

# 3. Switch to main and cherry-pick
git switch main
git cherry-pick abc1234

# 4. Bug fix applied to main
git log --oneline -1
```

---

## 4. git reset vs git revert

### git reset - Undo Commits (Delete History)

```bash
# soft: Undo commit only (keep changes staged)
git reset --soft HEAD~1

# mixed (default): Undo commit + staging (keep changes unstaged)
git reset HEAD~1
git reset --mixed HEAD~1

# hard: Delete everything (⚠️ Changes deleted too!)
git reset --hard HEAD~1
```

### Reset Visualization

```
Before: A---B---C---D (HEAD)

git reset --soft HEAD~2
After:  A---B (HEAD)
        C, D changes are staged

git reset --mixed HEAD~2
After:  A---B (HEAD)
        C, D changes are unstaged

git reset --hard HEAD~2
After:  A---B (HEAD)
        C, D changes are deleted!
```

### git revert - Undo Commits (Keep History)

Creates a new commit that undoes changes. Use for undoing pushed commits.

```bash
# Revert specific commit
git revert <commit-hash>

# Revert recent commit
git revert HEAD

# Revert without committing
git revert --no-commit HEAD
```

### Revert Visualization

```
Before: A---B---C---D (HEAD)

git revert C
After:  A---B---C---D---C' (HEAD)
        C' = commit that undoes C
```

### Reset vs Revert Selection Criteria

| Situation | Use |
|------|------|
| Local commits not yet pushed | `reset` |
| Shared commits already pushed | `revert` |
| Want clean history | `reset` |
| Want record of undo | `revert` |

---

## 5. git reflog - Recover History

Shows all HEAD movement history. Can recover accidentally deleted commits.

### Basic Usage

```bash
# Check reflog
git reflog

# Output example:
# abc1234 HEAD@{0}: reset: moving to HEAD~1
# def5678 HEAD@{1}: commit: add new feature
# ghi9012 HEAD@{2}: checkout: moving from feature to main
```

### Recover Deleted Commits

```bash
# 1. Accidentally reset --hard
git reset --hard HEAD~3  # Oops! Mistake!

# 2. Check previous state with reflog
git reflog
# def5678 HEAD@{1}: commit: important work

# 3. Recover to that point
git reset --hard def5678

# Or recover to new branch
git branch recovery def5678
```

### Recover Deleted Branch

```bash
# 1. Delete branch
git branch -D important-feature  # Oops!

# 2. Find in reflog
git reflog | grep important-feature

# 3. Recover
git branch important-feature abc1234
```

---

## 6. Other Useful Commands

### git blame - Check Line Authors

```bash
# Check author of each line in file
git blame filename.js

# Specific line range only
git blame -L 10,20 filename.js
```

### git bisect - Find Bug-Introducing Commit

```bash
# Find bug commit with binary search
git bisect start
git bisect bad          # Current is buggy
git bisect good abc1234 # This commit was good

# Git moves to middle commit
# After testing:
git bisect good  # If good
git bisect bad   # If buggy

# Repeat to find bug-introducing commit
git bisect reset  # Exit
```

### git clean - Delete Untracked Files

```bash
# Preview files to be deleted
git clean -n

# Delete untracked files
git clean -f

# Include directories
git clean -fd

# Include .gitignore files
git clean -fdx
```

---

## Command Summary

| Command | Description |
|--------|------|
| `git stash` | Temporarily save work |
| `git stash pop` | Restore saved work |
| `git rebase main` | Rebase onto main |
| `git rebase -i HEAD~n` | Interactive rebase |
| `git cherry-pick <hash>` | Pick specific commit |
| `git reset --soft` | Undo commit only |
| `git reset --hard` | Delete everything |
| `git revert <hash>` | Create undo commit |
| `git reflog` | HEAD movement history |
| `git blame` | Line-by-line authors |
| `git bisect` | Find bug commit |

---

## Next Steps

Let's learn CI/CD automation in [07_GitHub_Actions.md](./07_GitHub_Actions.md)!
