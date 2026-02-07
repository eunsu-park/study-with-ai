# Getting Started with GitHub

## 1. What is GitHub?

GitHub is a web service that hosts Git repositories.

### Key Features of GitHub

- **Remote Repositories**: Back up code in the cloud
- **Collaboration Tools**: Pull Requests, Issues, Projects
- **Social Coding**: Explore and contribute to other developers' code
- **CI/CD**: Automation with GitHub Actions

### Create GitHub Account

1. Visit [github.com](https://github.com)
2. Click "Sign up"
3. Enter email, password, username
4. Complete email verification

---

## 2. SSH Key Setup (Recommended)

Using SSH keys means you don't have to enter your password every time.

### Generate SSH Key

```bash
# Generate SSH key (use your GitHub account email)
ssh-keygen -t ed25519 -C "your_email@example.com"

# Use default settings (press Enter 3 times)
```

### View SSH Key

```bash
# Display public key
cat ~/.ssh/id_ed25519.pub
```

### Register SSH Key on GitHub

1. GitHub → Settings → SSH and GPG keys
2. Click "New SSH key"
3. Paste public key content
4. Click "Add SSH key"

### Test Connection

```bash
ssh -T git@github.com

# Success message:
# Hi username! You've successfully authenticated...
```

---

## 3. Connecting Remote Repository

### Push New Repository to GitHub

```bash
# 1. Create new repository on GitHub (empty repository)

# 2. Add remote repository from local
git remote add origin git@github.com:username/repository.git

# 3. First push
git push -u origin main
```

### Clone Existing GitHub Repository

```bash
# SSH method (recommended)
git clone git@github.com:username/repository.git

# HTTPS method
git clone https://github.com/username/repository.git

# Clone with specific folder name
git clone git@github.com:username/repository.git my-folder
```

---

## 4. Managing Remote Repository

### View Remote Repository

```bash
# List remote repositories
git remote

# Detailed information
git remote -v
```

Output example:
```
origin  git@github.com:username/repo.git (fetch)
origin  git@github.com:username/repo.git (push)
```

### Add/Remove Remote Repository

```bash
# Add
git remote add origin URL

# Remove
git remote remove origin

# Change URL
git remote set-url origin new-URL
```

---

## 5. Push - Local → Remote

Upload local changes to remote repository.

```bash
# Basic push
git push origin branch-name

# Push main branch
git push origin main

# First push with -u option (set upstream)
git push -u origin main

# After upstream is set, simply
git push
```

### Push Flow Diagram

```
Local                              Remote (GitHub)
┌─────────────┐                  ┌─────────────┐
│ Working Dir │                  │             │
│     ↓       │                  │             │
│ Staging     │     git push     │  Remote     │
│     ↓       │ ───────────────▶ │  Repository │
│ Local Repo  │                  │             │
└─────────────┘                  └─────────────┘
```

---

## 6. Pull - Remote → Local

Fetch changes from remote repository to local.

```bash
# Fetch remote changes + merge
git pull origin main

# If upstream is set
git pull
```

### Fetch vs Pull

| Command | Action |
|--------|------|
| `git fetch` | Download remote changes only |
| `git pull` | fetch + merge (download + merge) |

```bash
# Fetch, then check, then merge
git fetch origin
git log origin/main  # Check remote changes
git merge origin/main

# Process at once
git pull origin main
```

---

## 7. Working with Remote Branches

### View Remote Branches

```bash
# All branches (local + remote)
git branch -a

# Remote branches only
git branch -r
```

### Fetch Remote Branch

```bash
# Fetch remote branch to local
git switch -c feature origin/feature

# Or
git checkout -t origin/feature
```

### Delete Remote Branch

```bash
# Delete remote branch
git push origin --delete branch-name
```

---

## 8. Practice Example: Complete Workflow

### Upload New Project to GitHub

```bash
# 1. Create project locally
mkdir my-github-project
cd my-github-project
git init

# 2. Create files and commit
echo "# My GitHub Project" > README.md
echo "node_modules/" > .gitignore
git add .
git commit -m "initial commit"

# 3. Create new repository on GitHub (on web)
# - Click New repository
# - Enter name: my-github-project
# - Create empty repository (uncheck README)

# 4. Connect remote repository and push
git remote add origin git@github.com:username/my-github-project.git
git push -u origin main

# 5. Check on GitHub!
```

### Collaboration Scenario

```bash
# Team member A: Make changes and push
echo "Feature A" >> features.txt
git add .
git commit -m "feat: add Feature A"
git push

# Team member B: Get latest code
git pull

# Team member B: Add own changes
echo "Feature B" >> features.txt
git add .
git commit -m "feat: add Feature B"
git push
```

### When Conflict Occurs

```bash
# Attempt push - rejected
git push
# Output: rejected... fetch first

# Solution: pull first
git pull

# If conflict exists, resolve then
git add .
git commit -m "merge: resolve conflicts"
git push
```

---

## Command Summary

| Command | Description |
|--------|------|
| `git remote -v` | View remote repository |
| `git remote add origin URL` | Add remote repository |
| `git clone URL` | Clone repository |
| `git push origin branch` | Local → remote |
| `git push -u origin branch` | Push + set upstream |
| `git pull` | Remote → local (fetch + merge) |
| `git fetch` | Download remote changes only |

---

## Next Steps

Let's learn collaboration methods using Fork, Pull Requests, and Issues in [05_GitHub_Collaboration.md](./05_GitHub_Collaboration.md)!
