# Git Basics

## 1. What is Git?

Git is a **Distributed Version Control System (DVCS)**. It tracks changes to files and enables multiple people to collaborate.

### Why Use Git?

- **Version Control**: Save all change history of files
- **Backup**: Store code safely
- **Collaboration**: Multiple people can work simultaneously
- **Experimentation**: Test new features safely

### Git vs GitHub

| Git | GitHub |
|-----|--------|
| Version control **tool** | Git repository **hosting service** |
| Works locally | Online platform |
| Used via command line | Provides web interface |

---

## 2. Installing Git

### macOS

```bash
# Install with Homebrew
brew install git

# Or install via Xcode Command Line Tools
xcode-select --install
```

### Windows

Download and install from [Git official website](https://git-scm.com/download/win)

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install git
```

### Verify Installation

```bash
git --version
# Output example: git version 2.43.0
```

---

## 3. Initial Git Setup

You need to configure your user information when using Git for the first time.

### Set Username and Email

```bash
# Set username
git config --global user.name "John Doe"

# Set email
git config --global user.email "john@example.com"
```

### Verify Configuration

```bash
# View all settings
git config --list

# Check specific settings
git config user.name
git config user.email
```

### Set Default Editor (Optional)

```bash
# Set VS Code as default editor
git config --global core.editor "code --wait"

# Use Vim
git config --global core.editor "vim"
```

---

## 4. Creating a Git Repository

### Method 1: Initialize a New Repository

```bash
# Create project folder
mkdir my-project
cd my-project

# Initialize Git repository
git init
```

Output:
```
Initialized empty Git repository in /path/to/my-project/.git/
```

### Method 2: Clone an Existing Repository

```bash
# Clone repository from GitHub
git clone https://github.com/username/repository.git
```

---

## 5. Git's Three Areas

Git manages files in three areas:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Working        │    │  Staging        │    │  Repository     │
│  Directory      │───▶│  Area           │───▶│  (.git)         │
│  (Work space)   │    │  (Staging)      │    │  (Repository)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
      ↑                      ↑                      ↑
   Edit files            git add               git commit
```

1. **Working Directory**: The space where you actually modify files
2. **Staging Area**: The space where files to be committed are gathered
3. **Repository**: The space where committed snapshots are stored

---

## Practice Examples

### Example 1: Create Your First Repository

```bash
# 1. Create and navigate to practice folder
mkdir git-practice
cd git-practice

# 2. Initialize Git repository
git init

# 3. Create file
echo "# My First Git Project" > README.md

# 4. Check status
git status
```

Expected output:
```
On branch main

No commits yet

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	README.md

nothing added to commit but untracked files present (use "git add" to track)
```

### Example 2: Check Configuration

```bash
# View current Git configuration
git config --list --show-origin
```

---

## Key Summary

| Concept | Description |
|------|------|
| `git init` | Initialize new Git repository |
| `git clone` | Clone remote repository |
| `git config` | Modify Git configuration |
| Working Directory | Space for modifying files |
| Staging Area | Space for commit queue |
| Repository | Space for storing change history |

---

## Next Steps

Let's learn basic commands like `add`, `commit`, `status`, `log` in [02_Basic_Commands.md](./02_Basic_Commands.md)!
