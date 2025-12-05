# Setting Up Git LFS for Large Data Files

Your data files are 5.7GB total, which exceeds GitHub's 100MB file size limit. To push them, you need Git LFS (Large File Storage).

## ⚠️ Important Limitations

- **GitHub Free Tier**: 1GB storage + 1GB bandwidth/month
- **Your Data**: 5.7GB (exceeds free tier)
- **Solution Options**:
  1. Use Git LFS with paid GitHub account ($4/month for 50GB)
  2. Host data files elsewhere (Google Drive, Dropbox, etc.) and link in README
  3. Use a subset of the data for the repository

## Installation & Setup

### Step 1: Install Git LFS

**On Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install git-lfs
```

**On macOS:**
```bash
brew install git-lfs
```

**On Windows:**
Download from: https://git-lfs.github.com/

### Step 2: Initialize Git LFS in your repository

```bash
cd /home/beys/ElYazısıTanıma
git lfs install
```

### Step 3: Track CSV files with Git LFS

```bash
git lfs track "data/*.csv"
git lfs track "data/emnist_source_files/**"
```

This creates a `.gitattributes` file.

### Step 4: Add and commit

```bash
git add .gitattributes
git add .gitignore
git add data/*.csv
git add data/emnist_source_files/
git commit -m "Add EMNIST dataset files using Git LFS"
git push origin main
```

## Alternative: Host Data Elsewhere

If you don't want to use Git LFS, you can:

1. Upload data files to Google Drive / Dropbox / OneDrive
2. Create a shareable link
3. Update README.md with download instructions
4. Users download the data separately

This is often better for very large datasets!

