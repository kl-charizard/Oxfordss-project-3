# Minimal WeChat Mini Program

This project is a clean scaffold with a single page to start building from scratch.

## Run
- Open this folder in WeChat Developer Tools
- Make sure the AppID in `project.config.json` matches your account (or use test number)
- Click Compile/Preview

## Structure
- `app.json`, `app.js`, `app.wxss`
- `pages/index/` with `index.wxml`, `index.wxss`, `index.js`, `index.json`

## Large model file not included

The file `Vocabbuddy/models/word_embeddings.npy` (~154MB) is not included in this repository because it exceeds GitHub's 100MB file size limit.

- To run features that depend on this file, place it locally at `Vocabbuddy/models/word_embeddings.npy`.
- Alternatively, add it via Git LFS and push using LFS.
